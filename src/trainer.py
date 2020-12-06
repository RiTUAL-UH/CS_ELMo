import os
import time
import torch
import torch.nn as nn
import copy
import numpy as np
import utilities as utils

from utilities import running_fmeasure, sklearn_fmeasure
from utilities import save_model
from torch.utils.data import DataLoader


def to_tensor(labels, label2index, pad_value=0, return_mask=False, device='cpu'):
    maxlen = max(map(len, labels))
    target = torch.zeros(len(labels), maxlen).long() + pad_value
    mask   = torch.zeros(len(labels), maxlen).byte()

    for i, labels_i in enumerate(labels):
        sample = [label2index[label] for label in labels_i]
        target[i, :len(sample)] = torch.tensor(sample, dtype=torch.long)
        mask[i, :len(sample)] = 1

    target = target.to(device)
    mask = mask.to(device)

    return (target, mask) if return_mask else target

class Stats(object):
    def __init__(self, index2label):
        self.loss_total = 0
        self.loss_factor = 0
        self.preds = []
        self.golds = []
        self.i2l = index2label

    def update(self, scores, target, mask, loss, loss_factor=1):
        self.loss_total += loss
        self.loss_factor += loss_factor

        preds = torch.max(scores, 2)[1]

        lenghts = mask.cpu().long().sum(1)
        for i in range(len(scores)):
            self.preds.append(preds[i, :lenghts[i]].cpu().tolist())
            self.golds.append(target[i, :lenghts[i]].cpu().tolist())

    def loss(self):
        return self.loss_total / self.loss_factor

    def metrics(self, task, verbose=False):
        decoded_preds = [[self.i2l[self.preds[i][j]] for j in range(len(self.preds[i]))] for i in range(len(self.preds))]
        decoded_golds = [[self.i2l[self.golds[i][j]] for j in range(len(self.golds[i]))] for i in range(len(self.golds))]

        if task.startswith('lid'): return sklearn_fmeasure(decoded_golds, decoded_preds, verbose=verbose, average_choice='weighted')
        if task.startswith('ner'): return running_fmeasure(decoded_golds, decoded_preds, verbose=verbose)
        if task.startswith('pos'): return sklearn_fmeasure(decoded_golds, decoded_preds, verbose=verbose, average_choice='micro')

        raise Exception('Unknown task: {}'.format(task))

def collate(batch):
    collated = {'tokens': [], 'langids': [], 'simplified': []}
    for sample in batch:
        collated['tokens'].append(sample['tokens'])
        collated['langids'].append(sample['langids'])
        collated['simplified'].append(sample['simplified'])
        if 'entities' in sample:
            if 'entities' in collated:
                collated['entities'].append(sample['entities'])
            else:
                collated['entities'] = [sample['entities']]
    return collated

class Trainer(object):
    def __init__(self, datasets, args):
        self.args = args

        self.index2langid = dict(enumerate(sorted({l for d in datasets.values() for l in utils.flatten(d.langids)})))
        self.index2simplified = dict(enumerate(sorted({l for d in datasets.values() for l in utils.flatten(d.simplified)})))

        self.langid2index = {value: key for key, value in self.index2langid.items()}
        self.simplified2index = {value: key for key, value in self.index2simplified.items()}

        self.starting_epoch = 0
        self.best_f1 = -1
        self.best_loss = np.Inf

        self.best_f1_state = None
        self.best_loss_state = None

        # TODO: try different weights when using MTL
        self.alpha = 1
        self.beta = 1

        self.dataloaders = dict()
        for dataset in datasets.keys():
            batch_size = args.training.batch_size if dataset == 'train' else args.evaluation.batch_size

            self.dataloaders[dataset] = DataLoader(datasets[dataset],
                                                   batch_size=batch_size,
                                                   shuffle=dataset == 'train',
                                                   collate_fn=collate,
                                                   num_workers=1)

    def scheduler_step(self, scheduler, optimizer, epoch, dev_loss, dev_f1):
        lr_changed = False
        for param_group in optimizer.param_groups:
            if not lr_changed and param_group['lr'] != self.args.training.optimizer.lr:
                lr_changed = True
            self.args.training.optimizer.lr = param_group['lr']  # Acknowledge the recent top lr

        if self.args.training.lr_scheduler.name == 'plateau':
            scheduler.step(dev_loss)
        elif self.args.training.lr_scheduler.name == 'plateau_f1':
            scheduler.step(dev_f1)
        elif self.args.training.lr_scheduler.name == 'slanted_triangular':
            scheduler.step(epoch=epoch)
            lr_changed = False  # for STLR, we don't restore the model because lr change every backprop iteration
        else:
            scheduler.step()

        return lr_changed

    def calculate_loss(self, result):
        loss = result['word_output']['loss']
        if self.args.model.charngrams.use_second_task:
            loss = loss * self.beta + result['char_output']['loss'] * self.alpha
        return loss

    def train(self, model, optimizer, scheduler=None):
        print("[LOG] Training model...")

        ehist = {}
        for epoch in range(self.starting_epoch, self.starting_epoch + self.args.training.epochs):
            stats = {}
            epoch_msg = 'Epoch {:04d}'.format(epoch)

            for dataset in ['train', 'dev']:
                if dataset == 'train':
                        model.train()
                        model.zero_grad()
                else:   model.eval()

                ehist[dataset] = {'loss': [], 'f1': []}
                stats[dataset] = Stats(self.index2langid)

                # ================================================================================================
                epoch_time = time.time()
                for batch in self.dataloaders[dataset]:
                    batch['langids']  = to_tensor(batch['langids'], self.langid2index, device=self.args.device)
                    batch['simplified'] = to_tensor(batch['simplified'], self.simplified2index, device=self.args.device)

                    result = model(batch)

                    loss = self.calculate_loss(result)
                    ntoks = torch.sum(result['mask']).item()
                    loss /= ntoks

                    # L2 regularization
                    if self.args.training.l2 > 0:
                        loss = loss + model.get_l2_reg(self.args.training.l2)

                    if dataset == 'train':
                        loss.backward()

                        # Clipping the norm ||g|| of gradient g before the optmizer's step
                        if self.args.training.clip_grad > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), self.args.training.clip_grad)

                        optimizer.step()
                        optimizer.zero_grad()
                        model.zero_grad()

                        if self.args.training.lr_scheduler.name == 'slanted_triangular':
                            for param_group in optimizer.param_groups:
                                self.args.training.optimizer.lr = param_group['lr']  # Acknowledge the most recent top lr
                            scheduler.step_batch()

                    stats[dataset].update(result['word_output']['logits'], batch['langids'], result['mask'].float(), loss.item() * ntoks, ntoks)
                epoch_time = time.time() - epoch_time
                # ================================================================================================

                epoch_acc, _, _, epoch_f1 = stats[dataset].metrics(task=self.args.task)
                epoch_loss = stats[dataset].loss()

                ehist[dataset]['loss'].append(epoch_loss)
                ehist[dataset]['f1'].append(epoch_f1)

                if dataset == 'train':
                      epoch_msg += '| [{}] Loss: {:.5f}, F1: {:.5f}, Time: {:6.2f}s'.format(dataset.upper(), epoch_loss, epoch_f1, epoch_time)
                else: epoch_msg += '| [{}] Loss: {:.5f}, F1: {:.5f}, Acc: {:.5f}'.format(dataset.upper(), epoch_loss, epoch_f1, epoch_acc)

            lr_changed = False
            if scheduler is not None:
                lr_changed = self.scheduler_step(scheduler, optimizer, epoch, stats['dev'].loss(), ehist['dev']['f1'][-1])

            epoch_msg += "| LR: {:.9f}".format(self.args.training.optimizer.lr)
            epoch_msg += self.track_best_model(model, optimizer, scheduler, ehist['dev']['f1'][-1], ehist['dev']['loss'][-1], epoch, lr_changed)

            print("[LOG] {}".format(epoch_msg))

        return ehist

    def predict(self, model, dataset):
        print("[LOG] ============================================")
        print("[LOG] {} PREDICTIONS".format(dataset.upper()))
        print("[LOG] ============================================")

        model.eval()

        results = {
            'stats': Stats(self.index2langid),
            'preds': [],
            'ngram': {'sentences': []},
        }

        for batch in self.dataloaders[dataset]:
            batch_langids = batch['langids']

            batch['langids'] = to_tensor(batch['langids'], self.langid2index, device=self.args.device)
            batch['simplified'] = to_tensor(batch['simplified'], self.simplified2index, device=self.args.device)

            result = model(batch)

            loss = self.calculate_loss(result).item()
            ntoks = torch.sum(result['mask']).item()

            results['stats'].update(result['word_output']['logits'], batch['langids'], result['mask'].float(), loss, ntoks)

            for sent_ix in range(len(result['word_output']['tags'])):
                length = result['mask'][sent_ix].sum().item()
                labels = list(map(self.index2langid.get, result['word_output']['tags'][sent_ix][:length]))
                results['preds'].append(labels)

                if 'char_output' in result and ('lo_attention' in result['char_output'] or 'hi_attention' in result['char_output']):
                    sentence = []

                    assert len(batch['tokens'][sent_ix]) == len(labels)

                    for word_ix in range(len(batch['tokens'][sent_ix])):
                        charmeta_for_word = {
                            'word': batch['tokens'][sent_ix][word_ix],
                            'label': batch_langids[sent_ix][word_ix],
                            'pred': labels[word_ix]
                        }

                        if 'lo_attention' in result['char_output']:
                            charlen = len(charmeta_for_word['word'])

                            for ngram_order in range(4):
                                if ngram_order >= charlen:
                                    break

                                char_ngram_att = result['char_output']['lo_attention'][ngram_order][sent_ix][word_ix, :charlen - ngram_order]
                                charmeta_for_word[f'char_{ngram_order+1}gram_attentions'] = char_ngram_att.tolist()

                        if 'hi_attention' in result['char_output']:
                            cross_ngram_att = result['char_output']['hi_attention'][sent_ix, word_ix]
                            # assert sum(cross_ngram_att) > 1.0 - 1e-6

                            charmeta_for_word['char_nrgam'] = cross_ngram_att.tolist()

                        sentence.append(charmeta_for_word)
                    results['ngram']['sentences'].append(sentence)

        acc, _, _, f1 = results['stats'].metrics(task=self.args.task, verbose=True)

        print("[LOG] Loss: {:.5f}, F1: {:.5f}, Acc: {:.5f}".format(results['stats'].loss(), f1, acc))

        return results


    def track_best_model(self, model, optimizer, scheduler, val_f1, val_loss, epoch, lr_changed):
        message = ""
        loss_improved = self.best_loss > val_loss
        f1_improved = self.best_f1 < val_f1

        if not loss_improved and not f1_improved:
            if lr_changed:
                chckpt_fullpath = os.path.join(self.args.checkpoints, '{}.bestf1.pt'.format(self.args.experiment))
                utils.try_load_model(chckpt_fullpath, model, verbose=False)
                return "| Restoring model"
            else:
                return message

        state = {
            'epoch': epoch,
            'f1': val_f1,
            'loss': val_loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()

        if f1_improved:
            message = "F1 improved"
            self.best_f1 = val_f1

            chckpt_fullpath = os.path.join(self.args.checkpoints, '{}.bestf1.pt'.format(self.args.experiment))
            save_model(chckpt_fullpath, state)

        if loss_improved:
            message = "Loss improved"
            self.best_loss = val_loss

            chckpt_fullpath = os.path.join(self.args.checkpoints, '{}.bestloss.pt'.format(self.args.experiment))
            save_model(chckpt_fullpath, state)

        if loss_improved and f1_improved:
            message = "F1 & Loss improved"

        return "| {}...".format(message)
