import re
import numpy as np
import random
import os
import subprocess
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import sklearn
import warnings
import json

from modeling.seqtagger import SequenceTagger
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from allennlp.training.learning_rate_schedulers import CosineWithRestarts, SlantedTriangular
from seqeval.metrics import f1_score as ner_f1, precision_score as ner_prec, recall_score as ner_rec, accuracy_score as ner_acc
from seqeval.metrics import classification_report as ner_classification_report


def flatten(elems):
    return [e for elem in elems for e in elem]

def get_optimizer(model, args):
    optim_args = args.training.optimizer

    if args.training.lr_scheduler.name == 'slanted_triangular':
        print('[LOG] Using grouped parameters for STLR scheduler')
        params = model.get_param_groups()
    else:
        params = model.parameters() #; print('USING model.parameters()')

    # params = list(filter(lambda p: p.requires_grad, params))

    if optim_args.name == "sgd":
        optimizer = optim.SGD(params,
                              lr=optim_args.lr,
                              momentum=optim_args.momentum,
                              weight_decay=optim_args.weight_decay)
    elif optim_args.name == "asgd":
        optimizer = optim.ASGD(params,
                               lr=optim_args.lr,
                               weight_decay=optim_args.weight_decay)
    elif optim_args.name == "adam":
        optimizer = optim.Adam(params,
                               lr=optim_args.lr,
                               weight_decay=optim_args.weight_decay,
                               betas=(optim_args.beta1, optim_args.beta2))
    else:
        raise Exception("Opimizer '{}' not found".format(optim_args.name))

    return optimizer


def get_lr_scheduler(optimizer, train_size, args):
    lrs_args = args.training.lr_scheduler
    if lrs_args.name == "cos":
        scheduler = CosineWithRestarts(optimizer, t_initial=lrs_args.t_initial, t_mul=lrs_args.t_mul)  # t_initial=10,  t_mul=2)

    elif lrs_args.name == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lrs_args.step_size) # step_size=15)

    elif lrs_args.name == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_args.factor, patience=lrs_args.patience) # factor=0.3, patience=5)

    elif lrs_args.name == "plateau_f1":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_args.factor, patience=lrs_args.patience, mode='max')

    elif lrs_args.name == 'slanted_triangular':
        stepsperepoch = get_steps_per_epoch(train_size, args.training.batch_size)
        scheduler = SlantedTriangular(optimizer,
                                      num_epochs=args.training.epochs,
                                      num_steps_per_epoch=stepsperepoch,
                                      gradual_unfreezing=lrs_args.gradual_unfreezing,
                                      discriminative_fine_tuning=lrs_args.discriminative_fine_tuning)
    elif lrs_args.name == "none":
        scheduler = None
    else:
        raise Exception("Scheduler '{}' not found".format(choice))
    return scheduler


def get_steps_per_epoch(dataset_size, batch_size):
    if dataset_size % batch_size == 0:
        return dataset_size // batch_size
    else:
        return dataset_size // batch_size + 1


def save_model(filename, state):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def try_load_model(filename, model, optimizer=None, trainer=None, scheduler=None, verbose=True):
    if os.path.exists(filename):
        print(f"[LOG] Loading model from {filename}")
        state = torch.load(filename, map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

        if trainer is not None:
            trainer.best_f1 = state['f1']
            trainer.best_loss = state['loss']
            trainer.starting_epoch = state['epoch'] + 1

        model.load_state_dict(state['model'])

        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])

        if scheduler is not None and 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])

        if verbose:
            print("[LOG] Loading model... Epoch: {:03d}, F1: {:.5f}, Loss: {:.5f}".format(
                state['epoch'], state['f1'], state['loss']))
        return True
    else:
        if verbose:
            print("[LOG] No previous model found")
        return False


def load_model_only(filename, model):
    if os.path.exists(filename):
        state = torch.load(filename, map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(state['model'])
        print("[LOG] Loading LID model... Epoch: {:03d}, F1: {:.5f}, Loss: {:.5f}".format(state['epoch'], state['f1'], state['loss']))


def require_grad(parameters, required=True):
    for p in parameters:
        p.requires_grad = required

def choose_model(args, n_classes):
    if args.name == 'elmo':
        model = SequenceTagger(n_classes=n_classes,
                               word_hidden_dim=args.lstm_hidden_dim,
                               use_lstm=args.use_lstm,
                               use_position=args.charngrams.use_position,
                               use_second_task=args.charngrams.use_second_task,
                               charngram_mechanism=args.charngrams.mechanism,
                               ngram_order=args.charngrams.ngram_order,
                               dropout=args.dropout,
                               embeddings=args.embeddings,
                               use_ngram_vectors=args.charngrams.use_at_last_layer,
                               elmo_requires_grad=args.elmo_requires_grad,
                               elmo_version=args.version,
                               use_crf=args.use_crf)
    else:
        raise Exception('Unknown model: {}'.format(args.name))
    return model

def get_pretrained_model_architecture(config_path, include_pretraining=True):
    from src.data.dataset import CSDataset
    from src.modeling.main import Arguments

    args = Arguments(config_path=config_path)
    n_langids = len(set(flatten(CSDataset(args.dataset.train).langids)))

    model = SequenceTagger(n_classes=n_langids,
                           word_hidden_dim=args.model.lstm_hidden_dim,
                           use_lstm=args.model.use_lstm,
                           use_position=args.model.charngrams.use_position,
                           use_second_task=args.model.charngrams.use_second_task,
                           charngram_mechanism=args.model.charngrams.mechanism,
                           ngram_order=args.model.charngrams.ngram_order,
                           dropout=args.model.dropout,
                           embeddings=args.model.embeddings,
                           use_ngram_vectors=args.model.charngrams.use_at_last_layer,
                           elmo_requires_grad=args.model.elmo_requires_grad,
                           elmo_version=args.model.version,
                           use_crf=args.model.use_crf)

    if include_pretraining:
        bestpath = os.path.join(args.checkpoints, f'{args.experiment}.bestloss.pt')
        if os.path.exists(bestpath):
            load_model_only(bestpath, model)
            print(f"[LOG] Successfully loaded the CS-adapted model from {bestpath}")
        else:
            raise Exception(f"[ERROR] There is not checkpoint at '{bestpath}'")
    else:
        print("[LOG] Returning model without CS pretraining (only ELMo pretrained knowledge)")
    return model


def save_history(history, args):
    hist_path = '{}/'.format(args.experiment)
    hist_file = '{}.history.txt'.format(args.experiment)

    os.makedirs(os.path.join(args.history, hist_path), exist_ok=True)

    with open(os.path.join(args.history, hist_path, hist_file), 'a+') as fp:
        for i in range(len(history['train']['f1'])):
            train = '{}\t{}\t'.format(history['train']['loss'][i], history['train']['f1'][i])
            valid = '{}\t{}\t'.format(history['dev']['loss'][i], history['dev']['f1'][i])
            fp.write(train + valid)


def save_json(filename, content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w+') as fp:
        json.dump(content, fp)


def sklearn_fmeasure(gold, pred, verbose, average_choice='weighted'):
    y_true = flatten(gold)
    y_pred = flatten(pred)

    assert len(y_true) == len(y_pred)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
        prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, average=average_choice)
        acc = accuracy_score(y_true, y_pred)

        if verbose:
            report = classification_report(y_true, y_pred, digits=5)
            print(report)
            print('[LOG] Accuracy: {:.5f}'.format(acc))

    return acc, prec, rec, f1

def save_predictions(filename, words, golds, preds):
    with open(filename, 'w+') as fp:
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                line = '{}\t{}\t{}\n'.format(words[i][j], golds[i][j], preds[i][j])
                fp.write(line)
            fp.write('\n')

def running_fmeasure(gold, pred, verbose=False):
    if verbose:
        print(ner_classification_report(gold, pred, digits=5))

    f1 = ner_f1(gold, pred, average='micro')
    prec = ner_prec(gold, pred, average='micro')
    rec = ner_rec(gold, pred, average='micro')
    acc = ner_acc(gold, pred)

    return acc, prec, rec, f1


def get_character_ngrams(characters, n_order):
    charlen = len(characters)
    assert 1 <= n_order <= charlen and isinstance(n_order, int)
    ngrams = []
    for i in range(charlen - n_order + 1):
        ngrams.append(characters[i:i+n_order])
    return ngrams
