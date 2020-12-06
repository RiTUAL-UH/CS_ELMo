import os
import utilities as utils

from collections import Counter
from modeling.seqtagger import SequenceTagger
from trainer import Trainer
from dataset import CSDataset


def main(args):
    print("[LOG] LANGUAGE IDENTIFICATION EXPERIMENT")
    print("[LOG] {}".format('=' * 40))

    datasets = {
        'train': CSDataset(args.dataset.train),
        'dev': CSDataset(args.dataset.dev),
        'test': CSDataset(args.dataset.test)
    }

    print("[LOG] CORPUS SAMPLES:")
    print("[LOG] Train -> Posts: {:5,} Tokens: {:7,}".format(len(datasets['train']), len(utils.flatten(datasets['train'].tokens))))
    print("[LOG] Dev   -> Posts: {:5,} Tokens: {:7,}".format(len(datasets['dev']), len(utils.flatten(datasets['dev'].tokens))))
    print("[LOG] Test  -> Posts: {:5,} Tokens: {:7,}".format(len(datasets['test']), len(utils.flatten(datasets['test'].tokens))))
    print("[LOG] {}".format('=' * 40))

    print("[LOG] CORPUS LID DISTRIBUTION")
    print("[LOG] Train -> {}".format(Counter(utils.flatten(datasets['train'].langids)).most_common()))
    print("[LOG] Dev   -> {}".format(Counter(utils.flatten(datasets['dev'].langids)).most_common()))
    print("[LOG] Test  -> {}".format(Counter(utils.flatten(datasets['test'].langids)).most_common()))
    print("[LOG] {}".format('=' * 40))

    n_langids = len(set(utils.flatten(datasets['train'].langids + datasets['dev'].langids + datasets['test'].langids)))
    n_simplified = len(set(utils.flatten(datasets['train'].simplified + datasets['dev'].simplified + datasets['test'].simplified)))
    n_entities = len(set(utils.flatten(datasets['train'].entities + datasets['dev'].entities + datasets['test'].entities)))
    n_postags = len(set(utils.flatten(datasets['train'].postags + datasets['dev'].postags + datasets['test'].postags)))

    print("[LOG] CORPUS LABELS")
    print("[LOG]    LangID classes:", n_langids)
    print("[LOG] SimpleLID classes:", n_simplified)
    print("[LOG]       NER classes:", n_entities)
    print("[LOG]       POS classes:", n_postags)
    print("[LOG] {}".format('=' * 40))

    def get_charset(tokens):
        return set(utils.flatten(utils.flatten(tokens)))

    corpus_charset = get_charset(datasets['train'].tokens) & \
                     get_charset(datasets['dev'].tokens) & \
                     get_charset(datasets['test'].tokens)

    print("[LOG] Charset length:", len(corpus_charset))
    print("[LOG] {}".format('=' * 40))

    model = utils.choose_model(args.model, n_classes=n_langids)

    print("[LOG] {}".format(model))
    print("[LOG] {}".format('=' * 40))

    model.to(args.device)

    trainer = Trainer(datasets, args)
    optimizer = utils.get_optimizer(model, args)
    scheduler = utils.get_lr_scheduler(optimizer, len(datasets['train']), args)
    bestpath = os.path.join(args.checkpoints, f'{args.experiment}.bestf1.pt')

    if args.mode == 'train':
        if os.path.exists(bestpath):
            option = input("[LOG] Found a checkpoint! Choose an option:\n"
                           "\t0) Train from scratch and override the previous checkpoint\n"
                           "\t1) Load the checkpoint and train from there\nYour choice: ")

            assert option in {"0", "1"}, "Unexpected choice"

            if option == "1":
                utils.try_load_model(bestpath, model, optimizer, trainer, scheduler)

        history = trainer.train(model, optimizer, scheduler)
        utils.save_history(history, args)

    utils.try_load_model(bestpath, model, optimizer, trainer, scheduler)

    dev_output = trainer.predict(model, 'dev')
    tst_output = trainer.predict(model, 'test')

    # Saving attention
    utils.save_json(os.path.join(args.attentions, f'{args.experiment}.dev.json'), dev_output['ngram'])
    utils.save_json(os.path.join(args.attentions, f'{args.experiment}.test.json'), tst_output['ngram'])

    # Save predictions
    datasets['dev'].save(os.path.join(args.predictions, f'{args.experiment}.dev.txt'), dev_output['preds'])
    datasets['test'].save(os.path.join(args.predictions, f'{args.experiment}.test.txt'), tst_output['preds'])



