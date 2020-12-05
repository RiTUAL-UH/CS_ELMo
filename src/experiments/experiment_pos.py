import os
import re
import json
import random
import numpy as np
import torch
import torch.nn as nn
import utilities as utils
import globals as glb

from collections import Counter
from modeling.seqtagger import SequenceTagger
from trainer import Trainer
from dataset import CSDataset
from allennlp.modules import ConditionalRandomField


def main(args):
    # =========================================================================================================
    # THE POS DATASET
    # =========================================================================================================

    print("[LOG] =====================")
    print("[LOG] PART-OF-SPEECH TAGGING EXPERIMENT")

    datasets = {
        'train': CSDataset(args.dataset.train, pos_index=2),
        'dev': CSDataset(args.dataset.dev, pos_index=2),
        'test': CSDataset(args.dataset.test, pos_index=2)
    }

    print("[LOG] =====================")
    print("[LOG] CORPUS SAMPLES")
    print("[LOG] Train -> Posts: {:5,} Tokens: {:7,}".format(len(datasets['train']), len(utils.flatten(datasets['train'].tokens))))
    print("[LOG] Dev   -> Posts: {:5,} Tokens: {:7,}".format(len(datasets['dev']), len(utils.flatten(datasets['dev'].tokens))))
    print("[LOG] Test  -> Posts: {:5,} Tokens: {:7,}".format(len(datasets['test']), len(utils.flatten(datasets['test'].tokens))))

    print("[LOG] =====================")
    print("[LOG] CORPUS LID DISTRIBUTION")
    print("[LOG] Train -> {}".format(Counter(utils.flatten(datasets['train'].langids)).most_common()))
    print("[LOG] Dev   -> {}".format(Counter(utils.flatten(datasets['dev'].langids)).most_common()))
    print("[LOG] Test  -> {}".format(Counter(utils.flatten(datasets['test'].langids)).most_common()))

    print("[LOG] =====================")
    print("[LOG] CORPUS POS DISTRIBUTION")
    print("[LOG] Train -> {}".format(Counter(utils.flatten(datasets['train'].postags)).most_common()))
    print("[LOG] Dev   -> {}".format(Counter(utils.flatten(datasets['dev'].postags)).most_common()))
    print("[LOG] Test  -> {}".format(Counter(utils.flatten(datasets['test'].postags)).most_common()))

    n_langids = len(set(utils.flatten(datasets['train'].langids + datasets['dev'].langids + datasets['test'].langids)))
    n_simplified = len(set(utils.flatten(datasets['train'].simplified + datasets['dev'].simplified + datasets['test'].simplified)))
    n_entities = len(set(utils.flatten(datasets['train'].entities + datasets['dev'].entities + datasets['test'].entities)))
    n_postags = len(set(utils.flatten(datasets['train'].postags + datasets['dev'].postags + datasets['test'].postags)))

    print("[LOG] =====================")
    print("[LOG] CORPUS LABELS")
    print("[LOG] LangID classes:", n_langids)
    print("[LOG] Cluster classes:", n_simplified)
    print("[LOG] Entity classes:", n_entities)
    print("[LOG] POSTag classes:", n_postags)
    print("[LOG] =====================")

    # TODO: improve this temporal hack; for now it makes it compatible with the training pipeline
    datasets['train'].langids_tmp = datasets['train'].langids
    datasets['dev'].langids_tmp = datasets['dev'].langids
    datasets['test'].langids_tmp = datasets['test'].langids

    datasets['train'].langids = datasets['train'].postags
    datasets['dev'].langids = datasets['dev'].postags
    datasets['test'].langids = datasets['test'].postags

    # =========================================================================================================
    # PREPARING THE MODEL
    # =========================================================================================================

    # Load the pretrained model
    if args.pretrained_config is None:
        print("[LOG] No pretrained config was specified. Creating model from main config only...")
        model = utils.choose_model(args.model, n_classes=n_postags)
    else:
        # ================================================================================
        # Building model and choosing parameters

        print(f"[LOG] Model will be built in mode '{args.pretrained_config.pretrained_part}'")

        if args.pretrained_config.pretrained_part == 'full':
            # model = pretrained_model
            # model.proj = nn.Linear(model.output_size, n_postags)
            # model.crf = ConditionalRandomField(n_postags)
            raise NotImplementedError
        elif args.pretrained_config.pretrained_part == 'elmo':
            # Load the pretrained LID model
            print("[LOG] Loading pretrained model...")
            pretrained_model = utils.get_pretrained_model_architecture(args.pretrained_config.path, include_pretraining=True)

            model = utils.choose_model(args.model, n_classes=n_postags)
            model.elmo = pretrained_model.elmo

            del pretrained_model # free memory
        else:
            raise Exception(f"Unknown pretrained part: {args.pretrained_config.pretrained_part}")

        # ================================================================================
        # Fixing parameters according to fine-tuning mode

        print(f"[LOG] Model will be fine-tuned in mode '{args.pretrained_config.finetuning_mode}'")

        if args.pretrained_config.finetuning_mode == 'fully_trainable':

            utils.require_grad(model.parameters(), True)

        elif args.pretrained_config.finetuning_mode == 'frozen_elmo':
            utils.require_grad(model.parameters(), True)
            utils.require_grad(model.elmo.parameters(), False)

        elif args.pretrained_config.finetuning_mode == 'inference':
            # Fixed all the parameters first
            utils.require_grad(model.parameters(), False)

            # Unfreeze the last layers for the POS task
            if hasattr(model, 'proj'): utils.require_grad(model.proj.parameters(), True)
            if hasattr(model, 'crf'): utils.require_grad(model.crf.parameters(), True)
        else:
            raise Exception(f"[ERROR] Unknown finetuning mode: {args.pretrained_config.finetuning_mode}")

    # Move to CUDA if available
    if torch.cuda.is_available():
        model.cuda()

    # =========================================================================================================
    # TRAINING THE MODEL
    # =========================================================================================================

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

    # TODO: save predictions
    # _ = trainer.predict(model, 'train')
    dev_output = trainer.predict(model, 'dev')
    tst_output = trainer.predict(model, 'test')


def require_grad(parameters, required=True):
    for p in parameters:
        p.requires_grad = required

