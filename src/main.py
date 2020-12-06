import os
import re
import json
import argparse
import random
import numpy as np
import torch

import experiments.experiment_langid as experiment_lid
import experiments.experiment_ner as experiment_ner
import experiments.experiment_pos as experiment_pos

from types import SimpleNamespace as Namespace

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Arguments(object):
    def __init__(self, config_path=None):
        if config_path is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', help="provide a relative path to a JSON config file from the configs directory")
            parser.add_argument('--mode', choices=['train', 'eval'], default='train', help="Specify whether to train or evaluate the model")
            parser.add_argument('--gpu', type=int, default=-1, help="The GPU label number. By default, the code runs on CPU")
            parser.add_argument('--seed', type=int, default=42)

            args = parser.parse_args()

            # Fields expected from the command line
            self.config = os.path.join(PROJ_DIR, args.config)
            self.mode = args.mode
            self.gpu = args.gpu
            self.seed = args.seed
        else:
            self.gpu = -1
            self.mode = 'eval'
            self.seed = 42
            self.config = os.path.join(PROJ_DIR, config_path)

        assert os.path.exists(self.config) and self.config.endswith('.json'), f'Bad config path: {self.config}'

        # Read the parameters from the JSON file and skip comments
        with open(self.config, 'r') as f:
            params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])

        arguments = json.loads(params, object_hook=lambda d: Namespace(**d))

        # Must-have fields expected from the JSON config file
        self.experiment = arguments.experiment
        self.description = arguments.description
        self.task = arguments.task
        self.dataset = arguments.dataset
        self.model = arguments.model
        self.training = arguments.training
        self.evaluation = arguments.evaluation

        # Checking that the JSON contains at least the fixed fields
        assert all([hasattr(self.dataset, name) for name in {'train', 'dev', 'test'}])
        assert all([hasattr(self.model, name) for name in {'name'}])
        assert all([hasattr(self.training, name) for name in {'epochs', 'batch_size', 'optimizer', 'lr_scheduler', 'l2', 'clip_grad'}])
        assert all([hasattr(self.training.optimizer, name) for name in {'name', 'lr'}])
        assert all([hasattr(self.training.lr_scheduler, name) for name in {'name'}])
        assert all([hasattr(self.evaluation, name) for name in {'batch_size'}])

        self._format_datapaths()
        self._add_extra_fields()
        self._add_transfer_learning_fields(arguments)

    def _add_transfer_learning_fields(self, args):
        if hasattr(args, "pretrained_config"):
            self.pretrained_config = args.pretrained_config
        if hasattr(args, "transfer_mode"):
            self.transfer_mode = args.transfer_mode
        if hasattr(args, "restore_model"):
            self.restore_model = args.restore_model

    def _format_datapaths(self):
        self.dataset.train = os.path.join(PROJ_DIR, 'data', self.dataset.train)
        self.dataset.dev   = os.path.join(PROJ_DIR, 'data', self.dataset.dev)
        self.dataset.test  = os.path.join(PROJ_DIR, 'data', self.dataset.test)

    def _add_extra_fields(self):
        self.checkpoints = os.path.join(PROJ_DIR, 'checkpoints', self.experiment)
        self.figures     = os.path.join(PROJ_DIR, 'reports/figures', self.experiment)
        self.history     = os.path.join(PROJ_DIR, 'reports/history', self.experiment)
        self.predictions = os.path.join(PROJ_DIR, 'reports/predictions', self.experiment)
        self.attentions  = os.path.join(PROJ_DIR, 'reports/attentions', self.experiment)


def main():
    args = Arguments()

    if torch.cuda.is_available() and args.gpu >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True

        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
    else:
        args.device = torch.device("cpu")

    print("[LOG] {}".format('=' * 40))
    print("[LOG] {: >15}: '{}'".format("Experiment ID", args.experiment))
    print("[LOG] {: >15}: '{}'".format("Description", args.description))
    print("[LOG] {: >15}: '{}'".format("Task", args.task.upper()))
    for key, val in vars(args.dataset).items():
        print("[LOG] {: >15}: {}".format(key, val))
    print("[LOG] {: >15}: '{}'".format("Modeling", vars(args.model)))
    print("[LOG] {: >15}: '{}'".format("Training", vars(args.training)))
    print("[LOG] {: >15}: '{}'".format("Evaluation", vars(args.evaluation)))
    print("[LOG] {: >15}: '{}'".format("Device", args.device))
    print("[LOG] {}".format('=' * 40))

    if   args.task.startswith('lid'): experiment_lid.main(args)
    elif args.task.startswith('ner'): experiment_ner.main(args)
    elif args.task.startswith('pos'): experiment_pos.main(args)
    else: raise Exception('Unexpected task: {}'.format(args.task))


if __name__ == '__main__':
    main()
