import os
import re
import json
import argparse

import globals as glb
import globals as glb
import experiments.experiment_langid as experiment_lid
import experiments.experiment_ner as experiment_ner
import experiments.experiment_pos as experiment_pos

from types import SimpleNamespace as Namespace


class Arguments(object):
    def __init__(self, config_path=None):
        if config_path is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config')
            parser.add_argument('--mode', choices=['train', 'eval'], default='train')
            parser.add_argument('--gpu', type=int, default=0)
            parser.add_argument('--replicable', action='store_true')

            args = parser.parse_args()

            # Fields expected from the command line
            self.config = os.path.join(glb.PROJ_DIR, args.config)
            self.mode = args.mode
            self.gpu = args.gpu
            self.replicable = args.replicable
        else:
            self.gpu = -1
            self.mode = 'eval'
            self.replicable = False
            self.config = os.path.join(glb.PROJ_DIR, config_path)

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
        self.dataset.train = os.path.join(glb.DATA_DIR, self.dataset.train)
        self.dataset.dev = os.path.join(glb.DATA_DIR, self.dataset.dev)
        self.dataset.test = os.path.join(glb.DATA_DIR, self.dataset.test)

    def _add_extra_fields(self):
        self.checkpoints = os.path.join(glb.CHECKPOINT_DIR, self.experiment)
        self.figures = os.path.join(glb.FIGURE_DIR, self.experiment)
        self.history = os.path.join(glb.HISTORY_DIR, self.experiment)
        self.predictions = os.path.join(glb.PREDICTIONS_DIR, self.experiment)
        self.attentions = os.path.join(glb.ATTENTIONS_DIR, self.experiment)


def main():
    args = Arguments()

    import numpy as np
    import random
    import torch

    if args.replicable:
        seed_num = 123
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)
            torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and args.gpu >= 0:
        glb.DEVICE = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(glb.DEVICE)
    else:
        glb.DEVICE = torch.device("cpu")

    print("[LOG] {}".format('=' * 40))
    print("[LOG] {: >15}: '{}'".format("Experiment ID", args.experiment))
    print("[LOG] {: >15}: '{}'".format("Description", args.description))
    print("[LOG] {: >15}: '{}'".format("Task", args.task.upper()))
    for key, val in vars(args.dataset).items():
        print("[LOG] {: >15}: {}".format(key, val))
    print("[LOG] {: >15}: '{}'".format("Modeling", vars(args.model)))
    print("[LOG] {: >15}: '{}'".format("Training", vars(args.training)))
    print("[LOG] {: >15}: '{}'".format("Evaluation", vars(args.evaluation)))
    print("[LOG] {: >15}: '{}'".format("Device", glb.DEVICE))
    print("[LOG] {}".format('=' * 40))


    if args.task.startswith('lid'):
        experiment_lid.main(args)

    elif args.task.startswith('ner'):
        experiment_ner.main(args)

    elif args.task.startswith('pos'):
        experiment_pos.main(args)

    else:
        raise Exception('Unexpected task: {}'.format(args.task))


if __name__ == '__main__':
    main()
