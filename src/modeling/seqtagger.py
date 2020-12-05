import torch
import torch.nn as nn
import numpy as np
import globals as glb

from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.data import Sentence

from modeling.attention import NgramEnhancer
from modeling.elmo import Elmo, batch_to_ids
from allennlp.modules import ConditionalRandomField

ELMO_SETTINGS = {
    "small": {
        "projection": 128 * 2,
        "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
        "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    },
    "medium": {
        "projection": 256 * 2,
        "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json",
        "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
    },
    "original": {
        "projection": 512 * 2,
        "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
        "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    }
}

class CharNgramTagger(nn.Module):
    def __init__(self, input_size):
        super(CharNgramTagger, self).__init__()

        self.input_size = input_size
        self.n_classes = 3

        self.clf = nn.Linear(self.input_size, self.n_classes)
        self.xent = nn.CrossEntropyLoss(reduction='sum')

    def xentropy_loss(self, logits, target, mask):
        predicted_tags = logits.max(-1)[1].cpu().tolist()
        loss = self.xent(logits[mask == 1], target[mask == 1])

        return {'loss': loss,
                'logits': logits,
                'tags': predicted_tags}

    def forward(self, inputs, labels, mask):
        return self.xentropy_loss(self.clf(inputs), labels, mask)


class SequenceTagger(nn.Module):
    def __init__(self, n_classes, word_hidden_dim,
                 use_lstm=False,
                 use_position=False,
                 use_second_task=False,
                 charngram_mechanism='none',
                 ngram_order=7,
                 dropout=0.4,
                 embeddings=None,
                 use_ngram_vectors=False,
                 elmo_requires_grad=True,
                 use_crf=True,
                 elmo_version='original'):

        super(SequenceTagger, self).__init__()

        # ELMo settings: output size and conv channels
        self.elmo_hidden_size = ELMO_SETTINGS[elmo_version]['projection']
        self.elmo_convolutions = [32, 32, 64, 128, 256, 512, 1024]

        self.word_hidden_size = word_hidden_dim
        self.n_classes = n_classes
        self.use_lstm = use_lstm
        self.use_crf = use_crf
        self.use_char_tagger = use_second_task
        self.use_ngram_vectors = use_ngram_vectors
        self.has_embeddings = embeddings is not None and len(embeddings) > 0
        self.elmo_requires_grad = elmo_requires_grad
        self.charngram_mechanism = charngram_mechanism

        if charngram_mechanism != 'none':
            char_enhancer = NgramEnhancer(variable_dims=self.elmo_convolutions[:ngram_order],
                                          ngram_order=ngram_order,
                                          attention_type=charngram_mechanism,
                                          use_position=use_position)
        else:
            char_enhancer = None

        self.elmo = Elmo(ELMO_SETTINGS[elmo_version]["options"],
                         ELMO_SETTINGS[elmo_version]["weights"],
                         num_output_representations=2,
                         dropout=dropout,
                         requires_grad=self.elmo_requires_grad,
                         char_enhancer=char_enhancer)
        if self.has_embeddings:
            print("[LOG] Stacking embeddings: {}".format(embeddings))
            self.embeddings = StackedEmbeddings([
                FlairEmbeddings(emb.split(":")[-1]) if emb.startswith('flair:') else
                WordEmbeddings(emb)  # 'glove', 'crawl', 'twitter'
                for emb in embeddings
            ])
            self.elmo_hidden_size += self.embeddings.embedding_length

        if self.use_char_tagger:
            self.charclf = CharNgramTagger(sum(self.elmo_convolutions[:ngram_order]))

        if self.use_lstm:
            self.lstm = nn.LSTM(self.elmo_hidden_size, self.word_hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.output_size = self.word_hidden_size + sum(self.elmo_convolutions[:ngram_order]) if self.use_ngram_vectors else self.word_hidden_size
            self.proj = nn.Linear(self.output_size, self.n_classes)
        else:
            self.output_size = self.elmo_hidden_size + sum(self.elmo_convolutions[:ngram_order]) if self.use_ngram_vectors else self.elmo_hidden_size
            self.proj = nn.Linear(self.output_size, self.n_classes)

        self.drop = nn.Dropout(dropout)

        if self.use_crf:
              self.crf = ConditionalRandomField(self.n_classes, constraints=None, include_start_end_transitions=True)
        else: self.xent = nn.CrossEntropyLoss(reduction='sum')


    def get_representations(self, inputs):
        """Get embedding character and word representations for the given input batch"""
        encoded_chars = batch_to_ids(inputs['tokens'])
        encoded_chars = encoded_chars['elmo_tokens']

        batch_size = encoded_chars.size(0)  # number of samples in batch
        sent_length = encoded_chars.size(1) # max number of words in a sentence
        word_length = encoded_chars.size(2) # max number of characters in a word

        # building character length tensor
        char_lenghts = torch.zeros(batch_size, sent_length).long()
        for i in range(len(inputs['tokens'])):
            for j in range(len(inputs['tokens'][i])):
                char_lenghts[i, j] = min(len(inputs['tokens'][i][j]), word_length)

        encoded_chars = encoded_chars.to(glb.DEVICE)
        char_lenghts = char_lenghts.to(glb.DEVICE)

        outputs = self.elmo.forward(encoded_chars, char_lengths=char_lenghts)

        return {
            'char_enhancement': outputs['char_enhancement'],  # convolutions of different kernel sizes from (1,) to (7,)
            'word_syn': outputs['elmo_representations'][0], # first layer has syntactic features
            'word_sem': outputs['elmo_representations'][1], # second layer has semantic features
            'mask': outputs['mask'],
            'char_lengths': char_lenghts.view(batch_size * sent_length).long()
        }

    @staticmethod
    def _get_param_group1():
        param_group = [
            ['_elmo_lstm._token_embedder._char_embedding_weights'],
            ['_elmo_lstm._token_embedder.char_conv_0.weight', '_elmo_lstm._token_embedder.char_conv_0.bias'],
            ['_elmo_lstm._token_embedder.char_conv_1.weight', '_elmo_lstm._token_embedder.char_conv_1.bias'],
            ['_elmo_lstm._token_embedder.char_conv_2.weight', '_elmo_lstm._token_embedder.char_conv_2.bias'],
            ['_elmo_lstm._token_embedder.char_conv_3.weight', '_elmo_lstm._token_embedder.char_conv_3.bias'],
            ['_elmo_lstm._token_embedder.char_conv_4.weight', '_elmo_lstm._token_embedder.char_conv_4.bias'],
            ['_elmo_lstm._token_embedder.char_conv_5.weight', '_elmo_lstm._token_embedder.char_conv_5.bias'],
            ['_elmo_lstm._token_embedder.char_conv_6.weight', '_elmo_lstm._token_embedder.char_conv_6.bias'],
            ['_elmo_lstm._token_embedder._highways._layers.0.weight', '_elmo_lstm._token_embedder._highways._layers.0.bias'],
            ['_elmo_lstm._token_embedder._projection.weight', '_elmo_lstm._token_embedder._projection.bias'],
            ['_elmo_lstm._elmo_lstm.forward_layer_0.input_linearity.weight', '_elmo_lstm._elmo_lstm.forward_layer_0.state_linearity.weight', '_elmo_lstm._elmo_lstm.forward_layer_0.state_linearity.bias', '_elmo_lstm._elmo_lstm.forward_layer_0.state_projection.weight'],
            ['_elmo_lstm._elmo_lstm.backward_layer_0.input_linearity.weight', '_elmo_lstm._elmo_lstm.backward_layer_0.state_linearity.weight', '_elmo_lstm._elmo_lstm.backward_layer_0.state_linearity.bias', '_elmo_lstm._elmo_lstm.backward_layer_0.state_projection.weight'],
            ['_elmo_lstm._elmo_lstm.forward_layer_1.input_linearity.weight', '_elmo_lstm._elmo_lstm.forward_layer_1.state_linearity.weight', '_elmo_lstm._elmo_lstm.forward_layer_1.state_linearity.bias', '_elmo_lstm._elmo_lstm.forward_layer_1.state_projection.weight'],
            ['_elmo_lstm._elmo_lstm.backward_layer_1.input_linearity.weight', '_elmo_lstm._elmo_lstm.backward_layer_1.state_linearity.weight', '_elmo_lstm._elmo_lstm.backward_layer_1.state_linearity.bias', '_elmo_lstm._elmo_lstm.backward_layer_1.state_projection.weight'],
            ['scalar_mix_0.gamma', 'scalar_mix_0.scalar_parameters.0', 'scalar_mix_0.scalar_parameters.1','scalar_mix_0.scalar_parameters.2'],
            ['scalar_mix_1.gamma', 'scalar_mix_1.scalar_parameters.0', 'scalar_mix_1.scalar_parameters.1','scalar_mix_1.scalar_parameters.2']
        ]
        return param_group

    @staticmethod
    def _get_param_group2():
        param_group = [
            # Character embedding weights
            ['_elmo_lstm._token_embedder._char_embedding_weights'],

            # Convolutional layer weights
            ['_elmo_lstm._token_embedder.char_conv_0.weight', '_elmo_lstm._token_embedder.char_conv_0.bias',
             '_elmo_lstm._token_embedder.char_conv_1.weight', '_elmo_lstm._token_embedder.char_conv_1.bias',
             '_elmo_lstm._token_embedder.char_conv_2.weight', '_elmo_lstm._token_embedder.char_conv_2.bias',
             '_elmo_lstm._token_embedder.char_conv_3.weight', '_elmo_lstm._token_embedder.char_conv_3.bias',
             '_elmo_lstm._token_embedder.char_conv_4.weight', '_elmo_lstm._token_embedder.char_conv_4.bias',
             '_elmo_lstm._token_embedder.char_conv_5.weight', '_elmo_lstm._token_embedder.char_conv_5.bias',
             '_elmo_lstm._token_embedder.char_conv_6.weight', '_elmo_lstm._token_embedder.char_conv_6.bias'],

            # Highway network weights
            ['_elmo_lstm._token_embedder._highways._layers.0.weight', '_elmo_lstm._token_embedder._highways._layers.0.bias'],

            # Token projection weights
            ['_elmo_lstm._token_embedder._projection.weight', '_elmo_lstm._token_embedder._projection.bias'],

            # First bidirectional LSTM
            ['_elmo_lstm._elmo_lstm.forward_layer_0.input_linearity.weight', '_elmo_lstm._elmo_lstm.forward_layer_0.state_linearity.weight',
             '_elmo_lstm._elmo_lstm.forward_layer_0.state_linearity.bias', '_elmo_lstm._elmo_lstm.forward_layer_0.state_projection.weight',
             '_elmo_lstm._elmo_lstm.backward_layer_0.input_linearity.weight', '_elmo_lstm._elmo_lstm.backward_layer_0.state_linearity.weight',
             '_elmo_lstm._elmo_lstm.backward_layer_0.state_linearity.bias', '_elmo_lstm._elmo_lstm.backward_layer_0.state_projection.weight'],

            # Second bidirectional LSTM
            ['_elmo_lstm._elmo_lstm.forward_layer_1.input_linearity.weight', '_elmo_lstm._elmo_lstm.forward_layer_1.state_linearity.weight',
             '_elmo_lstm._elmo_lstm.forward_layer_1.state_linearity.bias', '_elmo_lstm._elmo_lstm.forward_layer_1.state_projection.weight',
             '_elmo_lstm._elmo_lstm.backward_layer_1.input_linearity.weight', '_elmo_lstm._elmo_lstm.backward_layer_1.state_linearity.weight',
             '_elmo_lstm._elmo_lstm.backward_layer_1.state_linearity.bias', '_elmo_lstm._elmo_lstm.backward_layer_1.state_projection.weight'],

            # Scalar mixers
            ['scalar_mix_0.gamma', 'scalar_mix_0.scalar_parameters.0', 'scalar_mix_0.scalar_parameters.1','scalar_mix_0.scalar_parameters.2',
             'scalar_mix_1.gamma', 'scalar_mix_1.scalar_parameters.0', 'scalar_mix_1.scalar_parameters.1','scalar_mix_1.scalar_parameters.2']
        ]
        return param_group

    def get_param_groups(self):
        elmo_param_groups = self._get_param_group2()

        # There are at least len(elmo_param_groups) groups
        params = [{'params': []} for _ in range(len(elmo_param_groups))]

        # We also need to collect the character enhacer parameters within ELMo
        enhancer_params = []

        # Separating elmo layers in multiple param groups to unfreeze gradually
        for name, param in self.elmo.named_parameters():
            group_index = None
            for i, group_names in enumerate(elmo_param_groups):
                if name in group_names:
                    group_index = i
                    break
            if group_index is None:
                if '.char_enhancer.' in name:
                    enhancer_params.append(param)
                    continue
                else:
                    raise Exception("[ERROR] Parameter not found in groups: {}".format(name))
            params[group_index]['params'].append(param)

        assert all(len(p['params']) > 0 for p in params), "There shouldn't be empty groups at this point!"

        # The rest must be all in the last group, so that they get unfrozen first (why? because these parameters are not pretrained)
        params.append({'params': []})
        params[-1]['params'].extend(self.proj.parameters())

        if self.use_lstm:         params[-1]['params'].extend(self.lstm.parameters())
        if self.use_crf:          params[-1]['params'].extend(self.crf.parameters())
        if self.use_char_tagger:  params[-1]['params'].extend(self.charclf.parameters())
        if self.has_embeddings:   params[-1]['params'].extend(self.embeddings.parameters())
        if enhancer_params:       params[-1]['params'].extend(enhancer_params)

        # The default empty parameter group
        params.append({'params': []})

        return params

    def get_l2_reg(self, l2_lambda, include_elmo=True, include_crf=False):
        def sum_l2reg(parameters, accumulated=None):
            for W in parameters:
                if W.requires_grad:
                    if accumulated is None:
                        accumulated = W.norm(2)
                    else:
                        accumulated = accumulated + W.norm(2)
            return accumulated

        l2_sum = sum_l2reg(self.proj.parameters()) * l2_lambda

        if self.elmo_requires_grad and include_elmo:
            l2_sum = l2_lambda * sum_l2reg(self.elmo.parameters(), l2_sum)
        if self.use_crf and include_crf:
            l2_sum = l2_lambda * sum_l2reg(self.crf.parameters(), l2_sum)

        if self.use_lstm:        l2_sum = l2_lambda * sum_l2reg(self.lstm.parameters(), l2_sum)
        if self.use_char_tagger: l2_sum = l2_lambda * sum_l2reg(self.charclf.parameters(), l2_sum)
        if self.has_embeddings:  l2_sum = l2_lambda * sum_l2reg(self.embeddings.parameters(), l2_sum)

        return l2_sum

    def crf_loss(self, logits, target, mask):
        best_paths = self.crf.viterbi_tags(logits, mask)
        predicted_tags = [x for x, y in best_paths]

        loss = -self.crf.forward(logits, target, mask)  # neg log-likelihood loss

        logits = logits * 0.0
        for i, instance_tags in enumerate(predicted_tags):
            for j, tag_id in enumerate(instance_tags):
                logits[i, j, tag_id] = 1

        return {'loss': loss, 'logits': logits, 'tags': predicted_tags}

    def xent_loss(self, logits, target, mask):
        predicted_tags = logits.max(-1)[1].cpu().tolist()
        loss = self.xent(logits[mask == 1], target[mask == 1])

        return {'loss': loss, 'logits': logits, 'tags': predicted_tags}

    def get_embedding(self, inputs):
        sentences = [Sentence(' '.join(s)) for s in inputs['tokens']]
        maxlen = max([len(s) for s in sentences])

        self.embeddings.embed(sentences)
        output = torch.zeros(len(sentences), maxlen, self.embeddings.embedding_length).to(glb.DEVICE)

        for i, sent in enumerate(sentences):
            output[i, :len(sent)] = torch.cat([t.embedding.view(1, self.embeddings.embedding_length) for t in sent], dim=0)

        return output

    def forward(self, inputs):
        """
        :param inputs: a batch of raw text
        :return: dictionary with keys 'char_output', 'word_output', and 'mask'. The first two entries contain
        dictionaries with loss, predicted tags, and logits
        """
        results = dict()
        representations = self.get_representations(inputs)

        word_embedding = representations['word_sem']

        if self.has_embeddings:
            stacked_embedding = self.get_embedding(inputs)
            stacked_embedding = self.drop(stacked_embedding)
            word_embedding = torch.cat([word_embedding, stacked_embedding], dim=-1)

        if self.use_lstm:
              outputs, _  = self.lstm(word_embedding)
        else: outputs = word_embedding

        outputs = self.drop(outputs)

        if self.use_char_tagger:
            results['char_output'] = self.charclf.forward(representations['char_enhancement']['convolutions'],
                                                          inputs['simplified'],
                                                          representations['mask'])
        if self.use_ngram_vectors:
            outputs = torch.cat([outputs, representations['char_enhancement']['convolutions']], dim=-1)
            outputs = self.drop(outputs)

        word_logits = self.proj(outputs)
        results['mask'] = representations['mask']

        if self.use_crf:
              results['word_output'] = self.crf_loss(word_logits, inputs['langids'], representations['mask'])
        else: results['word_output'] = self.xent_loss(word_logits, inputs['langids'], representations['mask'])

        return results







