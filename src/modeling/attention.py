import torch
import torch.nn as nn
# from src.utilities import get_sinusoid_encoding_table


def stable_softmax(scores, mask=None, epsilon=1e-9):
    """
    :param scores: tensor of shape (batch_size, sequence_length)
    :param mask: (optional) binary tensor in case of padded sequences, shape: (batch_size, sequence_length)
    :param epsilon: epsilon to be added to the normalization factor
    :return: probability tensor of shape (batch_size, sequence_length)
    """
    batch, seq = scores.size()

    # Numerically stable masked-softmax
    maxvec, _ = scores.max(dim=1, keepdim=True)
    scores = torch.exp(scores - maxvec)  # greatest exp value is up to 1.0 to avoid overflow

    if mask is not None:
        scores = scores * mask.view(batch, seq).float()

    sums = torch.sum(scores, dim=1, keepdim=True) + epsilon  # add epsilon to avoid div by zero in case of underflow
    prob = scores / sums

    return prob


class NgramEnhancer(nn.Module):
    def __init__(self, variable_dims, ngram_order, attention_type, use_position):
        super(NgramEnhancer, self).__init__()

        self.ngram_order = ngram_order
        self.dims = variable_dims
        self.attention_type = attention_type

        self.lo_enhancer = None
        self.hi_enhancer = None

        assert attention_type in {'none', 'lo_attention', 'hi_attention', 'hier_attention'}

        if attention_type == 'hier_attention' or attention_type == 'lo_attention':
            self.lo_enhancer = LowNgramAttention(variable_dims, attn_size=100, use_position=use_position)

        if attention_type == 'hier_attention' or attention_type == 'hi_attention':
            self.hi_enhancer = HighNgramAttention(variable_dims, attn_size=256,  ngram_space=128)

    def lo_forward(self, convolved, n, char_lengths):
        if self.lo_enhancer is None or n >= self.ngram_order:
            # we do max pooling if no enhancer or ngram order is beyond the limit
            attention = None
            output, _ = torch.max(convolved, dim=-1)
        else:
            output, attention = self.lo_enhancer.forward_ngram_order(convolved, n, char_lengths)
        return output, attention

    def hi_forward(self, convolutions, mask):
        if self.hi_enhancer is None:
            attention = None
            # convolutions = torch.cat(convolutions, dim=-1)
        else:
            convolutions, attention = self.hi_enhancer.forward(convolutions, mask)
        return convolutions, attention

    def forward(self, inputs, mask, char_lengths):
        raise NotImplementedError('This class only supports calls for low- or high-level forward')


class LowNgramAttention(nn.Module):
    """Applies attention across the ngram vector sets (e.g., the trigrams are represented by a single vector"""
    def __init__(self, variable_dims, attn_size, use_position):
        """
        :param attn_size: int, specifies the attention space to which the inputs are projected
        :param variable_dims: a list with the expected dimensions of the ngram convolutions (n_filters)
        """
        super(LowNgramAttention, self).__init__()

        self.dims = variable_dims
        self.da = attn_size
        self.ngram_space = sum(self.dims)
        self.use_position = use_position

        self.W = nn.ModuleList([nn.Linear(dim, self.da) for dim in self.dims])
        self.v = nn.ModuleList([nn.Linear(self.da, 1) for _ in range(len(self.dims))])

        if self.use_position:
            self.positions = nn.ModuleList(
                [nn.Embedding(50 - i + 1, channels, padding_idx=0)
                 for i, channels in enumerate(self.dims)])

            # self.positions = nn.ModuleList(
            #     [nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position=50 - i + 1, d_hid=channels, padding_idx=0), freeze=True)
            #      for i, channels in enumerate(self.dims)])
        else:
            self.positions = None


    @staticmethod
    def _build_char_mask(char_lengths, i, ngramsize, n_words):
        lengths_i = char_lengths - i
        lengths_i[lengths_i < 0] = 0  # because of padding, 'char_length - i' will contain negative values
        lengths_i = lengths_i.view(n_words, 1).long()

        indexes = torch.arange(0, ngramsize).long().to(char_lengths.device)  # (1, ngram_size)
        char_mask = (indexes < lengths_i).float()  # broadcasting compares against all the words

        return char_mask

    def _positional_encoding(self, n, ngram_size, device):
        indexes = torch.arange(1, ngram_size + 1).view(1, ngram_size).to(device) # (1, ngram_size)
        pos_enc = self.positions[n](indexes)
        return pos_enc.unsqueeze(0) # (1, 1, ngram_size) -> unsqueeze to account for batch size


    def forward_ngram_order(self, inputs, n, char_lengths):
        inputs = inputs.transpose(1, 2)
        n_words, ngramsize, channels = inputs.size()
        batch, seqlen = char_lengths.size()

        has_boundaries = (n_words // batch) == (seqlen + 2)

        if has_boundaries:
            char_bos_indexes = 0
            char_eos_indexes = (char_lengths > 0).long().sum(dim=1) + 1  # +1 to account for bos

            temp = torch.zeros(batch, seqlen + 2).long().to(char_lengths.device)
            temp[:, 1:-1] += char_lengths
            temp[range(batch), char_bos_indexes] = 1  # include the bos token
            temp[range(batch), char_eos_indexes] = 1  # include the eos token

            _char_lengths = temp
        else:
            _char_lengths = char_lengths

        char_mask = self._build_char_mask(_char_lengths, n, ngramsize, n_words)
        pos_enc = 0
        if self.use_position:
            pos_enc = self._positional_encoding(n, ngramsize, char_mask.device) # (1, 1, ngramsize, channels)
            pos_enc = pos_enc.view(1, ngramsize, channels).repeat(n_words, 1, 1) # (n_words, ngramsize, channels)

        scores = self.v[n](torch.tanh(self.W[n](inputs + pos_enc)))  # (n_words, ngrams, 1)
        scores = scores.view(n_words, ngramsize)  # (n_words, ngrams)
        a = stable_softmax(scores, mask=char_mask)

        inputs = inputs * a.view(n_words, ngramsize, 1)  # (n_words, ngrams, channels)
        outputs = torch.sum(inputs, dim=1, keepdim=False)  # (n_words, channels)

        if has_boundaries:
            a = a.view(batch, seqlen + 2, ngramsize)
        else:
            a = a.view(batch, seqlen, ngramsize)

        return outputs, a.data.cpu()

    def forward(self, inputs, mask, char_lengths):
        """
        :param inputs: list of N convolutions with different filter size: (batch, sequence, ngrams, channels for the i-th width)
        :param mask: tensor with word mask (batch, word_sequence)
        :param char_lengths: tensor with character lengths (batch * word_sequence)
        :return:
        """
        attentions = []
        for i, channels in enumerate(self.dims):
            assert channels == inputs[i].size(2), "Expecting inputs to have shape (batch, seqlen, ngramsize, channels)"
            inputs[i], a = self.forward_ngram_order(inputs[i], i, char_lengths)
            attentions.append(a)
        return inputs, attentions


class HighNgramAttention(nn.Module):
    """Applies attention across the ngram vector sets (e.g., the trigrams are represented by a single vector"""
    def __init__(self, variable_dims, ngram_space=128, attn_size=256):
        """
        :param attn_size: int, specifies the attention space to which the inputs are projected
        :param variable_dims: a list with the expected dimensions of the ngram convolutions (n_filters)
        """
        super(HighNgramAttention, self).__init__()

        self.dims = variable_dims
        self.dn = ngram_space
        self.da = attn_size

        self.W = nn.ModuleList([nn.Linear(dim, self.dn) for dim in self.dims])
        self.U = nn.Linear(self.dn, self.da)
        self.v = nn.Linear(self.da, 1)

    def forward(self, inputs, mask):
        """
        :param inputs: a list of N convolutions with different filters each: (batch * sequence, n_filters for the i-th width)
        :param mask:
        :return:
        """
        batch_size, seq_length = mask.size()

        projs = []
        for i, channels in enumerate(self.dims):
            inputs[i] = inputs[i].view(batch_size * seq_length, channels)
            proj = self.W[i](inputs[i]).view(batch_size * seq_length, 1, self.dn)  # (batch * seq, 1, attn)
            projs.append(proj)

        projs = torch.cat(projs, dim=1)  # (batch * seq, N, attn)

        u = self.v(torch.tanh(self.U(projs)))
        u = u.view(batch_size * seq_length, len(self.dims))
        a = stable_softmax(u)  # (batch * seq, N)

        # weight the original given channels
        for ngram_order in range(len(self.dims)):
            inputs[ngram_order] = inputs[ngram_order] * a[:, ngram_order].view(batch_size * seq_length, 1)

        # o = torch.cat(inputs, dim=-1).view(batch_size, seq_length, sum(self.dims))
        a = a.view(batch_size, seq_length, len(self.dims))

        return inputs, a.data.cpu()












