import os
import string
from torch.utils.data import DataLoader, Dataset


class CSDataset(Dataset):
    def __init__(self, dataset_path, tok_index=0, lid_index=1, ner_index=None, pos_index=None, debug=False):
        assert os.path.exists(dataset_path), 'File path not found: {}'.format(dataset_path)

        self.has_ner = ner_index is not None
        self.has_pos = pos_index is not None

        self.tok_index = tok_index
        self.lid_index = lid_index
        self.ner_index = ner_index
        self.pos_index = pos_index

        self.dataset_path = dataset_path

        self.tokens = []
        self.langids = []
        self.simplified = []
        self.entities = []
        self.postags = []

        for post in read_lines(self.dataset_path):
            toks_i = []
            lids_i = []
            ners_i = []
            poss_i = []

            for token_line in post:
                token_pack = token_line.split('\t')

                toks_i.append(token_pack[self.tok_index])
                lids_i.append(token_pack[self.lid_index])

                if len(token_pack) > 2:
                    if self.has_ner: ners_i.append(token_pack[self.ner_index])
                    if self.has_pos: poss_i.append(token_pack[self.pos_index])

            self.tokens.append(toks_i)
            self.langids.append(lids_i)
            self.simplified.append(map_langids(lids_i))

            if self.has_ner: self.entities.append(ners_i)
            if self.has_pos: self.postags.append(poss_i)

        self.ner_scheme = self.get_current_scheme() if self.has_ner else None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        sample = dict()
        sample['tokens'] = self.tokens[item]
        sample['langids'] = self.langids[item]
        sample['simplified'] = self.simplified[item]

        if self.has_ner: sample['entities'] = self.entities[item]
        if self.has_pos: sample['postags'] = self.postags[item]

        return sample

    def merge(self, dataset):
        self.tokens += dataset.tokens
        self.langids += dataset.langids
        self.simplified += dataset.simplified

        if self.has_ner and dataset.has_ner: self.entities += dataset.entities
        elif self.has_ner or dataset.has_ner: raise Exception('Both datasets are expected to have entities')

        if self.has_pos and dataset.has_pos: self.postags += dataset.postags
        elif self.has_pos or dataset.has_pos: raise Exception('Both datasets are expected to have POS tags')

    def get_current_scheme(self):
        if not self.has_ner:
            return None
        ner_scheme = set()
        for labels in self.entities:
            for label in labels:
                ner_scheme.add(label[0])
        ner_scheme = sorted(ner_scheme)
        if ner_scheme == sorted('BIO'):
            return 'BIO'
        elif ner_scheme == sorted('BIOES'):
            return 'BIOES'
        else:
            raise NotImplemented(f'Unknown scheme! {ner_scheme}')

    def save(self, filepath, predictions=None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w+') as fp:
            for i in range(len(self.tokens)):
                for j in range(len(self.tokens[i])):
                    line_template = "{}\t{}".format(self.tokens[i][j], self.langids[i][j])

                    if self.has_ner: line_template += '\t{}'.format(self.entities[i][j])
                    if self.has_pos: line_template += '\t{}'.format(self.postags[i][j])

                    if predictions is not None:
                        line_template += '\t{}'.format(predictions[i][j])

                    fp.write(line_template + '\n')
                fp.write('\n')

    def change_scheme(self, scheme):
        if not self.has_ner:
            print("[WARNING] Your dataset does not have entity labels!")
            return

        if scheme == 'BIO':
            if self.ner_scheme == 'BIOES':
                for i in range(len(self.entities)):
                    self.entities[i] = from_bioes_to_bio(self.entities[i])
        elif scheme == 'BIOES':
            if self.ner_scheme == 'BIO':
                for i in range(len(self.entities)):
                    self.entities[i] = from_bio_to_bioes(self.entities[i])
        else:
            raise NotImplemented(f"Scheme not implemented: {scheme}")

        self.ner_scheme = self.get_current_scheme()


    def sanity_check(self):
        assert len(self.tokens) == len(self.langids)
        assert len(self.tokens) == len(self.simplified)

        assert not self.has_ner or len(self.tokens) == len(self.entities)
        assert not self.has_pos or len(self.tokens) == len(self.postags)

        for i in range(len(self)):
            assert len(self.tokens[i]) == len(self.langids[i])
            assert len(self.tokens[i]) == len(self.simplified[i])

            assert not self.has_ner or len(self.tokens[i]) == len(self.entities[i])
            assert not self.has_pos or len(self.tokens[i]) == len(self.postags[i])


class RawDataset(Dataset):
    def __init__(self, dataset_path):
        assert os.path.exists(dataset_path), 'File path not found: {}'.format(dataset_path)

        self.dataset_path = dataset_path

        self.postids = []
        self.userids = []
        self.starts = []
        self.ends = []
        self.tokens = []
        self.labels1 = []
        self.labels2 = []

        self.postid_to_index = {}

        curr_post_id = ''
        curr_user_id = ''
        curr_start = -1

        toks_i = []
        bots_i = []
        eots_i = []
        lids_i = []
        ners_i = []

        with open(self.dataset_path, 'r') as stream:
            lines = [line.strip().split('\t') for line in stream] + [['']]

        for i, token_pack in enumerate(lines):

            if curr_post_id == '':
                curr_post_id = token_pack[0].strip()
                curr_user_id = token_pack[1].strip()

            if toks_i and \
                    (token_pack == [''] or
                    (curr_post_id and token_pack[0] != curr_post_id) or
                    (curr_start >= int(token_pack[2]))):

                self.postids.append(curr_post_id)
                self.userids.append(curr_user_id)
                self.postid_to_index[curr_post_id] = len(self.postid_to_index)

                self.starts.append(bots_i)
                self.ends.append(eots_i)
                self.tokens.append(toks_i)
                self.labels1.append(lids_i)
                if ners_i:
                    self.labels2.append(ners_i)

                toks_i = []
                bots_i = []
                eots_i = []
                lids_i = []
                ners_i = []

                if token_pack != ['']:
                    curr_post_id = token_pack[0].strip()
                    curr_user_id = token_pack[1].strip()
                    curr_start = int(token_pack[2].strip())

            if token_pack == ['']:
                break

            bots_i.append(token_pack[2].strip())
            eots_i.append(token_pack[3].strip())
            toks_i.append(token_pack[4].strip())
            lids_i.append(token_pack[5].strip())

            if len(token_pack) > 6:
                ners_i.append(token_pack[6].strip())

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        sample = dict()

        item = item if isinstance(item, int) else self.postid_to_index[item]

        sample['postid'] = self.postids[item]
        sample['userid'] = self.userids[item]
        sample['starts'] = self.starts[item]
        sample['ends'] = self.ends[item]
        sample['tokens'] = self.tokens[item]
        sample['labels1'] = self.labels1[item]

        if self.labels2:
            sample['labels2'] = self.labels2[item]

        return sample

    def merge(self, dataset):
        for i in range(len(dataset)):
            self.postid_to_index[dataset.postids[i]] = len(self.postid_to_index)
            self.postids.append(dataset.postids[i])
            self.userids.append(dataset.userids[i])
            self.starts.append(dataset.starts[i])
            self.ends.append(dataset.ends[i])
            self.tokens.append(dataset.tokens[i])
            self.labels1.append(dataset.labels1[i])
            if dataset.labels2:
                self.labels2.append(dataset.labels2)

    def save(self, filepath, labels1first=True):
        with open(filepath, 'w+') as fp:
            for i in range(len(self.tokens)):
                for j in range(len(self.tokens[i])):
                    if self.labels2:
                        fp.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            self.postids[i],
                            self.userids[i],
                            self.starts[i][j],
                            self.ends[i][j],
                            self.tokens[i][j],
                            self.labels1[i][j] if labels1first else self.labels2[i][j],
                            self.labels2[i][j] if labels1first else self.labels1[i][j]))
                    else:
                        fp.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            self.postids[i],
                            self.userids[i],
                            self.starts[i][j],
                            self.ends[i][j],
                            self.tokens[i][j],
                            self.labels1[i][j]))

    def save_conll(self, filepath, labels1first=True):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w+') as fp:
            for i in range(len(self.tokens)):
                for j in range(len(self.tokens[i])):
                    if self.labels2:
                        fp.write('{}\t{}\t{}\n'.format(
                            self.tokens[i][j],
                            self.labels1[i][j] if labels1first else self.labels2[i][j],
                            self.labels2[i][j] if labels1first else self.labels1[i][j]))
                    else:
                        fp.write('{}\t{}\n'.format(
                            self.tokens[i][j],
                            self.labels1[i][j]))
                fp.write('\n')


def read_lines(filepath):
    lines = [line.strip() for line in open(filepath, 'r')] + ['']
    post = []

    for line in lines:
        if not line:
            if post:
                yield post
            post = []
        else:
            post.append(line)

def from_bio_to_bioes(labels):
    for i in range(len(labels)):
        last_i = i == len(labels) - 1

        if labels[i].startswith('B'):
            if last_i or labels[i + 1] == 'O' or labels[i + 1].startswith('B'):
                labels[i] = 'S' + labels[i][1:]  # Single-token entity
        elif labels[i].startswith('I'):
            if last_i or labels[i + 1] == 'O' or labels[i + 1].startswith('B'):
                labels[i] = 'E' + labels[i][1:]  # Ending of a multi-token entity
    return labels

def from_bioes_to_bio(labels):
    for i in range(len(labels)):
        if labels[i].startswith('E'):
            labels[i] = 'I' + labels[i][1:]
        elif labels[i].startswith('S'):
            labels[i] = 'B' + labels[i][1:]
    return labels

def map_langids(langids):
    new_langids = []
    for lid in langids:
        if lid == 'lang1' or lid == 'mixed' or lid == 'eng' or lid == 'en':
            new_langids.append(lid)
        else:
            new_langids.append('other')
    return new_langids
