from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
from mol2vec.features import *
from mol2vec.helpers import *
from rdkit import Chem, RDLogger
import numpy as np
import collections
import pickle
from copy import deepcopy
from tqdm import tqdm


class DataCenter():

    def __init__(self, args):

        self.parse_args(args)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = self.load_data()
        self.split_data(input_ids, attention_mask, labels)
        self.sample_neighbors()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.training_ratio = args.training_ratio
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.max_mol_length = args.max_mol_length
        self.max_text_length = args.max_text_length
        self.aggregator = args.aggregator

    def load_tokenizer(self):

        bert_tokenizer = PreTrainedTokenizerFast(tokenizer_file='../pretrained_bert/vocab.json')
        bert_tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                           'mask_token': '[MASK]',
                                           'cls_token': '[CLS]',
                                           'sep_token': '[SEP]',
                                           'unk_token': '[UNK]', })
        self.bert_vocab_size = bert_tokenizer.vocab_size

        # make sure GPT2 appends EOS in begin and end
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            return outputs

        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_tokenizer.add_special_tokens({'bos_token': '[S]',
                                           'eos_token': '[\S]',
                                           'pad_token': '[PAD]'})

        return bert_tokenizer, gpt2_tokenizer

    def load_data(self):

        self.bert_tokenizer, self.gpt2_tokenizer = self.load_tokenizer()

        with open('../data/' + self.dataset_name + '.pickle', 'rb') as file:
            data = pickle.load(file)
        smileses, texts, kg_triples, chebi_ids = data['smileses'], data['texts'], data['kg_triples'], data['chebi_ids']
        self.num_mols = len(smileses)
        self.chebi_ids = chebi_ids

        self.construct_kg(chebi_ids, kg_triples)

        input_ids, attention_mask, substrs = [], [], []
        for smiles in smileses:
            smiles2mol = Chem.MolFromSmiles(smiles)
            try:
                row = mol2alt_sentence(smiles2mol, 1)
                row = ' '.join(str(i) for i in row)
            except:
                row = substrs[0]
            substrs.append(row)
            if self.aggregator == 'vt':
                row = '[PAD] [CLS] ' + row.strip()
            else:
                row = '[CLS] ' + row.strip()
            output = self.bert_tokenizer.batch_encode_plus([row], max_length=self.max_mol_length, padding='max_length', truncation=True)
            input_ids.extend(output['input_ids'])
            attention_mask.extend(output['attention_mask'])
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)

        decoder_input_ids, decoder_attention_mask = [], []
        for text in texts:
            output = self.gpt2_tokenizer.batch_encode_plus([text], max_length=self.max_text_length, padding='max_length', truncation=True)
            decoder_input_ids.extend(output['input_ids'])
            decoder_attention_mask.extend(output['attention_mask'])
        decoder_input_ids = np.array(decoder_input_ids)
        decoder_attention_mask = np.array(decoder_attention_mask)
        labels = np.copy(decoder_input_ids).tolist()
        labels = np.array([
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
            [zip(masks, labels) for masks, labels in zip(decoder_attention_mask, labels)]
        ])

        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels

    def split_data(self, input_ids, attention_mask, labels):

        num_training_mols = int(self.num_mols * self.training_ratio)
        self.training_input_ids, self.training_attention_mask = input_ids[:num_training_mols], attention_mask[:num_training_mols]
        self.test_input_ids, self.test_attention_mask = input_ids[num_training_mols:], attention_mask[num_training_mols:]
        self.training_labels, self.test_labels = labels[:num_training_mols], labels[num_training_mols:]
        self.training_mol_ids, self.test_mol_ids = np.arange(num_training_mols), np.arange(num_training_mols, self.num_mols)
        self.training_chebi_ids, self.test_chebi_ids = self.chebi_ids[:num_training_mols], self.chebi_ids[num_training_mols:]

    def construct_kg(self, chebi_ids, kg_triples):

        kg_np = []
        for kg_triple in kg_triples:
            for hop in kg_triple.keys():
                kg_np.extend(kg_triple[hop])
        kg_np = np.unique(kg_np, axis=0)
        entities = np.array(np.unique(np.concatenate([kg_np[:, 0], kg_np[:, 2]])), dtype=int)
        relations = np.unique(np.concatenate([kg_np[:, 1], np.array(['self_loop'])]))
        self.num_entities, self.num_relations = len(entities), len(relations)

        chebi_id2idx = {}
        for idx, chebi_id in enumerate(chebi_ids):
            chebi_id2idx[int(chebi_id)] = idx
        count = len(chebi_ids)
        for entity in entities:
            entity = int(entity)
            if entity not in chebi_id2idx:
                chebi_id2idx[entity] = count
                count += 1

        relation2idx = {}
        for idx, relation in enumerate(relations):
            relation2idx[relation] = idx

        self.kg = collections.defaultdict(list)
        for triple in kg_np:
            head_idx = chebi_id2idx[int(triple[0])]
            relation_idx = relation2idx[triple[1]]
            tail_idx = chebi_id2idx[int(triple[2])]
            self.kg[head_idx].append([relation_idx, tail_idx])

        if len(self.kg) < self.num_entities:
            for entity in entities:
                entity = int(entity)
                entity_idx = chebi_id2idx[entity]
                if entity_idx not in self.kg:
                    relation_idx = relation2idx['self_loop']
                    self.kg[entity_idx].append([relation_idx, entity_idx])

    def sample_neighbors(self):

        head_ids = np.arange(self.num_entities)
        self.sampled_tail_ids, self.sampled_relation_ids = [], []
        for head_id in head_ids:
            replace = len(self.kg[head_id]) < self.num_sampled_neighbors
            neighbor_idx = np.random.choice(len(self.kg[head_id]), size=self.num_sampled_neighbors, replace=replace)
            self.sampled_tail_ids.append([self.kg[head_id][idx][1] for idx in neighbor_idx])
            self.sampled_relation_ids.append([self.kg[head_id][idx][0] for idx in neighbor_idx])
        self.sampled_tail_ids, self.sampled_relation_ids = np.array(self.sampled_tail_ids), np.array(self.sampled_relation_ids)


class Data(Dataset):

    def __init__(self, args, mode, data):

        super(Data, self).__init__()
        self.mode = mode
        self.data = data

        if self.mode == 'train':
            self.input_ids = self.data.training_input_ids
            self.attention_mask = self.data.training_attention_mask
            self.labels = self.data.training_labels
            self.mol_ids = self.data.training_mol_ids
            self.chebi_ids = self.data.training_chebi_ids
        else:
            self.input_ids = self.data.test_input_ids
            self.attention_mask = self.data.test_attention_mask
            self.labels = self.data.test_labels
            self.mol_ids = self.data.test_mol_ids
            self.chebi_ids = self.data.test_chebi_ids

    def __len__(self):

        return len(self.input_ids)

    def __getitem__(self, idx):

        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx], self.mol_ids[idx], self.chebi_ids[idx]