import torch
import torch.nn as nn
import numpy as np
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer, BertEmbeddings, BertPreTrainedModel
from transformers import BertConfig, RobertaConfig, RobertaModel, BertGenerationEncoder, GPT2LMHeadModel, EncoderDecoderModel, EncoderDecoderConfig
from encoder import GraphTRMEncoder
from kgnn import KGEmbeddingLayer


class Model(nn.Module):

    def __init__(self, args, data, bert_tokenizer, gpt2_tokenizer):

        super(Model, self).__init__()
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer
        self.parse_args(args)
        if args.local_rank in [-1, 0]:
            self.show_config()
        self.load_model(args)

    def parse_args(self, args):

        self.current_device = args.device
        self.mode = args.mode
        self.ddp = args.distributed_training
        if self.ddp:
            self.world_size = args.world_size
        self.dataset_name = args.dataset_name
        self.training_ratio = args.training_ratio
        self.num_mols = self.data.num_mols
        self.num_entities = self.data.num_entities
        args.num_entities = self.num_entities
        self.num_relations = self.data.num_relations
        args.num_relations = self.num_relations
        self.max_text_length = args.max_text_length
        self.max_mol_length = args.max_mol_length
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.num_negative_samples = args.num_negative_samples
        self.reg_kge = args.reg_kge
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.finetune_lr = args.finetune_lr
        self.minibatch_size = args.minibatch_size
        self.aggregator = args.aggregator
        self.bert_vocab_size = self.data.bert_vocab_size

    def show_config(self):

        print('******************************************************')
        print('torch version:', torch.__version__)
        print('np version:', np.__version__)
        print('device:', self.current_device)
        print('distributed training:', self.ddp)
        if self.ddp:
            print('world size:', self.world_size)
        print('dataset name:', self.dataset_name)
        print('training ratio:', self.training_ratio)
        print('max text length:', self.max_text_length)
        print('max mol length:', self.max_mol_length)
        print('#molecules:', self.num_mols)
        print('#entities:', self.num_entities)
        print('#relations:', self.num_relations)
        print('#sampled neighbors:', self.num_sampled_neighbors)
        print('#negaive samples:', self.num_negative_samples)
        print('regularizer for kge:', self.reg_kge)
        print('#epochs:', self.num_epochs)
        print('learning rate:', self.learning_rate)
        print('finetune learning rate:', self.finetune_lr)
        print('minibatch size:', self.minibatch_size)
        print('aggregator:', self.aggregator)
        print('******************************************************')

    def load_model(self, args):

        config = BertConfig.from_pretrained('../pretrained_bert/checkpoint-5500', output_hidden_states=True, output_attentions=True)
        self.encoder = GraphTRMEncoder.from_pretrained('../pretrained_bert/checkpoint-5500', config=config, args=args, data=self.data)
        decoder = GPT2LMHeadModel.from_pretrained('gpt2', add_cross_attention=True, is_decoder=True)
        decoder.resize_token_embeddings(len(self.gpt2_tokenizer))
        decoder.config.bos_token_id = self.gpt2_tokenizer.bos_token_id
        decoder.config.eos_token_id = self.gpt2_tokenizer.eos_token_id
        decoder.config.vocab_size = len(self.gpt2_tokenizer)
        model_config = EncoderDecoderConfig.from_encoder_decoder_configs(self.encoder.config, decoder.config)
        self.model = EncoderDecoderModel(encoder=self.encoder, decoder=decoder, config=model_config)

        self.model.config.decoder_start_token_id = self.gpt2_tokenizer.bos_token_id
        self.model.config.eos_token_id = self.gpt2_tokenizer.eos_token_id
        self.model.config.pad_token_id = self.bert_tokenizer.pad_token_id
        self.model.config.no_repeat_ngram_size = 3
        self.model.length_penalty = 2.0
        self.model.num_beams = 4

        self.kge_layer = KGEmbeddingLayer(config, args)

    def forward(self, input_ids, attention_mask, labels, mol_ids):

        mol_ids = mol_ids.cpu()

        input_ids = np.concatenate([np.expand_dims(mol_ids, axis=1), input_ids.cpu()], axis=1)
        outputs = self.model(input_ids=torch.tensor(input_ids).to(self.current_device),  # may need to change to LongTensor
                             attention_mask=torch.tensor(attention_mask).to(self.current_device),
                             labels=torch.tensor(labels).to(self.current_device),
                             return_dict=True,
                             )

        encoder = self.encoder.encoder
        entity_emb, relation_emb, relation_matrix = encoder.entity_emb[0].weight, encoder.relation_emb[0].weight, encoder.relation_matrix
        if self.aggregator == 'concat':
            mol_emb = outputs.encoder_last_hidden_state[:, 1, :]
        else:
            mol_emb = outputs.encoder_last_hidden_state[:, 0, :]
        kge_loss = self.kge_layer(mol_emb, entity_emb, relation_emb, relation_matrix, mol_ids, self.data)

        loss = outputs.loss + self.reg_kge * kge_loss

        return [loss, outputs.loss, self.reg_kge * kge_loss]

    def inference(self, input_ids, attention_mask, mol_ids, num_beams=4):

        input_ids = np.concatenate([np.expand_dims(mol_ids.cpu(), axis=1), input_ids.cpu()], axis=1)
        outputs = self.model.generate(
            input_ids=torch.tensor(input_ids).to(self.current_device),
            attention_mask=torch.tensor(attention_mask).to(self.current_device),
            max_length=self.max_mol_length,
            num_beams=num_beams,
            bos_token_id=self.gpt2_tokenizer.bos_token_id,
            eos_token_id=self.gpt2_tokenizer.eos_token_id,
            inputs_embeds=None,
        )

        return outputs