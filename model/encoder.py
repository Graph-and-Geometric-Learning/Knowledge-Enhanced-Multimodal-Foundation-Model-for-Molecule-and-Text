import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer, BertEmbeddings, BertPreTrainedModel
from transformers.models.bert_generation import BertGenerationPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import numpy as np
import collections
from kgnn import GraphConvEncoder, KGEmbeddingLayer


class GraphTRMLayerConcat(nn.Module):

    def __init__(self, config, args, data):

        super(GraphTRMLayerConcat, self).__init__()
        self.data = data
        self.aggregator = args.aggregator
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.minibatch_size = args.minibatch_size
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.emb_dim = config.hidden_size
        self.current_device = args.device

        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.graph_conv_layer = nn.ModuleList([GraphConvEncoder(config, args) for _ in range(config.num_hidden_layers - 1)])
        self.entity_emb = nn.ModuleList([nn.Embedding(self.num_entities, self.emb_dim).to(self.current_device)])
        self.relation_emb = nn.ModuleList([nn.Embedding(self.num_relations, self.emb_dim).to(self.current_device)])
        self.relation_matrix_ = nn.ModuleList([nn.Embedding(self.num_relations, self.emb_dim * self.emb_dim).to(self.current_device)])
        self.relation_matrix = torch.reshape(self.relation_matrix_[0].weight, [self.num_relations, self.emb_dim, self.emb_dim])
        if self.aggregator == 'sum':
            self.linear_layers = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim),
                                                nn.Linear(self.emb_dim, self.emb_dim)])
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.identity = nn.Identity()

    def forward(self, hidden_states, attention_mask, mol_ids):

        all_hidden_states, all_attentions = (), ()
        [all_nodes_num, seq_length, emb_dim] = hidden_states.size()

        for layer_id, bert_layer in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if layer_id > 0:
                cls_emb = hidden_states[:, 0, :]
                kg_emb = self.graph_conv_layer[layer_id - 1](cls_emb, self.entity_emb, self.relation_emb, self.relation_matrix, mol_ids, self.data)
                if self.aggregator == 'vt':
                    hidden_states[:, 0, :] = kg_emb
                elif self.aggregator == 'sum':
                    kg_emb = torch.tile(torch.unsqueeze(kg_emb, dim=1), [1, seq_length, 1])
                    hidden_states = hidden_states + kg_emb
                    for i in range(len(self.linear_layers)):
                        hidden_states = self.linear_layers[i](hidden_states)
                hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)
                layer_outputs = bert_layer(hidden_states, attention_mask=attention_mask, output_attentions=True)
            else:
                attention_mask_tmp = attention_mask.clone()
                if self.aggregator == 'vt':
                    attention_mask_tmp[:, :, :, 0] = -10000.0
                layer_outputs = bert_layer(hidden_states, attention_mask=attention_mask_tmp, output_attentions=True)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GraphTRMEncoder(BertGenerationPreTrainedModel):

    def __init__(self, config, args, data):

        super(GraphTRMEncoder, self).__init__(config)
        self.aggregator = args.aggregator
        self.minibatch_size = args.minibatch_size
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.hidden_size = config.hidden_size
        self.data = data

        self.embeddings = BertEmbeddings(config=config)
        self.encoder = GraphTRMLayerConcat(config, args, data)
        self.kge_layer = KGEmbeddingLayer(config, args)

    def forward(self, input_ids, attention_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict):

        mol_ids = input_ids[:, 0].cpu()
        if self.aggregator == 'vt':
            input_ids = input_ids[:, 2:]
        else:
            input_ids = input_ids[:, 1:]

        [all_nodes_num, seq_length] = input_ids.size()

        embedding_output = self.embeddings(input_ids=input_ids)

        if self.aggregator == 'vt':
            attention_mask[:, 0] = 1

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        if self.aggregator == 'vt':
            station_placeholder = torch.zeros([all_nodes_num, 1, embedding_output.size(-1)]).type(embedding_output.dtype).to(embedding_output.device)
            embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 1+L D

        encoder_outputs = self.encoder(hidden_states=embedding_output,
                                       attention_mask=extended_attention_mask,
                                       mol_ids=mol_ids)

        if not return_dict:
            return encoder_outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1],
            attentions=encoder_outputs[2],
        )