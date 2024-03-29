import torch
import torch.nn as nn
import numpy as np
import collections


class ConvLayer(nn.Module):

    def __init__(self, dim_in, dim_out):

        super(ConvLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout_prob = 0.0
        self.generate_modules()

    def generate_modules(self):

        self.linear_layer = nn.ModuleList([nn.Linear(self.dim_in, self.dim_out)])
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_prob)

    def message_passing(self, self_emb, neigh_emb, relation_emb, relation_matrix, act):

        att = self.att(self_emb, neigh_emb, relation_emb, relation_matrix)
        self_emb_agg = self.agg(self_emb, neigh_emb, att, act)

        return self_emb_agg

    def att(self, self_emb, neigh_emb, relation_emb, relation_matrix):

        n = torch.div(neigh_emb.size(0), self_emb.size(0)).int()

        self_emb = torch.reshape(torch.tile(torch.unsqueeze(self_emb, dim=1), [1, n, 1]), neigh_emb.size())
        Rh = torch.squeeze(torch.matmul(relation_matrix, torch.unsqueeze(self_emb, dim=2)))
        Rt = torch.squeeze(torch.matmul(relation_matrix, torch.unsqueeze(neigh_emb, dim=2)))
        att = torch.sum(Rt + self.tanh(Rh + relation_emb), dim=-1)
        att = torch.reshape(att, [-1, n])
        att = self.softmax(att)

        return att

    def agg(self, self_emb, neigh_emb, att, act):

        att = torch.unsqueeze(att, dim=1)
        neigh_emb = torch.reshape(neigh_emb, [att.size(0), att.size(-1), -1])
        neigh_emb_agg = torch.squeeze(torch.matmul(att, neigh_emb))
        self_emb_agg = act(self.linear_layer[0](self_emb + neigh_emb_agg))

        return self_emb_agg

    def forward(self, self_emb, neigh_emb, relation_emb, relation_matrix, act):

        self_emb_agg = self.message_passing(self_emb, neigh_emb, relation_emb, relation_matrix, act)

        return self_emb_agg


class GraphConvEncoder(nn.Module):

    def __init__(self, config, args):

        super(GraphConvEncoder, self).__init__()
        self.current_device = args.device
        self.emb_dim = config.hidden_size
        self.num_conv_layers = args.num_conv_layers
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.minibatch_size = args.minibatch_size
        self.generate_modules(args)

    def generate_modules(self, args):

        self.conv_dim = [self.emb_dim] * (self.num_conv_layers + 1)

        self.conv_layers = nn.ModuleList()
        for layer_id in range(self.num_conv_layers):
            dim_in, dim_out = self.conv_dim[layer_id], self.conv_dim[layer_id + 1]
            self.conv_layers += nn.ModuleList([ConvLayer(dim_in, dim_out)])

        self.acts = nn.ModuleList([nn.LeakyReLU()] * (self.num_conv_layers + 1))

    def get_neighbors(self, mol_ids):

        neighbors = collections.defaultdict(list)
        neighbors['mol_ids'] = [mol_ids]
        neighbors['relation_ids'] = []
        for layer_id in range(self.num_conv_layers):
            neighbor_mol_ids = np.reshape(self.data.sampled_tail_ids[neighbors['mol_ids'][layer_id]], [-1])
            neighbors['mol_ids'].append(neighbor_mol_ids)
            neighbor_relation_ids = np.reshape(self.data.sampled_relation_ids[neighbors['mol_ids'][layer_id]], [-1])
            neighbors['relation_ids'].append(neighbor_relation_ids)

        return neighbors

    def get_features(self, mol_ids):

        neighbors = self.get_neighbors(mol_ids)

        features = collections.defaultdict(list)
        for layer_id in range(self.num_conv_layers + 1):
            feat_entity_emb = self.entity_emb[0].weight[neighbors['mol_ids'][layer_id]].to(self.current_device)
            feat_entity_emb = torch.reshape(feat_entity_emb, [-1, self.emb_dim])
            features['entity_emb'].append(feat_entity_emb)
            if layer_id < self.num_conv_layers:
                feat_relation_emb = self.relation_emb[0].weight[neighbors['relation_ids'][layer_id]].to(self.current_device)
                feat_relation_emb = torch.reshape(feat_relation_emb, [-1, self.emb_dim])
                features['relation_emb'].append(feat_relation_emb)
                feat_relation_matrix = self.relation_matrix[neighbors['relation_ids'][layer_id]].to(self.current_device)
                feat_relation_matrix = torch.reshape(feat_relation_matrix, [-1, self.emb_dim, self.emb_dim])
                features['relation_matrix'].append(feat_relation_matrix)

        return features

    def graph_conv_encoder(self, mol_ids):

        features = self.get_features(mol_ids)

        emb = features['entity_emb']
        for layer_id in range(self.num_conv_layers):
            next_emb = []
            for hop in range(self.num_conv_layers - layer_id):
                emb_agg = self.conv_layers[layer_id](self_emb=emb[hop],
                                                     neigh_emb=emb[hop + 1],
                                                     relation_emb=features['relation_emb'][hop],
                                                     relation_matrix=features['relation_matrix'][hop],
                                                     act=self.acts[layer_id])
                next_emb.append(emb_agg)
            emb = next_emb

        return emb[0]

    def forward(self, mol_emb, entity_emb, relation_emb, relation_matrix, mol_ids, data):

        self.data = data
        self.mol_emb, self.entity_emb, self.relation_emb, self.relation_matrix = mol_emb, entity_emb, relation_emb, relation_matrix
        emb = self.graph_conv_encoder(mol_ids)

        return emb


class KGEmbeddingLayer(nn.Module):

    def __init__(self, config, args):

        super(KGEmbeddingLayer, self).__init__()
        self.current_device = args.device
        self.num_conv_layers = args.num_conv_layers
        self.num_negative_samples = args.num_negative_samples
        self.emb_dim = config.hidden_size
        self.reg_kge = args.reg_kge
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.functional.binary_cross_entropy

    def kg_embedding_layer(self, mol_emb, entity_emb, relation_emb, relation_matrix, mol_ids):

        neigh_mol_ids = np.reshape(self.data.sampled_tail_ids[mol_ids], [-1])
        neigh_relation_ids = np.reshape(self.data.sampled_relation_ids[mol_ids], [-1])

        neigh_mol_emb = entity_emb[neigh_mol_ids].to(self.current_device)
        neigh_mol_emb = torch.reshape(neigh_mol_emb, [-1, self.emb_dim])
        neigh_relation_emb = relation_emb[neigh_relation_ids].to(self.current_device)
        neigh_relation_emb = torch.reshape(neigh_relation_emb, [-1, self.emb_dim])
        neigh_relation_matrix = relation_matrix[neigh_relation_ids].to(self.current_device)
        neigh_relation_matrix = torch.reshape(neigh_relation_matrix, [-1, self.emb_dim, self.emb_dim])

        loss = self.one_embedding_layer(mol_emb, neigh_mol_emb, neigh_relation_emb, neigh_relation_matrix)

        return loss

    def one_embedding_layer(self, self_emb, neigh_emb, relation_emb, relation_matrix):

        n = torch.div(neigh_emb.size(0), self_emb.size(0)).int()

        self_emb = torch.reshape(torch.tile(torch.unsqueeze(self_emb, dim=1), [1, n, 1]), neigh_emb.size())
        Rh = torch.squeeze(torch.matmul(relation_matrix, torch.unsqueeze(self_emb, dim=2)))
        Rt = torch.squeeze(torch.matmul(relation_matrix, torch.unsqueeze(neigh_emb, dim=2)))
        pos_score = torch.sum(torch.square(Rh + relation_emb - Rt), dim=-1)

        neg_indices = torch.randint(neigh_emb.size(0), size=[self.num_negative_samples * neigh_emb.size(0)])
        neigh_emb_neg = neigh_emb[neg_indices]

        self_emb = torch.reshape(torch.tile(torch.unsqueeze(self_emb, dim=1), [1, self.num_negative_samples, 1]), neigh_emb_neg.size())
        relation_emb = torch.reshape(torch.tile(torch.unsqueeze(relation_emb, dim=1), [1, self.num_negative_samples, 1]), neigh_emb_neg.size())
        relation_matrix = torch.reshape(torch.tile(torch.unsqueeze(relation_matrix, dim=1), [1, self.num_negative_samples, 1, 1]), [neigh_emb_neg.size(0), self.emb_dim, self.emb_dim])
        Rh = torch.squeeze(torch.matmul(relation_matrix, torch.unsqueeze(self_emb, dim=2)))
        Rt = torch.squeeze(torch.matmul(relation_matrix, torch.unsqueeze(neigh_emb_neg, dim=2)))
        neg_score = torch.sum(torch.square(Rh + relation_emb - Rt), dim=-1)

        pos_loss = torch.reshape(pos_score, [-1, n])
        neg_loss = torch.reshape(torch.mean(torch.reshape(neg_score, [-1, self.num_negative_samples]), dim=-1), [-1, n])
        loss = - torch.log(self.sigmoid(neg_loss - pos_loss) + 1e-12)
        loss = torch.mean(torch.sum(loss, dim=-1))

        return loss

    def forward(self, mol_emb, entity_emb, relation_emb, relation_matrix, mol_ids, data):

        self.data = data
        loss = self.kg_embedding_layer(mol_emb, entity_emb, relation_emb, relation_matrix, mol_ids)

        return loss