import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers.CRFKLwoLCT import CRF
from .layers.CRFKL import CRF
# from .layers.CRFKLwoLCC import CRF
from transformers import BertModel,BertPreTrainedModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

class BertCrfForMultiLabel(BertPreTrainedModel):
    def __init__(self, config, hidden_dim=768):
        super(BertCrfForMultiLabel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        print("config.num_labels", config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()
        
        self.config = config
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.normalize = nn.BatchNorm1d(18, eps=1e-5, momentum=0.1, affine=True)
        self.labels_tiled = None
    
    def draw_matrix(self, matrix, title=''): 
        matrix = F.normalize(matrix)
        a = matrix.cpu().detach().numpy()
        ax = plt.matshow(a)
        plt.colorbar(ax.colorbar, fraction=0.025)
        plt.title(title)
        plt.savefig(title+'.jpg')
        plt.close()
    def save_matrix(self, matrix, title): 
        a = matrix.detach().cpu().numpy()
        data = pd.DataFrame(a)
        w = pd.ExcelWriter(title+'.xlsx')
        data.to_excel(w, title, float_format='%.4f')
        w.save()
        w.close()
    def read_csv(self, filename): 
        result = [[1]+[0 for i in range(17)]]
        with open(filename) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            for i in range(17):
                result.append([0]+[float(i) for i in next(reader)[1:]])
        return result

    def LCTLayer(self, label_representations, labels, Lambda=1): 
        history = self.read_csv('../ngram-label-similarity/similarity/Cosine(2-gram).csv')
        history = torch.FloatTensor(history).to(labels.device)
        # print(history)
        labels_similarity = torch.matmul(label_representations, label_representations.permute(0,2,1))
        Z = torch.sqrt((label_representations**2).sum(-1))
        Z = torch.matmul(Z.unsqueeze(1), Z.unsqueeze(2))
        labels_similarity = (labels_similarity / Z + history) / 2
        # self.draw_matrix(labels_similarity.mean(0), "LCT confusion matrix")
        # self.save_matrix(labels_similarity.mean(0), 'LCT confusion matrix')
        # print(labels.shape, labels_similarity.shape)
        # labels_tiled = labels + Lambda * torch.matmul(labels, labels_similarity)
        labels_tiled = Lambda * torch.matmul(labels, labels_similarity) + labels
        self.labels_tiled = labels_tiled
        # print(labels_tiled[0], labels[0])
        # labels_tiled = self.normalize(labels_tiled.transpose(1,2)).transpose(1,2)
        # labels_tiled = torch.nn.functional.softmax(labels_tiled, dim=-1)
        return labels_tiled

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, output_mask=None, labels=None, multi_labels=None, all_output_mask=None, head_mask=None):
        # print(input_ids.shape)
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,position_ids=position_ids)
        last_hidden_state = outputs.last_hidden_state
        labels_representations_ = last_hidden_state[:, -19:-1, :]
        # last_hidden_state, _ = self.lstm(last_hidden_state)
        last_hidden_state = torch.masked_select(last_hidden_state.permute(2,0,1), output_mask.bool()).reshape(self.config.hidden_size, -1).permute(1,0)
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        mask_sum = output_mask.sum(dim=1)
        tags = labels[:, :max(mask_sum)]
        labels = torch.zeros(tags.size(0),tags.size(1),18).to(tags.device).scatter_(-1,tags.unsqueeze(-1),1)
        max_mask_sum = max(mask_sum)
        mask_idx = [0]+[i for i in mask_sum]
        for i in range(1, len(mask_idx)): 
            mask_idx[i] = mask_idx[i] + mask_idx[i-1]
        logits = torch.stack([torch.cat([logits[mask_idx[i]:mask_idx[i+1]],torch.zeros(max_mask_sum-mask_sum[i],18).to(logits.device)],dim=0) for i in range(len(mask_idx)-1)],dim=0)
        
        if tags is not None:
            crf_input_mask = torch.stack([torch.arange(max_mask_sum).to(logits.device)]*logits.shape[0])<mask_sum.unsqueeze(1)
            # w/o LCT
            # labels_tiled = self.LCTLayer(label_representations=labels_representations_, labels=labels, Lambda=0)
            # loss = self.crf(emissions=logits, tags=tags, mask=crf_input_mask, labels_tiled=labels)
            # full model
            labels_tiled = self.LCTLayer(label_representations=labels_representations_, labels=labels, Lambda=1.0)
            loss = self.crf(emissions=logits, tags=tags, mask=crf_input_mask, labels_tiled=labels_tiled)
            outputs =(loss, outputs)
        return outputs # (loss), scores

