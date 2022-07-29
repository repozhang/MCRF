import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
# from .layers.crfwithlct import CRF
# from .layers.crfwithlcc import CRF
from transformers import BertModel,BertPreTrainedModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy

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
        self.normalize = nn.LayerNorm([18])

    def LCTLayer(self, label_representations, labels, Lambda=0.6): 
        labels_similarity = torch.matmul(label_representations, label_representations.permute(0,2,1))
        # print(labels.shape, labels_similarity.shape)
        labels_tiled = (1-Lambda)*labels + Lambda * torch.matmul(labels, labels_similarity) / labels_similarity.shape[1]
        # labels_tiled = torch.nn.functional.softmax(labels_tiled, dim=-1)
        return labels_tiled

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, output_mask=None, labels=None, multi_labels=None, all_output_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
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
        labels_tiled = self.LCTLayer(label_representations=labels_representations_, labels=labels, Lambda=1)
        if tags is not None:
            crf_input_mask = torch.stack([torch.arange(max_mask_sum).to(logits.device)]*logits.shape[0])<mask_sum.unsqueeze(1)
            loss = self.crf(emissions=logits, tags=tags, mask=crf_input_mask, labels_tiled=labels_tiled)
            outputs =(loss, outputs)
        return outputs # (loss), scores

