import torch
import torch.nn as nn
# from transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers import BertPreTrainedModel, BertModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
# import sys
# sys.path.append(r"./pybert/model") 
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

class BertForMultiLabel(BertPreTrainedModel):
    def __init__(self, config, hidden_dim=768):
        super(BertForMultiLabel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.normalize = nn.LayerNorm([18])
        self.labels_tiled = None

    def freeze_layers(self): # !!! freeze classification model layers
        # Eventual fine-tuning for self-confid
        # LOGGER.info("Freezing every layer except uncertainty")

        for param in self.named_parameters():
            print('parameters include',param[0]) # ++
            if "uncertainty" in param[0]:
                print(param[0], "kept to testing")
                continue
            # print('--------------') # ++
            # print(param[1]) # ++
            param[1].requires_grad = False
            
    def unfreeze_layers(self):

        for param in self.named_parameters():
            print('parameters include',param[0]) # ++
            if "uncertainty" in param[0]:
                print(param[0], "kept to training")
                continue
            # print('--------------') # ++
            # print(param[1]) # ++
            param[1].requires_grad = True


    def disable_bn(self, verbose=False):
        # Freeze also BN running average parameters
        # if verbose:
        #     LOGGER.info("Keeping original BN parameters")
        for layer in self.named_modules():
            if "bert" in layer[0] or "classifier" in layer[0] or 'dropout' in layer[0]:
                if verbose:
                    print(layer[0], "original BN setting")
                # layer[1].momentum = 0
                layer[1].eval()
    def enable_bn(self, verbose=False):
        # Freeze also BN running average parameters
        # if verbose:
        #     LOGGER.info("Keeping original BN parameters")
        for layer in self.named_modules():
            if "bert" in layer[0] or "classifier" in layer[0] or 'dropout' in layer[0]:
                if verbose:
                    print(layer[0], "original BN setting")
                # layer[1].momentum = 0
                layer[1].train()

    def draw_matrix(self, matrix): 
        a = matrix.cpu().detach().numpy()
        ax = plt.matshow(a)
        plt.colorbar(ax.colorbar, fraction=0.025)
        plt.title('LCT')
        plt.savefig('LCT.jpg')
        plt.close()

    def LCTLayer(self, label_representations, labels, Lambda=0.6): 
        labels_similarity = torch.matmul(label_representations, label_representations.permute(0,2,1))
        Z = torch.sqrt((label_representations**2).sum(-1))
        Z = torch.matmul(Z.unsqueeze(1), Z.unsqueeze(2))
        labels_similarity = labels_similarity / Z
        # print(labels.shape, labels_similarity.shape)
        # labels_tiled = labels + Lambda * torch.matmul(labels, labels_similarity)
        labels_tiled = Lambda * torch.matmul(labels, labels_similarity) + labels
        labels_tiled = F.normalize(labels_tiled, p=2, dim=-1)
        # print(labels_tiled[0], labels[0])
        # labels_tiled = self.normalize(labels_tiled.transpose(1,2)).transpose(1,2)
        # labels_tiled = torch.nn.functional.softmax(labels_tiled, dim=-1)
        return labels_tiled

    def LCCLayer(self): 
        pass

    def get_lct_result(self, input_ids, token_type_ids=None, attention_mask=None, output_mask=None, labels=None, multi_labels=None, all_output_mask=None, head_mask=None):
        pass

    def get_results(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, output_mask=None, labels=None, multi_labels=None, all_output_mask=None, head_mask=None):

        # outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        labels_representations_ = last_hidden_state[:, -19:-1, :]
        # last_hidden_state, _ = self.lstm(last_hidden_state)
        last_hidden_state = torch.masked_select(last_hidden_state.permute(2,0,1), output_mask.bool()).reshape(self.config.hidden_size, -1).permute(1,0)
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)

        # LCT
        self.labels_tilde = self.LCTLayer(labels_representations_.mean(dim=0), labels, Lambda=1)
        return logits

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
        max_mask_sum = max(mask_sum)
        mask_idx = [0]+[i for i in mask_sum]
        for i in range(1, len(mask_idx)): 
            mask_idx[i] = mask_idx[i] + mask_idx[i-1]
        logits = torch.stack([torch.cat([logits[mask_idx[i]:mask_idx[i+1]],torch.zeros(max_mask_sum-mask_sum[i],18).to(logits.device)],dim=0) for i in range(len(mask_idx)-1)],dim=0)
        self.labels_tiled = self.LCTLayer(label_representations=labels_representations_, labels=labels)
        return logits

    def _output_select(self, output, output_mask): 
        output = torch.masked_select(output.permute(2,0,1), output_mask.bool()).view(18, -1).permute(1,0)
        output_ix = torch.argmax(output, dim=1)
        idxs = torch.sum(output_mask, dim=1)
        max_len = max(idxs)
        output_logits = torch.zeros([len(idxs), max_len]).to(device)
        s = 0
        for ii, i in enumerate(idxs): 
            for j in range(i): 
                output_logits[ii][j] = output_ix[s]
                s += 1
        return output_logits

    def _get_result(self, input_ids, token_type_ids=None, attention_mask=None, output_mask=None, labels=None, multi_labels=None, all_output_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        idxs = self._output_select(logits, output_mask)
        return idxs
        

    def unfreeze(self,start_layer,end_layer):
        def children(m):  
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)