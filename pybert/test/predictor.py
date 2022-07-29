#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,model,logger,n_gpu,epoch_metrics):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.epoch_metrics = epoch_metrics

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        self.epoch_reset()
        all_logits = None
        all_labels = None
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_lens, input_ids, input_mask, segment_ids, labels, multi_labels, output_mask, all_output_mask = batch
                ll = []
                for l in labels:
                    for i in l:
                        if i!=-1: 
                            ll.append(int(i))
                lll = torch.LongTensor(ll).cuda()
                labels_s = torch.zeros(lll.size(0),18).cuda().scatter_(1,lll.reshape(-1,1),1)
                logits = self.model(input_ids, segment_ids, input_mask, output_mask, labels_s)
                logits = logits.sigmoid()

                # if all_logits is None:
                #     all_logits = logits.detach().cpu().numpy()
                # else:
                #     all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()], axis=0)
                # print(multi_labels.shape)
                # print(np.concatenate([[i for i in j if i[0]>=0] for j in multi_labels]))
                labels = []
                for i in multi_labels:
                    for j in i:
                        if j[0]>=0: 
                            labels.append([j.detach().cpu().numpy()])
                labels = np.concatenate(labels)
                print(type(labels))
                # if all_labels is None: 
                #     all_labels = np.concatenate(labels)
                # else:
                #     all_labels = np.concatenate([all_labels,np.concatenate(labels)], axis=0)
                pbar(step=step)
            self.outputs.append(logits)
            self.targets.append(torch.from_numpy(labels))
        self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
        self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
        # if 'cuda' in str(self.device):
        #     torch.cuda.empty_cache()
        # print(all_logits.shape, all_labels.shape)
        # return all_logits, all_labels
        print("------------- valid result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result


class Single_Predictor(object):
    def __init__(self,model,logger,n_gpu, epoch_metrics):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.epoch_metrics = epoch_metrics

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc="Testing")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_lens, input_ids, input_mask, segment_ids, labels, multi_labels, output_mask, all_output_mask = batch
                # logits = self.model(input_ids, segment_ids,input_mask, output_mask, labels)
                ll = []
                for l in labels:
                    for i in l:
                        if i!=-1: 
                            ll.append(int(i))
                lll = torch.LongTensor(ll).cuda()
                labels = torch.zeros(lll.size(0),18).cuda().scatter_(1,lll.reshape(-1,1),1)
                logits = self.model(input_ids, segment_ids, input_mask, output_mask, labels)
            # logits = torch.cat(logits_list, dim=0)
            # labels = torch.cat([torch.zeros(i.size(0),18).cuda().scatter_(1,i.reshape(-1,1),1) for i in labels_list], dim=0)
            # print(logits)

        #     def del_tensor_ele(arr,index):
        #         arr1 = arr[0:index]
        #         arr2 = arr[index+1:]
        #         return torch.cat((arr1,arr2),dim=0)
        #     multi_labels = multi_labels.reshape(-1, multi_labels.size(2))
        #     # print("multi_labels: ", multi_labels.size())
        #     del_ids = []
        #     for i in range(multi_labels.size(0)-1, -1, -1): 
        #         if multi_labels[i][0] == -1: 
        #             del_ids.append(i)
        #     for i in del_ids:
        #         multi_labels = del_tensor_ele(multi_labels, i)
        #     # print("multi_labels: ", multi_labels.size())
            self.outputs.append(logits.cpu().detach())
            self.targets.append(labels.cpu().detach())
            pbar(step=step)
        self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
        self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
        
        # loss = self.criterion(target = self.targets, output=self.outputs)
        # self.result['valid_loss'] = loss.item()
        print("------------- valid result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result



