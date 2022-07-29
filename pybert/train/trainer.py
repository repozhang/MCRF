import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import model_device
from ..common.tools import summary
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
def show_results(one_zero_vectors, name=''): 
    one_zero_vectors_t2n = one_zero_vectors.clone().detach().cpu().numpy()
    indices = np.argwhere(one_zero_vectors_t2n==1)
    result = [[[] for j in range(10)] for i in range(16)]
    print("-----",name,"------")
    for item in indices: 
        result[item[0]][item[1]].append(item[2])
    new2 = []
    for sam in result:
        if sam:
            new1 = []
            for x in sam:
                if x:
                    new1.append(x)
            new2.append(new1)
    return new2
    for i in range(len(result)): 
        print(result[i])
    print("---------------")

import csv 
def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas


def process_labels(labels, output_mask): 
    batch_size = labels.shape[0]
    idx = max(output_mask.sum(dim=1))
    return labels[:, :idx]

class Trainer(object):
    def __init__(self,args,model,logger,criterion,optimizer,scheduler,early_stopping,epoch_metrics,
                 batch_metrics,verbose = 1,training_monitor = None,model_checkpoint = None
                 ):
        self.args = args
        self.model = model
        self.logger =logger
        self.verbose = verbose
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.start_epoch = 1
        self.global_step = 0
        self.model, self.device = model_device(n_gpu = args.n_gpu, model=self.model)
        print(self.device)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if args.resume_path:
            self.logger.info(f"\nLoading checkpoint: {args.resume_path}")
            resume_dict = torch.load(args.resume_path / 'checkpoint_info.bin')
            best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{args.resume_path}' and epoch {self.start_epoch} loaded")

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self,epoch,best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model":model_save,
                 'epoch':epoch,
                 'best':best}
        return state

    def test_epoch(self, data):
        pbar = ProgressBar(n_total=len(data),desc="Testing")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch
                
                final_mask = output_mask.sum(dim=1)
                tags = labels[:, :max(final_mask)]
                labels = torch.zeros(tags.size(0),tags.size(1),18).to(self.device).scatter_(-1,tags.unsqueeze(-1),1)
                logits = self.model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)
                # ----- -----
                logits = logits.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                labels = labels.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                # -------------------
                logits = logits.reshape(-1, logits.shape[-1])
                labels = labels.reshape(-1, labels.shape[-1])
                # 事中模式
                # final_mask = output_mask.sum(dim=1)
                # tags = labels.gather(1,(final_mask-1).unsqueeze(1))
                # labels = torch.zeros(tags.shape[0],18).to(self.device).scatter_(-1,tags, 1)
                # logits = self.model(input_ids, segment_ids, input_mask, output_mask, labels)
                # logits = logits.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                # logits = logits.squeeze(1)
                # logits = self.model(input_ids, segment_ids,input_mask, output_mask, labels)
                # ll = []
                # for l in labels:
                #     for i in l:
                #         if i!=-1: 
                #             ll.append(int(i))
                # lll = torch.LongTensor(ll).to(self.device)
                # labels = torch.zeros(lll.size(0),18).to(self.device).scatter_(1,lll.reshape(-1,1),1)
                # logits = self.model(input_ids, segment_ids, input_mask, output_mask, labels)
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
        
        loss = self.criterion(target = self.targets, output=self.outputs)
        self.result['test_loss'] = loss.item()
        print("------------- test result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                print(metric.name(), value)
                if value:
                    self.result[f'test_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def test_multi_epoch(self, data):
        pbar = ProgressBar(n_total=len(data),desc="Testing")
        self.epoch_reset()
        outputs_1 = []
        outputs_2 = []
        outputs_3 = []
        outputs_4 = []
        multi_targets_ = []
        tags_cases = []
        labels_cases = []
        estim_cases = []
        num_cases = []
        turns_cases = []
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch

                final_mask = output_mask.sum(dim=1)
                tags = labels[:, :max(final_mask)]
                multi_labels_ = multi_labels[:, :max(final_mask), :]
                labels = torch.zeros(tags.size(0),tags.size(1),18).to(self.device).scatter_(-1,tags.unsqueeze(-1),1)
                logits = self.model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)

                logits = logits.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                labels = labels.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                multi_labels_ = multi_labels_.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                tags2_r = show_results((logits>0.5).long(), name='predict @ 2')
                label_r = show_results(multi_labels, name="multi_labels")
                tags_cases.append(tags2_r)
                labels_cases.append(label_r)
                num_cases.append(num.detach().cpu().tolist())
                turns_cases.append(turn.detach().cpu().tolist())


                tags0 = logits.topk(k=4, dim=2).indices.permute(2,0,1)
            self.outputs.append(logits.reshape(-1, logits.shape[-1]).cpu().detach())
            self.targets.append(labels.reshape(-1, labels.shape[-1]).cpu().detach())
            multi_targets_.append(multi_labels_.reshape(-1,18))
            pbar(step=step)
        cases = []
        for a,b,c,d in zip(num_cases, turns_cases, tags_cases, labels_cases):
            for e,f,g,h in zip(a,b,c,d):
                cases.append({'Number':e, 'Turn':f, 'predicts':g, 'labels':h[-1]})
                print(h)
        write_to_tsv('./bert_case_study.tsv', ['Number', 'Turn', 'predicts', 'labels'], cases)
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()

        multi_targets0 = torch.cat(multi_targets_, dim=0)
        
        loss = self.criterion(target = self.targets.cpu().detach(), output=self.outputs)
        self.result['test_loss'] = loss.item()
        print("------------- test result --------------")
        if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    metric(logits=self.outputs.cpu().detach(), target=multi_targets0.cpu().detach())
                    value = metric.value()
                    print(metric.name(), value)
                    if value:
                        self.result[f'test_{metric.name()}'] = value
                    num = multi_targets0.cpu().detach().sum(-1)
                    multi_targets_z1 = torch.masked_select(self.outputs.cpu().detach(), (num==1).repeat(self.outputs.cpu().detach().shape[-1],1).t()).reshape(-1,self.outputs.cpu().detach().shape[-1])
                    multi_targets1 = torch.masked_select(multi_targets0.cpu().detach(), (num==1).repeat(multi_targets0.cpu().detach().shape[-1],1).t()).reshape(-1,multi_targets0.cpu().detach().shape[-1])
                    metric(logits=multi_targets_z1, target=multi_targets1)
                    value = metric.value()
                    print("label-1", metric.name(), value)
                    num = multi_targets0.cpu().detach().sum(-1)
                    multi_targets_z1 = torch.masked_select(self.outputs.cpu().detach(), (num==2).repeat(self.outputs.cpu().detach().shape[-1],1).t()).reshape(-1,self.outputs.cpu().detach().shape[-1])
                    multi_targets1 = torch.masked_select(multi_targets0.cpu().detach(), (num==2).repeat(multi_targets0.cpu().detach().shape[-1],1).t()).reshape(-1,multi_targets0.cpu().detach().shape[-1])
                    metric(logits=multi_targets_z1, target=multi_targets1)
                    value = metric.value()
                    print("label-2", metric.name(), value)
                    num = multi_targets0.cpu().detach().sum(-1)
                    multi_targets_z1 = torch.masked_select(self.outputs.cpu().detach(), (num==3).repeat(self.outputs.cpu().detach().shape[-1],1).t()).reshape(-1,self.outputs.cpu().detach().shape[-1])
                    multi_targets1 = torch.masked_select(multi_targets0.cpu().detach(), (num==3).repeat(multi_targets0.cpu().detach().shape[-1],1).t()).reshape(-1,multi_targets0.cpu().detach().shape[-1])
                    metric(logits=multi_targets_z1, target=multi_targets1)
                    value = metric.value()
                    print("label-3", metric.name(), value)
                    num = multi_targets0.cpu().detach().sum(-1)
                    multi_targets_z1 = torch.masked_select(self.outputs, (num==4).repeat(self.outputs.shape[-1],1).t()).reshape(-1,self.outputs.shape[-1])
                    multi_targets1 = torch.masked_select(multi_targets0.cpu().detach(), (num==4).repeat(multi_targets0.cpu().detach().shape[-1],1).t()).reshape(-1,multi_targets0.cpu().detach().shape[-1])
                    metric(logits=multi_targets_z1, target=multi_targets1)
                    value = metric.value()
                    print("label-4", metric.name(), value)
                    
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def valid_epoch(self, data):
        pbar = ProgressBar(n_total=len(data),desc="Evaluating")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch
                # 事后模式
                final_mask = output_mask.sum(dim=1)
                tags = labels[:, :max(final_mask)]
                labels = torch.zeros(tags.size(0),tags.size(1),18).to(self.device).scatter_(-1,tags.unsqueeze(-1),1)
                logits = self.model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)
                # ----- 事中测评 -----
                logits = logits.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                labels = labels.gather(1,(final_mask-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
                # -------------------
                logits = logits.reshape(-1, logits.shape[-1])
                labels = labels.reshape(-1, labels.shape[-1])
            print(logits.shape, final_mask)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(labels.cpu().detach())
            pbar(step=step)
        self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
        self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
        
        loss = self.criterion(target = self.targets, output=self.outputs)
        self.result['valid_loss'] = loss.item()
        print("------------- valid result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                print(metric.name(), value)
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train_epoch(self, data):
        pbar = ProgressBar(n_total = len(data),desc='Training')
        tr_loss = AverageMeter()
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.batch_reset()
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch


            final_mask = output_mask.sum(dim=1)
            tags = labels[:, :max(final_mask)]
            labels = torch.zeros(tags.size(0),tags.size(1),18).to(self.device).scatter_(-1,tags.unsqueeze(-1),1)
            logits = self.model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1, labels.shape[-1])
            loss = self.criterion(output=logits, target=labels) # bug ++
            if len(self.args.n_gpu) >= 2:
                loss = loss.mean()
                # mean_loss = mean_loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
                # mean_loss = mean_loss / self.args.gradient_accumulation_steps
            if self.args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clip)
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            # mean_loss.backward()
            # print(self.model.parameters())
            clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            if self.batch_metrics:
                for metric in self.batch_metrics:
                    metric(logits = logits,target = labels)
                    self.info[metric.name()] = metric.value()
            self.info['loss'] = loss.item()
            tr_loss.update(loss.item(),n = 1)
            if self.verbose >= 1:
                pbar(step= step,info = self.info)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(labels.cpu().detach())
        print("\n------------- train result --------------")
        # epoch metric
        self.outputs = torch.cat(self.outputs, dim =0).cpu().detach()
        self.targets = torch.cat(self.targets, dim =0).cpu().detach()
        self.result['loss'] = tr_loss.avg
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'{metric.name()}'] = value
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self,train_data,valid_data,test_data):
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 a
        for epoch in range(self.start_epoch,self.start_epoch+self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            train_log = self.train_epoch(train_data)
            valid_log = self.valid_epoch(valid_data)

            logs = dict(train_log,**valid_log)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch,best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.bert_epoch_step(current=logs[self.model_checkpoint.monitor],state = state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
        test_log = self.test_epoch(test_data)
        print(test_log)
        test_multi_log = self.test_multi_epoch(test_data)
        print(test_multi_log)






