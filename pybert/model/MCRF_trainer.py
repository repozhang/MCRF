import glob
import logging
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from models.bert_MCRF import BertCrfForMultiLabel
from processors.n_seq import n_processors as processors
from tools.finetuning_argparse import get_argparse
from metrics0 import F1Score, JaccardScore
import numpy as np
import pandas as pd

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForMultiLabel, BertTokenizer),
}

def train(args, train_dataloader, valid_dataloader, test_dataloader, model, tokenizer):
    """ Train the model """
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
    #                               collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    # linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    if args.save_steps==-1 and args.logging_steps==-1:
        args.logging_steps=len(train_dataloader)
        args.save_steps = len(train_dataloader)
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch
            final_mask = output_mask.sum(dim=1)
            outputs = model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            # if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                print(" ")
                if args.local_rank == -1:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # evaluate(args, valid_dataloader, model, tokenizer)
                    print("----- valid epoch {}-----".format(epoch+1))
                    predict(args, valid_dataloader, model, tokenizer)
                    predict_multi(args, valid_dataloader, model, tokenizer)
                    print("--------------------")
                    print("----- test epoch {}-----".format(epoch+1))
                    predict(args, test_dataloader, model, tokenizer)
                    predict_multi(args, test_dataloader, model, tokenizer)
                    print("--------------------")
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                tokenizer.save_vocabulary(output_dir)
                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, eval_dataloader, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    # eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    outputs_ = []
    targets_ = []
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch
            tmp_eval_loss, outputs = model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)
            last_hidden_state = outputs.last_hidden_state
            labels_representations_ = last_hidden_state[:, -19:-1, :]
            # last_hidden_state, _ = self.lstm(last_hidden_state)
            last_hidden_state = torch.masked_select(last_hidden_state.permute(2,0,1), output_mask.bool()).reshape(model.config.hidden_size, -1).permute(1,0)
            last_hidden_state = model.dropout(last_hidden_state)
            logits = model.classifier(last_hidden_state)
            logits = model.softmax(logits)
            mask_sum = output_mask.sum(dim=1)
            tags_ = labels[:, :max(mask_sum)]
            labels = torch.zeros(tags_.size(0),tags_.size(1),18).to(tags_.device).scatter_(-1,tags_.unsqueeze(-1),1)
            max_mask_sum = max(mask_sum)
            mask_idx = [0]+[i for i in mask_sum]
            for i in range(1, len(mask_idx)): 
                mask_idx[i] = mask_idx[i] + mask_idx[i-1]
            logits = torch.stack([torch.cat([logits[mask_idx[i]:mask_idx[i+1]],torch.zeros(max_mask_sum-mask_sum[i],18).to(logits.device)],dim=0) for i in range(len(mask_idx)-1)],dim=0)
            #############################################
            crf_input_mask = torch.stack([torch.arange(max_mask_sum).to(logits.device)]*logits.shape[0])<mask_sum.unsqueeze(1)
            tags = model.crf.decode(logits, crf_input_mask)
            # print(tags[0], tags_[0])
        # print("tags:", tags)
        tags_1 = torch.zeros(tags.shape[1], tags.shape[2], 18).to(tags.device).scatter_(-1,tags[0].unsqueeze(-1),1)
        tags_ = torch.zeros(tags_.shape[0], tags_.shape[1], 18).to(tags_.device).scatter_(-1,tags_.unsqueeze(-1),1)
        tags_1 = tags_1.gather(1,(mask_sum-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
        tags_ = tags_.gather(1,(mask_sum-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
        # -------------------
        outputs_.append(tags_1.reshape(-1, 18))
        targets_.append(tags_.reshape(-1, 18))
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)
    metrics = [F1Score(thresh=0.7, average='micro', task_type='multiclass'), 
                F1Score(thresh=0.7, average='macro', task_type='multiclass'), 
                JaccardScore(thresh=0.7, average='macro', task_type='multiclass'), 
                nDCG(thresh=0.7, average='macro', task_type='multiclass')]
    outputs0 = torch.cat(outputs_, dim=0)
    targets0 = torch.cat(targets_, dim=0)
    # outputs1 = torch.zeros(outputs0.shape[0], 18).to(outputs0.device).scatter_(dim=-1, index=outputs0.unsqueeze(-1), value=1)
    # targets1 = torch.zeros(targets0.shape[0], 18).to(targets0.device).scatter_(dim=-1, index=targets0.unsqueeze(-1), value=1)
    for metric in metrics:
        metric(logits=outputs0, target=targets0)
        value = metric.value()
        print("validing", metric.name(), value)
    logger.info("\n")


def predict(args, test_dataloader, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataloader))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    outputs_ = []
    targets_ = []
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch
            tmp_eval_loss, outputs = model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)
            #######################
            last_hidden_state = outputs.last_hidden_state
            labels_representations_ = last_hidden_state[:, -19:-1, :]
            # last_hidden_state, _ = self.lstm(last_hidden_state)
            last_hidden_state = torch.masked_select(last_hidden_state.permute(2,0,1), output_mask.bool()).reshape(model.config.hidden_size, -1).permute(1,0)
            last_hidden_state = model.dropout(last_hidden_state)
            logits = model.classifier(last_hidden_state)
            logits = model.softmax(logits)
            mask_sum = output_mask.sum(dim=1)
            tags_ = labels[:, :max(mask_sum)]
            labels = torch.zeros(tags_.size(0),tags_.size(1),18).to(tags_.device).scatter_(-1,tags_.unsqueeze(-1),1)
            max_mask_sum = max(mask_sum)
            mask_idx = [0]+[i for i in mask_sum]
            for i in range(1, len(mask_idx)): 
                mask_idx[i] = mask_idx[i] + mask_idx[i-1]
            logits = torch.stack([torch.cat([logits[mask_idx[i]:mask_idx[i+1]],torch.zeros(max_mask_sum-mask_sum[i],18).to(logits.device)],dim=0) for i in range(len(mask_idx)-1)],dim=0)
            #######################
            crf_input_mask = torch.stack([torch.arange(max_mask_sum).to(logits.device)]*logits.shape[0])<mask_sum.unsqueeze(1)
            tags0 = model.crf.decode(logits, crf_input_mask)
            tags = tags0.squeeze(0).cpu().numpy().tolist()
        tags_1 = torch.zeros(tags0.shape[1], tags0.shape[2], 18).to(tags0.device).scatter_(-1,tags0[0].unsqueeze(-1),1)
        tags_ = torch.zeros(tags_.shape[0], tags_.shape[1], 18).to(tags_.device).scatter_(-1,tags_.unsqueeze(-1),1)
        tags_1 = tags_1.gather(1,(mask_sum-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
        tags_ = tags_.gather(1,(mask_sum-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
        # -------------------
        outputs_.append(tags_1.reshape(-1, 18))
        targets_.append(tags_.reshape(-1, 18))
        preds = tags[0][1:-1] 
        pbar(step)
    metrics = [F1Score(thresh=0.7, average='micro', task_type='multiclass'), 
                F1Score(thresh=0.7, average='macro', task_type='multiclass'), 
                JaccardScore(thresh=0.7, average='macro', task_type='multiclass'), 
                nDCG(thresh=0.7, average='macro', task_type='multiclass')]
    outputs0 = torch.cat(outputs_, dim=0)
    targets0 = torch.cat(targets_, dim=0)
    for metric in metrics:
        metric(logits=outputs0, target=targets0)
        value = metric.value()
        print("testing", metric.name(), value)

    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')

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


def predict_multi(args, test_dataloader, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    # test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # # Note that DistributedSampler samples randomly
    # test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataloader))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    nbest = 10
    outputs_ = []
    outputs_1 = []
    outputs_2 = []
    outputs_3 = []
    outputs_4 = []
    outputs_z = [[] for i in range(nbest)]
    multi_targets_ = []
    tags_cases = []
    labels_cases = []
    estim_cases = []
    num_cases = []
    turns_cases = []
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            batch = tuple(t.to(args.device) for t in batch)
            input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = batch
            # print(multi_labels, multi_labels.shape)
            tmp_eval_loss, outputs = model(input_ids, segment_ids, pos_ids, input_mask, output_mask, labels)
            #######################
            last_hidden_state = outputs.last_hidden_state
            labels_representations_ = last_hidden_state[:, -19:-1, :]
            # last_hidden_state, _ = self.lstm(last_hidden_state)
            last_hidden_state = torch.masked_select(last_hidden_state.permute(2,0,1), output_mask.bool()).reshape(model.config.hidden_size, -1).permute(1,0)
            last_hidden_state = model.dropout(last_hidden_state)
            logits = model.classifier(last_hidden_state)
            logits = model.softmax(logits)
            mask_sum = output_mask.sum(dim=1)
            tags_ = labels[:, :max(mask_sum)]
            multi_labels_ = multi_labels[:, :max(mask_sum), :]
            labels = torch.zeros(tags_.size(0),tags_.size(1),18).to(tags_.device).scatter_(-1,tags_.unsqueeze(-1),1)
            max_mask_sum = max(mask_sum)
            mask_idx = [0]+[i for i in mask_sum]
            for i in range(1, len(mask_idx)): 
                mask_idx[i] = mask_idx[i] + mask_idx[i-1]
            logits = torch.stack([torch.cat([logits[mask_idx[i]:mask_idx[i+1]],torch.zeros(max_mask_sum-mask_sum[i],18).to(logits.device)],dim=0) for i in range(len(mask_idx)-1)],dim=0)
            #######################
            crf_input_mask = torch.stack([torch.arange(max_mask_sum).to(logits.device)]*logits.shape[0])<mask_sum.unsqueeze(1)
            
            tags0 = model.crf.decode(logits, crf_input_mask, nbest=nbest)
            tags_z = []
            tags_m = torch.zeros(tags0.shape[1], tags0.shape[2], 18).to(tags0.device)
            for i in range(nbest):
                tags_m = tags_m.clone().scatter_(-1,tags0[i].unsqueeze(-1),1)
                t = []
                for i in tags_m:
                    i[i.sum(dim=-1)>1, 0] = 0
                    t.append(i.detach().clone())
                tags_m = torch.stack(t).to(tags0.device)
                tags_z.append(tags_m.clone())

        tags_z = [i.gather(1,(mask_sum-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18)) for i in tags_z]
        multi_labels_ = multi_labels_.gather(1,(mask_sum-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,18))
        tags2_r = show_results(tags_z[2], name='predict @ 3')
        label_r = show_results(multi_labels, name="multi_labels")
        tags_cases.append(tags2_r)
        labels_cases.append(label_r)
        num_cases.append(num.detach().cpu().tolist())
        turns_cases.append(turn.detach().cpu().tolist())
        for i in range(nbest): 
            # print(tags_z[i].squeeze(1).shape)
            outputs_z[i].append(tags_z[i].squeeze(1))
        multi_targets_.append(multi_labels_.reshape(-1,18))
        pbar(step)
    cases = []
    for a,b,c,d in zip(num_cases, turns_cases, tags_cases, labels_cases):
        for e,f,g,h in zip(a,b,c,d):
            cases.append({'Number':e, 'Turn':f, 'predicts':g, 'labels':h[-1]})
    write_to_tsv('./case_study.tsv', ['Number', 'Turn', 'predicts', 'labels'], cases)
    metrics = [F1Score(thresh=0.7, average='micro', task_type='multiclass'), 
                F1Score(thresh=0.7, average='macro', task_type='multiclass'), 
                JaccardScore(thresh=0.7, average='macro', task_type='multiclass')]
    multi_targets_z = [torch.cat(i, dim=0) for i in outputs_z]
    # multi_outputs = [multi_outputs_1, multi_outputs_2, multi_outputs_3, multi_outputs_4]
    multi_targets0 = torch.cat(multi_targets_, dim=0)
    for i in range(nbest):
        print("@", i+1) 
        for metric in metrics:
            # print(multi_targets_z[i], multi_targets0)
            metric(logits=multi_targets_z[i], target=multi_targets0)
            value = metric.value()
            print("testing multi", metric.name(), value)
            num = multi_targets0.sum(-1)
            multi_targets_z1 = torch.masked_select(multi_targets_z[i], (num==1).repeat(multi_targets_z[i].shape[-1],1).t()).reshape(-1,multi_targets_z[i].shape[-1])
            multi_targets1 = torch.masked_select(multi_targets0, (num==1).repeat(multi_targets0.shape[-1],1).t()).reshape(-1,multi_targets0.shape[-1])
            metric(logits=multi_targets_z1, target=multi_targets1)
            value = metric.value()
            print("label-1", metric.name(), value)
            multi_targets_z2 = torch.masked_select(multi_targets_z[i], (num==2).repeat(multi_targets_z[i].shape[-1],1).t()).reshape(-1,multi_targets_z[i].shape[-1])
            multi_targets2 = torch.masked_select(multi_targets0, (num==2).repeat(multi_targets0.shape[-1],1).t()).reshape(-1,multi_targets0.shape[-1])
            metric(logits=multi_targets_z2, target=multi_targets2)
            value = metric.value()
            print("label-2", metric.name(), value)
            multi_targets_z3 = torch.masked_select(multi_targets_z[i], (num==3).repeat(multi_targets_z[i].shape[-1],1).t()).reshape(-1,multi_targets_z[i].shape[-1])
            multi_targets3 = torch.masked_select(multi_targets0, (num==3).repeat(multi_targets0.shape[-1],1).t()).reshape(-1,multi_targets0.shape[-1])
            metric(logits=multi_targets_z3, target=multi_targets3)
            value = metric.value()
            print("label-3", metric.name(), value)
            multi_targets_z4 = torch.masked_select(multi_targets_z[i], (num==4).repeat(multi_targets_z[i].shape[-1],1).t()).reshape(-1,multi_targets_z[i].shape[-1])
            multi_targets4 = torch.masked_select(multi_targets0, (num==4).repeat(multi_targets0.shape[-1],1).t()).reshape(-1,multi_targets0.shape[-1])
            metric(logits=multi_targets_z4, target=multi_targets4)
            value = metric.value()
            print("label-4", metric.name(), value)


def run_bert(data_name='MDMDdata', max_len=512, batch_size=32, n_gpu='0', arch='bert', mode='train', sorted='False'):
    import sys
    sys.path.append(r'../../')
    from pybert.io.task_data import TaskData
    # from pybert.test.predictor import Predictor
    from pybert.io.bert_processor import BertProcessor
    from pybert.configs.basic_config import config
    from pybert.io.utils import collate_fn
    from torch.utils.data import RandomSampler, SequentialSampler
    from torch.utils.data import DataLoader
    from pybert.model.bert_for_multi_label import BertForMultiLabel
    from pybert.common.tools import init_logger, logger
    from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport, F1Score, JaccardScore, Hamming, ExactRatio, PatK, nDCG
    
    data = TaskData()
    processor = BertProcessor(vocab_path="../.." / config['bert_vocab_path'], do_lower_case=True)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}

    print("load train data")
    train_data = processor.get_train("../.." / config['data_dir'] / f"{data_name}.train.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file="../.." / config[
                                                    'data_dir'] / f"cached_train_examples_{arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=max_len,
                                               cached_features_file="../.." / config[
                                                    'data_dir'] / "cached_train_features_{}_{}".format(
                                                   max_len, arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=False)
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    if sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,
                                  collate_fn=collate_fn)

    print("load valid data")
    valid_data = processor.get_dev("../.." / config['data_dir'] / f"{data_name}.valid.pkl")
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file="../.." / config[
                                                'data_dir'] / f"cached_valid_examples_{arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=max_len,
                                               cached_features_file="../.." / config[
                                                'data_dir'] / "cached_valid_features_{}_{}".format(
                                                   max_len, arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size,
                                  collate_fn=collate_fn)

    print("load test data")
    test_data = processor.get_test("../.." / config['data_dir'] / f"{data_name}.test.pkl")
    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file="../.." / config[
                                            'data_dir'] / f"cached_test_examples_{arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=max_len,
                                              cached_features_file="../.." / config[
                                            'data_dir'] / "cached_test_features_{}_{}".format(
                                                  max_len, arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size,
                                 collate_fn=collate_fn)
    bert_model = BertForMultiLabel.from_pretrained("../.." / config['checkpoint_dir'] / "bert", num_labels=len(label_list))
    return train_dataloader, valid_dataloader, test_dataloader, bert_model

def save_matrix(matrix, title): 
    a = matrix.detach().cpu().numpy()
    data = pd.DataFrame(a)
    w = pd.ExcelWriter(title+'.xlsx')
    data.to_excel(w, title, float_format='%.4f')
    w.save()
    w.close()
def main():
    args = get_argparse().parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)
    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    # label_list = processor.get_labels()
    label_list = [str(i) for i in range(18)]
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path,num_labels=num_labels,)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    total_size = 0
    print("----- Parameter Size -----")
    for name,parameters in model.named_parameters():
        t = 1
        for i in range(len(parameters.size())): 
            t *= parameters.size()[i]
        total_size += t
    print("model parameters size:", total_size)
    print("-"*20)
    logger.info("Training/evaluation parameters %s", args)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_data_li, valid_data_li, test_data_li, bert_model = run_bert(batch_size=args.train_batch_size)
    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_data_li, valid_data_li, test_data_li, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        tokenizer = None
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, valid_data_li, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        #     results.update(result)
        # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     for key in sorted(results.keys()):
        #         writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            # predict(args, test_data_li, model, tokenizer, prefix=prefix)
            predict_multi(args, valid_data_li, model, tokenizer, prefix=prefix)
            predict_multi(args, test_data_li, model, tokenizer, prefix=prefix)
            # model.draw_matrix(model.crf.transitions_prime, "LCC T'")
            # save_matrix(model.crf.transitions_prime, title="LCC T'")
            # model.draw_matrix(model.crf.transitions, "LCC T")
            # save_matrix(model.crf.transitions, title="LCC T")


if __name__ == "__main__":
    main()
