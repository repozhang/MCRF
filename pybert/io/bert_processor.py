import csv
import torch
import numpy as np
from ..common.tools import load_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

class InputExample(object):
    def __init__(self, num, num1, num2, turn, guid, text, labels, multi_labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid   = guid
        self.num = num
        self.num1 = num1
        self.num2 = num2
        self.turn = turn
        self.text = text
        self.labels  = labels
        self.multi_labels = multi_labels
        

class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self,num, num1, num2, turn, input_ids,input_mask,segment_ids,pos_ids,labels, multi_labels,input_len, output_mask, all_output_mask):
        self.num=num, 
        self.num1 = num1, 
        self.num2 = num2, 
        self.turn = turn, 
        self.input_ids   = input_ids
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.labels = labels
        self.multi_labels = multi_labels
        self.input_len = input_len
        self.output_mask = output_mask
        self.all_output_mask = all_output_mask

class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,vocab_path,do_lower_case):
        self.tokenizer = BertTokenizer(vocab_path,do_lower_case)
        # self.abs_position = [(0,0),(1,1),(2,2),(3,3),(3,4),(6,5),(5,6),(10,7),(6,8),(5,9),(10,10),(7,11),(6,12),(4,13),(8,14),(2,15),(9,16),(10,17)]
        self.abs_position = [(0,0,0),(1,0,0),(1,1,0),(1,2,0),(1,2,1),(1,5,2),(1,4,0),(1,9,0),(1,5,0),(1,4,1),(1,9,1),(1,6,0),(1,5,1),(1,3,0),(1,7,0),(1,1,1),(1,8,0),(1,9,2)]

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self,data_file):
        return self.read_data(data_file)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        # return ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
        return ['non-malevolence', 'unconcernedness', 'detachment', 'digust', 'blame', 'arrogance', 'anger', 'dominance', 'violence', 'NIA', 'phobia', 'anti-authority', 'obscenity', 'jealousy', 'self-hurt', 'deceit', 'privacy invasion', 'immoral & illegal']

    @classmethod
    def read_data(cls, input_file,quotechar = None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self,lines,example_type,cached_examples_file):
        '''
        Creates examples for data
        '''
        pbar = ProgressBar(n_total = len(lines),desc='create examples')
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i,line in enumerate(lines):
#                 print(line)
                guid = '%s-%d'%(example_type,i)
                num = [i[0] for i in line]
                num1 = [0 for i in line]
                num2 = [0 for i in line]
                turn = [i[3] for i in line]
                text = [i[4] for i in line]
                # print(num, turn, text)
                if len(text)==0: 
                    continue
                labels = [i[5] for i in line]
                multi_labels = [i[6] for i in line]
                if isinstance(labels,str):
                    labels = [np.float(x) for x in labels.split(",")]
                else:
                    labels = [np.float(x) for x in list(labels)]
                example = InputExample(guid=guid,num=num, num1=num1, num2=num2, turn=turn, text=text, labels=labels, multi_labels=multi_labels)
                examples.append(example)
                pbar(step=i)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features_(self,examples,max_seq_len,cached_features_file):

        pbar = ProgressBar(n_total=len(examples),desc='create features')
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id,example in enumerate(examples):
                num_0 = example.num
                num1_0 = example.num1
                num2_0 = example.num2
                turn_0 = example.turn

                tokens0 = [self.tokenizer.tokenize(i) for i in example.text]
                labels0 = example.labels
                multi_labels0 = example.multi_labels
                tokens1 = [['[CLS]'] + token + ['[SEP]'] for token in tokens0]
                segment_ids1 = [[0] * len(tokens1[i]) if i%2==0 else [1] * len(tokens1[i]) for i in range(len(tokens1))]
                # pos_ids1 = [[self.abs_position[labels0[i]]] * len(tokens1[i]) for i in range(len(tokens1))]

                output_mask1 = [[1]+[0]*(len(token)-1) for token in tokens1]
                labels_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]+labels0+[self.tokenizer.convert_tokens_to_ids('[SEP]')]
                output_label_mask = [[1] + [0]*(len(labels_ids)-1)]
                
                tokens, segment_ids, pos_ids, output_mask = [], [], [], []
                for t in tokens1:
                    tokens += t
                for s in segment_ids1: 
                    segment_ids += s
                # segment_ids = segment_ids + [len(tokens1)%2]*len(labels_ids) # multi-label
                # segment_ids = segment_ids # single-label
                for p in pos_ids1: 
                    pos_ids += p
                for o in output_mask1: 
                    output_mask += o
                all_output_mask = output_mask[:]
                all_output_mask += output_label_mask[0]
                # input_ids = self.tokenizer.convert_tokens_to_ids(tokens) + labels_ids # multi-label
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens) # single-label
                input_mask = [1] * len(input_ids)
                padding = [0] * (max_seq_len - len(input_ids))
                input_len = len(input_ids)
#                 print(len(input_ids), len(input_mask), len(segment_ids))
                input_ids   += padding
                input_mask  += padding
                segment_ids += padding
                pos_ids += padding
                output_mask += [0] * (max_seq_len - len(output_mask))
                all_output_mask += [0] * (max_seq_len - len(all_output_mask))
                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len
                assert len(pos_ids) == max_seq_len
                assert len(output_mask) == max_seq_len
                assert len(all_output_mask) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {example.guid}" % ())
                    logger.info(f"num: {example.num}" % ())
                    logger.info(f"num1: {example.num1}" % ())
                    logger.info(f"num2: {example.num2}" % ())
                    logger.info(f"turn: {example.turn}" % ())

                    logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
                    logger.info(f"pos_ids: {' '.join([str(x) for x in pos_ids])}")
                    logger.info(f"output_mask: {' '.join([str(x) for x in output_mask])}")
                    logger.info(f"all_output_mask: {' '.join([str(x) for x in all_output_mask])}")

                feature = InputFeature(num=example.num, 
                                    num1=example.num1, 
                                    num2=example.num2, 
                                    turn=example.turn, 
                                    input_ids = input_ids,
                                    input_mask = input_mask,
                                    segment_ids = segment_ids,
                                    pos_ids = pos_ids, 
                                    labels = labels0,
                                    multi_labels = multi_labels0, 
                                    input_len = input_len, 
                                    output_mask = output_mask, 
                                    all_output_mask = all_output_mask)
                features.append(feature)
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_features(self,examples,max_seq_len,cached_features_file):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] u1 [SEP] [CLS] u2 [SEP] . . . [CLS] l1 l2 l3 ... l18 [SEP]
        #  type_ids:   0   0    0     1   1    1           0    0  0  0      0   0   
        '''
        pbar = ProgressBar(n_total=len(examples),desc='create features')
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id,example in enumerate(examples):
                tokens0 = [self.tokenizer.tokenize(i) for i in example.text]
                labels0 = example.labels
                multi_labels0 = example.multi_labels
#                 if example.text_b:
#                     tokens_b = self.tokenizer.tokenize(example.text_b)
#                     # Modifies `tokens_a` and `tokens_b` in place so that the total
#                     # length is less than the specified length.
#                     # Account for [CLS], [SEP], [SEP] with "- 3"
#                     self.truncate_seq_pair(tokens_a,tokens_b,max_length = max_seq_len - 3)
#                 else:
#                     # Account for [CLS] and [SEP] with '-2'
#                     if len(tokens_a) > max_seq_len - 2:
#                         tokens_a = tokens_a[:max_seq_len - 2]
                tokens1 = [['[CLS]'] + token + ['[SEP]'] for token in tokens0]
                segment_ids1 = [[0] * len(tokens1[i]) if i%2==0 else [1] * len(tokens1[i]) for i in range(len(tokens1))]
                pos_ids1 = [[int(labels0[i])+(self.abs_position[int(labels0[i])][0]+self.abs_position[int(labels0[i])][1])*10]*len(tokens1[i]) for i in range(len(tokens1))]
                output_mask1 = [[1]+[0]*(len(token)-1) for token in tokens1]
                labels_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]+labels0+[self.tokenizer.convert_tokens_to_ids('[SEP]')]
                output_label_mask = [[1] + [0]*(len(labels_ids)-1)]
                
                tokens, segment_ids, pos_ids, output_mask = [], [], [], []
                for t in tokens1:
                    tokens += t
                for s in segment_ids1: 
                    segment_ids += s
                # segment_ids = segment_ids + [len(tokens1)%2]*len(labels_ids) # multi-label
                # segment_ids = segment_ids # single-label
                for p in pos_ids1: 
                    pos_ids += p
                for o in output_mask1: 
                    output_mask += o
                all_output_mask = output_mask[:]
                all_output_mask += output_label_mask[0]
                # input_ids = self.tokenizer.convert_tokens_to_ids(tokens) + labels_ids # multi-label
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens) # single-label
                input_mask = [1] * len(input_ids)
                padding = [0] * (max_seq_len - len(input_ids) - 20)
                input_len = len(input_ids)
#                 print(len(input_ids), len(input_mask), len(segment_ids))
                input_ids   += padding + [101] + [i for i in range(18)] + [102]
                input_mask  += padding + [0] + [0 for i in range(18)] + [0]
                segment_ids += padding + [0] + [0 for i in range(18)] + [0]
                pos_ids += padding + [0] + [0 for i in range(18)] + [0]
                output_mask += [0] * (max_seq_len - len(output_mask))
                all_output_mask += [0] * (max_seq_len - len(all_output_mask))
                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len
                assert len(pos_ids) == max_seq_len
                assert len(output_mask) == max_seq_len
                assert len(all_output_mask) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {example.guid}" % ())
                    logger.info(f"num: {example.num}" % ())
                    logger.info(f"num1: {example.num1}" % ())
                    logger.info(f"num2: {example.num2}" % ())
                    logger.info(f"turn: {example.turn}" % ())
                    logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
                    logger.info(f"pos_ids: {' '.join([str(x) for x in pos_ids])}")
                    logger.info(f"output_mask: {' '.join([str(x) for x in output_mask])}")
                    logger.info(f"all_output_mask: {' '.join([str(x) for x in all_output_mask])}")

                feature = InputFeature(num=example.num, 
                                    num1=example.num1, 
                                    num2=example.num2, 
                                    turn=example.turn, 
                                    input_ids = input_ids,
                                    input_mask = input_mask,
                                    segment_ids = segment_ids,
                                    pos_ids = pos_ids, 
                                    labels = labels0,
                                    multi_labels = multi_labels0, 
                                    input_len = input_len, 
                                    output_mask = output_mask, 
                                    all_output_mask = all_output_mask)
                features.append(feature)
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self,features,is_sorted = False):
        # Convert to Tensors and build dataset
        if is_sorted:
            logger.info("sorted data by th length of input")
            features = sorted(features,key=lambda x:x.input_len,reverse=True)
        # for f in features: 
        #     print(f.num, f.num1, f.num2, f.turn)
        num_s = torch.tensor([f.num[0][0] for f in features], dtype=torch.long)
        num1_s = torch.tensor([-1 for f in features], dtype=torch.long)
        num2_s = torch.tensor([f.num2[0][0] for f in features], dtype=torch.long)
        turn_s = torch.tensor([f.turn[0][-1] for f in features], dtype=torch.long)
        input_ids_s = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask_s = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        output_mask_s = torch.tensor([f.output_mask for f in features], dtype=torch.long)
        all_output_mask_s = torch.tensor([f.all_output_mask for f in features], dtype=torch.long)
        segment_ids_s = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        pos_ids_s = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
#         all_label_ids = torch.tensor([f.labels_ids for f in features],dtype=torch.long)
        input_lens_s = torch.tensor([f.input_len for f in features], dtype=torch.long)
        max_label_len = max([len(f.labels) for f in features])
        labels_s = torch.tensor([f.labels+[0]*(max_label_len-len(f.labels)) for f in features], dtype=torch.long)
        multi_labels_s = torch.tensor([f.multi_labels+[[0]*18]*(max_label_len-len(f.multi_labels)) for f in features], dtype=torch.long)
        dataset = TensorDataset(input_lens_s, num_s, num1_s, num2_s, turn_s, input_ids_s, input_mask_s, segment_ids_s, pos_ids_s, labels_s, multi_labels_s, output_mask_s, all_output_mask_s)
        return dataset

