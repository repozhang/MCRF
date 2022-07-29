import _pickle as cPickle
import sys
from emoji_to_text import del_emoji
sys.path.append('../_tools/')
import random
from itertools import chain
from csv_io import CsvProcessDict, CsvProcess
from split_utils import Csv_utils,Nlp_utils


def get_data(inputfile, split_idlist):

    cp1 = CsvProcessDict(inputfile, None)
    # print(cp.header())
    f1 = cp1.read_csv_dict()

    text_label_list=[[] for x in range(18)]
    for line in f1:
        if line['split'] == 'train':
            for k in range(0,18):
                if int(line['single_label'])==k:
                    text_label_list[k].append(del_emoji(line['utterance']))

    text_label_list_1=[]
    for i in text_label_list:
        text_label_list_1.append(' '.join(i))
    # print(len(text_label_list_1))
    return text_label_list_1 


if __name__=='__main__':
    inputfile = '../mturk/out1.tsv'
    split1_idlist = list(set(Csv_utils.load_pickle('../data_input/split_id/test_id_list.pickle')))
    split2_idlist = list(set(Csv_utils.load_pickle('../data_input/split_id/dev_id_list.pickle')))
    split_idlist=split1_idlist+split2_idlist
    get_data(inputfile,split_idlist)
