import pickle
import pandas
import torch
import random
import re

class Nlp_utils():
    def __int__(self):
        pass

    # demoji.download_codes()
    # @staticmethod
    def del_emoji(self,emojisentence):
        """
        tweet = "startspreadingthenews yankees win great start by 🎅🏾 going 5strong innings with 5k’s🔥 🐂"
        return: startspreadingthenews yankees win great start by  going 5strong innings with 5k’s

        """
        replace = demoji.replace(emojisentence)
        return (replace)

    # @staticmethod
    def tweet_tokenizer(self,data):
        tknzr = TweetTokenizer()
        new=tknzr.tokenize(self.del_emoji(data))
        return(' '.join(new))

    # @staticmethod
    def processTweet(self,tweet):
        tweet = tweet.lower()
        # tweet = " " + tweet
        tweet=re.sub('thats',"that's",tweet)
        tweet = re.sub('\(@\)', '@', tweet)
        tweet = re.sub(' u ', " you ", tweet)
        tweet = re.sub(' im ', " i'm ", tweet)
        tweet = re.sub('smarter then', "smarter than", tweet)
        tweet = re.sub('isnt', "isn't", tweet)
        tweet = re.sub('youre', "you're", tweet)
        tweet = re.sub('\\\/w', " ", tweet)
        tweet = re.sub(r'[^\x00-\x7F]+','', tweet)
        #tweet = tweet.replace(" rt "," ")
        # tweet = re.sub('@XXX','',tweet)
        # tweet = re.sub(' rt ','', tweet)
        tweet = re.sub('(\.)+','.', tweet)
        #tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+) | (http://[^\s]+))','URL',tweet)
        tweet = re.sub('((www\.[^\s]+))','',tweet)
        tweet = re.sub('((http://[^\s]+))','',tweet)
        tweet = re.sub('((https://[^\s]+))','',tweet)
        tweet = re.sub('@[^\s]+','',tweet)
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('_','',tweet)
        tweet = re.sub('\$','',tweet)
        tweet = re.sub('#', ' ', tweet)
        tweet = re.sub('%','',tweet)
        tweet = re.sub('^','',tweet)
        tweet = re.sub('&',' ',tweet)
        tweet = re.sub('\*','',tweet)
        tweet = re.sub('\(','',tweet)
        tweet = re.sub('\)','',tweet)
        tweet = re.sub('-','',tweet)
        tweet = re.sub('«', '', tweet)
        tweet = re.sub('»', '', tweet)
        tweet = re.sub('\+','',tweet)
        tweet = re.sub('=','',tweet)
        tweet = re.sub('"','',tweet)
        tweet = re.sub('~','',tweet)
        tweet = re.sub('`','',tweet)
        # tweet = re.sub('!',' ',tweet)
        # tweet = re.sub(':',' ',tweet)
        tweet = re.sub('^-?[0-9]+$','', tweet)
        tweet = tweet.strip('\'"')
        return tweet


    def nltk_tokenizer(self,data):
        tokens = nltk.word_tokenize(self.del_emoji(data))
        out=' '.join(tokens)
        out=out.replace('XXX','USER')
        out=out.replace('`','')
        out=out.replace('\t','')
        return out

def one_hot(x, class_count): 
    return torch.eye(class_count)[x, :]

def split_data(data, prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def one_hot(vector, num_labels): 
    o = [0] * num_labels
    for i in vector: 
        o[int(i)] = 1
    return o

lines = pandas.read_csv('data_input/multi-label-n/MDMDdata.tsv', sep='\t')
print(max(lines['Number']))
data = []
for item in lines:
    data.append(list(lines[item]))


data_trans = [[i[j] for i in data] for j in range(len(data[0]))]

# data_dict = [[i] for i in range(1, 6182)]
train_data_dict = [[] for i in range(1, max(lines['Number'])+1)]
dev_data_dict = [[] for i in range(1, max(lines['Number'])+1)]
test_data_dict = [[] for i in range(1, max(lines['Number'])+1)]
test_data_dict1 = [[] for i in range(1, max(lines['Number'])+1)]
test_data_dict2 = [[] for i in range(1, max(lines['Number'])+1)]
test_data_dict3 = [[] for i in range(1, max(lines['Number'])+1)]
test_data_dict4 = [[] for i in range(1, max(lines['Number'])+1)]

nlp_utils = Nlp_utils()
for d in data_trans: 
    if d[6] == 'train': 
        train_data_dict[d[0]-1].append([d[0], d[1], d[2], d[3], nlp_utils.processTweet(d[4]), int(d[5]), one_hot([int(d[5])], 18)]) 
    elif d[6] == 'dev': 
        labels = [int(i) for i in d[5].split(',')]
        dev_data_dict[d[0]-1].append([d[0], d[1], d[2], d[3], nlp_utils.processTweet(d[4]), labels[0], one_hot(labels, 18)]) 
    else: 
        labels = [int(i) for i in d[5].split(',')]
        test_data_dict[d[0]-1].append([d[0], d[1], d[2], d[3], nlp_utils.processTweet(d[4]), labels[0], one_hot(labels, 18)]) 

train_set = []
val_set = []
test_set = []
for i in train_data_dict:
    i = sorted(i, key=lambda x:x[0])
    if len(i) <= 1: 
        continue
    for k in [i[:j] for j in range(1,len(i)+1)]: 
        train_set.append(k)
for i in dev_data_dict:
    i = sorted(i, key=lambda x:x[0])
    if len(i) <= 1: 
        continue
    for k in [i[:j] for j in range(1,len(i)+1)]: 
        val_set.append(k)
for i in test_data_dict:
    i = sorted(i, key=lambda x:x[0])
    if len(i) <= 1: 
        continue
    for k in [i[:j] for j in range(1,len(i)+1)]: 
        test_set.append(k)

def train_test_split(dataset,test_pct):
    train, test = split_data(dataset, 1 - test_pct)
    return train, test

print(len(train_set), len(val_set), len(test_set))

with open("./data_set/MDMDdata.train.pkl", "wb") as ftr:
    pickle.dump(train_set, ftr)
with open("./data_set/MDMDdata.valid.pkl", "wb") as fte: 
    pickle.dump(val_set, fte)
with open("./data_set/MDMDdata.test.pkl", "wb") as ftt: 
    pickle.dump(test_set, ftt)
