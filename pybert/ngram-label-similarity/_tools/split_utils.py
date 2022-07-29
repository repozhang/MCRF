from itertools import chain
import pickle
from nltk.tokenize import TweetTokenizer
import demoji
import nltk

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

class Csv_utils():
    def __int__(self):
        pass


    # 处理csv dictreader，一般用来生成  {key:[list]}
    @staticmethod
    def get_number_list_dict(data,key,value):
        """
        :param data:  csv dict reader
        :param key: 根据key进行分类
        :param value: value append为一个list
        :return: [3254: [0, 0, 0]]
        """
        datadict={}
        for line in data:
            datadict.setdefault((line[key]), []).append(line[value])
        return datadict

    @staticmethod
    def save_pickle(data,filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            out = pickle.load(f)
            return out



class Dict_utils():
    def __int__(self):
        pass


    #反查key
    @staticmethod
    def get_key_sum(mydict, value):
        # for k, v in mydict.items():
        #     print(k,v)
        #     print(sum([int(i) for i in v]))
        #     if sum([int(i) for i in v]) == value:
        #         return k
        #     else:
        #         return None
        return [k for k, v in mydict.items() if sum([int(i) for i in v]) == value]

    @staticmethod
    def get_key(mydict, value):
        return [k for k, v in mydict.items() if v == value]


if __name__=="__main__":
    testdict1 = {'hi': ['1', '2']}
    out=Dict_utils.get_key_sum(testdict1,3)
    print(out)