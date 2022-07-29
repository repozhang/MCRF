import re
import math
import numpy as np
from itertools import chain
from collections import Counter
import nltk
from nltk.util import ngrams  # This is the ngram magic.
from textblob import TextBlob
from get_label_ngram_feature2 import get_data
import sys
sys.path.append('../_tools/')
from split_utils import Csv_utils, Nlp_utils

NGRAM = 2

re_sent_ends_naive = re.compile(r'[.\n]')
re_stripper_alpha = re.compile('[^a-zA-Z]+')
re_stripper_naive = re.compile('[^a-zA-Z\.\n]')

splitter_naive = lambda x: re_sent_ends_naive.split(re_stripper_naive.sub(' ', x))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def get_tuples_nosentences(txt):
    """Get tuples that ignores all punctuation (including sentences)."""
    if not txt: return None
    ng = ngrams(re_stripper_alpha.sub(' ', txt).split(), NGRAM)
    return list(ng)


def get_tuples_manual_sentences(txt):
    """Naive get tuples that uses periods or newlines to denote sentences."""
    if not txt: return None
    sentences = (x.split() for x in splitter_naive(txt) if x)
    ng = (ngrams(x, NGRAM) for x in sentences if len(x) >= NGRAM)
    return list(chain(*ng))


def get_tuples_nltk_punkt_sentences(txt):
    """Get tuples that doesn't use textblob."""
    if not txt: return None
    sentences = (re_stripper_alpha.split(x) for x in sent_detector.tokenize(txt) if x)
    # Need to filter X because of empty 'words' from punctuation split
    ng = (ngrams(filter(None, x), NGRAM) for x in sentences if len(x) >= NGRAM)
    return list(chain(*ng))


def get_tuples_textblob_sentences(txt):
    """New get_tuples that does use textblob."""
    if not txt: return None
    tb = TextBlob(txt)
    ng = (ngrams(x.words, NGRAM) for x in tb.sentences if len(x.words) > NGRAM)
    return [item for sublist in ng for item in sublist]


def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(a)
    b = set(b)
    return 1.0 * len(a & b) / len(a | b)


def cosine_similarity_ngrams(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


def test():
    paragraph = """It was the best of times, it was the worst of times.
               It was the age of wisdom? It was the age of foolishness!
               I first met Dr. Frankenstein in Munich; his monster was, presumably, at home."""
    print(paragraph)
    _ = get_tuples_nosentences(paragraph);
    print("Number of N-grams (no sentences):", len(_));
    _

    _ = get_tuples_manual_sentences(paragraph);
    print("Number of N-grams (naive sentences):", len(_));
    _

    _ = get_tuples_nltk_punkt_sentences(paragraph);
    print("Number of N-grams (nltk sentences):", len(_));
    _

    _ = get_tuples_textblob_sentences(paragraph);
    print("Number of N-grams (TextBlob sentences):", len(_));
    _

    a = get_tuples_nosentences("It was the best of times.")
    b = get_tuples_nosentences("It was the worst of times.")
    print("Jaccard: {}   Cosine: {}".format(jaccard_distance(a, b), cosine_similarity_ngrams(a, b)))

    a = get_tuples_nosentences("Above is a bad example of four-gram similarity.")
    b = get_tuples_nosentences("This is a better example of four-gram similarity.")
    print("Jaccard: {}   Cosine: {}".format(jaccard_distance(a, b), cosine_similarity_ngrams(a, b)))

    a = get_tuples_nosentences("Jaccard Index ignores repetition repetition repetition repetition repetition.")
    b = get_tuples_nosentences("Cosine similarity weighs repetition repetition repetition repetition repetition.")
    print("Jaccard: {}   Cosine: {}".format(jaccard_distance(a, b), cosine_similarity_ngrams(a, b)))


def test(a,b):
    a = get_tuples_nosentences(a)
    b = get_tuples_nosentences(b)
    jaccard_score=jaccard_distance(a, b)
    cosine_score=cosine_similarity_ngrams(a,b)
    return jaccard_score, cosine_score

def get_score(inputfile, split_idlist):

    data=get_data(inputfile, split_idlist)
    mylist=range(1,18)
    import itertools
    # cc = list(itertools.permutations(mylist, 2))
    cc = [p for p in itertools.product(mylist, repeat=2)]
    # cc = list(itertools.combinations_with_replacement(mylist, 2))
    print(len(cc))
    # d = deque(mylist)
    # d.rotate(-1)
    # for i,j in zip(d,mylist):
    #     print(i,j)
    import numpy
    # x = np.empty((17,17), dtype = float)
    # print(x)
    jaccard_score_list=[]
    cosine_score_list=[]
    for i in cc:
        jaccard_score, cosine_score=test(data[i[0]],data[i[1]])
        # print(i[0],i[1],jaccard_score,cosine_score)
        jaccard_score_list.append(jaccard_score)
        cosine_score_list.append(cosine_score)
    jaccard_score_np = np.array(jaccard_score_list)
    cosine_score_np = np.array(cosine_score_list)
    # x[1][1]=jaccard_score_list[1]
    # print(x)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    array_cos = cosine_score_np.reshape(17,17)
    array_jac = jaccard_score_np.reshape(17,17)

    df_cos= pd.DataFrame(array_cos, range(17), range(17))
    df_jac= pd.DataFrame(array_jac, range(17),range(17))
    df_jac.columns = [str(i) for i in range(1,18)]
    df_jac.index = [str(i) for i in range(1,18)]
    # df_jac=df_jac.set_axis([str(i) for i in range(1,18)], axis='index')
    df_cos.columns = [str(i) for i in range(1,18)]
    df_cos.index = [str(i) for i in range(1, 18)]

    df_cos.to_csv('Cosine({}-gram).csv'.format(NGRAM),index=True)
    df_jac.to_csv('Jaccard({}-gram).csv'.format(NGRAM),index=True)

    # plt.figure(figsize=(10,7))
    f = plt.figure(1)
    sn.set(font_scale=1.4)  # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    # cmap color: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    sn.heatmap(df_cos, annot=False, annot_kws={"size": 10},cmap="OrRd")  
    plt.suptitle('Cosine similarity ({}-gram)'.format(NGRAM))
    f.savefig('Cosine({}-gram).png'.format(NGRAM))

    g = plt.figure(2)
    sn.heatmap(df_jac, annot=False, annot_kws={"size": 10},cmap="OrRd") 
    plt.suptitle('Jaccard similarity ({}-gram)'.format(NGRAM))
    g.savefig('Jaccard({}-gram).png'.format(NGRAM))


if __name__=="__main__":
    # a="Jaccard Index ignores repetition repetition repetition repetition repetition."
    # b="Cosine similarity weighs repetition repetition repetition repetition repetition."
    # jaccard_score,cosine_score=test(a,b)
    # print(jaccard_score,cosine_score)
    split1_idlist = list(set(Csv_utils.load_pickle('test_id_list.pickle')))
    split2_idlist = list(set(Csv_utils.load_pickle('dev_id_list.pickle')))
    split_idlist = split1_idlist + split2_idlist
    inputfile='final-2.tsv'
    get_score(inputfile, split_idlist)