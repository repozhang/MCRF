r"""Functional interface"""
import torch
from tqdm import tqdm
import numpy as np
# from sklearn.metrics import  
from sklearn.metrics import f1_score, classification_report, dcg_score, precision_score, recall_score

__call__ = ['Accuracy','AUC','F1Score','JaccardScore','EntityScore','ClassReport','MultiLabelReport','AccuracyThresh','ExactRatio','PatK', 'nDCG']

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class Accuracy(Metric):
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK
        self.reset()

    def __call__(self, logits, target):
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct_k = correct[:self.topK].view(-1).float().sum(0)
        self.total = target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k)  / self.total

    def name(self):
        return 'accuracy'


class AccuracyThresh(Metric):
    def __init__(self,thresh = 0.5):
        super(AccuracyThresh,self).__init__()
        self.thresh = thresh
        self.reset()

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid()
        self.y_true = target

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        data_size = self.y_pred.size(0)
        acc = np.mean(((self.y_pred>self.thresh)==self.y_true.byte()).float().cpu().numpy(), axis=1).sum()
        return acc / data_size

    def name(self):
        return 'accuracy_thresh'


class AUC(Metric):

    def __init__(self,task_type = 'binary',average = 'binary'):
        super(AUC, self).__init__()

        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average

    def __call__(self,logits,target):
        if self.task_type == 'binary':
            self.y_prob = logits.sigmoid().data.cpu().numpy()
        else:
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def value(self):
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        return auc

    def name(self):
        return self.average + '_auc'

class F1Score(Metric):
    def __init__(self,thresh = 0.5, normalizate = False, task_type = 'binary',average = 'binary',search_thresh = False):
        super(F1Score).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(float)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            # self.y_pred = np.argmax(y_prob, 1)
            self.y_pred = (y_prob > self.thresh).astype(float) # multi-label
            # self.y_pred = torch.zeros(torch.from_numpy(y_prob).shape).scatter_(dim=1,index=torch.argmax(torch.from_numpy(y_prob),dim=1).unsqueeze(1),value=1).numpy().astype(int) # single-label

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        print(self.y_true.shape)
        print(sum(sum(self.y_true)))
        p = precision_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        r = recall_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        # print(self.average, p, r)
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return p,r,f1
        # return f1

    def name(self):
        return self.average + '_f1'

class JaccardScore(Metric):
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(JaccardScore).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average
        self.inte = 0
        self.unit = 1

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target
        y_prob = logits

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - Jaccard Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            # self.y_pred = np.argmax(y_prob, 1)
            self.y_pred = (y_prob > self.thresh).long() # multi-label
            # self.y_pred = y_prob # single_label
            self.inte = torch.sum((self.y_pred == target) * self.y_pred, dim=1)
            self.unit = torch.sum(self.y_pred, dim=1) + torch.sum(target, dim=1)
            


    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        jaccard = self.inte / self.unit
        return torch.mean(jaccard)

    def name(self):
        return 'jaccard'

class Hamming(Metric):
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(Hamming).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average
        self.inte = 0

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target
        y_prob = logits

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - Jaccard Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            self.y_pred = y_prob
            self.inte = torch.sum(self.y_pred == target, dim=1) / self.y_pred.size(-1)
            


    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        
        hamming = self.inte
        return torch.mean(hamming)

    def name(self):
        return 'hamming'

class ExactRatio(Metric):
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(ExactRatio).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average
        self.inte = 0

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target
        y_prob = logits

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - Jaccard Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            # self.y_pred = np.argmax(y_prob, 1)
            self.y_pred = (y_prob > self.thresh).long() # multi-label
            # self.y_pred = y_prob # single-label
            # print(self.y_pred, target)
            self.inte = torch.sum((self.y_pred == target) * self.y_pred, dim=1) / self.y_pred.size(-1)
            


    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        
        exactratio = self.inte
        return torch.mean(exactratio)

    def name(self):
        return 'exactratio'

class PatK(Metric):
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(PatK).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average
        self.inte = 0

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target
        y_prob = logits.softmax(dim=1)

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - Jaccard Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            # self.y_pred = np.argmax(y_prob, 1)
            # self.y_pred = (y_prob > self.thresh).long() # multi_label
            self.y_pred = y_prob # single-label
            self.inte = torch.sum(self.y_pred * target, dim=1) / self.y_pred.size(-1)

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        
        PatK = self.inte
        return torch.topK(PatK)

    def name(self):
        return 'Pat1'



class nDCG(Metric):
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(nDCG).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh ).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            # self.y_pred = np.argmax(y_prob, 1)
            self.y_pred = (y_prob > self.thresh).astype(float)
            # self.y_pred = y_prob

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        # print(self.y_true)
        # print(self.y_pred)
        ndcg = dcg_score(y_true=self.y_true, y_score=self.y_pred)
        return ndcg

    def name(self):
        return 'nDCG'


class PatK(Metric):
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(PatK).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh ).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            # self.y_pred = np.argmax(y_prob, 1)
            self.y_pred = (y_prob > self.thresh).astype(float) # multi-label
            # self.y_pred = torch.zeros(torch.from_numpy(y_prob).shape).scatter_(dim=1,index=torch.argmax(torch.from_numpy(y_prob),dim=1).unsqueeze(1),value=1).numpy().astype(int) # single-label

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        # print(self.y_true)
        # print(self.y_pred)
        PatK = precision_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return PatK

    def name(self):
        return 'PatK'


class ClassReport(Metric):
    '''
    class report
    '''
    def __init__(self,target_names = None):
        super(ClassReport).__init__()
        self.target_names = target_names

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        score = classification_report(y_true = self.y_true,
                                      y_pred = self.y_pred,
                                      target_names=self.target_names)
        print(f"\n\n classification report: {score}")

    def __call__(self,logits,target):
        _, y_pred = torch.max(logits.data, 1)
        self.y_pred = y_pred.cpu().numpy()
        self.y_true = target.cpu().numpy()

    def name(self):
        return "class_report"

class MultiLabelReport(Metric):
    '''
    multi label report
    '''
    def __init__(self,id2label = None):
        super(MultiLabelReport).__init__()
        self.id2label = id2label

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def __call__(self,logits,target):

        self.y_prob = logits.sigmoid().data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        for i, label in self.id2label.items():
            # print('self.y_prob[:, i]',self.y_prob[:, i])
            # print('self.y_true[:, i]',self.y_true[:, i])
            try:
                auc = roc_auc_score(y_score=self.y_prob[:, i], y_true=self.y_true[:, i])
            except:
                auc = 0
            # print(f"label:{label} - auc: {auc:.4f}")

    def name(self):
        return "multilabel_report"

