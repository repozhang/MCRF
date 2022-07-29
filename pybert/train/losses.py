from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss, BCELoss


__call__ = ['CrossEntropy','BCEWithLogLoss', 'BCE_Loss']

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss

class BCE_Loss(object):
    def __init__(self):
        self.loss_fn = BCELoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss
