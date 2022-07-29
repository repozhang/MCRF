
import numpy as np

def atl(input):
    # a = np.array([[1, 2], [3, 4]])
    # >>> list(a)
    # [array([1, 2]), array([3, 4])]
    # out=list(input)
    out=input.tolist()
    return out
    # print(out)

if __name__=="__main__":
    input=np.array([[1, 2], [3, 4]])
    out=atl(input)
    print(out)