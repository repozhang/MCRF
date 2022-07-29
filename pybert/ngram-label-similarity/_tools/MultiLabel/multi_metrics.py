from sklearn.metrics import hamming_loss
import numpy as np

def hmloss():
    y_true = np.array([[1, 1], [1, 1]])
    y_pred = np.array([[0, 1], [1, 1]])
    output=hamming_loss(y_true, y_pred)
    print(output)


# y_true = [2, 2, 3, 4]
# y_pred = [1, 2, 3, 4]
# hamming_loss(y_true, y_pred)
if __name__=="__main__":
    hmloss()