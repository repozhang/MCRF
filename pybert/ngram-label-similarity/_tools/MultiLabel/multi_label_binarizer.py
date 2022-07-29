

from sklearn.preprocessing import MultiLabelBinarizer

def binarizer(input,label_class):
    # label_class=['0','1','2','3','4','5','6','7','8','9','10', '11', '12', '13', '14', '15', '16', '17']
    mlb = MultiLabelBinarizer(classes=label_class)
    out_array=mlb.fit_transform(input)
    class_order=mlb.classes_
    return out_array,class_order

if __name__=="__main__":
    input=[(1, 2), (3,)]
    label_class=[1,2,3]
    print(binarizer(input,label_class))
