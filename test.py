import pickle
from dnn_model import *

with open('parameters.model', 'rb') as file:
    parameters = pickle.load(file)
    
X = np.array([[1.4, 2.7, 3.3, 2.9]]).T
y = np.array([[0, 0, 1, 0]]).T
y_class = 2
pred_test, topk_pred, _, _ = predict(X, y, parameters)
pred_location = np.array(np.where(pred_test.T==1)).T
lens = len(pred_location)
for l in range(lens):
    print("第"+str(l+1)+"张图片是"+classes[int(pred_location[l][1])])
    print("第"+str(l+1)+"张图片实际是"+classes[y_class]+'\n')
    print("第"+str(l+1)+"张图片的前"+str(topk)+"种可能情况为:")
    topk_class = topk_pred[l]
    print(str(classes[topk_class])+'\n')
