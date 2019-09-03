import time
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_model import *
from load_data import *

topk = 3
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
clas = 4 #num of classes
train_x_orig, train_y, train_y_onehot, test_x_orig, test_y, test_y_onehot, classes = load_datasets()
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_y = train_y.reshape(1, train_y.shape[0])
test_y = test_y.reshape(1, test_y.shape[0])
#train_x.shape = (dim, m) 12288=64*64*3
train_x = train_x_flatten
test_x = test_x_flatten

layers_dims = [train_x.shape[0], 10, 10, 10, clas] #5-layer model

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []    
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i%100 == 0:
            print("Cost after iteration {}:{}" .format(i, np.squeeze(cost)))
            costs.append(cost)
    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per tennns)')
    plt.title("Learning rate=" + str(learning_rate))
    plt.show()
    return parameters

if __name__ == '__main__':
    parameters = L_layer_model(train_x, train_y_onehot, layers_dims, num_iterations=5000, print_cost=True)
    pred_train, _, train_set_Top1, train_set_Top5 = predict(train_x, train_y_onehot, parameters)
    pred_test, topk_pred, test_set_Top1, test_set_Top5 = predict(test_x, test_y_onehot, parameters)
    
    pred_location = np.array(np.where(pred_test.T==1)).T
    lens = len(pred_location)
    for l in range(lens):
        print("第"+str(l+1)+"张图片是"+classes[int(pred_location[l][1])])
        print("第"+str(l+1)+"张图片的前"+str(topk)+"种可能情况为:")
        topk_class = topk_pred[l]
        print(str(classes[topk_class]))
        print("第"+str(l+1)+"张图片实际是"+classes[int(train_y[0][l])]+'\n')
        
    with open('parameters.model', 'wb') as file:
        pickle.dump(parameters, file)
        file.close()
        
    print("Train set Top1 Accuracy: " + str(train_set_Top1*100)+'%')
    print("Train set Top5 Accuracy: " + str(train_set_Top5*100)+'%')
    print("Test set Top1 Accuracy: " + str(test_set_Top1*100)+'%')
    print("Test set Top5 Accuracy: " + str(test_set_Top5*100)+'%')
    
    test_total, test_rate = compute_rate(test_y_onehot)
    pred_total, pred_rate = compute_rate(pred_test)
    print('\n')
    for l in range(clas):
        print(classes[l]+"的比例为:"+str(pred_rate[l][0]*100)+'%')
    print('\n')
    for l in range(clas):
        print(classes[l]+"的预测数量为:"+str(pred_total[l][0])+",真实数量为:"+str(int(test_total[l][0])))
#    print_mislabeled_images(classes, test_x, test_y, pred_test)