from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os
import urllib.request
 
import numpy as np
import tensorflow as tf

clas = 4
def onehot_encode(Y, C):

    Y_onehot = (np.arange(C)[:,None]  == Y[None,:]).astype(np.float32)    #broadcasting
    Y_onehot = np.squeeze(Y_onehot)
    
    return Y_onehot

def load_datasets():
    
    train_dataset = "iris_training.csv"
    test_dataset = "iris_test.csv"
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=train_dataset,
      target_dtype=np.int,
      features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=test_dataset,
      target_dtype=np.int,
      features_dtype=np.float32)
    
    train_set_x_orig = training_set.data
    train_set_y_orig = training_set.target
    test_set_x_orig = test_set.data
    test_set_y_orig = test_set.target
    train_y_onehot = onehot_encode(train_set_y_orig, C=clas)
    test_y_onehot = onehot_encode(test_set_y_orig, C=clas)
    classes = np.array(['玫瑰', '菊花', '鸡冠花', '康乃馨', '红掌', '牡丹', '紫罗兰', '郁金香', '鸢尾花', '荷花',
                        '栀子花', '铃兰', '牵牛花', '一串红', '百合', '茉莉花', '杜鹃花', '桂花', '马蹄莲', '杏花',
                        '梨花', '马兰花'])
    
    return train_set_x_orig, train_set_y_orig, train_y_onehot, test_set_x_orig, test_set_y_orig, test_y_onehot, classes

def compute_rate(test_y_onehot):
    classes_total = np.sum(test_y_onehot, axis=1, keepdims=True)
    m = test_y_onehot.shape[1]
    classes_rate = classes_total/m
    return classes_total, classes_rate

if __name__ == '__main__':
    train_x, train_y, train_y_onehot, test_x, test_y, test_y_onehot, classes = load_datasets()
    s, b = compute_rate(test_y_onehot)
    print(s.shape)
    print(b.shape)