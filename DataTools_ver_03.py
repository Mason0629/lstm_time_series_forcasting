#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:45:15 2019

@author: wangderi
"""
import os
import numpy as np
import pandas as pd

def SearchObjects(directory, endwith='.csv'):
    """
    function uses to search files
    
    Args
    directory: the location of flies
    endwith: the extension of flies
    
    Returns
    labels & objects
    """
    directory = os.path.normpath(directory)
    if not os.path.isdir(directory):
        raise IOError("The directory " + directory + " is not exist")
    objects = {}
    for curpath, subdirs, files in os.walk(directory):
        for fileType in (file for file in files if file.endswith(endwith)):
            path = os.path.join(curpath, fileType)
            label = path.split(os.path.sep)[-2]
            if label not in objects:
                objects[label] = []
            objects[label].append(path)
            
    return objects[label]

#---------------------------------------------------------------------------------------------------------------

def GenerateData(file_name, n_inputs=10, pred_step=1, step=1):
    '''
    function uses to generate inputs for basical RNNs Model training or prediction.
    
    Args
    file_name : A csv file, shape = (Data, columName), last colum should be the target series
    n_inputs : RNN input window size, type int
    pred_step : RNN output step, type int
    
    Returns 
    return: train_x & train_y --> shape (dataSize, n_steps, n_inputs)
    type: ndarray 
    mode : "MinMax", "Standard"
    '''
    if type(file_name) is str:
        print("Read data from:" + file_name)
        df = pd.read_csv(file_name)
        array = np.array(df.iloc[:, 1:])
    else:
        print("Read data from an array...")
        array = file_name
    
    train_x, train_y = [], []
    if array.shape[1] > 1:#-->array为包含feature, label的二维数组
        #fetch features
        input_x = array[:, :-1].astype(np.float32)
        #fetch target series
        input_y = array[:, -1:].astype(np.float32)
        
    else:#-->array为只包含label的二维数组, 输入的array shape 为（-1, 1)
        #fetch features
        input_x = array.astype(np.float32)
        #fetch target series
        input_y = array.astype(np.float32)
    
    #可截取的大小
    assert input_x.shape[0] - n_inputs - pred_step + 1 > 0, '\n 错      误：数组无法切分，可能原因是序列长度不足 \n 解决办法：缩短输入/预测序列的长度'
    
    #cut the input_x & input_y to train_x & train_y 
    for i in range(0, input_x.shape[0] - n_inputs - pred_step + 1):
        indices = range(i, i + n_inputs, step)
        #train_x.append(input_x[i : i + n_inputs, :])
        train_x.append(input_x[indices])
        train_y.append(input_y[i + n_inputs : i + n_inputs + pred_step])
    train_x , train_y = np.array(train_x).astype(np.float32), np.array(train_y).reshape(-1, pred_step).astype(np.float32)

    return train_x, train_y

#---------------------------------------------------------------------------------------------------------------

def GenerateDataNLP(arr, seq_length):
    """
    """
    assert len(arr) - seq_length > 0, '\n 错      误：数组无法切分，可能原因是序列长度不足 \n 解决办法：缩短输入序列的长度'
    
    inputs = []
    targets = []
    for i in range(0, len(arr) - seq_length):
        inputs.append(arr[i : i + seq_length])
        targets.append(arr[i + 1 : i + seq_length + 1])
    
    inputs, targets = np.array(inputs).astype(np.float32), np.array(targets).astype(np.float32)
    
    return inputs, targets

#---------------------------------------------------------------------------------------------------------------

def FwdNormal(array, mode="Standard"):
    """
    Args
    array: An numpy array, shape = (?, n), last colum should be the target series
    
    Returns
    return: An array , input_array[:, -1]'s std & mean
    shape: array's shape will stay still
    """
    K, B = [], []
    
    if mode == "Standard":
        print("Enable Standard Processing...")
        for i in range(array.shape[1]):          
            K.append((array[:, i].std() + 0.001))
            B.append(array[:, i].mean())
            
        K = np.array(K)
        B = np.array(B)
        
        array = (array - B) / K
        
    elif mode == "MinMax":
        print("Enable MinMax Processing...")
        for i in range(array.shape[1]):         
            K.append(array[:, i].min())
            B.append(array[:, i].max())
            
        K = np.array(K)
        B = np.array(B)
        
        array = (array - K) / (B - K)
        
    
    return array, K[-1], B[-1]

#---------------------------------------------------------------------------------------------------------------

def RevNormal(norArray, K, B, mode="Standard"):
    """
    Args
    norArray : Model predicted Data shape(?, 1) or (1, ?)
    std : A return value from FwdNormal func, type float??
    mean : A return value from FwdNormal func, type float??
    
    Returns
    An array, which have a standard deviation is std and mean value is mean
    """
    if mode == "Standard":
        print("Enable Standard Rev...")
        revArray = norArray * K + B
    
    elif mode == "MinMax":
        print("Enable MinMax Rev...")
        revArray = norArray * (B - K) + K
        
    else:
        print("Check your paramaters...")
    
    return revArray