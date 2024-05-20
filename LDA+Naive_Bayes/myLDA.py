# -*- coding: utf-8 -*-
"""PA-4-Problem-1-Task-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mXitR2UOX25-B5eP5P86Ap8V2hydcgbn
"""

#This is a helper code for problem-1 (Task-1) of PA-4
#Complete this code by writing the function definations
#Compute following terms and print them:\\
#1. Difference of class wise means = ${m_1-m_2}$\\
#2. Total Within-class Scatter Matrix $S_W$\\
#3. Between-class Scatter Matrix $S_B$\\
#4. The EigenVectors of matrix $S_W^{-1}S_B$ corresponding to highest EigenValue\\
#5. For any input 2-D point, print its projection according to LDA.

import csv
import numpy as np
import matplotlib as plt

def ComputeMeanDiff(X):
  m1 = np.array([0.0, 0.0])
  m2 = np.array([0.0, 0.0])
  n1 = 0
  n2 = 0
  for sample in X:
    if sample[2]==0:
      m1+=sample[:2]
      n1+=1
    else:
      m2+=sample[:2]
      n2+=1
  m1 = m1/n1
  m2 = m2/n2  
  return m1 - m2

def ComputeSW(X):
  sw1 = np.zeros((2, 2))
  sw2 = np.zeros((2, 2))
  m1 = np.array([0.0, 0.0])
  m2 = np.array([0.0, 0.0])
  n1 = 0
  n2 = 0
  for sample in X:
    if sample[2]==0:
      m1+=sample[:2]
      n1+=1
    else:
      m2+=sample[:2]
      n2+=1
  m1 = m1/n1
  m2 = m2/n2
  # Compute scatter matrices
  for sample in X:
        if sample[2] == 0:
            diff = (sample[:2] - m1)
            sw1 += np.dot(diff.reshape(-1, 1), diff.reshape(1, -1))
        else:
            diff = (sample[:2] - m2)
            sw2 += np.dot(diff.reshape(-1, 1), diff.reshape(1, -1))
  return sw1 + sw2

def ComputeSB(X):
  m1 = np.array([0.0, 0.0])
  m2 = np.array([0.0, 0.0])
  n1 = 0
  n2 = 0
  for sample in X:
    if sample[2]==0:
      m1+=sample[:2]
      n1+=1
    else:
      m2+=sample[:2]
      n2+=1
  m1 = m1/n1
  m2 = m2/n2
  sb = np.outer((m1-m2),np.transpose(m1-m2))
  return sb

def GetLDAProjectionVector(X):
    sw = ComputeSW(X)
    sb = ComputeSB(X)
    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(sw), sb))
    i = np.argmax(eigvals)
    return eigvecs[:, i]

def project(x,y,w):
    point = np.array([x, y])
    return np.dot(w, point)

#########################################################
###################Helper Code###########################
#########################################################

X = np.empty((0, 3))
with open('data.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for sample in csvFile:
        sample = [float(val) for val in sample]  # Convert elements to float
        X = np.vstack((X, sample))

print(X)
print(X.shape)
# X Contains (x,y) and class label 0.0 or 1.0

opt=int(input("Input your option (1-5): "))

match opt:
  case 1:
    meanDiff=ComputeMeanDiff(X)
    print(meanDiff)
  case 2:
    SW=ComputeSW(X)
    print(SW)
  case 3:
    SB=ComputeSB(X)
    print(SB)
  case 4:
    w=GetLDAProjectionVector(X)
    print(w)
  case 5:
    x=int(input("Input x dimension of a 2-dimensional point :"))
    y=int(input("Input y dimension of a 2-dimensional point:"))
    w=GetLDAProjectionVector(X)
    print(project(x,y,w))