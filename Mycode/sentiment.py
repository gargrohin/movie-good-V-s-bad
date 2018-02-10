#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
#import sys
from time import time
#sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf=SVC(C=10000,kernel="rbf")
#features_train=features_train[:int(len(labels_train)/100)]
#labels_train=labels_train[:int(len(labels_train)/100)]
to=time()
clf.fit(features_train,labels_train)
print('training time:',round(time()-to,3),'s')
#print(clf.score(features_test,labels_test))
from sklearn.metrics import accuracy_score
pred=clf.predict(features_test)
print(accuracy_score(pred,labels_test))
#print(pred[26],pred[50],pred[10])
#count=0
#for i in range(len(labels_test)):
#	if pred[i]==0:
#		count+=1
#print(count)
#print(len(labels_test))
#########################################################


