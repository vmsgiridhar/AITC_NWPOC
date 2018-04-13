# -*- coding: utf-8 -*-
"""
Licensed
Created on Mon Jul 17 21:20:22 2017

@author: C5232886

Script to perform CRUD operation on C5232886 Schema of BFA Server TICKET_TRAIN, TICKET_TEST tables
Script works as a backend controller for InTellect Ticket Tool.
"""
# Importing the required Libraries
import pyodbc
#from __future__ import print_function
from time import time
import itertools
import sys
import os
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB      
# End of Importing the required Libraries 

# Connecting to SAP HANA BFA Server
connString = 'DRIVER={HDBODBC};SERVERNODE=vhmcdbfadb.mcd.rot.hec.sap.biz:30415;SERVERDB=BFA;UID=;PWD='
conn = pyodbc.connect(connString, autocommit=True)
print(conn)
#try:
#    conn = pyodbc.connect(connString, autocommit=True) #Open connection to SAP HANA , 
#    print('##### Connection Successful ! #####')
#except pyodbc.Error as excep:
#    sqlstate = excep.args[0]
#    if sqlstate == 'HY000':
#        print('##### Connection to Database failed #####')
#        print('#####           Error HY000         #####')
#        print('#########################################')
cur = conn.cursor() #Open a cursor  
cur1 = conn.cursor()
# End of Connecting to SAP HANA BFA Server

# Data feeding into the Database Tables
#file = open('C:/Users/C5232886/Desktop/HANA/MachineLearning_Learn/ML_SVM/TACP/news20/MT01/train/srsmteam/srsm_2.txt', 'rb') #Open file in read-only and binary   
#content = file.read() #Save the content of the file in a variable  
#cur.execute("INSERT INTO C5232886.TICKET_TRAIN VALUES(?,?,?)", (content,'srsmteam','2')) #Save the content to the table  
#cur.execute("COMMIT") #Save the content to the table
# Data feeding into the Database Tables

# Data selections and learning from TICKET_TRAIN table
print("=================================================")
print("Loading tickets training set... ")
print("=================================================")
readquerytrain = "SELECT * FROM C5232886.TICKET_TRAIN"
readquerytraintargetnames = "SELECT DISTINCT COMP_TRAIN FROM C5232886.TICKET_TRAIN"
cur.execute(readquerytrain)
cur1.execute(readquerytraintargetnames)
trainrows = cur.fetchall()
traintargetnamesrows = cur1.fetchall()
#Trying python lists to collect data
ticket_train_data = []
ticket_train_num = []
ticket_train_component = []
ticket_train_target = []
#End of Trying python Lists to collect data
for i in trainrows:
    ticket_train_data.append(i.TICKET_TRAIN_DATA)
    ticket_train_num.append(i.TICKET_TRAIN_NUM)
    if i.COMP_TRAIN == "camteam":
        ticket_train_target.append(0)
    if i.COMP_TRAIN == "databasemanagement":
        ticket_train_target.append(1)
    if i.COMP_TRAIN == "servermanagement":
        ticket_train_target.append(2)
    if i.COMP_TRAIN == "srsmteam":
        ticket_train_target.append(3)
    #print(i.TICKET_TRAIN_NUM, i.TICKET_TRAIN_DATA, i.COMP_TRAIN)
    #print('---------------------------------------------------')
print(ticket_train_num, ticket_train_data, ticket_train_target)
for i in traintargetnamesrows:
    ticket_train_component.append(i.COMP_TRAIN)
print(ticket_train_component)
print("%d Training tickets" % len(ticket_train_data))
print("%d Training categories" % len(ticket_train_component))
print("Extracting features from the dataset using a sparse vectorizer")
vectorizer = TfidfVectorizer(encoding='utf-8') #latin1, utf-8
X_train = vectorizer.fit_transform(ticket_train_data)
print("n_samples: %d, n_features: %d" % X_train.shape)
assert sp.issparse(X_train)
y_train = ticket_train_target

# Continue work from here 18-July-2017
# End of Data selections and learning from TICKET_TRAIN table

# Data selections and prediction from TICKET_TEST table
# TICKET_TEST will have the ticket data, component, ticket number. This Python script runs and predicts the output and insert into the component.
#readquerytest = "SELECT * FROM C5232886.TICKET_TEST WHERE TICKET_TEST_NUM = 'CAM1'" # To Read content of test ticket.
#cur.execute(readquerytest) # To Read the test ticket data
#testrow = cur.fetchone()
#print(testrow.TICKET_TEST_NUM, testrow.TICKET_TEST_DATA, testrow.COMP_TEST) # COMP_TEST would be empty. Not predicted yet.
print("=================================================")
print("Loading tickets testing set... ")
print("=================================================")
readquerytest = "SELECT TICKET_TEST_NUM, TICKET_TEST_DATA FROM C5232886.TICKET_TEST"
cur3 = conn.cursor()
cur3.execute(readquerytest)
testrows = cur3.fetchall()
#Trying python lists to collect data
ticket_test_data = []
ticket_test_num = []
#End of Trying python Lists to collect data
for i in testrows:
    ticket_test_data.append(i.TICKET_TEST_DATA)
    ticket_test_num.append(i.TICKET_TEST_NUM)
print(ticket_test_num, ticket_test_data)
print("Extracting features from the dataset using the same vectorizer")
X_test = vectorizer.transform(ticket_test_data)
print("n_samples: %d, n_features: %d" % X_test.shape)

# End of Data selections and prediction from TICKET_TEST table

# Benchmark classifiers

def benchmark(clf_class, params, name):
    clf = clf_class(**params).fit(X_train, y_train)
    pred = clf.predict(X_test)
    if pred == 0:
        valuetoinsertintodb = "camteam"
    if pred == 1:
        valuetoinsertintodb = "databasemanagement"
    if pred == 2:
        valuetoinsertintodb = "servermanagement"
    if pred == 3:
        valuetoinsertintodb = "srsmteam"
    return valuetoinsertintodb
#Change content in database: UPDATE "C5232886"."TICKET_TEST" SET "TICKET_TEST_DATA" = 'Sybase' WHERE "TICKET_TEST_NUM" = 'CAM1';
# End of Benchmark classifiers

# Inserting the data into COMP_TEST column

# End of Inserting the data into COMP_TEST column
print("Testbenching a MultinomialNB classifier...")
parameters = {'alpha': 0.01}

predvalue = benchmark(MultinomialNB, parameters, 'MultinomialNB')
cur4 = conn.cursor()
insertpredquery = "UPDATE C5232886.TICKET_TEST SET COMP_TEST = '"+predvalue+"' WHERE TICKET_TEST_NUM = 'CAM1'"
print(insertpredquery)
cur4.execute(insertpredquery)
#Closing all the connections
#file.close() #Close the file
cur.close() #Close the cursor  
cur1.close()
cur3.close()
cur4.close()
#conn.close() #Close the connection 
#End of Closing all the connections

#Commands
#insert into "C5232886"."TICKET_TEST" values('CAM Issue','','CAM1')
