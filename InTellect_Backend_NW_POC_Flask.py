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
import pyhdb
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
from flask import Flask,redirect,url_for,jsonify
from flask.ext.cors import CORS, cross_origin
# End of Importing the required Libraries 

# Connecting to SAP HANA BFA Server
#connString = 'DRIVER={HDBODBC};SERVERNODE=vhmcdbfadb.mcd.rot.hec.sap.biz:30415;SERVERDB=BFA;UID=c5259455;PWD=' # Enter your CUSER and Password
#conn = pyodbc.connect(connString, autocommit=True)
#print(conn)
'''app = Flask(__name__)
CORS(app)
@app.route('/',methods=['GET','OPTIONS'])
def AITC():'''
conn = pyhdb.connect(
        host="vhmcdbfadb.mcd.rot.hec.sap.biz",
        port=30415,
        user="C5232886",
        password="Giri2017@Bfa",
        autocommit=True
)
    #try:
    #    conn = pyodbc.connect(connString, autocommit=True) #Open connection to SAP HANA C5233174, Ch@nd8985
    #    print('##### Connection Successful ! #####')
    #except pyodbc.Error as excep:
    #    sqlstate = excep.args[0]
    #    if sqlstate == 'HY000':
    #        print('##### Connection to Database failed #####')
    #        print('#####           Error HY000         #####')
    #        print('#########################################')
cur = conn.cursor() #Open a cursor  
#query = "Select * from NOC_DEV.NOC_LOGS"
cur1 = conn.cursor()
# End of Connecting to SAP HANA BFA Server
# Data selections and learning from TICKET_TRAIN table
print("=================================================")
print("Loading tickets training set... ")
print("=================================================")
readquerytrain = "SELECT * FROM C5232886.TICKET_TRAIN_NWPOC"
readquerytraintargetnames = "SELECT DISTINCT COMP_TRAIN FROM C5232886.TICKET_TRAIN_NWPOC"
cur.execute(readquerytrain)
cur1.execute(readquerytraintargetnames)
trainrows = cur.fetchall()
#print trainrows
traintargetnamesrows = cur1.fetchall()
#Trying python lists to collect data
ticket_train_data = []
ticket_train_num = []
ticket_train_component = []
ticket_train_target = []
ticket_category_dict ={0:"Acceleration",
                       1:"DNS",
                       2:"Internet"}
#End of Trying python Lists to collect data
for i in trainrows:
    ticket_train_data.append(i[0])
    ticket_train_num.append(i[2])
    if i[1] == ticket_category_dict[0]:
        ticket_train_target.append(0)
    if i[1] == ticket_category_dict[1]:
        ticket_train_target.append(1)
    if i[1] == ticket_category_dict[2]:
        ticket_train_target.append(2)
#    if i.COMP_TRAIN == ticket_category_dict[3]:
#        ticket_train_target.append(3)
#    if i.COMP_TRAIN == ticket_category_dict[4]:
#        ticket_train_target.append(4)
#    if i.COMP_TRAIN == ticket_category_dict[5]:
#        ticket_train_target.append(5)
#    if i.COMP_TRAIN == ticket_category_dict[6]:
#        ticket_train_target.append(6)
#    if i.COMP_TRAIN == ticket_category_dict[7]:
#        ticket_train_target.append(7)
    #print(i.TICKET_TRAIN_NUM, i.TICKET_TRAIN_DATA, i.COMP_TRAIN)
    #print('---------------------------------------------------')
print(ticket_train_num, ticket_train_target)
for i in traintargetnamesrows:
    ticket_train_component.append(i)
print(ticket_train_component)
print("%d Training tickets" % len(ticket_train_data))
print("%d Training categories" % len(ticket_train_component))
print("Extracting features from the dataset using a sparse vectorizer")
vectorizer = TfidfVectorizer(encoding='utf-8', analyzer='word',stop_words='english') #latin1, utf-8
X_train = vectorizer.fit_transform(ticket_train_data)

'''print ticket_train_data
print '\n'*3
print X_train
print '\n'*3'''
print("n_samples: %d, n_features: %d" % X_train.shape)
print(X_train)
'''assert sp.issparse(X_train)

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
readquerytest = "SELECT TICKET_TEST_NUM, TICKET_TEST_DATA FROM C5232886.TICKET_TEST_NWPOC"
cur3 = conn.cursor()
cur3.execute(readquerytest)
testrows = cur3.fetchall()
#Trying python lists to collect data
ticket_test_data = []
ticket_test_num = []
#End of Trying python Lists to collect data
for i in testrows:
    ticket_test_data.append(i[1])
    ticket_test_num.append(i[0])
print(ticket_test_num, ticket_test_data)
print("Extracting features from the dataset using the same vectorizer")
X_test = vectorizer.transform(ticket_test_data)
print X_test
print("n_samples: %d, n_features: %d" % X_test.shape)

# End of Data selections and prediction from TICKET_TEST table

# MultinomialNB classifiers

def MultinomialNBFunc(clf_class, params, name):
    clf = clf_class(**params).fit(X_train, y_train)
    pred = clf.predict(X_test)
    if pred == 0:
        valuetoinsertintodb = ticket_category_dict[0]
    if pred == 1:
        valuetoinsertintodb = ticket_category_dict[1]
    if pred == 2:
        valuetoinsertintodb = ticket_category_dict[2]
#    if pred == 3:
#        valuetoinsertintodb = ticket_category_dict[3]
#    if pred == 4:
#        valuetoinsertintodb = ticket_category_dict[4]
#    if pred == 5:
#        valuetoinsertintodb = ticket_category_dict[5]
#    if pred == 6:
#        valuetoinsertintodb = ticket_category_dict[6]
#    if pred == 7:
#        valuetoinsertintodb = ticket_category_dict[7]
    return valuetoinsertintodb
print("Testbenching a MultinomialNB classifier...")
parameters = {'alpha': 0.001}

predvalue = MultinomialNBFunc(MultinomialNB, parameters, 'MultinomialNB')
print predvalue
cur4 = conn.cursor()
insertpredquery = "UPDATE C5232886.TICKET_TEST_NWPOC SET COMP_TEST = '"+predvalue+"' WHERE TICKET_TEST_NUM = 'CAM1'"
#insertpredquery = "UPDATE C5232886.TICKET_TEST_NWPOC SET COMP_TEST ='Acceleration' WHERE TICKET_TEST_NUM = 'CAM1'"
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
#insert into "C5232886"."TICKET_TEST" values('CAM Issue','','CAM1')'''
    
