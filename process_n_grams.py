import csv
import re
import nltk
import numpy as np
from nltk.util import ngrams
from scipy.stats.stats import pearsonr
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from textblob import TextBlob
from nltk import ngrams

        
def c_to_int(string):
    try:
        return int(string)
    except ValueError:
        return 0
        
train_temp = []
train = []
label = []


unigram_map = dict()
bigram_map = dict()
trigram_map = dict()

def add_to_unigram_map(unigrams):
    for uni in unigrams:
        unigram_map[uni] = 0
        
            

def add_to_bigram_map(bigrams):
    for bi in bigrams:
        bigram_map[bi] = 0

def add_to_trigram_map(trigrams):
    for tri in trigrams:
        trigram_map[tri] = 0


def return_n_grams_with_nouns_replaced(txt):
    blob = TextBlob(txt)
    np_array = blob.noun_phrases
    text = nltk.word_tokenize(txt)
    n_array = nltk.pos_tag(text)
    txt = txt.lower()
    for np in np_array:
        txt = txt.replace(np, 'N',1)
    for n in n_array:
        if (n[1] == 'NN' or n[1] == 'NNP' or n[1] == 'NNS' or n[1] == 'NNPS'):
            txt = txt.replace(n[0].lower(),'N',1)
    add_to_unigram_map(list(ngrams(nltk.word_tokenize(txt), 1))) 
    add_to_bigram_map(list(ngrams(nltk.word_tokenize(txt), 2)))
    add_to_trigram_map(list(ngrams(nltk.word_tokenize(txt), 3)))
    return [list(ngrams(nltk.word_tokenize(txt), 1)), list(ngrams(nltk.word_tokenize(txt), 2)),list(ngrams(nltk.word_tokenize(txt), 3)) ]

#row['comments'] 42320

with open('DataSet2.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0
    for row in reader:
        count = count + 1
#        print count
        if row['grade_for_reviewer'] != 'NULL':
            noofcomments = len(row['scores'].split(","))
            if row['comments'] != '':
                uni_code_text = ''.join(i for i in row['comments'] if ord(i)<128)
                list_of_uni_bi_tri = return_n_grams_with_nouns_replaced(uni_code_text)
                train_temp.append(list_of_uni_bi_tri)
                label.append(int(row['grade_for_reviewer']))
            
                
                 
                 
print 'Step1 done'

for ts in train_temp:
    uni_dict_cl = unigram_map.copy()
    bi_dict_cl = bigram_map.copy()
    tri_dict_cl = trigram_map.copy()
    uni_list = ts[0]
    bi_list = ts[1]
    tri_list = ts[2]
    for uni in uni_list:
        uni_dict_cl[uni]= uni_dict_cl[uni]+1
    for bi in bi_list:
        bi_dict_cl[bi]= bi_dict_cl[bi]+1
    for tri in tri_list:
        tri_dict_cl[tri]= tri_dict_cl[tri]+1
    f_list = uni_dict_cl.values()
    f_list = f_list + bi_dict_cl.values()
    f_list = f_list + tri_dict_cl.values()
    train.append(f_list)
    
print 'Step2 done'

X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.33, random_state=42)
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
preds = clf.predict(X_test)

rms = sqrt(mean_squared_error(preds, y_test))

print rms

#print sugp
#print errp
#print negp
#print solp

#print pearsonr(x,z)           
#print pearsonr(y,z)        
#plt.figure(1)
#plt.subplot(211)
#plt.plot(x,z,'ro')
#plt.subplot(212)
#plt.plot(y,z,'ro')
#plt.axis([0, 10000, 0, 100])
#plt.subplots_adjust(hspace = 0.8)
#plt.savefig('visualization.png')

plt.figure(1)
plt.subplot(411)
plt.plot(sugp,z,'ro')
plt.xlabel('Suggestion Percent')
plt.subplot(412)
plt.plot(locp,z,'ro')
plt.xlabel('location percent')
plt.subplot(413)
plt.plot(errp,z,'ro')
plt.xlabel('error terms percent')
plt.subplot(414)
plt.plot(summp,z,'ro')
plt.xlabel('summarising terms percent')

plt.subplots_adjust(hspace = 0.8)
plt.savefig('visualization_1.png')

