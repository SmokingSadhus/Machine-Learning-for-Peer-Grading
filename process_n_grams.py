import csv
import re
import nltk
import numpy as np
from nltk.util import ngrams
from scipy.stats.stats import pearsonr
from nltk.stem import PorterStemmer
#import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from textblob import TextBlob
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from nltk import ngrams
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

        
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
        if uni in unigram_map:
            unigram_map[uni] = unigram_map[uni] + 1
        else:
            unigram_map[uni] = 1

def add_to_bigram_map(bigrams):
    for bi in bigrams:
        if bi in bigram_map:
            bigram_map[bi] = bigram_map[bi] + 1
        else:
            bigram_map[bi] = 1

def add_to_trigram_map(trigrams):
    for tri in trigrams:
        if tri in trigram_map:
            trigram_map[tri] = trigram_map[tri] + 1
        else:
            trigram_map[tri] = 1

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
    uni = list(ngrams(nltk.word_tokenize(txt), 1))
    uni_final = []
    for lst in uni:
        uni_final.append(lst[0])
    bi = list(ngrams(nltk.word_tokenize(txt), 2))
    bi_final = []
    for lst in bi:
        bi_final.append(lst[0] +' '+ lst[1])
    tri = list(ngrams(nltk.word_tokenize(txt), 3))
    tri_final = []
    for lst in tri:
        tri_final.append(lst[0] + ' '+lst[1] + ' '+ lst[2])
    add_to_unigram_map(uni_final) 
    add_to_bigram_map(bi_final)
    add_to_trigram_map(tri_final)
    return [uni_final, bi_final, tri_final]



def filter_stop_words():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in cachedStopWords]) 

#row['comments'] 42320

def process_grades(g):
	g = g.strip("%")
	g = g.split(".")
	return int(g[0])

with open('DataSet5.csv') as csvfile:
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
                label.append(process_grades(row['grade_for_reviewer']))
                 
print ('Step1 done')

final_gram_map = dict()

for key in unigram_map.keys():
    if(unigram_map[key] > 100):
        final_gram_map[key] = 0

for key in bigram_map.keys():
    if(bigram_map[key] > 200):
        final_gram_map[key] = 0

for key in trigram_map.keys():
    if(trigram_map[key] > 200):
        final_gram_map[key] = 0

#exit()

for ts in train_temp:
    final_dict_cl = final_gram_map.copy()
    uni_list = ts[0]
    bi_list = ts[1]
    tri_list = ts[2]
    for uni in uni_list:
        if uni in final_dict_cl:
            final_dict_cl[uni]= final_dict_cl[uni]+1
    for bi in bi_list:
        if bi in final_dict_cl:
            final_dict_cl[bi]= final_dict_cl[bi]+1
    for tri in tri_list:
        if tri in final_dict_cl:
            final_dict_cl[tri]= final_dict_cl[tri]+1
    f_list = final_dict_cl.values()
    list_added = []
    for lt in f_list:
        list_added.append(lt)
    train.append(list_added)
    
print ('Step2 done')

#train = np.array(train)
#label = np.array(label)


X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.33, random_state=42)

reg1 = tree.DecisionTreeRegressor()
reg1 = reg1.fit(X_train, y_train)
preds = reg1.predict(X_test)

rms = sqrt(mean_squared_error(preds, y_test))
print (rms)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
reg2 = MLPRegressor(hidden_layer_sizes=(500,))
reg2 = reg2.fit(X_train, y_train)
preds2 = reg2.predict(X_test)

rms = sqrt(mean_squared_error(preds2, y_test))
print (rms)
