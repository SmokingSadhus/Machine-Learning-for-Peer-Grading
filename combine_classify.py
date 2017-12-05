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
from nltk import ngrams
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import csv
import re
import nltk
import numpy as np
import inflect
from nltk.util import ngrams
from scipy.stats.stats import pearsonr
from nltk.stem import PorterStemmer
#import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.corpus import wordnet
from sklearn.feature_selection import mutual_info_classif

sno = nltk.stem.SnowballStemmer('english')
ps = nltk.stem.PorterStemmer()
ls = nltk.stem.LancasterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()
p = inflect.engine()
pc = 0

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

def get_class_value(score):
    if score <= 50:
        return 0
    elif score > 50 and score <=70:
        return 1
    elif score > 70 and score <=85:
        return 2
    elif score > 85 and score <=95:
        return 3
    elif score > 95:
        return 4

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
                label.append(get_class_value(process_grades(row['grade_for_reviewer'])))
            
                
                 
                 
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

train = np.array(train)
label = np.array(label)

#X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.33, random_state=42)

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X_train, y_train)
#preds = clf.predict(X_test)
#print (accuracy_score(preds, y_test))

#print ('Step3 done')

#gnb = GaussianNB()
#gnb = gnb.fit(X_train, y_train)
#preds = gnb.predict(X_test)
#print (accuracy_score(preds, y_test))

#print('Step4 done')
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(label)))
x_shuffled = train[shuffle_indices]
y_shuffled = label[shuffle_indices]

dev_sample_index = -1 * int(0.1 * float(len(label)))
X_train, X_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(500,16), random_state=1)
nn = nn.fit(X_train, y_train)

#pred_nn = nn.predict(X_test)
#print (accuracy_score(pred_nn, y_test))

def get_activations(clf, X):
    hidden_layer_sizes = clf.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [X.shape[1]] + hidden_layer_sizes + \
        [clf.n_outputs_]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(np.empty((X.shape[0], layer_units[i + 1])))
    clf._forward_pass(activations)
    return activations

acti_train = get_activations(nn,np.array(X_train))
acti_test = get_activations(nn,np.array(X_test))

nn_ops_train = acti_train[2]
nn_ops_test = acti_test[2]

###############Ngram over##################3
################KE start#######################################

x = []
y =[]
z = []
sugp=[]
locp=[]
errp=[]
idep=[]
negp=[]
posp=[]
summp=[]
nottp=[]
solp=[]

sug = ['should', 'must', 'might', 'could', 'need', 'needs', 'maybe', 'try', 'revision', 'want']
loc = ['page', 'paragraph', 'sentence']
err = ['error', 'mistakes', 'typo', 'problem', 'difficulties', 'conclusion']
ide = ['consider', 'mention']
neg = ['fail', 'hard', 'difficult', 'bad', 'short', 'little', 'bit', 'poor', 'few', 'unclear', 'only', 'more']
pos = ['great', 'good', 'well', 'clearly', 'easily', 'effective', 'effectively', 'helpful', 'very']
summ = ['main', 'overall', 'also', 'how', 'job']
nott = ['not', 'doesn\'t','don\'t']
sol = ['revision', 'specify', 'correction']

##### Stemming #################
#for i in range(0, len(sug)):
#    sug[i] = ls.stem(sug[i])
#for i in range(0, len(loc)):
#    loc[i] = ls.stem(loc[i])
#for i in range(0, len(err)):
#    err[i] = ls.stem(err[i])
#for i in range(0, len(ide)):
#    ide[i] = ls.stem(ide[i])
#for i in range(0, len(neg)):
#    neg[i] = ls.stem(neg[i])
#for i in range(0, len(pos)):
#    pos[i] = ls.stem(pos[i])
#for i in range(0, len(summ)):
#    summ[i] = ls.stem(summ[i])
#for i in range(0, len(nott)):
#    nott[i] = ls.stem(nott[i])
#for i in range(0, len(sol)):
#    sol[i] = ls.stem(sol[i])


def getngramscore(tokens,i):
    count = 0
    wlist = list()
    for ngram in ngrams(tokens, i):
        if ngram in wlist:
            count = count + 1
        else:
            wlist.append(ngram)
    return count * i
            
def getrepetitionvalue(tokens):
    ngramscore = 0
    for i in range(1 , len(tokens)):
        ngramscore = ngramscore + getngramscore(tokens, i)
    return ngramscore

def getcontents(tokens):
    tot = len(tokens)
    if tot == 0:
        return 0,0,0,0,0,0,0,0,0
    sugcount = 0
    loccount = 0
    errcount = 0
    idecount = 0
    negcount = 0
    poscount = 0
    summcount = 0
    nottcount = 0
    solcount = 0
    for token in tokens:
#        token = ls.stem(token)
        if token in sug:
            sugcount = sugcount + 1
        if token in loc:
            loccount = loccount + 1
        if token in err:
            errcount = errcount + 1
        if token in ide:
            idecount = idecount + 1
        if token in neg:
            negcount = negcount + 1
        if token in pos:
            poscount = poscount + 1
        if token in summ:
            summcount = summcount + 1
        if token in nott:
            nottcount = nottcount + 1
        if token in sol:
            solcount = solcount + 1
#    return (sugcount * 1.0)/(tot)  ,  (loccount * 1.0)/(tot), (errcount * 1.0)/(tot), (idecount * 1.0)/(tot), (negcount * 1.0)/(tot) , (poscount * 1.0)/(tot) , (summcount * 1.0)/(tot) ,  (nottcount * 1.0)/(tot) , (solcount * 1.0)/(tot)
    return (sugcount * 1.0)/(tot)  ,  (loccount * 1.0)/(tot), (errcount * 1.0)/(tot), (idecount * 1.0)/(tot), (negcount * 1.0)/(tot) , (poscount * 1.0)/(tot) , (summcount * 1.0)/(tot) ,  (nottcount * 1.0)/(tot) , (solcount * 1.0)/(tot)

def get_class_value(score):
    if score <= 50:
        return 0
    elif score > 50 and score <=70:
        return 1
    elif score > 70 and score <=85:
        return 2
    elif score > 85 and score <=95:
        return 3
    elif score > 95:
        return 4

def c_to_int(string):
    try:
        return int(string)
    except ValueError:
        return 0

train_temp = []        
train = []
label = []

assgn_score_similarit = dict()
assgn_subm_simmilarit = dict()

def populate_assgn_subm_similarit(assign, reviewee, scores):
    key = assign + '' + reviewee
    if key in assgn_subm_simmilarit:
        assgn_subm_simmilarit[key].append(scores)
    else:
        slist = []
        slist.append(scores)
        assgn_subm_simmilarit[key] = slist

def populate_assgn_score_similarit(assign , reviewer, scores):
    key = assign + '' + reviewer
    if key in assgn_score_similarit:
        assgn_score_similarit[key].append(scores)
    else:
        slist = []
        slist.append(scores)
        assgn_score_similarit[key] = slist
        
def compute_diff(s1, s2):
    s1_array = [c_to_int(scr) for scr in s1.split(",")]
    s2_array = [c_to_int(scr) for scr in s2.split(",")]
    if(len(s1_array) != len(s2_array)):
        return 'NA'
    else:
        return sqrt((sum([((s1_i - s2_i) * (s1_i - s2_i)) for s1_i, s2_i in zip(s1_array, s2_array)]))/len(s1_array))

def compute_diff_from_other_scores(scores,list_of_scores):
    count = 0
    diff_score = 0
    for sc in list_of_scores:
        val = compute_diff(scores, sc)
        if val != 'NA':
            diff_score = diff_score + val
            count = count + 1
    if count ==0 :
        return -1
    return (diff_score * 1.0)/count

def compute_score_similarity(list_of_scores):
    countn = 0
    sim_score = 0
    for i in range(0,len(list_of_scores)-1):
        for j in range(i+1,len(list_of_scores)):
            val_sc = compute_diff(list_of_scores[i], list_of_scores[j])
            if( val_sc != 'NA'):
                sim_score = sim_score + val_sc
                countn = countn + 1
    if countn == 0:
        return -1
    else:
        return (sim_score * 1.0)/countn

def count_wout_stop_words(sentence, no_of_comments):
    sentence = sentence.lower()
    words = re.findall(r'\w+', sentence)
    count = 0
    for word in words:
        if word not in  stopwords.words('english'):
            count = count + 1
    return count/no_of_comments

def process_grades(g):
	g = g.strip("%")
	g = g.split(".")
	return int(g[0])


########### Code for calculating QnA correlation ####################

def return_POS_tags(txt):
    blob = TextBlob(txt)
    np_array = blob.noun_phrases
    txt = txt.lower()
    ret_array = []
    for np in np_array:
        txt = txt.replace(np, 'N',1)
        ret_array.append([np,'NP'])
    text = nltk.word_tokenize(txt)
    tags = nltk.pos_tag(text)
    for word in tags:
        if word[0] not in stopwords.words('english') and word[0] != 'N':
            ret_array.append(word)
    return ret_array

def add_similar_words(word,pos_tag):
    ret_set = set()
    syns = []
    if pos_tag == 'NN' or pos_tag == 'NNS' or pos_tag == 'NNP' or pos_tag == 'NNPS':
        syns = wordnet.synsets(word,pos='n')
    elif pos_tag == 'VB' or pos_tag == 'VBD' or pos_tag == 'VBG' or pos_tag == 'VBN' or pos_tag == 'VBP' or pos_tag == 'VBZ':
        syns = wordnet.synsets(word,pos='v')
    elif pos_tag == 'JJ' or pos_tag == 'JJR' or pos_tag == 'JJS':
        syns = wordnet.synsets(word,pos='a') + wordnet.synsets(word,pos='s')
    elif pos_tag == 'RB' or pos_tag == 'RBR' or pos_tag == 'RBS':
        syns = wordnet.synsets(word,pos='r')
    elif pos_tag == 'NP':
        ret_set.add(word)
    for syn in syns:
        for l in syn.lemmas():
            name = l.name()
            name = name.replace('_',' ')
            ret_set.add(name)
    return ret_set


def count_occurences_ignoring_tense(array, word):
    count = 0
    for pl_word in array:
        if pl_word.startswith('\'') != True and p.compare(pl_word,word) != False:
            count = count + 1
    return count


def compute_qna_match_score(q,a):
    #print q + '/////' + a
    q = ''.join(i for i in q if ord(i)<128)
    a = ''.join(i for i in a if ord(i)<128)
    ori_q_len = len(nltk.word_tokenize(q))
    pos_array = return_POS_tags(q)
    a = a.lower()
    w_a = nltk.word_tokenize(a)
    main_ct = 0
    for p_w in pos_array:
        syns = add_similar_words(p_w[0],p_w[1])
        ct = 0
        for syn in syns:
            ct = ct + count_occurences_ignoring_tense(w_a, syn)
        main_ct = main_ct + ct
    return main_ct/(ori_q_len * 1.0)



def get_q_n_a_score(questions, answers):
    global pc
    q_array = questions.split('*!!')
    a_array = answers.split('*!!')
    if len(q_array) != len(a_array):
        pc = pc + 1
#        print 'Serious Problem in Data'
        return 0
    ret_score = 0
    for i in (0,len(q_array)-1):
        ret_score = ret_score + compute_qna_match_score(q_array[i],a_array[i])
    return ret_score/(len(a_array) * 1.0)



#quit()

########### Code for calculating QnA correlation ####################

  

with open('DataSet5.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        populate_assgn_score_similarit(row['assignment_id'],row['reviewer_uid'],row['scores'])
        populate_assgn_subm_similarit(row['assignment_id'],row['reviewee_team'], row['scores'])
        if row['grade_for_reviewer'] != 'NULL':
            noofcomments = len(row['scores'].split(","))
            if noofcomments != 0 :
#                print(row['assignment_id'],row['reviewer_uid'],row['reviewee_team'],len(re.findall(r'\w+', row['comments']))/noofcomments,getrepetitionvalue(re.findall(r'\w+', row['comments'])) ,row['grade_for_reviewer'])
                 words_per_comment = len(re.findall(r'\w+', row['comments']))/noofcomments
                 wrds_per_comment_filtered = count_wout_stop_words(row['comments'],noofcomments)
                 repetition_score = getrepetitionvalue(re.findall(r'\w+', row['comments']))
                 x.append(wrds_per_comment_filtered)
                 y.append(getrepetitionvalue(re.findall(r'\w+', row['comments'])))
                 z.append(process_grades(row['grade_for_reviewer']))
                 (sugc , locc , errc , idec , negc , posc , summc , nottc , solc) = getcontents(re.findall(r'\w+', row['comments']))
                 sugp.append(sugc)
                 locp.append(locc)
                 errp.append(errc)
                 idep.append(idec)
                 negp.append(negc)
                 posp.append(posc)
                 summp.append(summc)
                 nottp.append(nottc)
                 solp.append(solc)
                 stdev = np.std([c_to_int(scr) for scr in row['scores'].split(",")])
                 q_n_a_score = get_q_n_a_score(row['questions'],row['comments'])
                 clean = ''.join(i for i in row['comments'] if ord(i)<128)
                 blob = TextBlob(clean)
                 train_temp.append([row['assignment_id'],row['reviewer_uid'], row['reviewee_team'],row['scores'], wrds_per_comment_filtered,repetition_score,sugc,locc,errc,idec,negc,posc,summc,nottc,solc,stdev,q_n_a_score,blob.sentiment.subjectivity])
                 label.append(get_class_value(process_grades(row['grade_for_reviewer'])))


for ts in train_temp:
    key1 = ts[0] + ''+ ts[1]
    key2 = ts[0] + '' + ts[2]
    scores = ts[3]
    simi_score = compute_score_similarity(assgn_score_similarit[key1])
    diff_score = compute_diff_from_other_scores(scores, assgn_subm_simmilarit[key2])
    l_appen = list()
    for l in range(4, len(ts)):
        l_appen.append(ts[l])
    l_appen.append(simi_score)
    l_appen.append(diff_score)
    train.append(l_appen)

train = np.array(train)
label = np.array(label)


x_shuffled = train[shuffle_indices]
y_shuffled = label[shuffle_indices]

#dev_sample_index = -1 * int(0.1 * float(len(label)))
X_train_ke, X_test_ke = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train_ke, y_test_ke = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

#################################################################################################################

###################### Combine outputs from NN with those obtaiined from KE ##################################

comb_tr = []
comb_tst = []

for i in range(0,len(nn_ops_train)):
    comb_tr.append(np.concatenate((nn_ops_train[i], X_train_ke[i]), axis = 0))

for i in range(0,len(nn_ops_test)):
    comb_tst.append(np.concatenate((nn_ops_test[i] , X_test_ke[i]), axis = 0))

print (len(comb_tr))
print (len(comb_tst))

t = tree.DecisionTreeClassifier()
t = t.fit(comb_tr, y_train_ke)#same as y_train
preds = t.predict(comb_tst)
print (accuracy_score(preds, y_test_ke))#same as y_test
