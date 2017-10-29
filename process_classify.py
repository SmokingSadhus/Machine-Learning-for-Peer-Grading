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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords


sno = nltk.stem.SnowballStemmer('english')
ps = nltk.stem.PorterStemmer()
ls = nltk.stem.LancasterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()

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
        return sum([((s1_i - s2_i) * (s1_i - s2_i)) for s1_i, s2_i in zip(s1_array, s2_array)])

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
  

with open('DataSet4.csv') as csvfile:
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
                 train_temp.append([row['assignment_id'],row['reviewer_uid'], row['reviewee_team'],row['scores'], wrds_per_comment_filtered,repetition_score,sugc,locc,errc,idec,negc,posc,summc,nottc,solc,stdev])
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


X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print accuracy_score(preds, y_test)


gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)
print accuracy_score(preds, y_test)

#clf.predict()



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

