import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#natural language tootl kit
import nltk
from nltk import word_tokenize
import string
from nltk.stem import WordNetLemmatizer

import math

text = "We've learned about methods for regression and for classification involving predictors and for making predictions from our data. Well, ideally, we'd like to get a new sample from the population and see how well our predictions do. Well, we don't always have new data. And we can't use our training data just straight off, because it's going to be a little bit optimistic. So we're going to tell you about cross-validation which is very clever device for using the same training data to tell you how well your prediction method works."
print(text)

text = text.lower()
text1 = word_tokenize(text)
POS_tag = nltk.pos_tag(text1)
print(POS_tag)

wordnet_lemmatizer = WordNetLemmatizer()
adjective_tags = ['JJ','JJR','JJS']
lemmatized_text = []
for word in POS_tag:
    if word[1] in adjective_tags:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
    else:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun

POS_tag = nltk.pos_tag(lemmatized_text)

stopwords = []
wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW']
for word in POS_tag:
    if word[1] not in wanted_POS:
        stopwords.append(word[0])
not_required=['a', 'about', 'above', 'after' , 'again' , 'against', 'all', 'am', 'an' , 'and', 'any', 'are',
              'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could',
              'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
              'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself',
              'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it',  'it\'s', 'its', 'itself',
              'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
              'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so',
              'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these',
              'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t',
              'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which',
              'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re',
              'you\'ve', 'your', 'yours', 'yourself', 'yourselves']
fin_words=[]
for word in POS_tag:
    if word[0] not in not_required and word[0] not in stopwords:
        fin_words.append(word[0])
stopwordplus=[]
for word in POS_tag:
    if word[0] in not_required:
        stopwordplus.append(word[0])
stopwordplus=stopwords+stopwordplus

# generates uniques values
vocabulary = list(set(fin_words))
print (vocabulary)

# creating a weighted undirected graph
# If weighted_edge[i][j] is zero, it means no edge or connection is present between the words represented by index i and j.
vocab_len = len(vocabulary)
weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)
score = np.zeros((vocab_len),dtype=np.float32)
window_size = 3
covered_coocurrences = []
for i in range(0,vocab_len):
    score[i]=1
    for j in range(0,vocab_len):
        if j==i:
            weighted_edge[i][j]=0
        else:
            for window_start in range(0,(len(fin_words)-window_size)):

                window_end = window_start+window_size

                window = fin_words[window_start:window_end]

                if (vocabulary[i] in window) and (vocabulary[j] in window):

                    index_of_i = window_start + window.index(vocabulary[i])
                    index_of_j = window_start + window.index(vocabulary[j])

                    if [index_of_i,index_of_j] not in covered_coocurrences:
                        weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                        covered_coocurrences.append([index_of_i,index_of_j])

# inout[i] will contain the total no. of undirected connections\edges associated withe the vertex represented by i.
inout = np.zeros((vocab_len),dtype=np.float32)
for i in range(0,vocab_len):
    for j in range(0,vocab_len):
        inout[i]+=weighted_edge[i][j]

#giving score to each vertex depending on the number of connections it has
#d is the damping factor
# score[i] = (1-d) + d x [ Summation(j) ( (weighted_edge[i][j]/inout[j]) x score[j] ) ]
MAX_ITERATIONS = 50
d=0.85
threshold = 0.0001 #convergence threshold
for iter in range(0,MAX_ITERATIONS):
    prev_score = np.copy(score)
    for i in range(0,vocab_len):
        summation = 0
        for j in range(0,vocab_len):
            if weighted_edge[i][j] != 0:
                summation += (weighted_edge[i][j]/inout[j])*score[j]
        score[i] = (1-d) + d*(summation)
    if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
        print ("Converging at iteration " + str(iter) + "....")
        break

for i in range(0,vocab_len):
    print ("Score of "+vocabulary[i]+": "+str(score[i]))

#Paritioning lemmatized_text into phrases using the stopwords in it as delimeters. The phrases are also candidates for keyphrases to be extracted.
# phrases will be formed when there are no stop words in between the two words of lemmatized_text
phrases = []
phrase = " "
for word in lemmatized_text:
    if word in stopwordplus:
        if phrase!= " ":
            phrases.append(str(phrase).strip().split())
        phrase = " "
    elif word not in stopwordplus:
        phrase+=str(word)
        phrase+=" "
print (phrases)

unique_phrases = []
for phrase in phrases:
    if phrase not in unique_phrases:
        unique_phrases.append(phrase)
print (unique_phrases)

#scoring keyphases
#scoring is done on the basis of score of each vertex calculated above
phrase_scores = []
keywords = []
for phrase in unique_phrases:
    phrase_score=0
    keyword = ''
    for word in phrase:
        keyword += str(word)
        keyword += " "
        phrase_score+=score[vocabulary.index(word)]
    phrase_scores.append(phrase_score)
    keywords.append(keyword.strip())
i=0
for keyword in keywords:
    print ("Keyword: '"+str(keyword)+"', Score: "+str(phrase_scores[i]))
    i+=1

sorted_index = np.flip(np.argsort(phrase_scores),0)
keywords_num = 5
for i in range(0,keywords_num):
    print (str(keywords[sorted_index[i]])+", ")
