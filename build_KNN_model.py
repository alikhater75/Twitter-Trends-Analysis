import re
import random
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from gensim.models.doc2vec import Doc2Vec
from joblib import dump


path = 'the path your file are located at'
DocToVec = Doc2Vec.load(path + "DocToVec.d2v")



def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) # strip html tags
    sent = re.sub(r'(\w)\'(\w)', '\1\2', sent) # remove apostrophes
    sent = re.sub(r'\W', ' ', sent) # remove punctuation
    sent = re.sub(r'\s+', ' ', sent) # remove repeated spaces
    sent = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', sent) #remove any single letter
    sent = sent.strip()
    return sent.split()


sentences = []
sentvecs = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]: 
    with open("sentiment labelled sentences/%s_labelled.txt" % fname, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split('\t')
            sentences.append(line_split[0])
            words = extract_words(line_split[0])
            sentvecs.append(DocToVec.infer_vector(words, steps=10)) # create a vector for this document
            sentiments.append(int(line_split[1]))
            
# shuffle sentences, sentvecs, sentiments together
combined = list(zip(sentences, sentvecs, sentiments))
random.shuffle(combined)
sentences, sentvecs, sentiments = zip(*combined)
sentvecs = pd.DataFrame(sentvecs)
sentiments = pd.DataFrame(sentiments)



# Build kNearestNeighbors model
KNN_model = KNeighborsClassifier() 
#Hyper Parameters Set
param_grid = {
    'n_neighbors': list(range(1,30,3)),
    'n_jobs':[-1]
    }

#Making models with hyper parameters sets
Tuned_KNN = GridSearchCV(KNN_model, param_grid, cv=10)

#Learning
Tuned_KNN.fit(sentvecs, sentiments.values.ravel())

# saving the trained model
dump(Tuned_KNN, 'KNN_model.joblib') 
