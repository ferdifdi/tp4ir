#installing library
#!pip install lightgbm
#!pip install gensim

import subprocess
import lightgbm as lgb
import numpy as np
import random
import pickle
import os 
import re

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.models import Word2Vec
from gensim.corpora import Dictionary

from scipy.spatial.distance import cosine
from .bsbi import BSBIIndex
from .compression import VBEPostings

# training dataset 
def init_data():
    proc = subprocess.Popen("wget -c https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz -P data".split())
    proc.wait()
    proc = subprocess.Popen("tar -xvf data/nfcorpus.tar.gz".split())
    proc.wait()

def corpus_docs():
    documents = {}
    with open(os.path.dirname(__file__) +"/nfcorpus/train.docs") as file:
        for line in file:
            doc_id, content = line.split("\t")
            documents[doc_id] = content.split()
    return documents

def corpus_queries():
    queries = {}
    with open(os.path.dirname(__file__) +"/nfcorpus/train.vid-desc.queries", encoding='utf-8') as file:
        for line in file:
            q_id, content = line.split("\t")
            queries[q_id] = content.split()
    return queries

def make_dataset(documents, queries):
    NUM_NEGATIVES = 1

    q_docs_rel = {} # grouping by q_id terlebih dahulu
    with open(os.path.dirname(__file__) +"/nfcorpus/train.3-2-1.qrel") as file:
        for line in file:
            q_id, _, doc_id, rel = line.split("\t")
            if (q_id in queries) and (doc_id in documents):
                if q_id not in q_docs_rel:
                    q_docs_rel[q_id] = []
                q_docs_rel[q_id].append((doc_id, int(rel)))

    # group_qid_count untuk model LGBMRanker
    group_qid_count = []
    dataset = []
    for q_id in q_docs_rel:
        docs_rels = q_docs_rel[q_id]
        group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
        for doc_id, rel in docs_rels:
            dataset.append((queries[q_id], documents[doc_id], rel))
        # tambahkan satu negative (random sampling saja dari documents)
        dataset.append((queries[q_id], random.choice(list(documents.values())), 0))
    return dataset, group_qid_count

#Building LSI/LSA Model
NUM_LATENT_TOPICS = 200
dictionary = Dictionary()

def model(documents, modelName):   
    bow_corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
    
    if(modelName=='lsi_model'):
        myModel = LsiModel(bow_corpus, num_topics = NUM_LATENT_TOPICS) # 200 latent topics
        save_model(myModel,'lsi_model')

    return myModel

    # test melihat representasi vector dari sebuah dokumen & query
def vector_rep(text, model):
    rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
    return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

def features(query, doc, model):
    v_q = vector_rep(query, model)
    v_d = vector_rep(doc, model)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)
    return v_q + v_d + [jaccard] + [cosine_dist]

def separate_dataset(dataset, model):
    X = []
    Y = []
    for (query, doc, rel) in dataset:
        X.append(features(query, doc, model))
        Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

    
#Training the Ranker

def training_with_lgbm(X, Y, group_qid_count, modelName):
    ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)
    ranker.fit(X, Y,
           group = group_qid_count,
           verbose = 10)
    ranker_pred = ranker.predict(X)
    save_lgb_ranker(ranker, modelName)

def predict(query, docs, model, ranker):
    # bentuk ke format numpy array
    X_unseen = []
    for doc_id, doc in docs:
        X_unseen.append(features(query.split(), doc.split(), model))

    X_unseen = np.array(X_unseen)

    # hitung scores
    scores = ranker.predict(X_unseen)
    return scores

def explain_score(query, docs, scores):
    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    # print("query        :", query)
    # print("SERP/Ranking :")
    # for (did, score) in sorted_did_scores:
    #     print(did, score)

    return sorted_did_scores

def save_model(model, modelName):
    current_filename = os.path.dirname(__file__) +'/result/'+modelName+'.pkl'
    with open(current_filename, 'wb') as f:
        pickle.dump([model], f)

def load_model(modelName):
    current_filename = os.path.dirname(__file__) +'/result/'+modelName+'.pkl'
    with open(current_filename, 'rb') as f:
        return pickle.load(f)

def save_lgb_ranker(lgb_ranker, modelName):
    current_filename = os.path.dirname(__file__) +'/result/lgb_ranker_'+modelName+'.pkl'
    with open(current_filename, 'wb') as f:
        pickle.dump([lgb_ranker], f)

def load_lgb_ranker(modelName):
    current_filename = os.path.dirname(__file__) +'/result/lgb_ranker_'+modelName+'.pkl'
    with open(current_filename, 'rb') as f:
        return pickle.load(f)

def eval_whole(k=100,query='the crystalline lens in vertebrates, including humans.'):
    # documents = corpus_docs()
    # queries = corpus_queries()
    # dataset, group_qid_count = make_dataset(documents, queries)

    # lsi_model = model(documents, 'lsi_model')
    # X_lsi,Y_lsi = separate_dataset(dataset, lsi_model)

    # training_with_lgbm(X_lsi, Y_lsi, group_qid_count, 'lsi_model')

    lsi_model_saved = load_model("lsi_model")
    lgb_ranker_lsi_saved = load_lgb_ranker("lsi_model")  

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')
    
    list_of_docs = []
    docss = []
    docss_id = []

    try:
        for (_, doc) in BSBI_instance.retrieve_bm25(query, k = k):
            list_of_docs.append(r""+os.path.dirname(__file__) +"\\collection\\" + str(doc))
        for i in list_of_docs:
            with open(i, "r") as f:
                all_of_it= f.read()
                temp = int(i.split("\\")[-1].replace(".txt",""))
                docss_id.append(temp)
                docss.append(all_of_it)    

        docuss = list(zip(docss_id, docss))

        scores = predict(query, docuss, lsi_model_saved[0], lgb_ranker_lsi_saved[0])
        sorted_did_scores = explain_score(query, docuss, scores)
        return sorted_did_scores

    except Exception as e:
        return None
    
if __name__ == "__main__":
    # init_data()
    eval_whole()

    

    


