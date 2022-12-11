
import numpy as np
import pickle
import os 
from gensim.corpora import Dictionary

from scipy.spatial.distance import cosine
from .bsbi import BSBIIndex
from .compression import VBEPostings

# Create your models here.\

def vector_rep(text, model):
    dictionary = Dictionary()
    NUM_LATENT_TOPICS = 200
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

def load_model(modelName):
    current_filename = os.path.dirname(__file__) +'/result/'+modelName+'.pkl'
    with open(current_filename, 'rb') as f:
        return pickle.load(f)

def load_lgb_ranker(modelName):
    current_filename = os.path.dirname(__file__) +'/result/lgb_ranker_'+modelName+'.pkl'
    with open(current_filename, 'rb') as f:
        return pickle.load(f)

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


def eval_whole(k=100,query='the crystalline lens in vertebrates, including humans.'):
    lsi_model_saved = load_model("lsi_model")
    lgb_ranker_lsi_saved = load_lgb_ranker("lsi_model") 
    BSBI_instance = BSBIIndex(data_dir = os.path.dirname(__file__) +'/collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = os.path.dirname(__file__) +'/index') 
    
    list_of_docs = []
    docss = []
    docss_id = []

    

   
    for (_, doc) in BSBI_instance.retrieve_bm25(query, k = k):
        # d = doc.replace("\\", "/").split("collection")[1][1:]
        list_of_docs.append(r""+os.path.dirname(__file__) +"/collection/" + str(doc).replace("\\", "/")) 

    if len(list_of_docs) < 1:
        return None
    else:  

        for i in list_of_docs:
            with open(i, "r") as f:
                all_of_it= f.read()
                temp = int(i.split("/")[-1].replace(".txt",""))
                docss_id.append(temp)
                docss.append(all_of_it)  

        

        docuss = list(zip(docss_id, docss))

        scores = predict(query, docuss, lsi_model_saved[0], lgb_ranker_lsi_saved[0])
        sorted_did_scores = explain_score(query, docuss, scores)

        return sorted_did_scores