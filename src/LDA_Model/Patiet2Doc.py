'''
Created on Jun 30, 2017

@author: longgu
'''

import codecs
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation


import numpy as np
import pandas as pd


def load_Preprocess():
    
    df = pd.read_csv("~/data/hab_test.csv", dtype={'hab_test.tot_visit': np.float32,'hab_test.tot_visit_ed': np.float32,'hab_test.tot_visit_acute': np.float32,'hab_test.age_in_2015': np.float32}, low_memory=False)
    df.fillna(0)
    
    df.columns = [feature.split('.')[1] for (feature_idx, feature) in enumerate(df.columns)]
    
    #the total visiting number is not more than 5
    df.tot_visit[df.tot_visit <=5.0] = 0
    
    #the total visiting number is between 6 and 10
    df.tot_visit[(df.tot_visit > 5.0) & (df.tot_visit <=10.0)] = 1
    
    #the total visiting number is more than 10
    df.tot_visit[df.tot_visit >10.0] = 2
    
    #the age of patient is not more than 12
    df.age_in_2015 [df.age_in_2015 <=12.0] = 0
    #the age of patient is is between 13 and 19
    df.age_in_2015 [(df.age_in_2015 > 12.0) & (df.age_in_2015 <=19.0)] = 1
    #the age of patient is is between 20 and 30
    df.age_in_2015 [(df.age_in_2015 > 19.0) & (df.age_in_2015 <=30.0)] = 2
    #the age of patient is is between 31 and 45
    df.age_in_2015 [(df.age_in_2015 > 30.0) & (df.age_in_2015 <=45.0)] = 3
    #the age of patient is is between 46 and 60
    df.age_in_2015 [(df.age_in_2015 > 45.0) & (df.age_in_2015 <=60.0)] = 4
    #the age of patient is more than 60
    df.age_in_2015 [df.age_in_2015 > 60.0] = 5
    
    df = pd.get_dummies(data=df, columns=['tot_visit', 'age_in_2015','gender',  'latest_hospital_type','latest_adm_year',  'latest_adm_month'])
    df = df.drop(labels = ['row_number',  'person_id',  'tot_visit_ed', 'tot_visit_acute'], axis=1)
    
#     smaple = df.head(10)
    if os.path.exists("transformedDoc.txt"):
        os.remove("transformedDoc.txt")
    p2d  = codecs.open('transformedDoc.txt',"a+","utf-8") 
    docs = []
    for i in range(0,len( df.index)):
        cols =  df.columns[( df > 0 ).iloc[i]]
        vals = ' '.join(cols._data)
        vals = vals.replace("tot_visit_0.0","tot_visit_no_more_than_5")\
                     .replace("tot_visit_1.0","tot_visit_between_6_and_10")\
                     .replace("tot_visit_2.0","tot_visit_more_than_10")\
                     .replace("age_in_2015_0.0","age_no_more_than_12")\
                     .replace("age_in_2015_1.0","age_between_13_and_19")\
                     .replace("age_in_2015_2.0","age_between_20_and_30")\
                     .replace("age_in_2015_3.0","age_between_31_and_45")\
                     .replace("age_in_2015_4.0","age_between_46_and_60")\
                     .replace("age_in_2015_5.0","age_more_than_60")\
                     .replace("gender_3.0","gender_Other")\
                     .replace("gender_1.0","gender_Male")\
                     .replace("gender_2.0","gender_Female")\
                     .replace("latest_hospital_type_1","latest_hospital_Service_type_Acute_care")\
                     .replace("latest_hospital_type_2","latest_hospital_Service_type_Rehabilitation_care")\
                     .replace("latest_hospital_type_4","latest_hospital_Service_type_Geriatric_evaluation_and_management")\
                     .replace("latest_hospital_type_5","latest_hospital_Service_type_Psychogeriatric_care")\
                     .replace("latest_hospital_type_68","latest_hospital_Service_type_Other_admitted_care")\
                     .replace("latest_hospital_type_91","latest_hospital_Service_type_Other_admitted_care")
        docs.append(vals)
        print >> p2d, vals.decode('utf8') 

    print docs[1:10]
    p2d.close()
    
    return docs

def run_Model(documents):
    
#     no_features = 40
    no_topics = 5
    no_top_words = 5
    
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print "Topic %d:" % (topic_idx)
            print " ".join([feature_names[i]
                           for i in topic.argsort()[:-no_top_words - 1:-1]])
    
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    
    display_topics(nmf, tfidf_feature_names, no_top_words)
    
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    # Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    
    display_topics(lda, tf_feature_names, no_top_words)
    
    
def statistics(documents):
    
#     no_features = 40
    no_topics = 5
    
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    
    print tf[1:10,]
    
    # Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    
    print lda.components_
    
    doc_topic_distrib = lda.transform(tf)
    print doc_topic_distrib[1:10,]


if __name__ == "__main__":

#     documents = load_Preprocess()
    
    with open('transformedDoc.txt') as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        documents = [x.strip() for x in content] 
    
#     run_Model(documents)
    statistics(documents)