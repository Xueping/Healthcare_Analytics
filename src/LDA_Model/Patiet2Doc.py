'''
Created on Jun 30, 2017

@author: Xueping
'''

import codecs
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    df.age_in_2015 [df.age_in_2015 <=14.0] = 0
    #the age of patient is is between 13 and 19
    df.age_in_2015 [(df.age_in_2015 > 14.0) & (df.age_in_2015 <=24.0)] = 1
    #the age of patient is is between 20 and 30
    df.age_in_2015 [(df.age_in_2015 > 24.0) & (df.age_in_2015 <=54.0)] = 2
    #the age of patient is is between 31 and 45
    df.age_in_2015 [(df.age_in_2015 > 54.0) & (df.age_in_2015 <=64.0)] = 3
    #the age of patient is more than 60
    df.age_in_2015 [df.age_in_2015 > 64.0] = 4
    
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
                     .replace("age_in_2015_0.0","age_no_more_than_14")\
                     .replace("age_in_2015_1.0","age_between_15_and_24")\
                     .replace("age_in_2015_2.0","age_between_25_and_54")\
                     .replace("age_in_2015_3.0","age_between_55_and_64")\
                     .replace("age_in_2015_4.0","age_more_than_64")\
                     .replace("gender_3.0","gender_Other")\
                     .replace("gender_9.0","gender_Other")\
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

def load_Preprocess_ABS():
    
    df = pd.read_csv("~/data/hab_test.csv", dtype={'hab_test.tot_visit': np.float32,'hab_test.tot_visit_ed': np.float32,'hab_test.tot_visit_acute': np.float32,'hab_test.age_in_2015': np.float32}, low_memory=False)
    df.fillna(0)
    
    df.columns = [feature.split('.')[1] for (feature_idx, feature) in enumerate(df.columns)]
    
    df.tot_visit[df.tot_visit <=1.0] = 0
    df.tot_visit[df.tot_visit ==2.0] = 1
    df.tot_visit[df.tot_visit ==3.0] = 2
    df.tot_visit[df.tot_visit ==4.0] = 3
    df.tot_visit[df.tot_visit ==5.0] = 4
    df.tot_visit[(df.tot_visit > 5.0) & (df.tot_visit <=9.0)] = 5
    df.tot_visit[df.tot_visit >9.0] = 6
    

    df.age_in_2015 [df.age_in_2015 <=4.0] = 0
    df.age_in_2015 [(df.age_in_2015 >= 5.0) & (df.age_in_2015 <=9.0)] = 1
    df.age_in_2015 [(df.age_in_2015 >= 10.0) & (df.age_in_2015 <=14.0)] = 2
    df.age_in_2015 [(df.age_in_2015 >= 15.0) & (df.age_in_2015 <=19.0)] = 3
    df.age_in_2015 [(df.age_in_2015 >= 20.0) & (df.age_in_2015 <=24.0)] = 4
    df.age_in_2015 [(df.age_in_2015 >= 25.0) & (df.age_in_2015 <=29.0)] = 5
    df.age_in_2015 [(df.age_in_2015 >= 30.0) & (df.age_in_2015 <=34.0)] = 6
    df.age_in_2015 [(df.age_in_2015 >= 35.0) & (df.age_in_2015 <=39.0)] = 7
    df.age_in_2015 [(df.age_in_2015 >= 40.0) & (df.age_in_2015 <=44.0)] = 8
    df.age_in_2015 [(df.age_in_2015 >= 45.0) & (df.age_in_2015 <=49.0)] = 9
    df.age_in_2015 [(df.age_in_2015 >= 50.0) & (df.age_in_2015 <=54.0)] = 10
    df.age_in_2015 [(df.age_in_2015 >= 55.0) & (df.age_in_2015 <=59.0)] = 11
    df.age_in_2015 [(df.age_in_2015 >= 60.0) & (df.age_in_2015 <=64.0)] = 12
    df.age_in_2015 [(df.age_in_2015 >= 65.0) & (df.age_in_2015 <=69.0)] = 13
    df.age_in_2015 [(df.age_in_2015 >= 70.0) & (df.age_in_2015 <=74.0)] = 14
    df.age_in_2015 [(df.age_in_2015 >= 75.0) & (df.age_in_2015 <=79.0)] = 15
    df.age_in_2015 [(df.age_in_2015 >= 80.0) & (df.age_in_2015 <=84.0)] = 16
    df.age_in_2015 [df.age_in_2015 > 84.0] = 17
    
    df = pd.get_dummies(data=df, columns=['tot_visit', 'age_in_2015','gender',  'latest_hospital_type','latest_adm_year',  'latest_adm_month'])
    df = df.drop(labels = ['row_number',  'person_id',  'tot_visit_ed', 'tot_visit_acute'], axis=1)
    
#     smaple = df.head(10)
    if os.path.exists("transformedDoc_ABS.txt"):
        os.remove("transformedDoc_ABS.txt")
    p2d  = codecs.open('transformedDoc_ABS.txt',"a+","utf-8") 
    docs = []
    for i in range(0,len( df.index)):
        cols =  df.columns[( df > 0 ).iloc[i]]
        vals = ' '.join(cols._data)
        vals = vals.replace("tot_visit_0.0","tot_visit_1")\
                     .replace("tot_visit_1.0","tot_visit_2")\
                     .replace("tot_visit_2.0","tot_visit_3")\
                     .replace("tot_visit_3.0","tot_visit_4")\
                     .replace("tot_visit_4.0","tot_visit_5")\
                     .replace("tot_visit_5.0","tot_visit_6_9")\
                     .replace("tot_visit_6.0","tot_visit_10_plus")\
                     .replace("age_in_2015_0.0","age_0_4")\
                     .replace("age_in_2015_1.0","age_5_9")\
                     .replace("age_in_2015_2.0","age_10_14")\
                     .replace("age_in_2015_3.0","age_15_19")\
                     .replace("age_in_2015_4.0","age_20_24")\
                     .replace("age_in_2015_5.0","age_25_29")\
                     .replace("age_in_2015_6.0","age_30_34")\
                     .replace("age_in_2015_7.0","age_35_39")\
                     .replace("age_in_2015_8.0","age_40_44")\
                     .replace("age_in_2015_9.0","age_45_49")\
                     .replace("age_in_2015_10.0","age_50_54")\
                     .replace("age_in_2015_11.0","age_55_59")\
                     .replace("age_in_2015_12.0","age_60_64")\
                     .replace("age_in_2015_13.0","age_65_69")\
                     .replace("age_in_2015_14.0","age_70_74")\
                     .replace("age_in_2015_15.0","age_75_79")\
                     .replace("age_in_2015_16.0","age_80_84")\
                     .replace("age_in_2015_17.0","age_85_Plus")\
                     .replace("gender_3.0","gender_Other")\
                     .replace("gender_9.0","gender_Other")\
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

def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print "Topic %d:" % (topic_idx+1)
            print " ".join([feature_names[i]
                           for i in topic.argsort()[:-no_top_words - 1:-1]])+"\n"

def run_Model(documents):
    
#     no_features = 40
    no_topics = 5
    no_top_words = 5
    
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

    no_topics = 5
    no_top_words = 6
    
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    
    #get document-term matrix, the entry is term frequency in document
    tf = tf_vectorizer.fit_transform(documents)

    #run LDA model to get topics and distribution
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    #document-topic distribution matrix
    doc_topic_distrib = lda.transform(tf)
  
    #get term name
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    #term-topic distribution matrix
    term_distrib = tf.transpose()*doc_topic_distrib
    
    #Convert numpy ndarray to pandas dataframe and add index
    indexed_terms = pd.DataFrame(term_distrib, index=tf_feature_names,columns=['topic_1','topic_2','topic_3','topic_4','topic_5'])
    
    display_topics(lda, tf_feature_names, no_top_words)
    
    #show diagram for age group distribution
    age = indexed_terms[indexed_terms.index.str.contains("age_")].transpose()
    cols = age.columns.tolist()
#     cols = cols[-1:] + cols[:-1]
    cols = cols[0:1] + cols[11:12]+cols[1:11]+cols[12:]
    age = age[cols]
    print age.sum()
    
    age = age.plot(kind='bar', title ="Age Group Distribution on Topics", figsize=(15, 10), legend=True, fontsize=12)
#     age.set_xlabel("Topics", fontsize=12)
    age.set_ylabel("Patient Number", fontsize=12)
    
    #show diagram for gender distribution
    gender = indexed_terms[indexed_terms.index.str.contains("gender_")].transpose()
    gender = gender.plot(kind='bar', title ="Gender Distribution on Topics", figsize=(15, 10), legend=True, fontsize=12)
    gender.set_ylabel("Patient Number", fontsize=12)
    
    tot_visit = indexed_terms[indexed_terms.index.str.contains("tot_visit_")].transpose()
    cols = tot_visit.columns.tolist()
#     cols = cols[-1:] + cols[:-1]
    cols = cols[0:1] + cols[2:]+cols[1:2]
    tot_visit = tot_visit[cols]
    print tot_visit.sum()
    
    tot_visit = tot_visit.plot(kind='bar', title ="Distribution of Total Visit Hospital on Topics", figsize=(15, 10), legend=True, fontsize=12)
    tot_visit.set_ylabel("Patient Number", fontsize=12)
    

     
    latest_adm_year = indexed_terms[indexed_terms.index.str.contains("latest_adm_year_")].transpose()
    latest_adm_year = latest_adm_year.plot(kind='bar', title ="Latest Admission Year Distribution on Topics", figsize=(15, 10), legend=True, fontsize=12)
    latest_adm_year.set_ylabel("Patient Number", fontsize=12)

     
    latest_adm_month = indexed_terms[indexed_terms.index.str.contains("latest_adm_month_")].transpose()
    cols = latest_adm_month.columns.tolist()
    cols = cols[0:1] + cols[4:]+cols[1:4]
    latest_adm_month = latest_adm_month[cols]
    latest_adm_month = latest_adm_month.plot(kind='bar', title ="Latest Admission Month Distribution on Topics", figsize=(15, 10), legend=True, fontsize=12)
    latest_adm_month.set_ylabel("Patient Number", fontsize=12)
#     
#     latest_hospital_service = indexed_terms[indexed_terms.index.str.contains("latest_hospital_service_")]
#     latest_hospital_service.plot()

    latest_hospital_service = indexed_terms[indexed_terms.index.str.contains("latest_hospital_service_")].transpose()
    latest_hospital_service = latest_hospital_service.plot(kind='bar', title ="Latest Hospital Service Distribution on Topics", figsize=(15, 10), legend=True, fontsize=12)
    latest_hospital_service.set_ylabel("Patient Number", fontsize=12)
    
    plt.show()


if __name__ == "__main__":

#     documents = load_Preprocess()
    load_Preprocess_ABS()
    
#     with open('transformedDoc.txt') as f:
    with open('transformedDoc_ABS.txt') as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        documents = [x.strip() for x in content] 
    
#     run_Model(documents)
    statistics(documents)
