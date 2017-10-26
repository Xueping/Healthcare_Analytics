'''
Created on Jun 29, 2017

@author: longgu
'''

from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np

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
    #the age of patient is is between 31 and 60
    df.age_in_2015 [(df.age_in_2015 > 30.0) & (df.age_in_2015 <=60.0)] = 3
    #the age of patient is more than 60
    df.age_in_2015 [df.age_in_2015 > 60.0] = 4
    
    df = pd.get_dummies(data=df, columns=['tot_visit', 'age_in_2015','gender',  'latest_hospital_type','latest_adm_year',  'latest_adm_month'])
    df = df.drop(labels = ['row_number',  'person_id',  'tot_visit_ed', 'tot_visit_acute'], axis=1)
    
    print df.head(10)
    
    return df
    


def run_Model(df):
    
    tf_feature_names = dict((feature_idx, feature) for (feature_idx, feature) in enumerate(df.columns))

    no_topics = 5
      
      
    # Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(df.fillna(0))
      
    # def display_topics(model, feature_names, no_top_words):
    #     for topic_idx, topic in enumerate(model.components_):
    #         print "Topic %d:" % (topic_idx)
    #         print " ".join([feature_names[i]+":"+str(topic[i])
    #                         for i in topic.argsort()[:-no_top_words - 1:-1]])
            
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print "Topic %d:" % (topic_idx)
            print " ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]])
      
    no_top_words = 5
    display_topics(lda, tf_feature_names, no_top_words)

    
if __name__ == "__main__":

    df = load_Preprocess()
    run_Model(df)
