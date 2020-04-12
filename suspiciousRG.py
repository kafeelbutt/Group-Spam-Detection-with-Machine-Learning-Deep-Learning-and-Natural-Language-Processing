import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

l=[]
def lambdaFunction(prod1,time1,rating1,prod2,time2,rating2):
        if prod1==prod2:
            val=RSG(time1,rating1,time2,rating2)
            if val == 1:
                return 1
            return 0
        else:
            return -1
                        
def RSG(time1,rating1,time2,rating2):
    alpha=10
    if (abs(time1-time2))>alpha or (abs(rating1-rating2))>=2:
        return 0
    else:
        l.append(abs(time1-time2))
        return 1
        
#R_id,p_id,rating,label,date
def Graph():
    
    '''data1=pd.read_csv('metadata',names = ['reviewerID' , 'asin', 'overall' , 'label','reviewTime'],  sep="\t")
    data2=pd.read_csv('reviewContent',names = ['reviewerID' , 'asin', 'reviewTime' , 'reviewText'],  sep="\t")
    raw_data_df=data1.merge(data2)
    raw_data_df=raw_data_df[10000:11000]
    raw_data_df.to_csv('./raw_data2.csv', sep = '\t',index=False)'''
    
    raw_data_df = pd.read_csv("raw_data.csv", sep="\t")
    #raw_data_df=raw_data_df.append({'reviewerID' : R_id , 'asin' : p_id , 'label' : label , 'overall' : rating , 'reviewTime' : date} , ignore_index=True)
    #raw_data_df=raw_data_df.drop_duplicates() 
    #raw_data_df.to_csv('./raw_data.csv', sep = '\t',index=False)
    reviewers_df = pd.DataFrame(columns= raw_data_df.columns.values)
    for col in  raw_data_df.columns.values:
        if (col == "reviewerID"): continue
        reviewers_df[col] =  raw_data_df.groupby("reviewerID")[col].apply(list)
    date_conv = lambda row: [datetime.strptime(r, '%Y-%m-%d').date().toordinal() for r in row.reviewTime]
    reviewers_df['ordinal'] = reviewers_df.apply(date_conv, axis=1)
    reviewers_df.drop('reviewerID', axis=1, inplace=True)
    reviewers_df_copy=reviewers_df.copy()
    
    reviewers_df.reset_index(inplace = True)
    rowiter = reviewers_df.iterrows()
    co_review_similarity = []
    lambda_vectorize=np.vectorize(lambdaFunction,doc='Vectorized `lambdaFunction`',otypes=[np.integer])
    for index_reviewer1, row_reviewer1 in rowiter:
        nextrowiter = reviewers_df.iloc[index_reviewer1 + 1:, :].iterrows()
        for index_reviewer2, row_reviewer2 in nextrowiter:
                reviwer_pair=[]
                commonProducts=list(set(row_reviewer1['asin']).intersection(set(row_reviewer2['asin'])))
                length=len(commonProducts)
                if(length):
                    val=lambda_vectorize(row_reviewer1['asin'],row_reviewer1['ordinal'],row_reviewer1['overall'],row_reviewer2['asin'],row_reviewer2['ordinal'],row_reviewer2['overall'])
                    if 1 in val:
                        reviwer_pair.append(row_reviewer1['reviewerID'])
                        reviwer_pair.append(row_reviewer2['reviewerID'])
                        reviwer_pair.append(1)
                        reviwer_pair.append(length)
                        co_review_similarity.append(reviwer_pair)
    
    graph_df = pd.DataFrame(co_review_similarity)
    graph_df.columns = ['reviewer_1', 'reviewer_2', 'suspicious_score','commonproducts']
    graph_df.to_csv('./suspiciousGraph.csv', sep = '\t')
    return reviewers_df_copy,raw_data_df

if __name__ == "__main__":
    #m,m1=Graph()
    '''data1=pd.read_csv('metadata',names = ['reviewerID' , 'asin', 'overall' , 'label','reviewTime'],  sep="\t")
    data2=pd.read_csv('reviewContent',names = ['reviewerID' , 'asin', 'reviewTime' , 'reviewText'],  sep="\t")
    raw_data_df=data1.merge(data2)
    raw_data_df=raw_data_df[8436:9936]
    raw_data_df.to_csv('./raw_data.csv', sep = '\t',index=False)'''

    
    '''raw_data_df = pd.read_csv("raw_data.csv", sep="\t")
    
     
    X =raw_data_df['reviewText'][0]
    Y= raw_data_df['reviewText'][1]
      
    # tokenization 
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 
      
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
      
    # remove stop words from string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
      
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
      
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    print("similarity: ")'''
    
  