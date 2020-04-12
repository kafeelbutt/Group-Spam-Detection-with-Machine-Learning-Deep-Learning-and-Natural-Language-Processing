'''
This file will be used to calculate features for a reviewer
'''

import pandas as pd
from collections import defaultdict
from sklearn.neighbors.kde import KernelDensity
import numpy as np
from datetime import datetime
import math
import cliques
import itertools
from sklearn import metrics
import suspiciousRG

average = lambda n : sum(n) / len(n)
average2 = lambda n : sum(n) / len(n) if len(n) > 0 else 0

Products=defaultdict(list)
testLabel=[]
predictions=[]

def kde(grouped_df):
    product_ratings_list = []

    #Grouping ratings and dates by product ID
    for index, row in grouped_df.iterrows():
        for index_list, val in enumerate(row['asin']):
            product_ratings_list.append([val, grouped_df.loc[index,'overall'][index_list], grouped_df.loc[index,'reviewTime'][index_list]])
    product_ratings = pd.DataFrame(product_ratings_list, columns=["product_id", "rating", "date"])
    grouped_pr = pd.DataFrame(columns=product_ratings.columns.values)
    grouped_pr["rating"] = product_ratings.groupby("product_id")["rating"].apply(list)
    grouped_pr["date"] = product_ratings.groupby("product_id")["date"].apply(list)

    #Deleting product_id row as it's already indexed on product_id
    grouped_pr.drop('product_id', axis=1, inplace=True)


    #Now grouped_pr contains product_id, [list of ratings], [list of dates]

    kde_list = []
    p_id = []

    # Calculating kde for all dates in all products
    for index, row in grouped_pr.iterrows():
        p_id.append(index)
        for i in range(len(grouped_pr.loc[index,'date'])):
            #Changing date to ordinal format as kde accepts numerical values only
            grouped_pr.loc[index, 'date'][i] = [datetime.strptime(grouped_pr.loc[index,'date'][i], '%Y-%m-%d').date().toordinal()]
        grouped_pr.loc[index, 'date'].sort()
        date = np.array(grouped_pr.loc[index, 'date'])
        #Fitting KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(date)
        log_dens = kde.score_samples(date)
        kde_list.append(log_dens)
        
    
    #Converting to dataframe : Each columns is a product_id with list of kdes for each sorted date
    kde_df = (pd.DataFrame(kde_list)).transpose()
    kde_df.columns = p_id

    #Computing threshold
    kde_df['max'] = kde_df.max(axis = 1)
    sorted_by_max_of_each_row = pd.DataFrame()
    sorted_by_max_of_each_row = kde_df.sort_values(by='max', ascending=False, inplace=False).reset_index(drop=True)
    length=int(len(sorted_by_max_of_each_row['max'])/2)
    threshold = sorted_by_max_of_each_row['max'][length] #threshold is the top 50th value
    kde_df.drop('max', axis=1, inplace=True)

    #Filtering all indices with kde > threshold for each product
    intersection = []
    for product_kde in kde_df:
        temp_df = kde_df[kde_df[product_kde]>threshold][product_kde]
        intersection.append(temp_df.index.tolist())

    df_indices = (pd.DataFrame(intersection )).transpose()
    df_indices.columns = p_id


    grouped_pr['bursts'] = grouped_pr['date']
    for column in df_indices:
        b = []
        b_id = []
        id = 0
        a = pd.to_numeric(df_indices[column].dropna())
        if (len(a) == 0):
            grouped_pr.loc[column, 'bursts'] = []
            continue
        ini = a[0]
        for i in range(len(a)):

            if (i is (len(a) - 1)) :
                b.append([grouped_pr.loc[column,"date"][int(ini)],grouped_pr.loc[column,"date"][int(a[i])]])
                b_id.append([str(column) + '-' + str(id)])
                id += 1

            elif ((int(a[i]+1) is not int(a[i+1]))):
                b.append([grouped_pr.loc[column,"date"][int(ini)],grouped_pr.loc[column,"date"][int(a[i])]])
                b_id.append([str(column) + '-' + str(id)])
                id += 1
                ini = a[i+1]

        grouped_pr.loc[column,'bursts'] = b
        grouped_pr.loc[column, 'bursts_ids'] = b_id
    return grouped_pr

def reviewer_bursts(grouped_df, grouped_pr):
    #Adding ordinal column
    prods_df = grouped_df[['asin', 'reviewTime']]
    date_conv = lambda row: [datetime.strptime(r, '%Y-%m-%d').date().toordinal() for r in row.reviewTime]
    prods_df['ordinal'] = prods_df.apply(date_conv, axis=1)
    prods_df = prods_df.drop('reviewTime', axis=1)
    prods_df = prods_df.reset_index()

    grouped_pr = grouped_pr.drop(['rating', 'date'], axis=1)
    calc_count = lambda row: sum([sum([1 for b in grouped_pr['bursts'][prod] if o_time >= b[0][0] and o_time <= b[1][0]]) \
                                if grouped_pr['bursts'].get(prod) else 0 \
                                for prod, o_time in zip(row.asin,row.ordinal)])


    bursts = lambda row: list(filter(None, ([next((b_id for b, b_id \
                        in zip(grouped_pr['bursts'][prod], grouped_pr['bursts_ids'][prod]) \
                        if o_time >= b[0][0] and o_time <= b[1][0]), None) \
                        if grouped_pr['bursts'].get(prod) else None for prod, o_time in zip(row.asin,row.ordinal)])))
    prods_df['burst_ids'] = prods_df.apply(bursts, axis=1)
    prods_df['burst_count'] = prods_df.apply(calc_count, axis=1)

    return prods_df

def burst_ratio(prods_df):
    find_burst = lambda row: float(row['burst_count'])/len(row.ordinal)
    prods_df['burst_ratio'] = prods_df.apply(find_burst, axis=1)
    prods_df = prods_df.set_index('reviewerID')
    return prods_df

def penalty_function(Rg,Pg):
    Penalty_function=1/(1+math.exp(-(Rg+Pg-3)))
    return Penalty_function

def group_size(Candidate_Groups):
    Group_Size=[]
    for i in Candidate_Groups:
        Rg=len(Candidate_Groups[i])
        size=1/(1+math.exp(-(Rg-3)))
        Group_Size.append(size)
    
    return Group_Size

def Group_Rating_Deviation(average_rating_product,final_df,grouped_df,Candidate_Groups):
   
    Group_Deviation=[]
    a=[]
    for i in Candidate_Groups:
        for index, row in grouped_df.iterrows():
            if index in Candidate_Groups[i]:
                for index_list, val in enumerate(row['overall']):
                    variance=np.var(average_rating_product[grouped_df.loc[index, 'asin'][index_list]])
                    a.append(variance)
        Rg=Candidate_Groups[i]
        Pg=Products[i]
        grd=2*(1-(1/(1+math.exp(-average(a)))))*penalty_function(len(set(Rg)),len(set(Pg)))
        Group_Deviation.append(grd)
        a=[]
    final_df['group_deviation']=Group_Deviation
    return final_df

def avg_rating_deviation(grouped_df,Candidate_Groups,final_df):
    
    average_rating_product = defaultdict(list)
    for i in Candidate_Groups:
        for index, row in grouped_df.iterrows():
            if index in Candidate_Groups[i]:
                for index_list, val in enumerate(row['asin']):
                    average_rating_product[val].append(grouped_df.loc[index,'overall'][index_list])
     
    final_df=Group_Rating_Deviation(average_rating_product,final_df,grouped_df,Candidate_Groups)               
    for key in average_rating_product:
        average_rating_product[key] = sum(average_rating_product[key])/len(average_rating_product[key])

    a = []
    Avg_Dev=[]
    for i in Candidate_Groups:
        for index, row in grouped_df.iterrows():
            if index in Candidate_Groups[i]:
                for index_list, val in enumerate(row['overall']):
                    a.append(abs((val - average_rating_product[grouped_df.loc[index, 'asin'][index_list]])) / 4)
        Avg_Dev.append(average(a))
        a = []
    final_df['avg_rating_deviation']=Avg_Dev
    return final_df

def review_tightness(Candidate_Groups,grouped_df):
    
    no_of_reviews=0
    total_num_reviews=[]
    for group_reviewer in Candidate_Groups:
            for index,row in grouped_df.iterrows():
                if index in Candidate_Groups[group_reviewer]:
                    for index_list, val in enumerate(row['asin']):
                        no_of_reviews=no_of_reviews+1
                        Products[group_reviewer].append(val)
            
            total_num_reviews.append(no_of_reviews)
            no_of_reviews=0
    Review_tightness=[]
    for i in range(len(Candidate_Groups)):
        Rg=Candidate_Groups[i]
        Pg=Products[i]
        Vg=total_num_reviews[i]
        cartesian_product=len(set(itertools.product(Rg,Pg)))
        RT=(Vg/cartesian_product)*penalty_function(len(set(Rg)),len(set(Pg)))
        Review_tightness.append(RT)    
    return Review_tightness

def BST(grouped_df,Candidate_Groups):
    
    Group_Burst_Ratio=[]
    for i in Candidate_Groups:
        temp=[]
        for index, row in grouped_df.iterrows():
            if index in Candidate_Groups[i]:
                temp.append(row['burst_ratio'])
        Group_Burst_Ratio.append(average(temp))
    
    normalized = [float(i)/max(Group_Burst_Ratio) for i in Group_Burst_Ratio]
    return normalized

def Group_Support_Count(Candidate_Groups):
    
    Group_Support_Count=dict()
    for i in Candidate_Groups:
        g=Products[i]
        Group_Support_Count[i]=((len(set(g))))
    
    a=[k for k in Group_Support_Count.keys() if Group_Support_Count.get(k)==max([n for n in Group_Support_Count.values()])]
    t=Group_Support_Count[a[0]]
    
    list_temp=[]
    
    for i in range(len(Group_Support_Count)):
        list_temp.append(Group_Support_Count[i]/t)
    
    return list_temp

def Group_Size_Ratio(raw_data_df,Candidate_Groups):
    
    grouped_df = pd.DataFrame(columns= raw_data_df.columns.values)
    for col in  raw_data_df.columns.values:
        if (col == "asin"): continue
        grouped_df[col] =  raw_data_df.groupby("asin")[col].apply(list)

    grouped_df.drop('asin', axis=1, inplace=True)

    Products_id=defaultdict(list)
    for index,row in grouped_df.iterrows():
        for index_list, val in enumerate(row['reviewerID']):
            Products_id[index].append(val)
    
    Group_size_ratio=defaultdict(list)
    for index in Products_id:
        for i in Candidate_Groups:
            a=[]
            for val in Candidate_Groups[i]:
                if val in Products_id[index]:
                    a.append(val)
            if len(a)==0:
                continue
            else:
                Group_size_ratio[i].append(len(Candidate_Groups[i])/len(a))
                a=[]
    
    gsr=[]
    for i in Group_size_ratio:
        gsr.append(average(Group_size_ratio[i]))
     
    normalized = [float(i)/max(gsr) for i in gsr]
    return normalized

def getPredictions(grouped_df,Candidate_Groups):
    for i in Candidate_Groups:
        for r1 in Candidate_Groups[i]:
            for index,row in grouped_df.iterrows():
                if r1==row['reviewerID']:
                    if row['label']==1:
                        testLabel.append(1)
                    else:
                        testLabel.append(0)
                    predictions.append(1)
    
    


def compute_features(R_id,p_id,rating,label,date):
    
    grouped_df,raw_data_df=suspiciousRG.Graph(R_id,p_id,rating,label,date)
   
    Candidate_Groups=cliques.cliques()
    
    for i in Candidate_Groups:
        Candidate_Groups[i] = list(map(int, Candidate_Groups[i]))
    
    final_df=pd.DataFrame(columns=['Groups'])
    for i in Candidate_Groups:
        final_df.loc[i,'Groups']=Candidate_Groups[i]
    
    
    final_df['review_tightness']=review_tightness(Candidate_Groups,grouped_df)
    
    final_df = avg_rating_deviation(grouped_df,Candidate_Groups,final_df)
    
    grouped_pr = kde(grouped_df)
    prods_df = reviewer_bursts(grouped_df, grouped_pr)

    prods_df = burst_ratio(prods_df)
    grouped_df['burst_ratio'] = prods_df['burst_ratio']
    grouped_df['burst_ids'] = prods_df['burst_ids']
    
    final_df['burst_ratio']=BST(grouped_df,Candidate_Groups)
    
    final_df['group_size']=group_size(Candidate_Groups)
    
    final_df['group_support']=Group_Support_Count(Candidate_Groups)
    
    final_df['group_size_ratio']=Group_Size_Ratio(raw_data_df,Candidate_Groups)

    gd=final_df['group_deviation'].tolist()
    rd=final_df['avg_rating_deviation'].tolist()
    bst=final_df['burst_ratio'].tolist()
    gsr=final_df['group_size_ratio'].tolist()
    gc=final_df['group_support'].tolist()
    gs=final_df['group_size'].tolist()
    rt=final_df['review_tightness'].tolist()
    
    avg_score_groups=dict()
    for i in range(len(gd)):
        avg_score_groups[i]=(round((gd[i]+rd[i]+gsr[i]+bst[i]+gc[i]+gs[i]+rt[i])/7,2))
    
    final_df['suspicious_score']=list(avg_score_groups.values())
    sort_by_score = final_df.sort_values('suspicious_score',ascending=False)

    suspicious_score = 0
    Group_Rating_Dev = 0
    avg_rating_dev = 0
    burst_rt = 0
    Group_Size_Rt = 0
    review_tght = 0
    Group_Support = 0
    Group_size = 0
    Group_num=0
    Group_reviwers=0
    
    flag=0
    for index, row in sort_by_score.iterrows():
        for index_list, val in enumerate(row['Groups']):
            if R_id==val:
                if flag==1:
                    break
                flag=1
                Group_Rating_Dev = row['group_deviation']
                avg_rating_dev = row['avg_rating_deviation']
                burst_rt = row['burst_ratio']
                Group_size = row['group_size']
                Group_Size_Rt = row['group_size_ratio']
                review_tght = row['review_tightness']
                Group_Support = row['group_support']
                suspicious_score=row['suspicious_score']
                Group_num=index
                Group_reviwers=len(row['Groups'])
    if flag==1:
        #raw_data_df.loc[raw_data_df.reviewerID == R_id, 'label'] = 1
        #raw_data_df=raw_data_df.drop_duplicates() 
        #raw_data_df.to_csv('./raw_data.csv', sep = '\t',index=False)
        print("Group Spam belongs to group "+ str(Group_num) + " with Group members "+ str(Group_reviwers) +" having suspicious score "+ str(suspicious_score) +" with the following Indicators that makes this reviewer spam"
        +"\nGroup Deviation "+str(round(Group_Rating_Dev,2))+"\nAverage Rating Deviation "+str(round(avg_rating_dev,2))+"\nBurst Ratio "+str(round(burst_rt,2))+"\nGroup Size "+str(round(Group_size,2))+"\nGroup Size Ratio "+str(round(Group_Size_Rt,2))+"\nReview Tightness "+str(round(review_tght,2))+"\nGroup Support Count "+str(round(Group_Support,2)))
    else:
        print("Not Spam")
    
    getPredictions(raw_data_df,Candidate_Groups)
    
    print(metrics.accuracy_score(testLabel,predictions))
    print(metrics.confusion_matrix(testLabel,predictions))
    return sort_by_score

if __name__ == '__main__':
    
    sort1=compute_features(2,87,5.0,1,"2012-12-02")
    

        
                    
    
