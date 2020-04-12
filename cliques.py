import csv
import collections
import numpy as np
import networkx as nx


def Intersection(lst1, lst2): 
    return list(set(lst1).intersection(lst2))  

def cliques():
    
    groups=dict()
    cliques=dict()
    common_products=collections.defaultdict(dict)
    cliques_products=collections.defaultdict(dict)
    kcliques=[]
    fileName = 'suspiciousGraph.csv'
    with open(fileName, 'rt') as f:
        data = csv.reader(f, delimiter="\t")
        next(data, None)
        for row in data:
            reviewer1 = row[1]
            reviewer2 = row[2]
            product   = row[4] 
            if reviewer1 in groups:
                groups[reviewer1].append(reviewer2)
                common_products[reviewer1][reviewer2]=product
            else:
                groups[reviewer1]=[reviewer2]
                common_products[reviewer1][reviewer2]=product
    
    for reviewers in groups:
        for reviewer in groups[reviewers]:
            for keys in groups:
                if keys==reviewers:
                    continue
                elif keys==reviewer:
                    commonid=Intersection(groups[reviewers],groups[keys])
                    if not commonid:
                        continue
                    else:
                        if reviewers in cliques:
                            commonid.append(reviewer)
                            for i in commonid:
                                if i in cliques[reviewers]:
                                    continue
                                else:
                                    cliques[reviewers].append(i)
                        else:
                            cliques[reviewers]=commonid
                            cliques[reviewers].append(reviewer)
    
    for r1,v1 in common_products.items():
        for r2,v2 in cliques.items():
            if r1==r2:
                kcliques.append(len(cliques[r2])+1)
                for value1 in v1 :
                    if value1 in cliques[r2]:
                        cliques_products[r1][value1]=v1[value1]
            else:
                continue
    
    new_graph = nx.Graph()
    for source, targets in cliques_products.items():
        for source1,targets1 in cliques_products.items():
            if source==source1:
                continue
            else:
                 commonIntersect=list(set( cliques_products[source].items() ) & set( cliques_products[source1].items() ))
                 if not commonIntersect:
                     accumulatedProduct=0
                     new_graph.add_edge(str(source), str(source1),weight=int(accumulatedProduct))
                 else:
                     accumulatedProduct=0
                     for x in commonIntersect:
                         accumulatedProduct=accumulatedProduct + int(x[1])
                     new_graph.add_edge(str(source), str(source1),weight=int(accumulatedProduct))
                             
    adjacency_matrix = nx.adjacency_matrix(new_graph)  
    matrix=adjacency_matrix.todense()              
    
    k=5
    for i in range(len(kcliques)):
       if kcliques[i]<k:
           kcliques[i]=0
       else:
           kcliques[i]=1
    
    row,col = np.diag_indices(matrix.shape[0])
    matrix[row,col] = kcliques
    # threshold matrix
    for i in row:
        for j in col:
            if i==j:
                continue
            else:
                if matrix[i,j]<k-1:
                    matrix[i,j]=0
                else:
                    matrix[i,j]=1
    
    
    #Candidate Groups
    Adjacent_Groups=dict()
    Non_Adjacent_Groups=[]
    NodeList=list(new_graph.nodes())
    for i in row:
        reviwer1=NodeList[i]
        group=[]
        if kcliques[i]==0:
            continue
        for j in col:
            if i==j:
                continue
                    
            else:
                if matrix[i,j]==1 and kcliques[i]==1:
                    if reviwer1 in Adjacent_Groups:
                        Adjacent_Groups[reviwer1].append(NodeList[j])
                    else:
                        Adjacent_Groups[reviwer1]=[NodeList[j]]
                if kcliques[i]==1:
                    group.append(matrix[i,j])
        if all(v == 0 for v in group):
            Non_Adjacent_Groups.append(reviwer1)
    
    #corresponding cliques
    Adj_grp=Adjacent_Groups.copy()
    for reviewers in Adjacent_Groups:
        for reviewer in Adjacent_Groups[reviewers]:
            if reviewers not in Adj_grp:
                continue
            for keys in Adjacent_Groups:
                if keys==reviewers:
                    continue
                elif keys==reviewer:
                    for i in Adjacent_Groups[keys]:
                        if i in Adj_grp[reviewers]:
                            continue
                        else:
                            Adj_grp[reviewers].append(i)
                    st='Null'+str(keys)
                    Adj_grp[st] = Adj_grp[keys] 
                    del Adj_grp[keys]
    
    #making groups
    index_Adj=0
    Group_keys=list(Adj_grp.keys())
    Candidate_Groups=dict()
    #Candidate_Products=[]
    for i in range(len(Adj_grp)):
        #total=0
        key=Group_keys[i]
        if key not in cliques:
            continue
        Candidate_Groups[i]=[key]
        for values in cliques[key]:
            Candidate_Groups[i].append(values)
            #total=total+int(cliques_products[key][values])
        for values in Adj_grp[key]:
            for rev in cliques[values]:
                if rev in Candidate_Groups[i]:
                    continue
                else:
                    Candidate_Groups[i].append(rev)
                    #total=total+int(cliques_products[values][rev])
        #Candidate_Products.append(total)
        index_Adj=index_Adj+1                   
    
    for i in range(len(Non_Adjacent_Groups)):
        #total=0
        key=Non_Adjacent_Groups[i]
        Candidate_Groups[index_Adj]=[key]
        for values in cliques[key]:
            Candidate_Groups[index_Adj].append(values)
            #total=total+int(cliques_products[key][values])
        #Candidate_Products.append(total)
        index_Adj=index_Adj+1
    return Candidate_Groups   