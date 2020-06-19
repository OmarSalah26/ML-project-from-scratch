# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:22:28 2020

@author: LEGION
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

#-----------------------------------------------------------------------

#  select random points
def init_centorids(X,k):
#    m, n = X.shape
#    centroids = np.zeros((k, n))
#    idx = np.random.randint(0, m, k)
#    for i in range(k):
#        centroids[i,:]=X[idx[i],:]
#        
#        return centroids
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    
    return centroids

#function to selection step in K-mean algorithm
def find_closest_centroids(X,cendroid):
    m=X.shape[0]
    k=cendroid.shape[0]
    idx=np.zeros(m)
    for i in range(m):
        min_distance=1000000
        for j in range(k):
            distance=np.sum((X[i,:]-cendroid[j,:])**2)
            if distance<min_distance:
                min_distance=distance
                idx[i]=j
    return idx


#function to dispalcement step in K-mean algorithm

def compute_centroids(X, index , k):
    m,n =X.shape
    centroids=np.zeros((k,n))
    for i in range(k):
        indices=np.where(index==i)
        centroids[i,:]=(np.sum(X[indices,:],axis=1)/len(indices[0]))
    return centroids
        
    
#apply K-mean
def run_K_mean(X,initial_centroids,max_iters):
    m,n=X.shape
    k=initial_centroids.shape[0]
    idx=np.zeros(m)
    centroids=initial_centroids
    
    for i in range(max_iters):
        idx=find_closest_centroids(X,centroids)
        
        centroids=compute_centroids(X,idx,k)
    return idx,centroids
        
        
    
    
    
    
    











#load data
data=loadmat("ex7data2.mat")
print(data)
print(data['X'])
print(data['X'].shape)

#classify point
X=data['X']
  #select manual
#initial_centroids=np.array([[3,3],[6,2],[8,5]])
  #select random by function
initial_centroids=init_centorids(X,3)
print(initial_centroids)

# to selection step in K-mean algorithm

index=find_closest_centroids(X,initial_centroids)
print(index)


# to displacement step in K-mean algorithm
#calculate new centroid
c = compute_centroids(X, index , 3)
print(c)
 

    
#apply K-mean
for x in range(6):
    idx,centroids=run_K_mean(X,initial_centroids,x)
    print()
    print(centroids)
    
    
#    draw  it
    cluster1=X[np.where(idx==0)]
    cluster2=X[np.where(idx==1)]
    cluster3=X[np.where(idx==2)]
# same
#    cluster1=X[np.where(idx==0)[0],:]
#    cluster2=X[np.where(idx==1)[0],:]
#    cluster3=X[np.where(idx==2)[0],:]
    
    fig, ax = plt.subplots(figsize=(9,6))
    ax.scatter(cluster1[:,0],cluster1[:,1],s=30,color='red',label="Cluter1")
    ax.scatter(centroids[0,0],centroids[0,1],s=300,color='red')
    
    ax.scatter(cluster2[:,0],cluster2[:,1],s=30,color='green',label="Cluter2")
    ax.scatter(centroids[1,0],centroids[1,1],s=300,color='green')   
    
    ax.scatter(cluster3[:,0],cluster3[:,1],s=30,color='blue',label="cluster3")
    ax.scatter(centroids[2,0],centroids[2,1],s=300,color='blue')   
    ax.legend()
    










#draw data by me
print(pd.DataFrame(data['X']).iloc[:,0])
plt.scatter(pd.DataFrame(data['X']).iloc[:,0],pd.DataFrame(data['X']).iloc[:,1])
























