# -*- coding: utf-8 -*-
"""
Created on Sun May 10 01:12:07 2020

@author: LEGION
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat



#----------------------------------------------------




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
        







#----------------------------------------------------
Image_data=loadmat('bird_small.mat')
#print(Image_data['A'])
#print(Image_data['A'].shape)
A=Image_data['A']
plt.imshow(A)

#normalization value image
A=A/255


#reshape array to 2D
X=np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]))
#print(X.shape)

#randomal initialize the centroids
initial_centroids=init_centorids(X,16)
#print(initial_centroids)


#run algorithm
idx,centroids=run_K_mean(X,initial_centroids,10)
print()
#print(centroids)
    



idx=find_closest_centroids(X,centroids)

X_recovered=centroids[idx.astype(int),:]
X_recovered=np.reshape(X_recovered,(A.shape[0],A.shape[1],A.shape[2]))

plt.imshow(X_recovered)

#-----------------------------------------------------------------