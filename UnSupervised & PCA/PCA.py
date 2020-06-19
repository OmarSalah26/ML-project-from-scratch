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
        





def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    print('cov \n', cov)
    print()
    # perform SVD
    U, S, V = np.linalg.svd(cov) # singular value decomposition
    
    return U, S, V

def project_data(X, U, k):
    U_reduced = U[:,:k]
    print("U_reduced",U_reduced.shape)
    return np.dot(X, U_reduced)



def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

#----------------------------------------------------
    #apply pca

data=loadmat('ex7data1.mat')
print(data['X'])
print(data['X'].shape)
X=data['X']
fig, ax=plt.subplots(figsize=(9,6))
ax.scatter(X[:,0],X[:,1])



U, S, V = pca(X)
print("U",U.shape)
print()
print(S)
print()
print(V)



Z = project_data(X, U, 1)
print(Z)




X_recovered = recover_data(Z, U, 1)
print(X_recovered)
print(X_recovered.shape)


#-----------------------------------------------------------------

#anthor example


# Apply PCA on faces

faces = loadmat('ex7faces.mat')
X = faces['X']
print(X.shape)
plt.imshow(X)


#show one face
face = np.reshape(X[356,:], (32, 32))
plt.imshow(face)


U, S, V = pca(X)
Z = project_data(X, U, 100)

X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[356,:], (32, 32))
plt.imshow(face)








