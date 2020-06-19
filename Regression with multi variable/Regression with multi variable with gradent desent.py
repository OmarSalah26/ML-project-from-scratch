# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:11:06 2020

@author: LEGION
"""

#import library 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#import data 
dataset=pd.read_csv("dataSet.csv",header=None, names=['Size', 'Bedrooms', 'Price'])


#show data
print('data = ')
print(dataset.head(10) )
print()
print('data.describe = ')
print(dataset.describe())



#Rescaling data 
dataset=(dataset-dataset.mean())/dataset.std()

#adding X0

dataset.insert(0,"X0",1)
#split data
cols=dataset.shape[1]

X=dataset.iloc[:,:cols-1]
Y=dataset.iloc[:,cols-1:cols]

#convert to matrix
X=np.matrix(X)
Y=np.matrix(Y)
theta=np.matrix(np.array([0,0,0]))





def costFunction(X,Y,theta):
    return np.sum(np.power((X*theta.T)-Y,2)/(2*len(X)))

print("Cost function = ",costFunction(X,Y,theta))
print('theta \n',g)

def GradentDesent(X,Y,theta,alpha,iters):
    temp=np.zeros(theta.shape)
    parameters=theta.shape[1]
    cost=np.zeros(iters)
    
    for i in range(iters):
        
        error=np.dot(X,theta.T)-Y #(X*theta.T)-Y # # same np.dot(X,theta.T)ضرب خطي  
        for j in range(parameters):
            term=np.multiply(error,X[:,j]) #  or   np.array(error)*np.array(X[:,j])for use as array كل واحد في اخوه
            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term))
        
        theta=temp
        cost[i]=costFunction(X,Y,theta)
    
    return theta,cost




# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

g,cost=GradentDesent(X,Y,theta,alpha,iters)


#هنسرمه مره مع السعر ومره مع عدد الغرف 


print('g = ' , g)
print('cost  = ' , cost[900:1000] )
print('computeCost = ' , costFunction(X, Y, g))
print('**************************************')
    


# get best fit line for Size vs. Price

x = np.linspace(dataset.Size.min(), dataset.Size.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)



# draw the line for Size vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(dataset.Size, dataset.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


# get best fit line for Bedrooms vs. Price

x = np.linspace(dataset.Bedrooms.min(), dataset.Bedrooms.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)

# draw the line  for Bedrooms vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(dataset.Bedrooms, dataset.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')



# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')











 
