# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:03:24 2020

@author: LEGION
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#read data 
dataset=pd.read_csv("homePrice.csv",header=None , names=['Population','Profit'])


#show data details
print('dataset  \n',dataset.head(10))
print("********************************************")
print('dataset describe   \n',dataset.describe())
print("********************************************")

#draw data 
dataset.plot(kind="scatter", x='Population',y='Profit',figsize=(5,5))

##-=====================================================
dataset.insert(0,'ones',1)
print(' new dataset  \n',dataset.head(10))
print("********************************************")

# Separate X => training data  Form Y => (target data )
cols=dataset.shape[1]
X=dataset.iloc[:,:cols-1]
#or 
#x=dataset.iloc[:,[0,1]]
Y=dataset.iloc[:,cols-1:cols]

print('X  \n',X.head(10))
print("********************************************")
print('Y   \n',Y.head(10))
print("********************************************")

#convert form dataframe to numpy matrices

X=np.matrix(X.values)
Y=np.matrix(Y.values)
theta=np.matrix(np.array([0,0]))

#print('X \n',X)
#print('X.shape = ' , X.shape)
#print('theta \n',theta)
#print('theta.shape = ' , theta.shape)
#print('y \n',Y)
#print('y.shape = ' , Y.shape)
#print('**************************************')

#==================================================================

#cost function

def costfunction(X,Y,theta):
    z=np.power(((X*theta.T)-Y),2)
#    print("z \n",z)
#    print('m',len(X))
    return np.sum(z)/(2*len(X))

print("Compute Cost (X , Y , theta) =  ",costfunction(X,Y,theta))

#Gradent Desent Function 

#def GradentDesentFunction(X,Y,Theta,alpha,iters ):
#    temp1=theta[0]-(alpha/len(X))*np.sum((X*theta.T)-Y)
#    temp2=theta[1]-(alpha/len(X))*np.sum(((X*theta.T)-Y)X[1])

def GradentDesentFunction(X,Y,theta,alpha,iters ):
    temp=np.matrix(np.zeros(theta.shape))
#    print("temp", temp)
    parameters=int(theta.ravel().shape[1])
#    print("parameters", parameters)
    cost=np.zeros(iters)
    
    for i in range(iters):
        error=(X*theta.T)-Y
        for j in range(parameters):
            term=np.multiply(error ,X[:,j])
            print("term", term)

            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term))
            
        theta=temp
        cost[i]=costfunction(X,Y,theta)
    return theta, cost
    
            
# initialize variables for learning rate and iterations
alpha = 0.01
iters = 10
    
# perform gradient descent to "fit" the model parameters
g, cost = GradentDesentFunction(X,Y, theta, alpha, iters)

print('g = ' , g)
print('cost  = ' , cost[900:1000] )
print('compute Cost = ' , costfunction(X, Y, g))
print('**************************************')
    
#=========================================================================

# get best fit line

x = np.linspace(dataset.Population.min(), dataset.Population.max(), 100)
print('x \n',x)
print('g \n',g)

final = g[0, 0] + (g[0, 1] * x)
print('f \n',final)




# draw the line

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, final, 'r', label='Prediction')
ax.scatter(dataset.Population, dataset.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

    



