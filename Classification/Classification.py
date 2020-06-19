# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:29:32 2020

@author: LEGION
"""

#import library 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#import data 
dataset=pd.read_csv("data.csv",header=None, names=['Exam1', 'Exam2', 'Admitted'])


#show data
print('data = ')
print(dataset.head(10) )
print()
print('data.describe = ')
print(dataset.describe())


postive=dataset[dataset["Admitted"].isin([1])]
negative=dataset[dataset["Admitted"].isin([0])]

print("Admitted student \n",postive)
print("NonAdmitted student \n",negative )









#segmoid function 
print("================================================")
def sigmoid(z):
    return 1/(1+np.exp(-z))


nums=np.arange(-10,10)

plt.plot(nums,sigmoid(nums),"r")
plt.show()

#draw the data for positive and negative
print("================================================")

plt.scatter(postive.iloc[:,0],postive.iloc[:,1])
plt.scatter(negative.iloc[:,0],negative.iloc[:,1],color="red")

plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()


dataset.insert(0,"X0",1)

#set training data and target data
cols=dataset.shape[1]
X=dataset.iloc[:,:cols-1]
y=dataset.iloc[:,cols-1:cols]    
theta=np.zeros(3)




#convert to matrix 
X=np.array(X)
y=np.array(y)
theta=np.zeros(3)
print()
print('X.shape = ' , X.shape)
print('theta.shape = ' , theta.shape)
print('y.shape = ' , y.shape)
print()
print('X = ' , X)
print('theta = ' , theta)
print('y = ' , y)



#costFunction
def costFunction(theta, X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first=np.multiply( -y,np.log(sigmoid(np.dot(X,theta.T))))
#    print("first",first)
    second=np.multiply( (1-y),np.log(1-sigmoid(np.dot(X,theta.T))))
#    print("second",second)

    return np.sum(first-second)/len(X)


thiscost = costFunction(theta, X, y)
print()
print('cost = ' , thiscost)

#function to calc gradientDescent by function summation
'''def gradientDescent(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    grad=np.zeros(3)
    parameters=theta.shape[1]
    error=sigmoid(X*theta.T)-y
    for i in range(parameters):
        term=np.multiply(error,X[:,i])
        grad[i]=np.sum(term)/len(X)
        
    return grad'''
#function to calc gradientDescent by function Matrics

def gradientDescent(theta,X,y):
    thetav = np.matrix(theta)
    Xv = np.matrix(X)
    yv = np.matrix(y) 
    return  (X.T*(sigmoid(Xv*thetav.T)-yv)   ) /len(X)









    
#to find the miniumim theta using scipy.optimize by gradent Desecnt
    # هنا بيغنيك عن الفاااااااااااااااااااا و عدد اللفات طرح كل ثيتا من اللي قبلها في كل لفه 
import scipy.optimize as opt
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradientDescent, args=(X, y))    
    
CostAfterOptimize=costFunction(result[0],X,y)
print()
print('cost after optimize = ' , CostAfterOptimize)
print()    
# to predict the value and checkk
def predict(theta, X,y):
    
    return [1 if x>=0.5 else 0 for x in sigmoid(X*np.matrix(theta).T)]

prediction=predict(result[0], X,y)
#check
correct=[1 if((a==1 and b==1 )or (a==0 and b==0)) else 0 for a,b in zip(prediction,y)]

#accurncy 
print("accurency = %", (np.sum(correct)/len(correct)*100))

nums=np.arange(-10,10)

plt.plot(nums,sigmoid(nums),"r")
plt.show()

#draw the data for positive and negative
print("================================================")

plt.scatter(postive.iloc[:,0],postive.iloc[:,1])
plt.scatter(negative.iloc[:,0],negative.iloc[:,1],color="red")

plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()



#check for me 




#def gradientDescent(theta,X,y):
#    theta = np.matrix(theta)
#    X = np.matrix(X)
#    y = np.matrix(y)
#    grad=np.matrix(np.zeros(3))
#    parameters=theta.shape[1]
#    
#    for j in range(100000):
#        error=sigmoid(X*theta.T)-y
#        print(error)
#        for i in range(parameters):
#            term=np.multiply(error,X[:,i])
#            grad[0,i]=theta[0,i]-(( 0.001/len(X))*np.sum(term))
#        theta=grad
#        
#        
#    return theta

#costFunction(gradientDescent(theta,X,y),X,y)
