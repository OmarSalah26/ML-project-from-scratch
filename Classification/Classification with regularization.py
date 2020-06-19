
#import library 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#import data 
dataset=pd.read_csv("dataForRegulariztion.csv",header=None, names=['test1', 'test2', 'Accepted'])


#show data
print('data = ')
print(dataset.head(10) )
print()
print('data.describe = ')
print(dataset.describe())


postive=dataset[dataset["Accepted"].isin([1])]
negative=dataset[dataset["Accepted"].isin([0])]
 
print("Accepted student \n",postive)
print("NonAccepted student \n",negative )

#draw the data for positive and negative
print("================================================")

plt.scatter(postive.iloc[:,0],postive.iloc[:,1] ,label="Accepted")
plt.scatter(negative.iloc[:,0],negative.iloc[:,1],marker='x',
            label="NonAccepted")

plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()
print("================================================")
x1=dataset.iloc[:,0]
x2=dataset.iloc[:,1]
dataset.insert(3,"X0",1) #adding x0


'''
x1 + x1^2 + x1x2 + x1^3 + x1^2 x2 + x1 x2^2 + x1^4 + x1^3 x2 + x1^2 x2^2 + x1 x2^3


F10 = x1

F20 = x1^2
F21 = x1 x2

F30 = x1^3
F31 = x1^2 x2
F32 = x1 x2^2

F40 = x1^4
F41 = x1^3 x2
F42 = x1^2 x2^2
F43 = x1 x2^3 

'''
degree=5
for i in range(1, degree): # 1,2,3,4
    for j in range(0, i):  # 0 , 1 , 2 ,2
        dataset['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j) # i=3 , j=2

 

dataset.drop('test1', axis=1, inplace=True)
dataset.drop('test2', axis=1, inplace=True)

print('data \n' , dataset.head(10))

print('................................................')

#segmoid function 
print("================================================")
def sigmoid(z):
    return 1/(1+np.exp(-z))


nums=np.arange(-10,10)

plt.plot(nums,sigmoid(nums),"r")
plt.show()




#set training data and target data
cols=dataset.shape[1]
X=dataset.iloc[:,1:cols-1]
y=dataset.iloc[:,0:1]    
theta=np.zeros(X.shape[1])




#convert to matrix 
X=np.array(X)
y=np.array(y)
theta=np.zeros(X.shape[1])
print()
print('X.shape = ' , X.shape)
print('theta.shape = ' , theta.shape)
print('y.shape = ' , y.shape)
print()
print('X = ' , X)
print('theta = ' , theta)
print('y = ' , y)


learningRate=0.000001 #والله العظيم ده اللامضا وانا متاكد ان هشام غلطان 
#costFunction
def costFunction(theta, X,y,learningRate): #بس انا شايف انه يقصد بيها لامضا
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first=np.multiply( -y,np.log(sigmoid(np.dot(X,theta.T))))
#    print("first",first)
    second=np.multiply( (1-y),np.log(1-sigmoid(np.dot(X,theta.T))))
#    print("second",second)
    #thissssssssssssssssssssssssssssssssssssssssssssssssssss

    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
#thissssssssssssssssssssssssssssssssssssssssssssssssssss

    return (np.sum(first-second)/len(X) )+reg


thiscost = costFunction(theta, X,y,learningRate)
print()
print('cost = ' , thiscost)

#function to calc gradientDescent by function summation
def gradientDescentReg(theta,X,y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y= np.matrix(y)
    
    parameters=theta.shape[1]
    grad=np.zeros(parameters)

    error=sigmoid(X*theta.T)-y
    
    for i in range(parameters):
        term=np.multiply(error,X[:,i])
#        print("ttttttttttttttttttttttttttttttttttttttt",term)
#thissssssssssssssssssssssssssssssssssssssssssssssssssss
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i]=(np.sum(term)/len(X))+((learningRate/len(X))*theta[:,i])
        
    return grad






CostbeforOptimize=costFunction(theta,X,y,learningRate)
print()
print('cost befor optimize = ' , CostbeforOptimize)
print()   

    
#to find the miniumim theta using scipy.optimize by gradent Desecnt
    # هنا بيغنيك عن الفاااااااااااااااااااا و عدد اللفات طرح كل ثيتا من اللي قبلها في كل لفه 
import scipy.optimize as opt
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradientDescentReg, args=(X, y,learningRate))    
    
CostAfterOptimize=costFunction(result[0],X,y,learningRate)
print()
print('cost after optimize = ' , CostAfterOptimize)
print()    



# to predict the value and checkk
def predict(theta, X):
    
    return [1 if x>=0.5 else 0 for x in sigmoid(X*np.matrix(theta).T)]

prediction=predict(result[0], X)
#check
correct=[1 if((a==1 and b==1 )or (a==0 and b==0)) else 0 for a,b in zip(prediction,y)]

#accurncy 
print("accurency = %", (sum(map(int, correct))/len(correct))*100)
accuracy = (sum(map(int, correct)) % len(correct)) #  طريقه هشاااام هنا غلط عشان هنا المجموع بتاعي مش 100 زي المثال اللي فااات هنا المجموع 118 
print ('accuracy = {0}%'.format(accuracy))

#########################################################################
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



