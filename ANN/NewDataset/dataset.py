# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:48:09 2020

@author: INTENTODEMUSICO
"""

import cv2
import os.path
import numpy as np
import scipy.stats as sts
import univariate as univariate
from scipy.stats.mstats import gmean
#from math import exp as e

#%%
def hjorth_params(trace):
    return univariate.hjorth(trace)
def e(x):
    return x
fail=0
fallas=np.array(("Ubicación","Gmean","Morbility","Complexity"))

def attributes(location,kind):
    global fail
    global fallas
    img = cv2.imread(location,0)
      
    #Preprocessing
    #If image is monochromatic
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #Else 
    #Gray scale
    
    trace=hist.reshape(256)
    trace[trace!=-10000]+=1
    
    #gTrace=trace[trace>0]
    
    #Getting atributes
    attributes=np.zeros(10,dtype='<U256')#.astype(object)
    
    #Kurtosis 
    attributes[0]=str(sts.kurtosis(trace))
    #Skewness
    attributes[1]=str(sts.skew(trace))
    #Std
    attributes[2]=str(np.std(trace))
    #Range
    attributes[3]=str(np.ptp(trace))
    #Median 
    attributes[4]=str(np.median(trace))
    #Geometric_Mean 
    attributes[5]=str(gmean(trace))
    #Hjorth
    a,mor, comp= hjorth_params(trace)
    #Mobility 
    attributes[6]=str(mor)
    #Complexity
    attributes[7]=str(comp)
    attributes[8]=str(kind)
    
    attributes[9]=str(location)
    #print(attributes)
    if(str(comp)=='nan' or str(mor)=='nan' or str(attributes[5])=="nan"):
        a=np.array((location,str(attributes[5]),mor,comp))
        fallas=np.vstack((fallas,a))
        fail+=1
    return attributes
#%%
x=np.zeros(10)
test=np.zeros(10,dtype='<U256')#.astype(object)#,headers=['Kurtosis', 'Skewness', 'Std', 'Range', 'Median', 'Geometric_Mean', 'Mobility', 'Complexity','IsStego','Location'])
train=np.zeros(10,dtype='<U256')#.astype(object)#,headers=['Kurtosis', 'Skewness', 'Std', 'Range', 'Median', 'Geometric_Mean', 'Mobility', 'Complexity','IsStego','Location'])
print(np.shape(test))
steg="train_steg_0."
carr="train_carr"

stegTe="test_steg_0."
carrTe="test_carr"

pATestS='.\Testing\Stego'
pATrainS='.\Training\Stego'
pATestC='.\Testing\Carriers'
pATrainC='.\Training\Carriers'

#Training
for i in range(1,2001):
    if (i<401):
        pay=1    
    elif (i<801):
        pay=2 
    elif (i<1201):
        pay=3 
    elif (i<1601):
        pay=4 
    elif (i<2001):
        pay=5 #+1#(i//400)%5
    num=i
    strC=carr+str(i)+".bmp"
    imgLocationC=os.path.join(pATrainC,strC)
    strS=steg+str(pay)+"_"+str(num)+".bmp"
    imgLocationS=os.path.join(pATrainS,strS)
    aTrain=attributes(imgLocationC,0)
    train=np.vstack((train,aTrain))
    aTrain=attributes(imgLocationS,1)
    train=np.vstack((train,aTrain))

#Testing
for i in range(1,501):
    if (i<101):
        pay=1    
    elif (i<201):
        pay=2 
    elif (i<301):
        pay=3 
    elif (i<401):
        pay=4 
    elif (i<501):
        pay=5 #+1#(i//400)%5
    num=i
    strC=carrTe+str(i)+".bmp"
    imgLocationC=os.path.join(pATestC,strC)
    strS=stegTe+str(pay)+"_"+str(num)+".bmp"
    imgLocationS=os.path.join(pATestS,strS)
    aTest=attributes(imgLocationC,0)
    test=np.vstack((test,aTest))  
    aTest=attributes(imgLocationS,1)
    test=np.vstack((test,aTest))

#%%    
print(fail)
test=np.delete(test,0,0)
train=np.delete(train,0,0)
path='./Final'
pTrain=os.path.join(path,"train_5000.csv")
pTest=os.path.join(path,"test_5000.csv")
np.savetxt(pTrain,train, delimiter=",", fmt="%s")
np.savetxt(pTest,test, delimiter=",", fmt="%s")
asd="fallas.csv"
np.savetxt(asd,fallas,delimiter=",", fmt="%s")
