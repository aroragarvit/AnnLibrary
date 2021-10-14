##make inputs , and outputs  matrix ,  and number of samples ,  as well as number of features in input , number of outputs in single sample as data members also pass l (;;;;; for functions sequence and make it as data members )
"""
inputs and outputs are matrices , l=[1,1,0] , hidden_layers=[3,6]

self.num_inputs =np.shape(inputs)[0]
self.num_outputs=np.shape(outputs)[0]
number_of_samples=np.shape(inputs)[1]
self.m=number_of_samples



"""

# now make all the rest of functons arguments less 

import pandas as pd 

import numpy as np
from numpy import exp
np.set_printoptions(suppress=True)

from sklearn import preprocessing


train_x = pd.read_csv("train_x.csv")
headerList = ['A', 'B', 'C','D','E','F','G','H','I']
train_x.to_csv("train_x2.csv", header=headerList, index=False)
train_x2=pd.read_csv("train_x2.csv")
train_x2.drop('A', inplace=True, axis=1)
train_x2=train_x2.to_numpy()
train_x2=np.transpose(train_x2)
print('.....................................INPUT DATA ....................................................................')
print(train_x2.shape)
print(train_x2)


test_x = pd.read_csv("test_x.csv")
headerList = ['A', 'B', 'C','D','E','F','G','H','I']
test_x.to_csv("test_x2.csv", header=headerList, index=False)
test_x2=pd.read_csv("test_x2.csv")
test_x2.drop('A', inplace=True, axis=1)
test_x2=test_x2.to_numpy()
test_x2=np.transpose(test_x2)


train_y = pd.read_csv("train_y.csv")
headerList = ['A']
train_y.to_csv("train_y2.csv", header=headerList, index=False)
train_y2=pd.read_csv("train_y2.csv")
train_y2=train_y2.to_numpy()
train_y2=np.transpose(train_y2)
print("-------------------------------------------output Data -----------------------------------------------------------------------------")
print(train_y2.shape)
print(train_y2)

test_y = pd.read_csv("test_y.csv")
headerList = ['A']
test_y.to_csv("test_y2.csv", header=headerList, index=False)
test_y2=pd.read_csv("test_y2.csv")
test_y2=test_y2.to_numpy()
test_y2=np.transpose(test_y2)







import numpy as np
from numpy import exp
class ann:



    def __init__(self,inputs , outputs ,hidden_layers,l):
        self.l=l 

        self.inputs=inputs
        self.outputs=outputs

        self.num_inputs =np.shape(inputs)[0]
        self.num_outputs=np.shape(outputs)[0]
        
        number_of_samples=np.shape(inputs)[1]
        self.m=number_of_samples
        
        layers=[self.num_inputs]+hidden_layers+[self.num_outputs]  
        self.layers=layers 

        weights=[]     
        bias=[]
        biases=[]  
        error_wrt_a=[]
        error_wrt_z=[]
        error_wrt_w=[]
        error_wrt_b=[]
        for i in range(len(layers)-1):            
            a=np.random.rand (layers[i+1],layers[i])   
            weights.append(a)
            
            b=np.random.rand(layers[i+1],1) 
            bias.append(b)                                # biases are  probably not getting updated in train function so change bias as biases in our z only it will be better 
            biases.append(np.repeat(b , repeats =self.m, axis=1)) # check this also
            
            c=np.zeros(  ( (layers[i+1]),self.m        )     )    
            error_wrt_a.append(c)
            
            d=np.zeros(((layers[i+1]),self.m))
            error_wrt_z.append(d)
            
            e=np.zeros(((layers[i+1],layers[i])))
            error_wrt_w.append(e)
            
            f=np.zeros((layers[i+1],1))
            error_wrt_b.append(f)
            
        self.weights=weights
        self.bias=bias
        self.biases=biases
        
        self.activations=[]
        self.z=[]

        self.error_wrt_a=error_wrt_a
        self.error_wrt_z= error_wrt_z
        self.error_wrt_w= error_wrt_w
        self.error_wrt_b= error_wrt_b
        
        self.function_derivitive=[self.sigmoid_derivative,self.reluDerivative]
        self.function=[self.sigmoid,self.relu]
      

    def sigmoid_derivative(self, x):        
      

        return x * (1.0 - x)


    

       
    def sigmoid(self , x):
        y=1/1+np.exp(-x)                        
        return y

    def relu(self,x):

        return np.maximum(0.0, x)
    
    def reluDerivative(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x


    def softmax (self,x):
        e=exp(x)
        return e/e.sum()

    
    
  
        
    

     


 

    
    
    def forward_propogate(self):   
        self.activations.clear()
        self.z.clear()
        self.z.append(self.inputs)
        self.activations.append(self.inputs)        ### note z and a will contain all the layers from  1st given layer at indeex 0 to the last output predicted layer 
        for i in range (len(self.layers)-1):
            self.z.append(np.add( np.dot( self.weights[i],self.activations[i] ) , self.biases[i] ) )    
            self.activations.append(self.function[self.l[i]](self.z[i+1]  ) )            # check and show this line 
    
    def backpropogation(self ):
       
        self.error_wrt_z[len(self.layers)-2]=np.subtract(self.activations[len(self.layers)-1] , self.outputs)
       # self.error_wrt_a[len(self.len(self.layers)-2)]=np.subtract(np.divide(1-self.outputs,self.activations[len(self.layers-1)] ) , np.divide(self.outputs,self.activations[len(self.layers-1)] )  )  # in this case we are always taking last one as sigmoid 
        for i in reversed(range(len(self.layers) - 1)):
            self.error_wrt_w[i]=1/self.m*(np.dot(self.error_wrt_z[i],np.transpose(self.activations[i])))
            self.error_wrt_b[i]=1/self.m *(np.sum(self.error_wrt_z[i] , axis=1 ))
            if i>=1:
                self.error_wrt_z[i-1]=np.multiply (  np.dot ( np.transpose (self.weights[i]) , self.error_wrt_z[i] )  , self.function_derivitive[ self.l[i-1] ] ( self.function[ self.l[i-1] ]   (self.z[i]) ) )


            else:
                break

            
            
            
          





                

    def train(self,epoches,alpha):
        for i in range(epoches):
            self.forward_propogate()
            self.backpropogation()
            for i in range(len (self.layers)-1):
                self.weights[i]=np.subtract(self.weights[i],alpha*self.error_wrt_w[i])
                self.bias[i]=np.subtract(self.bias[i],alpha*self.error_wrt_b[i])

    def error(self):
        pass

    def accuracy():
        pass 


a=ann(inputs=train_x2 , outputs=train_y2 ,hidden_layers=[3,3,3,1],l=[1,1,1,1,0])
a.forward_propogate()     #  all  are already formed  in activations in z 


for i in  a.z:


    print("---------------------------------------------------------------------------------------------------------------------------------------------------")
    print(i.shape)
    print(i)

    print("------------------------------------------------------------------ACTIVATIONS------------------------------------------------------------------------" )
for i in  a.activations:   
    print(i.shape)
    print(i)

print("---------------------------------------------------------------------------------------------------------------------------------------------------")

for i in a.weights:
    print(i.shape)
    print(i)

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")

a.train(epoches=10,alpha=0.01)



for i in a.error_wrt_w:
    print(i)
    



