import numpy as np
def nonlin(y,deriv=False):
    x = np.copy(y)
    if(deriv==True):
        x[x>=0]=1
        x[x<0]=0
        return x
    return np.maximum(0,x)

#def nonlin(x,deriv=False):
 #   if(deriv==True):
  #        return 1
   # else:
    #    return x

  
   
#input data (instances)
x = np.array([[1,0.374186,0.904727],
[1,0.397672,0.074281],
[1,0.387948,0.153990],
[1,0.541511,0.148609],
[1,0.366906,0.760656],
[1,0.389160,0.570398],
[1,0.892068,0.926974],
[1,0.507588,0.752778],
[1,0.316727,0.477287],
[1,0.727478,0.414801]])

#output labels
y = np.array([[0.680328],
[0.905738],
[1.000000],
[0.025000],
[0.778689],
[0.536885],
[0.672131],
[0.885246],
[0.549180],
[0.598361]])

#seed the pseudo-random no. generator
np.random.seed(1)

#synapses (weights)
syn0 = np.random.random((3,1))

#forward propagation, training
for j in range(90000):
    #layers
    l0 = x #inputs
    l1 = nonlin(np.dot(l0,syn0)) #hidden layer
    
    #back propagation
    l1_error = y - l1
    if(j % 10000) == 0:
        print ('Error'+str(np.mean(np.abs  (l1_error))))
    
    l1_delta = .0001*l1_error*nonlin(l1,deriv=True)

    #learning rate
    
    syn0 += l0.T.dot(l1_delta)
    
print ('Output after training')
print (l1)