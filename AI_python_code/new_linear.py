import numpy as np
#sigmoid function and its derivative
# def nonlin(x,deriv=False):
#     if(deriv==True):
#         x[x<=0]=0
#         x[x>0]=1
#     return x

#     return np.maximum(x,0)

def nonlin(x,deriv=False):
    if(deriv==True):
          return 1
    else:
        return x

  
   
#input data (instances)
x = np.array([[0.374186,0.904727],
[0.397672,0.074281],
[0.387948,0.153990],
[0.541511,0.148609],
[0.366906,0.760656],
[0.389160,0.570398],
[0.892068,0.926974],
[0.507588,0.752778],
[0.316727,0.477287],
[0.727478,0.414801]])

#output labels
y = np.array([[0.680328],
[0.905738],
[1.000000],
[.025000],
[0.778689],
[0.536885],
[0.672131],
[0.885246],
[0.549180],
[0.598361]])

#seed the pseudo-random no. generator
np.random.seed(1)

print (x)
print
print (y)

#synapses (weights)
syn0 = 2*np.random.random((2,2))-1
syn1 = 2*np.random.random((2,1))-1

print
print (syn0)
print
print (syn1)

#forward propagation, training
for j in range(90000):
    #layers
    l0 = x #inputs
    l1 = nonlin(np.dot(l0,syn0)) #hidden layer
    l2 = nonlin(np.dot(l1,syn1)) #output
    #back propagation
    l2_error = y - l2
    if(j % 10000) == 0:
        print ('Error'+str(np.mean(np.abs  (l2_error))))
    l2_delta = l2_error*nonlin(l2,deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1,deriv=True)

    #learning rate
    syn1 += 0.01*l1.T.dot(l2_delta)
    syn0 += 0.01*l0.T.dot(l1_delta)
    
print ('Output after training')
print (l2)