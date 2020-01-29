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
    return x

   
#input data (instances)
x = np.array([[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]])

#output labels
y = np.array([[0],
[1],
[1],
[0]])

#seed the pseudo-random no. generator
np.random.seed(1)

print (x)
print
print (y)

#synapses (weights)
syn0 = 2*np.random.random((3,4))-1
syn1 = 2*np.random.random((4,1))-1

print
print (syn0)
print
print (syn1)

#forward propagation, training
for j in range(60000):
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
    
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print ('Output after training')
print (l2)
# print
# print (syn0)
# print
# print (syn1)
# print
# print (l2_error)
    
