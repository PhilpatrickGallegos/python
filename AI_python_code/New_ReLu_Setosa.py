import numpy as np
def nonlin(y,deriv=False):
    x = np.copy(y)
    if(deriv==True):
        x[x<0]=0
        x[x>=0]=1
        return x

    return np.maximum(0,x)

f = open ('iris_data.dat' ,'r')

data = f.read()
f.close()
#print (data)

#data[:,-1]

t= data.split('\n')
#print (t)
x = []
y = []
a = []
for i in range (len(t)):
    t[i]=t[i].split(',')
    a.append(t[i])

for i in range(len(a)):
    z = []
    x.append(z)
    for j in range(len(a[i])):
        #print (a[i][j])
        if (j >= 4):
            y.append(a[i][j])
        if (j < 4):
            x[i].append(float(a[i][j]))
#print(x)
k = [[1,0,0], [0,1,0], [0,0,1]]
for i in range(len(y)):
    if(y[i]=='setosa'):
        y[i] = k[0]
    if(y[i]=='versicolor'):
        y[i] = k[1]
    if(y[i]== 'virginica'):
        y[i] = k[2]
#print(y)
x = np.asarray(x)
y = np.asarray(y)
#seed the pseudo-random no. generator
np.random.seed(1)


#synapses (weights)
syn0 = 2*np.random.random((4,3))-1

#print
#print (syn0)
#print
#print (syn1)

#forward propagation, training
for j in range(60000):
    #layers
    l0 = x #inputs
    l1 = nonlin(np.dot(l0,syn0)) #hidden layer
    #back propagation
    l1_error = y - l1
    if(j % 10000) == 0:
       print ('Error'+str(np.mean(np.abs  (l1_error))))#chain rule
    l1_delta = l1_error*nonlin(l1,deriv=True)
    
    # syn updates
    syn0 += 0.0001*l0.T.dot(l1_delta)
    
print ('Output after training')
print (l1)
print (y)

