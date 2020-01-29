import numpy as np
#sigmoid function and its derivative
def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

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

print (x)
print
print (y)

#synapses (weights)
syn0 = 2*np.random.random((2,10))-1
syn1 = 2*np.random.random((10,1))-1

#print
#print (syn0)
#print
#print (syn1)

#forward propagation, training
for j in range(60000):
    #layers
    l0 = x #inputs
    l1 = nonlin(np.dot(l0,syn0)) #hidden layer
    l2 = nonlin(np.dot(l1,syn1)) #output
    #back propagation
    l2_error = y - l2
    if(j % 10000) == 0:
       print ('Error'+str(np.mean(np.abs  (l2_error))))#chain rule
    l2_delta = l2_error*nonlin(l2,deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1,deriv=True)
    
    # syn updates
    syn1 += 2*l1.T.dot(l2_delta)
    syn0 += 2*l0.T.dot(l1_delta)
    
print ('Output after training')
print (l2)