import numpy as np

#sigmoid function and its derivative. This is where we construct our activation function
def nonlin(x,deriv=False):
   if(deriv==True):
       return(x*(1-x))
   return(1/(1+np.exp(-x)))

# Read data from file 'filename.csv' 
# Control delimiters, rows, column names with read_csv (see later) 

data = np.loadtxt("breast_cancer_data.csv",delimiter=",") 

#features:


#y = Outcome

x = data[0:511,0:-1]         #reads in all the rows and most of the columns except the last.
y = np.array([data[0:511,-1]]).T  #reads in all the rows and only the last column 

x_max = np.amax(x,axis=0)    #takes the max value of each input column
x_min = np.amin(x,axis=0)

y_max = np.amax(y,axis=0)    #takes the max value of each output
y_min = np.amin(y,axis=0)

x = (x-x_min)/(x_max-x_min)
y = (y-y_min)/(y_max-y_min)                  #normalizes the y outputs

#seed the pseudo-random no. generator
np.random.seed(1)

#synapses (weights)
weight_0 = 2*np.random.random((30,40))-1  #element of R^30x40, 1st weight matrix 
weight_1 = 2*np.random.random((40,1))-1  #element of R^40x1, 2nd weight matrix

#... if more were needed

#print()
#print(weight_0) just to check if working and bringing the rt data for the first weighted dataset
#print()
#print(weight_1) just to check if working and bringing the rt data for the second weighted dataset
#...

#forward propagation, training
for j in range(1000):
    #layers
    u_0 = x #input layer
    u_1 = nonlin(np.dot(u_0,weight_0)) #hidden layer 0 = x_l*w_0 
    u_2 = nonlin(np.dot(u_1,weight_1)) #hidden layer 1 = u_0*w_1

    #print(u_1.shape) to ensure the rt size
    #back propagation
    u_2_error = y - u_2
    if(j % 1000) == 0:
       print('Error = '+str(np.mean(np.abs  (u_2_error))))


    eta = 0.009
    u_2_delta = eta*u_2_error*nonlin(u_2,deriv=True) #back propagation
    u_1_error = u_2_delta.dot(weight_1.T) # bringing in the weight for back propagation
    
    u_1_delta = eta*u_1_error*nonlin(u_1,deriv=True) # new sum after back propagation
    
    weight_1 += u_1.T.dot(u_2_delta) # summing new weight
    weight_0 += u_0.T.dot(u_1_delta)


u_2_max = np.amax(u_2,axis=0)    #takes the max value of each output
u_2_min = np.amin(u_2,axis=0)    #takes the min value of each output

y_normalized = (u_2-u_2_min)/(u_2_max-u_2_min)
    
print('Output after training')
print(u_2[1:20])

y_train = np.where(y_normalized >= 0.5, 1, 0)
print(y_train[1:20]) #results for outcome normalized with a threshold

#compare y_label to outcome for accuracy
correct = (y == y_train)
accuracy = int((correct.sum()/correct.size)*100)

print("Predictions meet a " + str(accuracy) + "% accuracy rate") # get accracy print out

x = data[0:511,0:-1]         #reads in all the rows and most of the columns except the last.
y = np.array([data[0:511,-1]]).T  #reads in all the rows of only the last column 



