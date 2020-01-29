import numpy as np
from PIL import Image

def padder(name):
  #import image into a numpy array
  im = Image.open(name) #find and grab data out of the matrix
  arr= np.array(im) #turn image data to numpy array
  temp = arr.shape # (5x5) matrix

  #create an empty matrix of zeros
  row = temp[0] + 2 #add two rows to the matrix
  col = temp[1] + 2 #add two columns to the matrix
  matrix = np.zeros((row,col)) #create an empty matrix of zeros

  #add the image matrix to the empty matrix
  for i in range(len(arr)): #loop through the image rows
    for j in range(len(arr[i])): #loop through the image column
      matrix[i+1][j+1] = arr[i][j] #add the image data in the center of the empty matrix
  #print matrix #now the image is placed in the center of the matrix with a padding od zeros
  return matrix, temp[0], temp[1] #send back

