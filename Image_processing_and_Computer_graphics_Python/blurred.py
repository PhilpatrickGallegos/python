import numpy as np
from PIL import Image
from padding import padder

def blur(kernel):#math function is the sumation of the array w/ the kernel
  count = 0
  for i in kernel:
    for j in i:
      count = count + j
  s = count/9
  return s

def kernel_action(matrix, o_row, o_col):#grabbing information from padded image
  temp = np.zeros((o_row+2, o_col+2)) #a empty matrix where results go
  for i in range(o_row):#length of row size of original image
    for j in range(o_col):#length of column size of original image
      kernel = np.ones((3,3))#create a kernal matrix of ones
      row = i+1#avoid selecting the padding as an important coordinate in row
      col = j+1#avoid selecting the padding as an important coordinate in column
      for x in range(-1,2,1): #grab surrounding rows
        for y in range(-1,2,1): #grab surrounding columns
          kernel[x+1][y+1] = matrix[row+x][col+y]# replace kernel values
      temp[row][col] = blur(kernel) #replace empty matrix with results
  return temp
matrix, o_row, o_col = padder("box_image.png") #call in the padded image
#THIS CAN BE LOOPED INDEFINITELY!###########
matrix = kernel_action(matrix, o_row, o_col) #call the matrix blur function
############################################

arr = np.zeros((o_row, o_col))#CREATE AN EMPTY ARRAY WITH THE ORIGINAL SIZE	
for i in range(o_row): #loop through the image rows
    for j in range(o_col): #loop through the image column
      arr[i][j] = matrix[i+1][j+1] #remove the padding
array = np.array(arr, dtype=np.uint8())#format the numpy array's data format
#some image magic going on here or the new image is placed in old array.
img = Image.fromarray(array)
img.save('new_image.png')
