from PIL import Image
import numpy as np

def image_gen(array, name, size):
  arr = []
  for i in range(size):
    arr.append(array)
  temp = np.asarray(arr, dtype=np.uint8())
  print temp
  im = Image.fromarray(temp)
  im.save(name)
arr1 = [255,255,255,0,0]
image_gen(arr1, "box_image.png", 5)
arr2 = [0, 255, 255, 0, 255]
image_gen(arr2, "gauss_image.png", 5)
