import scipy.misc
import scipy.ndimage 
from scipy.misc.pilutil import Image
# opening the image and converting it to grayscale 
a = Image.open('../Figures/ct_saltandpepper.png'). 
	convert('L') 
# performing the median filter 
b = scipy.ndimage.filters.median_filter(a,size=5,
	 footprint=None,output=None,mode='reflect', 
	 cval=0.0,origin=0) 
# b is converted from an ndarray to an image
b = scipy.misc.toimage(b) 
b.save('../Figures/median_output.png')