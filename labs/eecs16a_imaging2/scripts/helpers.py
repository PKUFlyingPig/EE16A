# Import necessary libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def simulateRealImaging(imagePath, matrix, saveName, height, width, sigma):
    simulateImaging(imagePath, matrix, saveName, height, width, sigma)
    
def simulateIdealImaging(imagePath, matrix, saveName, height, width):
    simulateImaging(imagePath, matrix, saveName, height, width)

def simulateImaging(imagePath, matrix, saveName, height, width, sigma = 0):
    #Read image in CWD using PIL (as grayscale) and convert to np array 
    img_arr = np.array(Image.open(imagePath).convert(mode = 'L'))

    plt.imshow(img_arr, cmap = 'gray')

    i_vec = np.reshape(img_arr, (height*width, 1))
    s_vec = np.dot(matrix, i_vec)
    if sigma:
        # Noise of mean 0, with standard deviation `sigma` is added to each element of the
        # original column vector s
        noise = np.random.normal(0, sigma, height * width)
        noise = np.reshape(noise, (height * width, 1))
        s_vec += noise
    np.save(saveName + '.npy', s_vec)

def imageResize(imagePath, saveName, newHeight, newWidth):
	newSize = (newWidth, newHeight)
	originalIm = Image.open(imagePath).convert(mode = 'L')		#Opening uploaded image as grayscale
	(width, height) = originalIm.size
	region = (0, 0, 0, 0)
	if width > height:										#Selecting centred square for wide images						
		diff = width - height
		region = (diff/2, 0, width - (diff/2), height)
	elif height > width:									#Selecting centred square for tall images
		diff = height - width
		region = (0, diff/2, width, height - (diff/2))		
	else:
		region = (0, 0, width, height)
	newIm = originalIm.resize(newSize)
	# newIm = originalIm.resize(newSize, box = region)
	newIm.save(saveName)