# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image
from IPython import display

# Inputs
#  `avg1sPerRow`: Average number of illuminated (white) pixels per mask (row of `H`)
#  `X`: Image width
#  `Y`: Image height
#  `okCondNum`: Don't worry about it :) -- leave it as default
# Outputs
#  `H`: Generated random binary mask matrix
def generateRandomBinaryMask(avg1sPerRow, X = 32, Y = 32, okCondNum = 20000, plot=True):

    numRows = X * Y
    numCols = X * Y

    HSize = numRows * numCols
    total1s = numRows * avg1sPerRow
    condNum = 100000
    while (condNum > okCondNum):
        H1D = np.zeros(HSize)

        # For a 1D vector of size `HSize`, randomly choose indices in the range
        # [0, `HSize`) to set to 1. In total, you want to select `total1s`
        # indices to set to 1.
        randomizedH1DIndices = np.random.permutation(HSize)
        rand1Locs = randomizedH1DIndices[:total1s]

        # `H1D` was originally initialized to all 0's. Now assign 1 to all locations
        # given by `rand1Locs`.
        H1D[rand1Locs] = 1

        # Reshape the 1D version of `H` (`H1D`) into its 2D form.
        H = np.reshape(H1D, (numRows, numCols))

        condNum = np.linalg.cond(H)

    if(plot):
        plt.imshow(H, cmap = 'gray', interpolation = 'nearest')
        plt.title('H')
        plt.show()

    # Extra information you haven't learned about yet
    # print('H Condition Number: ' + str(condNum))

    return H;

# Inputs
#  `vec`: 2-element array where vec[0] = X, vec[1] = Y for the point (X, Y)
#  `label`: String label for the vector
def plotVec2D(vec, label, color = "blue"):
    X = vec[0]
    Y = vec[1]
    # Plots a vector from the origin (0, 0) to (`X`, `Y`)
    q = plt.quiver(
        [0], [0], [X], [Y],
        angles = 'xy', scale_units = 'xy', scale = 1,
        color = color)
    plt.quiverkey(
        q, X, Y, 0,
        label = label, coordinates = 'data', labelsep = 0.15,
        labelcolor = color, fontproperties={'size': 'large'})
    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    # Adapt the current plot axes based off of the new vector added
    # Plot should be square
    min = np.amin([ymin, Y, xmin, X])
    max = np.amax([ymax, Y, xmax, X])
    # Add some margin so that the labels aren't cut off
    margin = np.amax(np.abs([min, max])) * 0.2
    min = min - margin
    max = max + margin
    plt.ylim(min, max)
    plt.xlim(min, max)
    plt.ylabel('Y')
    plt.xlabel('X')

# Inputs
#  `vec`: 2-element array representing the eigenvector, where vec[0] = X, vec[1] = Y
#  `eigenvalue`: eigenvalue corresponding to the eigenvector (label)
def plotEigenVec2D(vec, eigenvalue, color = "red"):
    plotVec2D(vec, r'$\lambda = $' + str(eigenvalue), color = color)

# Supplied mystery matrix :)
# Inputs
#  `shape`: 2-element array. shape[0] = # of rows; shape[1] = # of columns
# Outputs
#  `H`: 2D Hadamard matrix (Hadamard)
def createHadamardMatrix(shape, plot=True):
    from scipy.linalg import hadamard
    # Hadamard matrix with +1, -1 entries
    # Note that the matrix order should be 2^r
    H = hadamard(shape[0])
    # Needs to be binary
    H = (H + 1) / 2

    if(plot):
        plt.imshow(H, cmap = 'gray', interpolation = 'nearest')
        plt.title('Hadamard Matrix :)')
    return H;

# Inputs
#  `H`: Mask matrix to analyze
#  `matrixName`: Name of the matrix for the plot title (i.e. Mystery H or Random Binary H)
def eigenanalysis(H, matrixName):

    # Calculate the eigenvalues of `H`
    eigen, x = np.linalg.eig(H)

    # Get the absolute values (magnitudes) of the eigenvalues
    absEigenH = np.absolute(eigen)

    plt.figure(figsize = (5, 2))

    # Make a 100-bin histogram of |eigenvalues|
    plt.hist(absEigenH, 100)

    # Plot in log scale to see what's going on
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Eigenvalues of %s" % matrixName)
    plt.ylabel('Amount in Each Bin')
    plt.xlabel('|Eigenvalue| Bins')
    plt.show()


# Inputs
#  `numCols`: Number of columns needed
def generateLinearityMatrix(numCols):
    # Testing sensor readings for projector brightness values between 10-100
    brightnesses = range(10, 102, 2)

    # Half of the mask's pixels are "white". The other half are "black."
    # This represents the maximum # of illuminated pixels that we'll be using.
    checkerBoard0 = np.tile([0, 1], int(numCols / 2))
    checkerBoard1 = np.tile([1, 0], int(numCols / 2))

    # Take an average: illuminate one half of the image, then the other half
    checkerBoardRows = np.append([checkerBoard1], [checkerBoard0], axis = 0)

    # Create new rows whose "white" brightnesses are scaled
    for i in range(len(brightnesses)):
        newRows = brightnesses[i] / 100.0 * checkerBoardRows
        if (i == 0):
            linearityMatrix = newRows
        else:
            linearityMatrix = np.append(linearityMatrix, newRows, axis = 0)

    np.save('LinearityMatrix.npy', linearityMatrix)



# OMP ------------------------------------------------------

# OMP with Gram Schmidt Orthogonalization. Applies to images, sparse in the DCT domain  (adapted from Vasuki's code).
# Inputs
#  `imDims`: Image dimensions
#  `sparsity`: # of expected non-zero elements in the DCT domain (should be << # of pixels in the image)
#  `measurements`: Some elements of s (after offset removal)
#  `A`: Transformation matrix
#  `MIDCT`: IDCT dictionary
#   Note: OMP solves for x in b = Ax, where A = H * MIDCT. i = MIDCT * x (what we care about)
def OMP_GS(imDims, sparsity, measurements, A, MIDCT):

    numPixels = imDims[0] * imDims[1]

    r = measurements.copy()
    r = r.reshape(len(measurements))

    indices = []
    bases = []
    projection = np.zeros(len(measurements))

    # Threshold to check error. If error is below this value, stop.
    THRESHOLD = 0.01

    # For iterating to recover all signals
    i = 0

    b = r.copy()

    while i < sparsity and np.linalg.norm(r) > THRESHOLD:

        # Calculate the correlations
        print('%d - ' % i, end = "", flush = True)

        corrs = A.T.dot(r)

        # Choose highest-correlated pixel location and add to collection
        best_index = np.argmax(np.abs(corrs))

        indices.append(best_index)

        # Gram-Schmidt Method
        if i == 0:
            bases = A[:, best_index] / np.linalg.norm(A[:, best_index], 2)
        else:
            bases = Gram_Schmidt_rec(bases, A[:, best_index])

        if i == 0:
            newsubspace = bases
        else:
            newsubspace = bases[:, i]

        projection = projection + (newsubspace.T.dot(b)) * newsubspace

        # Find component parallel to subspace to use for next measurement
        r = b - projection

        Atrunc = A[:, indices]
        if i % 20 == 1 or i == sparsity - 1 or np.linalg.norm(r) <= THRESHOLD:
            xhat = np.linalg.lstsq(Atrunc, b)[0]

            # `recovered_signal` = x
            recovered_signal = np.zeros(numPixels)
            for j, x in zip(indices, xhat):
                recovered_signal[j] = x

            ihat =  np.dot(MIDCT, recovered_signal).reshape(imDims)

            plt.imshow(ihat, cmap = "gray", interpolation = 'nearest')
            plt.title('Estimated Image from %s Measurements' % str(len(measurements)))

            display.clear_output(wait = True)
            display.display(plt.gcf())

        i = i + 1

    display.clear_output(wait = True)
    return recovered_signal, ihat

def Gram_Schmidt_rec(orthonormal_vectors, new_vector):
    if len(orthonormal_vectors) == 0:
        new_orthonormal_vectors = new_vector / np.linalg.norm(new_vector, 2)
    else:
        if len(orthonormal_vectors.shape) == 1:
            ortho_vector = new_vector - orthonormal_vectors.dot(orthonormal_vectors.T.dot(new_vector))
        else:
             ortho_vector = new_vector - orthonormal_vectors.dot(cheap_o_least_squares(orthonormal_vectors, new_vector))
        normalized = ortho_vector / np.linalg.norm(ortho_vector, 2)
        new_orthonormal_vectors = np.column_stack((orthonormal_vectors, normalized))

    return new_orthonormal_vectors

def cheap_o_least_squares(A, b):
    return (A.T).dot(b)

def noiseMassage(i, H):
    mean = np.mean(i)
    std = np.std(i)
    numSigmas = 0.5
    for idx in range(H.shape[0]):
        pixelVal = i[idx]
        if (pixelVal > mean + numSigmas * std or pixelVal < mean - numSigmas * std):
            i[idx] = (i[idx - 1] + i[idx - 1])/2
    return i

# Inputs:
#  `i2D`: 2D image you're trying to capture
#  `H`: Mask matrix
#  `matrixName`: Name of mask matrix (for image title)
#  `sigma`: Amount of noise to add (noise standard deviation)
# Outputs:
#  `s`: Sensor reading column vector with noise added
def simulateCaptureWithNoise(i2D, H, matrixName, sigma):
    # Get ideal image capture (without noise)
    idealS = simulateIdealCapture(i2D = i2D, H = H, matrixName = matrixName, display = False)
    # Noise of mean 0, with standard deviation `sigma` is added to each element of the
    # original column vector s
    noise = np.random.normal(0, sigma, H.shape[0])
    noise = np.reshape(noise, (H.shape[0], 1))
    s = idealS + noise
    return s
    

def simulateIdealCapture(i2D, H, matrixName, display = True):
    # Number of pixels in your image = `iHeight` * `iWidth`
    iHeight = i2D.shape[0]
    iWidth = i2D.shape[1]
    iSize = iHeight * iWidth
    # Convert the 2D image `i2D` into a 1D column vector `i` 
    i = np.reshape(i2D, (iSize, 1))
    s = np.dot(H, i)
    if (display):   
        # Reshape the simulated sensor output `s` into an appropriately 
        # sized 2D matrix `s2D` and plots it
        s2D = np.reshape(s, (iHeight, iWidth))
        plt.imshow(s2D, cmap = 'gray', interpolation = 'nearest')
        plt.title('Ideal Sensor Output, Using %s' % matrixName)
        plt.show()
    return s;

def simulateRealImaging(imagePath, matrix, matrixName, saveName, sigma):
    
    #Read image in CWD using PIL (as grayscale) and convert to np array 
    img_arr = np.array(Image.open(imagePath).convert(mode = 'L'))
    
    s_vec = simulateCaptureWithNoise(img_arr, matrix, matrixName, sigma)
    
    np.save(saveName + '.npy', s_vec)

def plot_image_noise_visualization(i2D, noise, s, H, title=None):
    M, N = i2D.shape
    imax = np.max(i2D)

    # Multiply noise by H^{-1}
    Hinv_noise = np.linalg.inv(H).dot(noise.ravel()).reshape((M,N))

    # assemble noisey measurement
    s = H.dot(i2D.ravel()).reshape((M,N)) + noise

    # let the measurement saturate for low values
    # (this makes plots look better and is semi-physical?)
    s[s<0] = 0.0
    recovered_image = np.linalg.inv(H).dot(s.ravel()).reshape((M,N))

    # Display the images
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    f = plt.figure(figsize=(10,2.5))
    gs = GridSpec(1,4)
    ax0 = f.add_subplot(gs[0,0])
    ax1 = f.add_subplot(gs[0,1])
    ax2 = f.add_subplot(gs[0,2])
    ax3 = f.add_subplot(gs[0,3])

    # hide tick labels
    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')

    # Plot the "actual" image
    im0 = ax0.imshow(i2D, interpolation='nearest', cmap='gray')
    ax0.set_title('$\\vec{i}$', fontsize=24)

    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("bottom", size="7.5%", pad=0.05)
    cbar0 = f.colorbar(im0, orientation="horizontal", ax=ax0, cax=cax0)

    # Plot the noise
    im1 = ax1.imshow(Hinv_noise, interpolation='nearest', cmap='seismic')
    im1.set_clim([-imax, imax])
    ax1.set_title('$H^{-1}\\vec{w}$', fontsize=24)
    ax1.text(-15,M/2+2,'+', fontsize=24)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("bottom", size="7.5%", pad=0.05)
    cbar1 = f.colorbar(im1, orientation="horizontal", ax=ax1, cax=cax1)

    # Plot the noise overlayed on the image
    overlay = np.zeros((M,N,4)) # 3 color channels and an alpha channel
    overlay[:,:,0] = Hinv_noise > 0 # red
    overlay[:,:,2] = Hinv_noise < 0 # blue
    overlay[:,:,3] = np.abs(Hinv_noise) / imax #alpha

    im2 = ax2.imshow(i2D, interpolation='nearest', cmap='gray')
    im2_overlay = ax2.imshow(overlay)
    ax2.set_title('$\\vec{i} + H^{-1}\\vec{w}$', fontsize=24)
    ax2.text(-13,M/2,'=', fontsize=24)

    # Plot the "reconstructed" image
    im3 = ax3.imshow(recovered_image, interpolation='nearest', cmap='gray')
    ax3.set_title('$H^{-1}\\vec{s}$', fontsize=24)
    ax3.text(-13,M/2,'=', fontsize=24)

    if(title is not None):
        title = f.suptitle(title, fontsize=24)
        title.set_y(1.15)

    plt.tight_layout()
    plt.show()

# Inputs
#  `H`: Mask matrix
#  `matrixName`: Name of mask matrix (for image title)
#  `s`: Sensor reading column vector
#  `X`: Image width
#  `Y`: Image height
def idealReconstruction(H, matrixName, s, X = 32, Y = 32, realImaging = False):
    i = np.dot(np.linalg.inv(H), s)

    if realImaging:
        i = noiseMassage(i, H)

    # Reshape the column vector `i` to display it as a 2D image
    i2D = np.reshape(i, (Y, X))
    # We're going to exclude the top row and left-most column from display
    #plt.imshow(i2D[1:, 1:], cmap = 'gray', interpolation = 'nearest')
    plt.imshow(i2D, cmap = 'gray', interpolation = 'nearest')
    plt.title('Reconstructed Image, Using %s' % matrixName)
    plt.show()

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
	newIm.save(saveName)									#Saving image in CWD with name saveName, for eg., 'example.png'
	
# Inputs
#  `H1`: First mask matrix to analyze
#  `matrixName1`: Name of the first matrix for the plot title
#  `H2`: Second mask matrix to analyze
#  `matrixName2`: Name of the second matrix for the plot title
def eigenAnalysisComparison(H1, matrixName1, H2, matrixName2):

    # Calculate the eigenvalues of `H1` and `H2`
    eigen1, x1 = np.linalg.eig(H1)
    eigen2, x2 = np.linalg.eig(H2)

    # Get the absolute values (magnitudes) of the eigenvalues
    absEigenH1 = np.absolute(eigen1)
    absEigenH2 = np.absolute(eigen2)

    plt.figure(figsize = (5, 4))
    # Make a 100-bin histogram of |eigenvalues|
    # Plot in log scale to see what's going on

    #subplot 1
    ax1 = plt.subplot(211)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.hist(absEigenH1, 100)
    plt.ylabel('Amount in Each Bin')
    plt.title("Eigenvalues of %s" % matrixName1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    #subplot 2
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)  # share x and y axis scale
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.hist(absEigenH2, 100)
    plt.title("Eigenvalues of %s" % matrixName2)

    plt.ylabel('Amount in Each Bin')
    plt.xlabel('|Eigenvalue| Bins')
    plt.show()
