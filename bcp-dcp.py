import cv2
import numpy as np
import matplotlib.pyplot as plt


def estimatedarkchannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark


def estimatebrightchannel(im,sz):
    b,g,r = cv2.split(im)
    bc = cv2.max(cv2.max(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    bright = cv2.dilate(bc,kernel)
    return bright


def guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q


def get_atmosphere(I, brightch, p):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)
    flatbright = brightch.ravel() # make array flatten
    
    searchidx = (-flatbright).argsort()[:int(M*N*p)]  # find top M * N * p indexes. argsort() returns sorted (ascending) index.
    
    # return the mean intensity for each channel
    A = np.mean(flatI.take(searchidx, axis=0),dtype=np.float64, axis=0) # 'take' get value from index.
    
    return A


def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    
    init_t = (brightch-A_c)/(1.-A_c) # original
    
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # min-max normalization.


def correct_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im3 = np.empty(I.shape, I.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = I[:,:,ind]/A[ind]

    dark_t = 1 - omega*estimatedarkchannel(im3, w)
    # dark_t = (dark_t - np.min(dark_t))/(np.max(dark_t) - np.min(dark_t))
    
    corrected_t = init_t
    diffch = brightch - darkch
    
    diff_flatten = diffch.ravel()
    indices = np.where(diff_flatten<alpha)
    
    mask = np.zeros(diff_flatten.shape)
    mask[indices] = 1
    mask_2d = mask.reshape(diffch.shape)

    inv_mask_2d = 1 - mask_2d
    
    corrected_t = dark_t*init_t*mask_2d + init_t*inv_mask_2d   

    return np.abs(corrected_t)


def get_final_image(I, A, corrected_t, tmin):
    corrected_t_broadcasted = np.broadcast_to(corrected_t[:,:,None], (corrected_t.shape[0], corrected_t.shape[1], 3))
    J = (I-A)/(np.where(corrected_t_broadcasted < tmin, tmin, corrected_t_broadcasted)) + A
    #J = (I-A)/(np.where(corrected_t < tmin, tmin, corrected_t)) + A # this is used when corrected_t has 3 channels
    # print('J between [%.4f, %.4f]' % (J.min(), J.max()))
    
    return (J - np.min(J))/(np.max(J) - np.min(J)) # min-max normalization.


def full_brightness_enhance(im, w):
    tmin=0.1   # minimum value for t to make J image
    # w=3       # window size, which determine the corseness of prior images
    alpha=0.4  # threshold for transmission correction. range is 0.0 to 1.0. The bigger number makes darker image.
    omega=0.75 # this is for dark channel prior. change this parameter to arrange dark_t's range. 0.0 to 1.0. bigger is brighter
    p=0.1      # percentage to consider for atmosphere. 0.0 to 1.0
    eps=1e-3   # for J image
    
    # Pre-process
    I = np.asarray(im, dtype=np.float64)
    I = I[:,:,:3]/255
    
    # Get dark/bright channels
    Idark_ch = estimatedarkchannel(I, w)
    Ibright_ch = estimatebrightchannel(I, w)
    
    # Get atmosphere
    # white = np.full_like(Idark, L - 1)
    At = get_atmosphere(I, Ibright_ch, p)
    
    # Get initial transmission
    init_tr = get_initial_transmission(At, Ibright_ch)
    
    # Correct transmission
    corrected_tr = correct_transmission(I, At, Idark_ch, Ibright_ch, init_tr, alpha, omega, w)
    
    # Refine transmission
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    refined_tr = guidedfilter(gray, corrected_tr, w, eps)
    
    # Produce final result
    enhanced_image = get_final_image(I, At, refined_tr, tmin)
    
    return enhanced_image


if __name__ == "__main__":


	# Load image
	img_name = "val00-16-2-2-FRONT_LEFT"
	src = f"images/{img_name}.jpg"

	im = cv2.imread(src)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	plt.figure(figsize=(12,10))
	plt.imshow(im)


	# Configuration
	tmin=0.1   # minimum value for t to make J image

	# w=15       # window size, which determine the corseness of prior images
	w=3        # window size, which determine the corseness of prior images

	alpha=0.4  # threshold for transmission correction. range is 0.0 to 1.0. The bigger number makes darker image.
	omega=0.75 # this is for dark channel prior. change this parameter to arrange dark_t's range. 0.0 to 1.0. bigger is brighter
	p=0.1      # percentage to consider for atmosphere. 0.0 to 1.0
	eps=1e-3   # for J image

	I = np.asarray(im, dtype=np.float64) # Convert the input to an array.
	I = I[:,:,:3]/255 # stackoverflow.com/questions/44955656/how-to-convert-rgb-pil-image-to-numpy-array-with-3-channels


	# Get Dark/Bright Channels
	Idark_ch = estimatedarkchannel(I, w)
	Ibright_ch = estimatebrightchannel(I, w)


	# Get atmosphere
	At = get_atmosphere(I, Ibright_ch, p)


	# Get initial transmission and enhanced image
	init_tr = get_initial_transmission(At, Ibright_ch)


	# Correct transmission
	corrected_tr = correct_transmission(I, At, Idark_ch, Ibright_ch, init_tr, alpha, omega, w)


	# Guided filter
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	gray = np.float64(gray)/255
	refined_tr = guidedfilter(gray,corrected_tr,w,eps)


	# Restore image
	refined = get_final_image(I, At, refined_tr, tmin)


	# Write out images
	output = refined*255
	output = np.array(output, dtype=np.uint8)
	cv2.imwrite(f"images/{img_name}_enhanced.jpg", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


	plt.figure(figsize=(12,10))
	plt.imshow(refined)
	plt.show()
