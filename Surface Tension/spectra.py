from scipy import signal
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import polarTransform
from scipy.ndimage import rotate
from scipy.special import kn
from scipy.optimize import curve_fit
from scipy import signal
from scipy.special import lpmv
from decimal import Decimal, getcontext
getcontext().prec = 15

import inspect
from lmfit import Model
import math

import imageio
import tifffile as tif
from skimage import io

from PIL import Image
from pandas import DataFrame as df
import h5py

from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

import matplotlib as mpl
plt.style.use(['science','nature'])

def read_tiff(path):
	hyperstack_path = path
	hyperstack = io.imread(hyperstack_path,  plugin="tifffile")
	return hyperstack
def angle_trunc(a):
    while a < 0.0:
        a += np.pi * 2
    return a
def dot_python(a, b, start, stop, delay):
    """Return dot product of two sequences in range."""
    sum = 0
    for n in range(start, stop):
        sum += a[n + delay] * b[n]
    return sum


def correlate_python(a, b):
    """Return linear correlation of two sequences."""
    size = len(a)

    c = [0] * size  # allocate output array/list

    for index in range(size):
        delay = index - size // 2
        if delay < 0:
            c[index] = dot_python(a, b, -delay, size, delay)
        else:
            c[index] = dot_python(a, b, 0, size-delay, delay)

    return c

def correlate1D(a, b):
    """Return circular correlation of two arrays using DFT."""
    size = np.size(a)
    print(size)
    a_flucc = a - np.mean(a)
    b_flucc = b - np.mean(b)
    print(np.mean(a))
    # forward DFT
    af = np.fft.fft(a_flucc)
    bf = np.fft.fft(b_flucc)
    # reverse DFT 
    c = (np.fft.fftshift(np.fft.ifft(af*bf.conj()))).real
    # reverse DFT and normalized
    #c = (np.fft.fftshift(np.fft.ifft(af*bf.conj()))).real/(np.sqrt(np.sum(np.power(a_flucc,2)))*np.sqrt(np.sum(np.power(b_flucc,2))))
    # positive delays only
    #cutoff = size//2
    #c = c[cutoff:]
    return c
def get_segment(img, b_thr, f_thr):
	#Background of image (nucleoplasm)
	kernel = np.ones((7,7),np.uint8)
	ret1,th1 = cv.threshold(img,b_thr, 255,cv.THRESH_BINARY)
	#th1 = cv.dilate(th1,kernel,iterations = 1)

	#Foreground of image (nucleolus)
	ret2,th2 = cv.threshold(img,f_thr,255,cv.THRESH_BINARY)
	#th2 = cv.dilate(th2,kernel,iterations = 1)
	return th1, th2
def get_mean_roi(img, cx , cy):
	# get top left part of ROI of square region
	w=40
	h=40
	xt = int(cx - w/2)
	yt = int(cy - h/2)

	# get bottom right part of ROI of square
	xb = xt + w
	yb = yt + h
	ROI = img[yt:yb, xt:xb]
	roi_avg_intensity = np.mean(ROI)
	return roi_avg_intensity
def get_center(img):
	#Gaussian Blur Image
	img_blur = cv.GaussianBlur(img,(9,9),0)
	# Applying multi-Otsu threshold for the default value, generating n classes
	thresholds = threshold_multiotsu(img_blur, classes=4)
	img1, img2 = get_segment(img_blur, thresholds[0], thresholds[1])
	contours2, hierarchy2 = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	max_contour = max(contours2, key = cv.contourArea)
	B = cv.moments(max_contour)
	cbX = int(B["m10"] / B["m00"])
	cbY = int(B["m01"] / B["m00"])
	center=(cbX,cbY)
	return center
def polarTocartesian (XY, center, img_resol, finalRadius, polarImage):
	x = center[0] + (XY[:,0]*np.cos(XY[:,1]*2*np.pi/polarImage.shape[1]))*(finalRadius)/polarImage.shape[1]
	y = center[1] + (XY[:,0]*np.sin(XY[:,1]*2*np.pi/polarImage.shape[1]))*(finalRadius)/polarImage.shape[1]
	return x, y
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def find_contours(img, I_gas, I_liquid, center, finalRadius):
	polarImage, ptSettings = polarTransform.convertToPolarImage(img, center = center, initialRadius=0,
                                                        finalRadius=finalRadius, initialAngle=0,
                                                        finalAngle= 2*np.pi, radiusSize= 212, angleSize = 420, order=5)
	I_del = I_liq -I_gas
	contours = list() 
	for i in range(polarImage.shape[0]):
		I = np.sum(polarImage[i,:])
		h = (I-I_gas*polarImage.shape[1])/I_del
		contours.append([h, i])

	XY = np.array(contours)
	contours = np.array(contours).reshape((1,-1,2)).astype(np.int32)
	cv.polylines(polarImage, [contours], False, (255), thickness = 1)
	return XY, contours, polarImage, ptSettings
def equi_rad(XY, img_resol, finalRadius, polarImage):
	sum_ = 0
	theta_rad = XY[:,1]*2*np.pi/polarImage.shape[0]
	#print(polarImage.shape[0], XY.shape)
	delta_theta = 2*np.pi/polarImage.shape[0]
	for i in range(0,(XY.shape[0]-1)):
		sum_ += (XY[i,0]+XY[i+1,0])*delta_theta
	sum_ += (XY[0,0]+XY[-1,0])*delta_theta
	R= 1/(4*np.pi)*sum_
	polarImage = cv.line(polarImage, (int(R), 0), (int(R), (polarImage.shape[0]-1)), (255,0,0), 2)

	radius = XY[:,0] * (img_resol*finalRadius)/polarImage.shape[1]
	R0 = R * (img_resol*finalRadius)/polarImage.shape[1]
	return R0, theta_rad, radius, polarImage
def mode_amp(radius, n, l, R0):
	u_l=list()
	u_l_2= list()
	for i in l:
		ul = 0
		for m in range(n):
			ul += radius[m]*np.exp(-2*np.pi*1j*i*m/n)/(np.pi*R0*n)
		u_l_2.append(np.abs(ul)**2)
		u_l.append(np.abs(ul))
	return u_l, u_l_2
def an_bn(radius, theta_rad, n, l, R0):
	c_l= list()
	c_l_2 = list()
	delta_theta = 2*np.pi/n
	for i in l:
		a_l = 0
		b_l = 0
		for m in range(n-1):
			a_l += (radius[m]*np.cos(i*theta_rad[m])+radius[m+1]*np.cos(i*theta_rad[m+1]))*delta_theta/2
			b_l += (radius[m]*np.sin(i*theta_rad[m])+radius[m+1]*np.sin(i*theta_rad[m+1]))*delta_theta/2
		a_l += (radius[0]*np.cos(i*theta_rad[0])+radius[-1]*np.cos(i*theta_rad[-1]))*delta_theta/2
		a_l  = a_l/(np.pi*R0)
		b_l += (radius[0]*np.sin(i*theta_rad[0])+radius[-1]*np.sin(i*theta_rad[-1]))*delta_theta/2
		b_l  = b_l/(np.pi*R0)
		c_l.append(np.sqrt(a_l**2+b_l**2))
		c_l_2.append(a_l**2+b_l**2)
	return c_l, c_l_2
def fit_func(x, _lambda, kappa):
	global mean_R0
	kb = 1.380649e-23
	T  = 310
	L= 2*np.pi*mean_R0
	q_l = x/mean_R0
	#return (kb*T)/(2*_lambda*L)*((1/q_l)-(1/np.sqrt(_lambda/kappa + q_l**2)))
	return (kb*T)/(_lambda*np.pi*mean_R0**3)*((1/q_l)-(1/np.sqrt(_lambda/kappa + q_l**2)))
def fit_legendre(x, _sigma, kappa):
	global mean_R0
	global lmax
	kb = 1.380649e-23
	T  = 310
	v_q_2 =[]
	for q in x:
		v_q = 0
		for l in range(int(q),lmax):
			P_lq = Decimal(lpmv(q, l, 0))
			P_lq_sq = P_lq**2
			factorial_ = P_lq_sq * Decimal(math.factorial(l-q))/Decimal(math.factorial(l+q))
			#print(factorial_)
			factorial_ = float(factorial_)
			#print(factorial_)
			N_lq_sq = ((2*l+1)/(4*np.pi))* factorial_
			v_q += (kb*T/kappa) * N_lq_sq / ((l-1)*(l+2)*(l*(l+1)+_sigma*(mean_R0**2)/kappa))
			#v_q += Decimal(kb*T/(4*np.pi*kappa)*(2*l+1))*Decimal(math.factorial(l-q))/Decimal(math.factorial(l+q)) * Decimal(P_lq**2) /Decimal((l+2)*(l-1)*(l*(l+1)+(_sigma*(mean_R0**2)/kappa)))
			#v_q += (kb*T/kappa*(2*l+1)/(np.pi))* factorial_ / (l**2*(l+1)**2 - (2-(_sigma*(mean_R0**2)/kappa))*l*(l+1))
		#v_q = float(v_q)
		v_q_2.append(v_q)
	return v_q_2

#################main function#####################################

if __name__ == "__main__":

	########Change image Name and threshold const for each experiment################
	INPUT_FILE ='Cell1'
	original_stack= read_tiff(INPUT_FILE + '.tif')
	print(original_stack.shape)
	time_p = original_stack.shape[0]
	img= original_stack
	img_resol = 0.04159*1e-6 # 0.03119#(m/pixel)
	#I_gas =  0.001
	#I_liq = 22.5
	finalRadius = 120
	tyx_stack=np.zeros((time_p, original_stack.shape[1],original_stack.shape[2]), np.uint8)
	theta_contour=np.empty((420,time_p), dtype=float)
	radius_contour=np.empty((420,time_p), dtype=float)
	U_l = []
	U_l_2 = []
	C_l = []
	C_l_2 = []
	_R0 =[]
	_STD = []
	_excess=[]
	fig, ax = plt.subplots(figsize=(3,2),subplot_kw={'projection': 'polar'})

	for i in range(time_p):
		img = original_stack[i, :, :]
		#img=img*3
		#img=img.astype('uint8')
		center = get_center(img)
		img_blur = cv.medianBlur(img, 7)
		#img_blur = cv.GaussianBlur(img,(7,7),0)
		I_gas = get_mean_roi(img_blur, 220, 150)
		I_liq = get_mean_roi(img_blur, center[0], center[1])
		I_liq =I_liq*0.95
		#img =cv.GaussianBlur(img,(3,3),0)
		XY, contours, polarImage, ptSettings = find_contours(img_blur, I_gas, I_liq, center, finalRadius)
		#theta_contour[:,i] = np.array([XY[:,1]*2*np.pi/polarImage.shape[0]])
		#radius_contour[:,i] = np.array(XY[:,0]*(img_resol*finalRadius)/polarImage.shape[1])
		#ax.plot(XY[:,1]*2*np.pi/polarImage.shape[0], XY[:,0]*(img_resol*finalRadius)/polarImage.shape[1],'-',c='0.65', linewidth=0.5)
		#plt.show()
		_XY_ = polarTransform.getCartesianPointsImage(XY, ptSettings)
		X_, Y_ = polarTocartesian(XY, center, img_resol, finalRadius, polarImage)
		area = PolyArea(_XY_[:,0]* img_resol, _XY_[:,1]* img_resol)
	
		
		# This is for checking contour
		polarImage1, ptSettings1 = polarTransform.convertToPolarImage(img, center = center, initialRadius=0,
                                                        finalRadius=finalRadius, initialAngle=0,
                                                        finalAngle= 2*np.pi, radiusSize= 212, angleSize = 420, order=5)
		cv.polylines(polarImage1, [contours], False, (255), thickness = 1)
		#cv.polylines(img, np.int32([_XY_]), True, (255), thickness = 1)
		#save_img = rotate(polarImage1, 90)
		#plt.imshow(save_img)
		#plt.show()
		#plt.savefig('interface_control.jpg', dpi=300)
		

		
		
		R0, theta_rad, radius, polarImage = equi_rad(XY, img_resol, finalRadius, polarImage)
		ex_area = (area - np.pi*R0**2)/(np.pi*R0**2)
		print(ex_area)
		_excess.append(ex_area)

		ax.plot(theta_rad, radius,'-',c='0.65', linewidth=0.5)
		theta_contour[:,i] = theta_rad
		radius_contour[:,i] = radius
		_R0.append(R0)
		_STD.append(np.std(np.array([radius])))
		cartesianImage = ptSettings.convertToCartesianImage(polarImage1)
		tyx_stack[i,:,:] = cartesianImage
		#plt.imshow(cartesianImage)
		#plt.show()
		'''
		plt.figure(1)
		plt.plot(theta_rad, radius)
		plt.xlabel("$\\theta$")
		plt.ylabel('r ($\mu$m)')
		plt.ylim(0, 4)
		plt.savefig('hieght_vs_angle.jpg', dpi=300)
		'''
		
		n= polarImage.shape[0]
		l= np.arange(2,69,1)
		#u_l, u_l_2 = mode_amp(radius, n, l, R0)
		c_l, c_l_2 = an_bn(radius, theta_rad, n, l, R0)
		#U_l.append(u_l)
		#U_l_2.append(u_l_2)
		C_l.append(c_l)
		C_l_2.append(c_l_2)
	mean_theta = np.mean(theta_contour,axis=1)
	mean_theta = np.append(mean_theta, mean_theta[:1], axis = 0)
	mean_radius = np.mean(radius_contour,axis=1)
	mean_radius = np.append(mean_radius, mean_radius[:1], axis = 0)

	ax.plot(mean_theta, mean_radius,'r-')
	fig.savefig('polar_contour.jpg', dpi=300)
	
	tif.imwrite(INPUT_FILE + '_curve.tif', tyx_stack, imagej=True)
	
	plt.figure(2, figsize=(3,2))
	plt.errorbar(np.arange(time_p)* 0.05814, _R0, _STD, marker='o', linestyle ='',  mec='black', color='black',
             ecolor='black', elinewidth=0.1, markersize=3, capsize= 0.1, markeredgewidth=0.5, mfc='None')
	#plt.ylim([0, 5e-6])
	plt.xlabel('$Time (s)$')
	plt.ylabel('$R (m)$')
	mean_excess = np.mean(_excess, axis=0)
	mean_R0 = np.mean(_R0, axis=0)
	STD_R0 = np.std(np.array([_R0]))
	save_param = np.array([mean_R0, mean_excess])
	np.savetxt('save_param.txt', save_param, delimiter=",")
	print(mean_R0, mean_excess)
	plt.axhline(y = mean_R0, color = 'r', linestyle = '-')
	plt.axhline(y = mean_R0+STD_R0, color = 'b', linestyle = '--')
	plt.axhline(y = mean_R0-STD_R0, color = 'b', linestyle = '--')
	plt.savefig('R0.jpg', dpi=300)


	variance = (np.mean(C_l_2,axis=0)-(np.mean(C_l, axis=0))**2) 
	mean_C_l_2 = np.mean(C_l_2,axis=0)
	l3 = variance*l**3 
	wave_v = l/mean_R0
	plt.figure(5)
	plt.plot(wave_v, variance*0.5*np.pi*mean_R0**3, 'o')
	plt.yscale('log')
	#plt.xscale('log')
	plt.figure(3,figsize=(3,2))
	plt.plot(l, variance, 'o-')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('$q$')
	plt.ylabel('$\\langle|u_{q}|^2\\rangle$')
	plt.savefig('spectra.jpg', dpi=300)
	plt.figure(4,figsize=(3,2))
	plt.plot(l, l3, 'o-')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('$q$')
	plt.ylabel('$\\langle|u_{q}|^2\\rangle \\times q^3$')
	plt.savefig('spectra_q3.jpg', dpi=300)
	save_data = np.concatenate([np.array([l]).T, np.array([variance]).T, np.array([l3]).T], axis=1)
	#save_data_84 = np.concatenate([np.array([l]).T, np.array([l3]).T], axis=1)
	np.savetxt('save_data.txt', save_data, delimiter=",")
	#np.savetxt('save_data_l3_84.txt', save_data_84, delimiter=",")
	plt.show()

	
	
	