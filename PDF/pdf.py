import numpy as np
import cv2 as cv
import matplotlib 
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import colorsys
from matplotlib import ticker
from scipy.optimize import curve_fit
import cmcrameri.cm as cmc




import imageio
import tifffile as tif
from skimage import io
#plt.rcParams['text.usetex'] = True
#plt.style.use(['science','nature'])


def read_tiff(path):
	hyperstack_path = path
	hyperstack = io.imread(hyperstack_path,  plugin="tifffile")
	return hyperstack

if __name__ == "__main__":

	########Change image Name and threshold const for each experiment################
	INPUT_FILE    = 'NPM1#3-Experiment1-30z-sections-160ms-1f per min.czi #2-1'
	original_stack= read_tiff(INPUT_FILE + '.tif')
	print(original_stack.shape)
	time 	      = original_stack.shape[0]
	stack         = original_stack.shape[1]
	tzcyx_stack   =np.zeros((time, stack, 3, original_stack.shape[2],original_stack.shape[3]), np.uint8)
	time_array    =np.zeros((time,256), int)
	plt.figure(0)
	color = plt.cm.coolwarm(np.linspace(0,1,time))
	m_list = list()
	save_hist = np.zeros((256, time)) 
	plot_frame = 1
	melt_frame = 25
	for i in range(time):
		img= original_stack[i, :, :, :]
		img_flatten = img.flatten()
		hist, bins = np.histogram(img_flatten, bins=256, density=True)
		bin_centers = 0.5 * (bins[1:] + bins[:-1])
		if (i % plot_frame == 0) and i <= melt_frame:
			logX = np.log(range(80,160,1))
			logY = np.log(hist[80:160,])
			m, c = np.polyfit(logX, logY, 1) # fit log(y) = m*log(x) + c
			m_list.append(m)
			plt.loglog(bin_centers, hist, 'o', ms=0.5, color=color[i], label='t = %s mins' % i)
		elif (i % plot_frame == 0) and i > melt_frame:
			logX = np.log(range(70,110,1))
			logY = np.log(hist[70:110,])
			m, c = np.polyfit(logX, logY, 1) # fit log(y) = m*log(x) + c
			m_list.append(m)
			plt.loglog(bin_centers, hist, 'o', ms=0.5, color=color[i], label='t = %s mins' % i)
		
	
	
	#plt.rcParams.update({'font.size': 22})
	plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
	plt.gca().xaxis.set_minor_formatter(ticker.ScalarFormatter(useOffset=False))
	minorLocator   = ticker.MultipleLocator(10)
	majorLocator   = ticker.MultipleLocator(100)
	plt.gca().xaxis.set_minor_locator(minorLocator)
	plt.gca().xaxis.set_major_locator(majorLocator)
	plt.setp(plt.gca().get_xminorticklabels(), visible=False)
	plt.minorticks_off()

	plt.xlim([30, 252])
	plt.ylim([1e-7, 1.5e-3])
	# Normalizer
	norm = matplotlib.colors.Normalize(vmin=-25, vmax=11)
	# creating ScalarMappable
	sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
	sm.set_array([])
	cbar = plt.colorbar(sm, ticks=[-30, -20, -10, 0, 10, 20])
	cbar.minorticks_off()
	cbar.set_label('Time (min)')
	

	plt.xlabel('Intensity')
	plt.ylabel('PDF')
	time_series = range(0,time,1)
	diff = np.diff(m_list)
	min_index = np.argmin(diff, axis=0)
	print(min_index)
	x1=np.array(time_series[:min_index])
	y1=np.array(m_list[:min_index])
	x1 = x1[~np.isnan(y1)]
	y1 = y1[~np.isnan(y1)]
	m1, c1 = np.polyfit(x1, y1, 1)
	x2=np.array(time_series[(min_index-1):(min_index+2)])
	y2=np.array(m_list[(min_index-1):(min_index+2)])
	x2 = x2[~np.isnan(y2)]
	y2 = y2[~np.isnan(y2)]
	m2, c2 = np.polyfit(x2, y2, 1)
	plt.savefig('PDF.jpg', dpi=300)
	plt.savefig('PDF.svg', dpi=300)


	plt.figure(1)
	plt.plot(time_series, m_list, 'o', markersize=5)
	plt.plot(x1, m1 * x1 + c1, 'r')
	x2_fit = np.array(time_series[(min_index-2):(min_index+4)])
	plt.plot(x2_fit, m2 * x2_fit + c2, 'r')
	plt.xlabel('Time (min)')
	plt.ylabel(r"$\alpha$")
	plt.title('Slope1 = %.4f, Slope2 = %.4f' %(m1, m2))
	
	time_series = np.array([time_series])
	m_list = np.array([m_list])
	#save_data = np.concatenate([time_series.T, m_list.T], axis=1)
	#np.savetxt('Slope.csv', save_data, delimiter=",")
	#plt.savefig('Slope.jpg')

	plt.show()



			
	

    
