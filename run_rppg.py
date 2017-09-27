import os
import numpy as np
import scipy.signal as signal
from S2R import *

folder_dir = "/Users/sherry/data_temp/"
folders = os.listdir(folder_dir)
for folder in folders:
	if folder[0] == ".": continue
	p_blocked = S2R(folder_dir+folder)
	f, pxx = signal.periodogram(p_blocked, 50)
	pxx = np.array([10*math.log10(ele) for ele in pxx])
	peaks = peakdet(pxx, 0.5, x = None)[0]
	sort_ind = np.argsort(peaks[:,1])[::-1]
	peaks = peaks[sort_ind,:]
	
	peak_ratio = abs(np.prod(peaks[1:5,1])/pow(peaks[0,1],4))	
	print(folder+": "+str(peaks[0,1])+" "+str(peak_ratio))