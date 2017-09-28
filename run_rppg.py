import os
import numpy as np
import scipy.signal as signal
from S2R import *

folder_dirs = ["/home/zhenheng/datasets/3dmad/frames/Data_01/fcrop/", \
				"/home/zhenheng/datasets/3dmad/frames/Data_03/fcrop/"]
wf = open("./3dmad_peak_ratio.txt","w")
for folder_dir in folder_dirs:
	folders = os.listdir(folder_dir)
	for i,folder in enumerate(folders):
		# if i != 4: continue
		if folder[0] == ".": continue
		p_blocked = S2R(folder_dir+folder)
		f, pxx = signal.periodogram(p_blocked, 50)
		pxx = np.array([10*math.log10(ele+pow(10,-10)) for ele in pxx])
		peaks = peakdet(pxx, 0.5, x = None)[0]
		sort_ind = np.argsort(peaks[:,1])[::-1]
		peaks = peaks[sort_ind,:]
		
		peak_ratio = abs(np.prod(peaks[1:5,1])/pow(peaks[0,1],4))
		print(folder+": "+str(peaks[0,1])+" "+str(peak_ratio))
		wf.write(folder+": "+str(peaks[0,1])+" "+str(peak_ratio)+"\n")