import os
import numpy as np
import scipy.signal as signal
from S2R import *

folder_dir = "/home/zhenheng/datasets/3dmad/frames/Data_01/"
folders = os.listdir(folder_dir)
for folder in folders:
	p_blocked = S2R(folder_dir+folder)
	peakind = signal.find_peaks_cwt(p_blocked, np.arange(1,10))
	peaks = sorted(p_blocked[peakind])[::-1]
	print(folder+": "+str(peaks[0]+" "+str(peaks[0]/peaks[1])))