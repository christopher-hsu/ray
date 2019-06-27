import numpy as np
import os 
import pdb

fnames = ["reward_v2_minimaze.npy", "Q_minimaze.npy", "count_minimaze.npy",	"reward_minimaze.npy",	
"Q_v2_minimaze.npy"	,"count_v2_minimaze.npy"]
for f in fnames:
	# Qlearning eg, Qlearning btz, ADFQ eg, ADFQ BS
	data_adfq = np.load(os.path.join("results/Discrete/092418_moore_act", f))
	sh = data_adfq.shape 
	# Qlearning eg, Qlearning btz, ADFQ-Numeric eg, ADFQ-V2 eg, ADFQ-Numeric BS, ADFQ-V2 BS
	data_rest = np.load(os.path.join("results/Discrete/092418_moore_act_2",f))
	# KTD
	data_ktd = np.load(os.path.join("results/Discrete/minimaze_s_30K_init3/minimaze_s_30K_init3_ktd",f))
	data_new = np.concatenate((data_adfq[:-1,:,:], data_rest[(2,3),:,:]))
	data_new = np.concatenate((data_new, np.reshape(data_adfq[-1,:,:],(1,sh[1], sh[2]))))
	data_new = np.concatenate((data_new, data_rest[(4,5),:,:]))
	data_new = np.concatenate((data_new, data_ktd[(-2,-1),:,:]))

	os.system("mv "+os.path.join("results/Discrete/092418_moore_act", f)+" "+os.path.join("results/Discrete/092418_moore_act", f+".part"))
	pdb.set_trace()
	np.save( os.path.join("results/Discrete/092418_moore_act",f), data_new)