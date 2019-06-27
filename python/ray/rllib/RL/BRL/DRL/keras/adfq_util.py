import numpy as np
import tensorflow as tf
import pdb
def kl_divergence_loss(y_true, y_pred_mean, y_pred_rho, rhoTH, clipvalue = 1000.0):
	# y_true = |batch_size|*[true_means, true_rhos]
	# y_pred_mean = |batch_size|*[predicted_means]
	mean_pred = y_pred_mean
	sd_pred = tf.log(1+tf.exp(tf.maximum(rhoTH,y_pred_rho)))
	mean_true = y_true[:,0]
	sd_true = tf.log(1+tf.exp(tf.maximum(rhoTH,y_true[:,1])))
	loss = tf.reduce_mean(tf.contrib.distributions.kl_divergence(
						tf.distributions.Normal(loc=mean_true, scale=sd_true),
						tf.distributions.Normal(loc=mean_pred, scale=sd_pred)), name = 'loss')
	return loss
	#return tf.clip_by_value(loss, clip_value_min=-clipvalue, clip_value_max=clipvalue)
def maxGaussian_approx(n_means, n_vars, c_mean, c_var, rew, dis, isBatch=False):
	if isBatch:
		batch_size = len(n_means)
		c_mean = np.reshape(c_mean, (batch_size,1))
		c_var = np.reshape(c_var, (batch_size,1))
		rew = np.reshape(rew, (batch_size,1))
	target_mean = rew + dis*np.array(n_means, dtype=np.float32)
	target_var = dis*dis*np.array(n_vars, dtype = np.float32)

	bar_vars = 1.0/(1.0/c_var + 1.0/target_var)
	bar_means = bar_vars*(c_mean/c_var + target_mean/target_var)

	c_tmp = -(c_mean-target_mean)**2/2.0/(c_var + target_var)
	c_min = np.amin(c_tmp, axis=int(isBatch), keepdims = isBatch)
	c_max = np.amax(c_tmp, axis=int(isBatch), keepdims = isBatch)
	B = (c_max - c_min < 50.0).astype(np.float32) if isBatch else float(c_max-c_min < 50.0)
	weights = np.exp(c_tmp - c_min*B - c_max*(1-B))/np.sqrt(c_var+target_var)
	Z = np.sum(weights, axis = int(isBatch))
	mean_new = np.sum(weights*bar_means, axis = int(isBatch))/Z
	var_new = np.sum(weights*bar_vars, axis=int(isBatch))/Z
	if np.isnan(mean_new).any() or np.isnan(var_new).any():
		print("NaN value is detected!!!")
		pdb.set_trace()
	return mean_new, var_new 