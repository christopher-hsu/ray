import numpy as np

def quaternion_from_euler(r, p, y):
	# Output order : w, x, y, z

	cr = np.cos(.5 * r)
	cp = np.cos(.5 * p)
	cy = np.cos(.5 * y)
	sr = np.sin(.5 * r)
	sp = np.sin(.5 * p)
	sy = np.sin(.5 * y)
	q = np.array([cy * cp * sr - sy * sp * cr,
					sy * cp * sr + cy * sp * cr,
					sy * cp * cr - cy * sp * sr,
					cy * cp * cr + sy * sp * sr])
	return q
