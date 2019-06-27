import matplotlib.pyplot as plt 
import numpy as np 
import brl_util as util 
import brl
from scipy.stats import norm
import pdb

def true_posterior_update():
	c_mean = 0.0
	c_var = 0.5 
	n_means = np.array([-2.0, 4.5])
	n_vars = np.array([2.0, 0.5])
	gamma = 0.9
	r = 0.0
	MARKERSIZE = 8
	LINEWIDTH = 2.5

	all_ax = []
	#1 IVW
	x = np.arange(-6, 6.5, 0.08)
	ivw_v = 1./(1./c_var + 1./(gamma*gamma*n_vars))
	ivw_m = ivw_v*(c_mean/c_var + (r+gamma*n_means)/(gamma*gamma*n_vars))
	f_ivw0, ax_ivw0 = plt.subplots()
	ax_ivw0.plot(x, norm.pdf(x, c_mean, np.sqrt(c_var)), 'g', marker='+', markersize=MARKERSIZE)
	ax_ivw0.plot(x, norm.pdf(x, r+gamma*n_means[0], gamma*np.sqrt(n_vars[0])), 'b', marker='x', markersize=MARKERSIZE)
	ax_ivw0.plot(x, norm.pdf(x, ivw_m[0], np.sqrt(ivw_v[0])), 'r', marker='^', markersize=MARKERSIZE)
	ax_ivw0.set_ylim((0.0, 0.9))
	
	f_ivw1, ax_ivw1 = plt.subplots()
	ax_ivw1.plot(x, norm.pdf(x, c_mean, np.sqrt(c_var)), 'g', marker='+', markersize=MARKERSIZE)
	ax_ivw1.plot(x, norm.pdf(x, r+gamma*n_means[1], gamma*np.sqrt(n_vars[1])), 'b', marker='x', markersize=MARKERSIZE)
	ax_ivw1.plot(x, norm.pdf(x, ivw_m[1], np.sqrt(ivw_v[1])), 'r', marker='^', markersize=MARKERSIZE)
	ax_ivw1.set_ylim((0.0, 0.9))
	all_ax.append(ax_ivw0)
	all_ax.append(ax_ivw1)	

	#2 Weight
	td_err = r+gamma*n_means-c_mean
	td_err_var = c_var + gamma*gamma*n_vars
	xc = np.arange(-5, 5, 0.01)
	f_c0, ax_c0 = plt.subplots()
	ax_c0.plot(xc, norm.pdf(xc, 0.0, np.sqrt(td_err_var[0])), linewidth=LINEWIDTH)
	ax_c0.plot(td_err[0], norm.pdf(td_err[0], 0.0, np.sqrt(td_err_var[0])), 'ro', markersize=15)
	ax_c0.set_ylim((0.0, 0.45))

	f_c1, ax_c1 = plt.subplots()
	ax_c1.plot(xc, norm.pdf(xc, 0.0, np.sqrt(td_err_var[1])), linewidth=LINEWIDTH)
	ax_c1.plot(td_err[1], norm.pdf(td_err[1], 0.0, np.sqrt(td_err_var[1])), 'ro', markersize=15)
	ax_c1.set_ylim((0.0, 0.45))
	all_ax.append(ax_c0)
	all_ax.append(ax_c1)

	#3 Prod CDFs
	xcdf = np.arange(-6, 8, 0.05)
	xcdf2 = np.arange(-6, 8, 0.5)
	f_cdf0, ax_cdf0 = plt.subplots()
	ax_cdf0.plot(xcdf, norm.cdf(xcdf, r+gamma*n_means[0], gamma*np.sqrt(n_vars[0])), linewidth=LINEWIDTH)
	ax_cdf0.plot(xcdf, norm.cdf(xcdf, r+gamma*n_means[1], gamma*np.sqrt(n_vars[1])), linewidth=LINEWIDTH)
	ax_cdf0.plot(xcdf2, norm.cdf(xcdf2, r+gamma*n_means[1], gamma*np.sqrt(n_vars[1])), 'ro')

	f_cdf1, ax_cdf1 = plt.subplots()
	ax_cdf1.plot(xcdf, norm.cdf(xcdf, r+gamma*n_means[0], gamma*np.sqrt(n_vars[0])), linewidth=LINEWIDTH)
	ax_cdf1.plot(xcdf, norm.cdf(xcdf, r+gamma*n_means[1], gamma*np.sqrt(n_vars[1])), linewidth=LINEWIDTH)
	ax_cdf1.plot(xcdf2, norm.cdf(xcdf2, r+gamma*n_means[0], gamma*np.sqrt(n_vars[0])), 'ro')
	all_ax.append(ax_cdf0)
	all_ax.append(ax_cdf1)

	#4 
	x4 = np.arange(0, 4.0, 0.03)
	m_adf, v_adf = util.posterior_numeric_exact(n_means, n_vars, c_mean, c_var, r, gamma, 0)
	m_adfq, v_adfq, _ = util.posterior_adfq(n_means, n_vars, c_mean, c_var, r, gamma,0)

	true_post = norm.pdf(x4, ivw_m[0], np.sqrt(ivw_v[0]))*norm.cdf(x4, r+gamma*n_means[1], gamma*np.sqrt(n_vars[1]))\
		+ norm.pdf(x4, ivw_m[1], np.sqrt(ivw_v[1]))*norm.cdf(x4, r+gamma*n_means[0], gamma*np.sqrt(n_vars[0]))
	true_post = true_post/np.sum(true_post)/0.03
	f_dist, ax_dist = plt.subplots()
	ax_dist.plot(x4, norm.pdf(x4, m_adf, np.sqrt(v_adf)),'o')
	ax_dist.plot(x4, norm.pdf(x4, m_adfq, np.sqrt(v_adfq)),'^')
	ax_dist.plot(x4, true_post,'+')
	ax_dist.legend(['ADFQ-Numeric', 'ADFQ', 'True'])
	all_ax.append(ax_dist)

	for a in all_ax:
		a.grid()
		a.tick_params(axis='both',which='major',labelsize=12)

	plt.show()
	pdb.set_trace()


def plot_variance_update():
	mu_s_a = 2.
    var_s_a = 20.
    mu1 = 0
    mu2 = 1.
    var1 = 1.
    var2 = 2.
    gamma = 0.99
    r = 0.

    plt.figure(1)
    var_s_a_vec = np.linspace(0, 2.5, 100)
    plt.plot(var_s_a_vec, var_s_a_vec, '--', label='y=x')
    for mu2 in np.array([0., -10, -20, 10, 20, -100]):
        var_s_a_new_vec = var_curve(var_s_a_vec, mu_s_a, mu1, mu2, var1, var2, gamma, r)
        plt.plot(var_s_a_vec, var_s_a_new_vec, label=str(mu2-mu1))
    plt.legend(prop={'size':14})
    plt.xlabel(r"$(\sigma_{s, a}^{k})^2$", fontsize=18)#, fontname="Times New Roman")
    plt.ylabel(r"$(\sigma_{s, a}^{k+1})^2$", fontsize=18)#, fontname="Times New Roman")
    plt.savefig('mu_diff.png', bbox_inches='tight')

    # diff var2/var1
    mu_s_a = 2.
    var_s_a = 10.
    mu1 = 1.
    mu2 = 1.
    var1 = 1.
    var2 = 2.
    gamma = 0.9
    r = 0.

    plt.figure(2)
    var_s_a_vec = np.linspace(0, 2.5, 100)
    plt.plot(var_s_a_vec, var_s_a_vec, '--', label='y=x')

    for var2 in ([1., 0.1, 0.5, 2, 10, 100]):
        var_s_a_new_vec = var_curve(var_s_a_vec, mu_s_a, mu1, mu2, var1, var2, gamma, r)
        plt.plot(var_s_a_vec, var_s_a_new_vec, label=str(var2/var1))
    plt.legend(prop={'size':14})
    plt.xlabel(r"$(\sigma_{s, a}^{k})^2$", fontsize=18)#, fontname="Times New Roman")
    plt.ylabel(r"$(\sigma_{s, a}^{k+1})^2$", fontsize=18)#, fontname="Times New Roman")

    plt.savefig('var_ratio.png', bbox_inches='tight')
    # plt.plot(var_s_a_vec, var_s_a_new_vec, label='blah')
    plt.show()

def next_var_s_a(mu_s_a, var_s_a, mu1, mu2, var1, var2, gamma, r):
    var1_bar = 1.0 / ((1.0 / var_s_a) + (1.0 / (gamma*gamma*var1)))
    var2_bar = 1.0 / ((1.0 / var_s_a) + (1.0 / (gamma*gamma*var2)))

    mu1_bar = var1_bar * ((mu_s_a / var_s_a) + ((r+gamma*mu1) / (gamma*gamma*var1)))
    mu2_bar = var2_bar * ((mu_s_a / var_s_a) + ((r+gamma*mu2) / (gamma*gamma*var2)))

    td_error1 = r + gamma*mu1 - mu_s_a
    scale1 = np.sqrt(var_s_a + gamma*gamma*var1)
    c1 = scipy.stats.norm.pdf(td_error1, scale=scale1)

    td_error2 = r + gamma*mu2 - mu_s_a
    scale2 = np.sqrt(var_s_a + gamma*gamma*var2)
    c2 = scipy.stats.norm.pdf(td_error2, scale=scale2)

    td_error1 = r + gamma*mu2 - mu1_bar
    scale1 = np.sqrt(var1_bar + gamma*gamma*var2)
    small_phi1 = scipy.stats.norm.pdf(td_error1, scale=scale1)


    td_error2 = r + gamma*mu1 - mu2_bar
    scale2 = np.sqrt(var2_bar + gamma*gamma*var1)
    small_phi2 = scipy.stats.norm.pdf(td_error2, scale=scale2)


    big_phi1 = scipy.stats.norm.cdf(td_error1, scale=scale1)
    big_phi2 = scipy.stats.norm.cdf(td_error2, scale=scale2)


    Z = c1*big_phi1 + c2*big_phi2
    mean = (c1/Z) * (mu1_bar * big_phi1 - var1_bar*small_phi1) + \
           (c2/Z) * (mu2_bar * big_phi2 - var2_bar*small_phi2)

    x = (c1/Z) * ((mu1_bar**2 + var1_bar)*big_phi1 - 2*mu1_bar*var1_bar*small_phi1 - small_phi1 * (var1_bar**2 * (r+gamma*mu2-mu1_bar)) / (var1_bar + gamma*gamma*var2)) + \
        (c2/Z) * ((mu2_bar**2 + var2_bar)*big_phi2 - 2*mu2_bar*var2_bar*small_phi2 - small_phi2 * (var2_bar**2 * (r+gamma*mu1-mu2_bar)) / (var2_bar + gamma*gamma*var1)) - \
        mean**2

    return x


def var_curve(var_s_a_vec, mu_s_a, mu1, mu2, var1, var2, gamma, r):
    var_s_a_vec = np.linspace(0, 1, 100)
    var_s_a_new_vec = np.zeros(var_s_a_vec.shape)
    for idx, var_s_a in enumerate(var_s_a_vec):
        var_s_a_new_vec[idx] = next_var_s_a(mu_s_a, var_s_a, mu1, mu2, var1, var2, gamma, r)
    return var_s_a_new_vec



if __name__=="__main__":
	true_posterior_update()


	