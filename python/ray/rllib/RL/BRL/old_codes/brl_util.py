#import sys
#sys.path.insert(0,'/usr/local/lib/python2.7/site-packages')
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from scipy.misc import logsumexp
import pdb
from operator import itemgetter
from scipy.interpolate import spline
import copy
import time

COLORS = ['g','k','r','b','c','m','y','burlywood','chartreuse','0.8','0.6', '0.4', '0.2']
MARKER = ['-','--', '*-', '+-','1-','o-','x-','1','2','3']
T_chain = 5000
T_loop = 5000
T_grid5 = 15000
T_grid10 = 20000
T_minimaze = 20000
T_maze = 40000

EVAL_RUNS = 10
EVAL_NUM = 100
EVAL_STEPS = 50
EVAL_EPS = 0.0
REW_VAR = 0.00001
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def discrete_phi(state, action, dim, anum):
    phi = np.zeros(dim, dtype=np.float)
    phi[state*anum+action] = 1.0
    return phi

def img_preprocess(org_img):
    imgGray = cv2.cvtColor( org_img, cv2.COLOR_RGB2GRAY )
    resizedImg = cv2.resize(np.reshape(imgGray, org_img.shape[:-1]), (84, 110))
    cropped = resizedImg[18:102,:]
    cropped = cropped.astype(np.float32)
    cropped *= (1.0/255.0)
    return cropped
    
def rbf(state, action, dim, const=1.0):
    n = dim
    c1 = np.reshape(np.array([-np.pi/4.0, 0.0, np.pi/4.0]),(3,1)) # For inverted pendulum
    c2 = np.reshape(np.array([-1.0,0.0,1.0]), (1,3)) # For inverted pendulum
    #basis = 1/np.sqrt(np.exp((c1-state[0])**2)*np.exp((c2-state[1])**2))
    basis = np.exp(-0.5*(c1-state[0])**2)*np.exp(-0.5*(c2-state[1])**2)
    basis = np.append(basis.flatten(), const)
    phi = np.zeros(3*n, dtype=np.float32)
    phi[action*n:(action+1)*n] = basis

    return phi

def normalGamma(x1, x2, mu, l , a, b):
    const = np.sqrt(l/2/np.pi, dtype=np.float32)*(b**a)/gamma(a)
    exp_input = np.maximum(-10.0,-0.5*l*x2*(x1-mu)**2-b*x2)
    output = const*x2**(a-0.5)*np.exp(exp_input, dtype=np.float32)
    if(np.isnan(output)).any():
        pdb.set_trace()

    return output

def mean_max_graph(mu,test_var):
    d = np.arange(0.1,1.0,0.1)
    var = np.array([0.1,1.0,10.0,20.0,30.0,40.0,50.0,80.0,100.0])
    if not(test_var in var):
        raise ValueError('the input variance value does not exist')
    for (i,v) in enumerate(var):
        if test_var == v:
            idx = i

    r = len(var)
    c = len(d)
    d,var = np.meshgrid(d,var)
    mu_bar = d*(1+d)/(1+d**2)*mu
    var_bar = d**2/(1+d**2)*var

    v = np.sqrt(var*d**2 + var_bar)
    mean = np.zeros(d.shape)
    for i in range(r):
        for j in range(c):
            mean[i,j] = mu_bar[i,j] + var_bar[i,j]*norm.pdf(mu_bar[i,j],d[i,j]*mu, v[i,j])/norm.cdf(mu_bar[i,j],d[i,j]*mu,v[i,j])

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(d, var, mu*np.ones(d.shape), color='r', linewidth=0, antialiased=False)
    ax.plot_wireframe(d, var, mean)#, rstride=10, cstride=10)
    
    plt.figure(2)
    plt.plot(d[idx,:],mean[idx,:])
    plt.plot(d[idx,:],mu*np.ones((d[idx,:]).shape),'r')
    plt.title("For Mean="+str(mu)+" and Variance="+str(test_var))
    plt.xlabel("discount factor")
    plt.ylabel("Mean Value")
    plt.show()


def maxGaussian(means, sds):
    """
    INPUT:
        means: a numpy array of Gaussian mean values of (next state, action) pairs for all available actions.
        sds: a numpy array of Gaussian SD values of (next state,action) pairs for all available actions.
        
    obs:
        mean and variance of the distribution
    """
    num_interval = 500
    interval = 12.0*max(sds)/float(num_interval)
    x = np.arange(min(means-6.0*sds),max(means+6.0*sds),interval)
    eps = 1e-5*np.ones(x.shape) # 501X1
    max_p = np.zeros(x.shape) # 501X1
    cdfs = [np.maximum(eps,norm.cdf(x,means[i], sds[i])) for i in range(len(means))]
    for i in range(len(means)):
        max_p += norm.pdf(x,means[i],sds[i])/cdfs[i]
    max_p*=np.prod(np.array(cdfs),0)

    z = np.sum(max_p)*interval 
    max_p = max_p/z # Normalization
   
    max_mean = np.inner(x,max_p)*interval
    return max_mean,np.inner(x**2,max_p)*interval- max_mean**2 

def posterior_numeric(n_means, n_vars, c_mean, c_var, rew, dis, terminal, num_interval=2000, 
    width = 10.0, varTH = 1e-10, noise=0.0):
    # ADFQ-Numeric
    # Not for Batch
    #c_var = c_var + 0.01
    if terminal:
        new_var = 1.0/(1.0/c_var + 1.0/REW_VAR)
        new_sd = np.sqrt(new_var)
        new_mean = new_var*(c_mean/c_var + rew/REW_VAR)
        interval = width*new_sd/float(num_interval)
        x = np.arange(new_mean-0.5*width*new_sd,new_mean+0.5*width*new_sd, interval)
        return new_mean, new_var, (x, norm.pdf(x, new_mean, new_sd))
    target_means = rew + dis*np.array(n_means, dtype=np.float64)
    target_vars = dis*dis*(np.array(n_vars, dtype = np.float64) + noise)
    anum = len(n_means)
    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars) 

    add_vars = c_var+target_vars
    sd_range = np.sqrt(np.append(target_vars, bar_vars))
    mean_range = np.append(target_means, bar_means)
    x_max = max(mean_range+0.5*width*sd_range)
    x_min = min(mean_range-0.5*width*sd_range)
    interval = (x_max-x_min)/float(num_interval)
    x = np.arange(x_min,x_max, interval)
    x = np.append(x, x[-1]+interval)

    #mean_range = np.concatenate((target_means,[c_mean]))
    #sd_range = np.sqrt(np.concatenate((dis*dis*n_vars, [c_var])))
    #interval = (width*max(sd_range)+10.0)/float(num_interval)
    #x = np.arange(min(mean_range-0.5*width*sd_range-5.0), max(mean_range+0.5*width*sd_range+5.0), interval)
    cdfs = np.array([norm.cdf(x, target_means[i], np.sqrt(target_vars[i])) for i in range(anum)])
    nonzero_ids = []
    for a_cdf in cdfs:
        if a_cdf[0]>0.0:
            nonzero_ids.append(0)
        else:
            for (i,v) in enumerate(a_cdf):
                if v > 0.0:
                    nonzero_ids.append(i)
                    break
    if len(nonzero_ids) != anum:
        print("CDF peak is outside of the range")
        pdb.set_trace()
    log_probs = []
    min_id = len(x) # To find the maximum length non-zero probability vector over all actions.
    log_max_prob = -10**100
    for b in range(anum):
        min_id = min(min_id, nonzero_ids[b])
        idx = max([nonzero_ids[c] for c in range(anum) if c!=b]) # For the product of CDF part, valid id should consider all actions except b
        tmp = -np.log(2*np.pi)-0.5*np.log(add_vars[b])-0.5*np.log(bar_vars[b])-0.5*(c_mean-target_means[b])**2/add_vars[b] \
                - 0.5*(x[idx:]-bar_means[b])**2/bar_vars[b] \
                + np.sum([np.log(cdfs[c, idx:]) for c in range(anum) if c!=b], axis=0)
        log_max_prob = max(log_max_prob, max(tmp))
        log_probs.append(tmp)
    probs = [np.exp(lp-log_max_prob) for lp in log_probs]
    probs_l = []
    for p in probs:
        probs_l.append(np.concatenate((np.zeros(len(x) - min_id -len(p),), p)))
    prob_tot = np.sum(np.array(probs_l),axis=0)
    if np.sum(prob_tot) == 0.0:
        pdb.set_trace()
    prob_tot =  prob_tot/np.sum(prob_tot, dtype=np.float32)/interval
    x = x[min_id:]
    new_mean = interval*np.inner(x, prob_tot)
    new_var = interval*np.inner((x-new_mean)**2, prob_tot)
    if np.isnan(new_var):
        print("variance is NaN")
        pdb.set_trace()
    return new_mean, np.maximum(varTH, new_var), (x, prob_tot)

def posterior_approx(n_means, n_vars, c_mean, c_var, rew, dis, terminal, logEps = - 1e+20, 
    varTH = 1e-10, asymptotic=False, batch=False, noise=0.0):
    # ADFQ-Approx
    # TO DO : Convert To Batch-able
    if batch:
        batch_size = len(n_means)
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_var = np.reshape(c_var, (batch_size,1))
        rew = np.reshape(rew, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))
    else:
        if terminal==1:
            var_new = 1./(1./c_var + 1./REW_VAR)
            mean_new = var_new*(c_mean/c_var + rew/REW_VAR)
            return mean_new, var_new, (n_means, n_vars, np.ones(n_means.shape))
    target_means = rew + dis*np.array(n_means, dtype=np.float64)
    target_vars = dis*dis*(np.array(n_vars, dtype = np.float64) + noise)
    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)

    add_vars = c_var + target_vars
    sorted_idx = np.argsort(target_means, axis=int(batch))
    if batch:
        ids = range(0,batch_size)
        bar_targets = target_means[ids,sorted_idx[:,-1], np.newaxis]*np.ones(target_means.shape)
        bar_targets[ids, sorted_idx[:,-1]] = target_means[ids, sorted_idx[:,-2]]
    else:
        bar_targets = target_means[sorted_idx[-1]]*np.ones(target_means.shape)
        bar_targets[sorted_idx[-1]] = target_means[sorted_idx[-2]]
    thetas = np.heaviside(bar_targets-bar_means,0.0)            
    if asymptotic and (n_vars <= varTH).all() and (c_var <= varTH):
        min_b = np.argmin((target_means-c_mean)**2-2*add_vars*logEps*thetas)
        weights = np.zeros(np.shape(target_means))
        weights[min_b] = 1.0
        return bar_means[min_b], np.maximum(varTH, bar_vars[min_b]), (bar_means, bar_vars, weights)

    log_weights = -0.5*(np.log(2*np.pi)+np.log(add_vars)+(c_mean-target_means)**2/add_vars) + logEps*thetas
    log_weights = log_weights - np.max(log_weights, axis=int(batch), keepdims=batch)
    log_weights = log_weights - logsumexp(log_weights, axis=int(batch), keepdims=batch) # normalized! 
    weights = np.exp(log_weights, dtype=np.float64)

    mean_new = np.sum(np.multiply(weights, bar_means), axis=int(batch), keepdims=batch)
    var_new = np.maximum(varTH, np.sum(np.multiply(weights,bar_means**2+bar_vars), axis=int(batch), keepdims=batch) - mean_new**2)
    var_new  = (1.-terminal)*var_new + terminal*1./(1./c_var + 1./REW_VAR)
    mean_new = (1.-terminal)*mean_new + terminal*var_new*(c_mean/c_var + rew/REW_VAR)
    if np.isnan(mean_new).any() or np.isnan(var_new).any():
        pdb.set_trace()
    return mean_new, var_new, (bar_means, bar_vars, weights)

def posterior_approx_log(n_means, n_logvars, c_mean, c_logvar, rew, dis, terminal, eps = 0.01, 
    logvarTH = -100.0, batch=False):
    # ADFQ-Approx using log variance
    # TO DO : Convert To Batch-able
    if batch:
        batch_size = len(n_means)
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_logvar = np.reshape(c_logvar, (batch_size,1))
        rew = np.reshape(rew, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    target_logvars = 2*np.log(dis)+np.array(n_logvars, dtype = np.float32)

    bar_logvars = -np.array([logsumexp([-c_logvar, -tv]) for tv in target_logvars])
    bar_means = np.exp(bar_logvars-c_logvar)*c_mean + np.exp(bar_logvars-target_logvars)*target_means

    if (n_logvars <= logvarTH).all() and (c_logvar <= logvarTH):
        min_b = np.argmin(abs(target_means-c_mean))
        weights = np.zeros(np.shape(target_means))
        weights[min_b] = 1.0
        return bar_means[min_b], np.maximum(logvarTH, bar_logvars[min_b]), (bar_means, bar_logvars, weights)

    add_logvars = np.array([logsumexp([c_logvar, tv]) for tv in target_logvars])
    sorted_idx = np.argsort(target_means, axis=int(batch))
    bar_targets = target_means[sorted_idx[-1]]*np.ones(target_means.shape)
    bar_targets[sorted_idx[-1]] = target_means[sorted_idx[-2]]
    thetas = np.heaviside(bar_targets-bar_means,0.0)    
    if (add_logvars < logvarTH).any():
        print(add_logvars) 
    log_c = np.maximum(-20.0,-0.5*(np.log(2*np.pi)+add_logvars+((c_mean-target_means)**2)/np.exp(add_logvars))) #(batch_size X anum)
     #max_log_c = max(log_c)
    c = np.exp(log_c) #np.exp(log_c-max_log_c)
    logZ = np.log(np.dot(1. - (1.-eps)*thetas, c))
    logmean_new =  np.log(np.dot(bar_means*(1. - (1.-eps)*thetas), c)) - logZ
    
    log_moment2 =  np.log(np.dot((bar_means**2+np.exp(bar_logvars))*(1. - (1.-eps)*thetas), c)) - logZ
    min_term = min(log_moment2, 2*logmean_new)
    logvar_new = max(logvarTH, min_term + np.log(np.maximum(1e-10, np.exp(log_moment2-min_term) - np.exp(2*logmean_new-min_term)) ) )
    logvar_new  = (1.-terminal)*logvar_new - terminal*logsumexp([-c_logvar, -np.log(REW_VAR)])
    mean_new = (1.-terminal)*np.exp(logmean_new)+ terminal*(c_mean*np.exp(logvar_new-c_logvar) + rew*np.exp(logvar_new-np.log(REW_VAR)))
    if np.isnan(mean_new).any() or np.isnan(logvar_new).any() or np.isinf(-logvar_new).any():
        pdb.set_trace()
    weights = np.exp(log_c + np.log(1.-(1.-eps)*thetas) - logZ) #- max_log_c 
    return mean_new, logvar_new, (bar_means, bar_logvars, weights)

def posterior_approx_log_v2(n_means, n_logvars, c_mean, c_logvar, rew, dis, terminal, 
    logEps = -1e+20, logvarTH = -100.0, batch=False):
    # ADFQ-Approx using log variance
    # Version 2 - considering smaller epsilon
    # TO DO : Convert To Batch-able
    if batch:
        batch_size = len(n_means)
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_logvar = np.reshape(c_logvar, (batch_size,1))
        rew = np.reshape(rew, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    target_logvars = 2*np.log(dis)+np.array(n_logvars, dtype = np.float32)

    bar_logvars = -np.array([logsumexp([-c_logvar, -tv]) for tv in target_logvars])
    bar_means = np.exp(bar_logvars-c_logvar)*c_mean + np.exp(bar_logvars-target_logvars)*target_means

    if (n_logvars <= logvarTH).all() and (c_logvar <= logvarTH):
        min_b = np.argmin(abs(target_means-c_mean))
        weights = np.zeros(np.shape(target_means))
        weights[min_b] = 1.0
        return bar_means[min_b], np.maximum(logvarTH, bar_logvars[min_b]), (bar_means, bar_logvars, weights)

    add_logvars = np.array([logsumexp([c_logvar, tv]) for tv in target_logvars])
    sorted_idx = np.argsort(target_means, axis=int(batch))
    bar_targets = target_means[sorted_idx[-1]]*np.ones(target_means.shape)
    bar_targets[sorted_idx[-1]] = target_means[sorted_idx[-2]]
    thetas = np.heaviside(bar_targets-bar_means,0.0)    
    if (add_logvars < logvarTH).any():
        print(add_logvars) 
    log_weights = -0.5*(np.log(2*np.pi)+add_logvars+((c_mean-target_means)**2)/np.exp(add_logvars)) + logEps*thetas #(batch_size X anum)
    log_weights = log_weights - max(log_weights)
    log_weights = log_weights - logsumexp(log_weights) # normalized! 
    weights = np.exp(log_weights)
    logmean_new = np.log(np.dot(weights,bar_means))
    log_moment2 =  np.log(np.dot(bar_means**2+np.exp(bar_logvars), weights))

    min_term = min(log_moment2, 2*logmean_new)
    if np.exp(log_moment2-min_term) - np.exp(2*logmean_new-min_term) <= 0.0:
        logvar_new  = logvarTH
    else:
        logvar_new = max(logvarTH, min_term + np.log(np.exp(log_moment2-min_term) - np.exp(2*logmean_new-min_term)))

    logvar_new  = (1.-terminal)*logvar_new - terminal*logsumexp([-c_logvar, -np.log(REW_VAR)])
    mean_new = (1.-terminal)*np.exp(logmean_new)+ terminal*(c_mean*np.exp(logvar_new-c_logvar) + rew*np.exp(logvar_new-np.log(REW_VAR)))

    if np.isnan(mean_new).any() or np.isnan(logvar_new).any() or np.isinf(-logvar_new).any():
        pdb.set_trace()

    return mean_new, logvar_new, (bar_means, bar_logvars, weights)

def posterior_soft_approx(n_means, n_vars, c_mean, c_var, rew, dis, terminal, varTH = 1e-10, 
    matching=True, ratio = False, scale = False, c_scale = 1.0, asymptotic=False, plot=False, noise=0.0, batch=False):
    # ADFQ-SoftApprox
    # Need New name
    # Not batch yet

    if terminal == 1:
        var_new = 1./(1./c_var + c_scale/REW_VAR)
        if not(scale):
            var_new = np.maximum(varTH, var_new)
        mean_new = var_new*(c_mean/c_var + rew/REW_VAR*c_scale)
        return mean_new, var_new, (n_means, n_vars, np.ones(n_means.shape))

    target_means = rew + dis*np.array(n_means, dtype=np.float64)
    target_vars = dis*dis*(np.array(n_vars, dtype = np.float64) + noise/c_scale)
    if matching:
        if ratio or (asymptotic and (n_vars <= varTH).all() and (c_var <= varTH)):
            stats = posterior_match_ratio_helper(target_means, target_vars, c_mean, c_var, dis)
            b_rep = np.argmin(stats[:,2])
            weights = np.zeros((len(stats),))
            weights[b_rep] = 1.0
            return stats[b_rep, 0], np.maximum(varTH, stats[b_rep, 1]), (stats[:,0], stats[:,1], weights)
        elif scale :
            stats = posterior_match_scale_helper(target_means, target_vars, c_mean, c_var, dis, c_scale = c_scale, plot=plot) 
        else:
            stats = posterior_match_helper(target_means, target_vars, c_mean, c_var, dis, plot=plot)   
    else:
        stats = posterior_soft_helper(target_means, target_vars, c_mean, c_var)
    
    k = stats[:,2] - max(stats[:,2])
    weights = np.exp(k - logsumexp(k), dtype=np.float64) # normalized.
    mean_new = np.sum(weights*stats[:,0], axis = int(batch))
    if scale: 
        #var_new = np.dot(weights,stats[:,0]**2/c_scale + stats[:,1]) - mean_new**2/c_scale
        var_new = np.dot(weights, stats[:,1]) + (np.dot(weights,stats[:,0]**2) - mean_new**2)/c_scale
        if var_new <= 0.0:
            #print("variance equals or below 0")
            pdb.set_trace()
            var_new = varTH
    else:
        var_new = np.maximum(varTH, np.dot(weights,stat[:,0]**2 + stats[:,1]) - mean_new**2)
    if np.isnan(mean_new).any() or np.isnan(var_new).any():
        pdb.set_trace()
    return mean_new, var_new, (stats[:,0], stats[:,1], weights)

def posterior_match_helper(target_means, target_vars, c_mean, c_var, discount, plot=False):
    # Matching a Gaussian distribution with the first and second derivatives.
    dis2 = discount*discount
    sorted_idx = np.flip(np.argsort(target_means),axis=0) 
    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var+target_vars
    log_weights = -0.5*(np.log(2*np.pi) + np.log(add_vars) + (c_mean-target_means)**2/add_vars) #(batch_size X anum) 
    anum = len(sorted_idx)
    stats = []
    for b in sorted_idx:
        b_primes = [c for c in sorted_idx if c!=b] # From large to small.
        upper = 10000
        for i in range(anum):
            if i == (anum-1):
                lower = -10000
            else:
                lower = target_means[b_primes[i]]
            var_star = 1./(1/bar_vars[b]+sum(1/target_vars[b_primes[:i]]))
            mu_star = (bar_means[b]/bar_vars[b]+ sum(target_means[b_primes[:i]]/target_vars[b_primes[:i]]))*var_star
            if (np.float16(mu_star) >= np.float16(lower)) and (np.float16(mu_star) <= np.float16(upper)):
                k = log_weights[b]+0.5*(np.log(var_star)-np.log(bar_vars[b])) \
                     -0.5*(mu_star-bar_means[b])**2/bar_vars[b] \
                     -0.5*sum([np.maximum(target_means[c]-mu_star, 0.0)**2/target_vars[c] for c in b_primes])
                     #-0.5*sum((target_means[b_primes[:i]]-mu_star)**2/target_vars[b_priems[:i]])
                stats.append((mu_star, var_star, k))
                break
            upper = lower
    if not(len(stats)== anum):
        pdb.set_trace()
    if plot:
        x = np.arange(min(target_means)+2,max(target_means)+10,0.4)
        f, ax_set = plt.subplots(anum, sharex=True, sharey=False)
        for (i,b) in enumerate(sorted_idx):
            b_primes = [c for c in sorted_idx if c!=b]
            true = np.exp(log_weights[b])/np.sqrt(2*np.pi*bar_vars[b])*np.exp(-0.5*(x-bar_means[b])**2/bar_vars[b] \
                - 0.5*np.sum([(np.maximum(target_means[c]-x,0.0))**2/target_vars[c] for c in b_primes],axis=0))
            approx = np.exp(stats[i][2])/np.sqrt(2*np.pi*stats[i][1])*np.exp(-0.5*(x-stats[i][0])**2/stats[i][1])
            ax_set[b].plot(x,true)
            ax_set[b].plot(x, approx,'+', markersize=8)
        plt.show() 
    return np.array([stats[sorted_idx[b]] for b in range(anum)], dtype=np.float64)

def posterior_match_scale_helper(target_means, target_vars, c_mean, c_var, discount, c_scale, plot=False):
    # Matching a Gaussian distribution with the first and second derivatives.
    # target vars and c_var are not the true variance but scaled variance. 
    dis2 = discount*discount
    rho_vars = c_var/target_vars*dis2
    sorted_idx = np.flip(np.argsort(target_means),axis=0) 
    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars) # scaled
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var+target_vars # scaled
    anum = len(sorted_idx)
    stats = []
    log_weights = -0.5*(np.log(2*np.pi) + np.log(add_vars) + (c_mean-target_means)**2/add_vars)
    for (j,b) in enumerate(sorted_idx):
        b_primes = [c for c in sorted_idx if c!=b] # From large to small.
        upper = 10000
        tmp_vals = []
        for i in range(anum):
            lower = 1e-5 if i==(anum-1) else target_means[b_primes[i]]
            mu_star = np.float64((bar_means[b]+sum(target_means[b_primes[:i]]*rho_vars[b_primes[:i]])/(dis2+rho_vars[b])) \
                        /(1.+sum(rho_vars[b_primes[:i]])/(dis2+rho_vars[b])))
            tmp_vals.append((lower, mu_star, upper))
            if (np.float32(mu_star) >= np.float32(lower)) and (np.float32(mu_star) <= np.float32(upper)):
                var_star = 1./(1/bar_vars[b]+sum(1/target_vars[b_primes[:i]])) # scaled
                k = 0.5*(np.log(var_star) - np.log(bar_vars[b]) - np.log(2*np.pi) - np.log(add_vars[b])) \
                    -0.5/add_vars[b]/c_scale*((target_means[b]-c_mean)**2 + ((mu_star-bar_means[b])*(rho_vars[b]+dis2))**2/dis2/rho_vars[b] \
                        + sum( rho_vars[b_primes[:i]]*(target_means[b_primes[:i]] - mu_star)**2 )*(1./dis2 + 1./rho_vars[b]))
                stats.append((mu_star, var_star, k))
                break
            upper = lower
        if len(stats) <= j:
            print("Could not find a macthing q star")
            tmp_vals = np.array(tmp_vals, dtype=np.float64)
            i_u = np.argmin(abs(tmp_vals[:,2]-tmp_vals[:,1]))
            i_l = np.argmin(abs(tmp_vals[:,1]-tmp_vals[:,0]))
            i = i_u if (abs(tmp_vals[i_u,2]-tmp_vals[i_u,1]) < abs(tmp_vals[i_l,1]-tmp_vals[i_l,0])) else i_l
            mu_star = tmp_vals[i,1]
            var_star = 1./(1/bar_vars[b]+sum(1/target_vars[b_primes[:i]])) # scaled
            k = 0.5*(np.log(var_star) - np.log(bar_vars[b]) - np.log(2*np.pi) - np.log(add_vars[b])) \
                -0.5/add_vars[b]/c_scale*((target_means[b]-c_mean)**2 + ((mu_star-bar_means[b])*(rho_vars[b]+dis2))**2/dis2/rho_vars[b] \
                    + sum( rho_vars[b_primes[:i]]*(target_means[b_primes[:i]] - mu_star)**2 )*(1./dis2 + 1./rho_vars[b]))
            stats.append((mu_star, var_star, k))

    return np.array([stats[sorted_idx[b]] for b in range(anum)], dtype=np.float64)

def posterior_match_ratio_helper(target_means, target_vars, c_mean, c_var, discount, plot=False):
    # Matching a Gaussian distribution with the first and second derivatives.
    dis2 = discount*discount
    sorted_idx = np.flip(np.argsort(target_means),axis=0) 
    rho_vars = c_var/target_vars*dis2
    bar_means = dis2/(rho_vars+dis2)*c_mean + rho_vars/(rho_vars+dis2)*target_means
    anum = len(sorted_idx)
    stats = []
    for (j,b) in enumerate(sorted_idx):
        b_primes = [c for c in sorted_idx if c!=b] # From large to small.
        upper = 10000
        tmp_vals = []
        for i in range(anum):
            if i == (anum-1):
                lower = -10000
            else:
                lower = target_means[b_primes[i]]

            mu_star = np.float64((bar_means[b]+sum(target_means[b_primes[:i]]*rho_vars[b_primes[:i]])/(dis2+rho_vars[b])) \
                        /(1.+sum(rho_vars[b_primes[:i]])/(dis2+rho_vars[b])))
            tmp_vals.append((lower, mu_star, upper))
            if (np.float32(mu_star) >= np.float32(lower)) and (np.float32(mu_star) <= np.float32(upper)):
                var_star = c_var/(1.+rho_vars[b]/dis2 + sum(rho_vars[b_primes[:i]])/dis2)
                I = (target_means[b]-c_mean)**2 + ((mu_star-bar_means[b])*(rho_vars[b]+dis2))**2/dis2/rho_vars[b] \
                        + sum( rho_vars[b_primes[:i]]*(target_means[b_primes[:i]] - mu_star)**2 )*(1./dis2 + 1./rho_vars[b])
                stats.append((mu_star, var_star, I))
                break
            upper = lower
        if len(stats) <= j:
            #print("Could not find a macthing q star")
            tmp_vals = np.array(tmp_vals, dtype=np.float64)
            i_u = np.argmin(abs(tmp_vals[:,2]-tmp_vals[:,1]))
            i_l = np.argmin(abs(tmp_vals[:,1]-tmp_vals[:,0]))
            i = i_u if (abs(tmp_vals[i_u,2]-tmp_vals[i_u,1]) < abs(tmp_vals[i_l,1]-tmp_vals[i_l,0])) else i_l
            mu_star = tmp_vals[i,1]
            var_star = c_var/(1.+rho_vars[b]/dis2 + sum(rho_vars[b_primes[:i]])/dis2)
            I = (target_means[b]-c_mean)**2 + ((mu_star-bar_means[b])*(rho_vars[b]+dis2))**2/dis2/rho_vars[b] \
                        + sum( rho_vars[b_primes[:i]]*(target_means[b_primes[:i]] - mu_star)**2 )*(1./dis2 + 1./rho_vars[b])
            stats.append((mu_star, var_star, I))
    assert(len(stats)== anum)
    return np.array([stats[sorted_idx[b]] for b in range(anum)])
    

def posterior_soft_helper(target_means, target_vars, c_mean, c_var):
    # Select a representaves truncated Gaussian which peak is in its truncated region.
    sorted_idx = np.argsort(target_means)
    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var+target_vars
    stats = []
    for b in range(len(target_means)):
        if c_mean >= target_means[sorted_idx[-1]]:
            dominate = (bar_means[b] >= target_means[sorted_idx[-1]])
        elif (c_mean < target_means[sorted_idx[-2]]) and (bar_means[sorted_idx[-1]] < target_means[sorted_idx[-2]]):
            dominate = False
        else:
            dominate = (b == sorted_idx[-1])
        mean_b = bar_means[b]/bar_vars[b]
        log_coeff_b = -0.5*(np.log(2*np.pi*add_vars[b])+(c_mean-target_means[b])**2/add_vars[b]+bar_means[b]**2/bar_vars[b]+np.log(bar_vars[b]))
        lower = -1000.0
        found = False
        if dominate:
            var = bar_vars[b]
            mean =  bar_means[b]
            log_coeff = log_coeff_b + 0.5*(mean**2/var +np.log(var))
        else:
            for (i,b2) in enumerate(sorted_idx):
                if b2 != b:
                    upper = target_means[b2]
                    var = 1.0/(1.0/bar_vars[b] \
                            + np.sum([1.0/(np.pi/4*target_vars[b3]) for b3 in sorted_idx[i:] if b3!=b]))
                    mean = var*(mean_b + np.sum([target_means[b3]/(np.pi/4*target_vars[b3]) for b3 in sorted_idx[i:] if b3!=b]))
                    if (mean > lower) and (mean <= upper):
                        log_coeff = log_coeff_b - sum([1 for b3 in sorted_idx[i:] if b3!=b])*np.log(2) \
                            + 0.5*(np.log(var) + mean**2/var - np.sum([target_means[b3]**2/(np.pi/4*target_vars[b3]) for b3 in sorted_idx[i:] if b3!=b]))
                        found = True
                        break
                    lower = upper
            if not(found):
                var = bar_vars[b]
                mean =  bar_means[b]
                #coeff = coeff_b * np.exp(0.5*mean**2/var) * np.sqrt(var)
                log_coeff = log_coeff_b + 0.5*(mean**2/var +np.log(var))
        stats.append((mean, var, log_coeff))
    return np.array(stats)
        

def posterior_soft_true(target_means, target_vars, c_mean, c_var, plot_major = False, plot_ave = False):
    # True distribution - Mixture of Truncated Gaussian. 
    # Plotting purpose

    anum = len(target_means)
    sorted_idx = np.argsort(target_means)
    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var+target_vars
    mean_range = np.concatenate((target_means,[c_mean]))
    sd_range = np.sqrt(np.concatenate((target_vars, [c_var])))

    width = 8.0
    num_interval = 300
    interval = width*max(sd_range)/float(num_interval)
    x = np.arange(min(mean_range-0.5*width*sd_range), max(mean_range+0.5*width*sd_range), interval)
    truncated_x = []
    prev = min(mean_range-0.5*width*sd_range)
    for b in sorted_idx:
        truncated_x.append(np.arange(prev,target_means[b],interval))
        prev = target_means[b]
    truncated_x.append(np.arange(prev,max(mean_range+0.5*width*sd_range), interval))
    f, ax_set = plt.subplots(anum, sharex=True, sharey=False)
    for b in range(anum):
        mean_b = bar_means[b]/bar_vars[b]
        coeff_b = norm.pdf(c_mean-target_means[b], 0, np.sqrt(add_vars[b]))/np.sqrt(bar_vars[b]) * np.exp(-0.5*bar_means[b]**2/bar_vars[b])
        add_y = []
        lower = -1000.0
        rep = None
        stats_set = []
        for (i,b2) in enumerate(sorted_idx):
            if b2 != b:
                upper = target_means[b2]
                y = np.concatenate((add_y,truncated_x[i]))
                var = 1.0/(1.0/bar_vars[b] \
                        + np.sum([1.0/(np.pi/4*target_vars[b3]) for b3 in sorted_idx[i:] if b3!=b]))
                mean = var*(mean_b + np.sum([target_means[b3]/(np.pi/4*target_vars[b3]) for b3 in sorted_idx[i:] if b3!=b]))
                coeff = coeff_b/(2**sum([1 for b3 in sorted_idx[i:] if b3!=b])) * np.sqrt(var) \
                        * np.exp(0.5*mean**2/var - 0.5*np.sum([target_means[b3]**2/(np.pi/4*target_vars[b3]) for b3 in sorted_idx[i:] if b3!=b]))
                stats_set.append((mean,var,coeff))
                if (mean > lower) and (mean <= upper):
                    rep = (mean, var, coeff)
                ax_set[b].plot(y,coeff*norm.pdf(y,mean,np.sqrt(var)))
                add_y = []
                lower = upper
            else:
                add_y = truncated_x[i]
        y = np.concatenate((add_y,truncated_x[-1]))
        var = bar_vars[b]
        mean =  bar_means[b]
        coeff = coeff_b * np.exp(0.5*mean**2/var) * np.sqrt(var)
        stats_set.append((mean,var,coeff))
        ax_set[b].plot(y,coeff*norm.pdf(y,mean,np.sqrt(var)))
        if plot_major:
            if rep is None:
                rep = (mean, var, coeff)
            ax_set[b].plot(x, rep[2]*norm.pdf(x, rep[0], np.sqrt(rep[1])),'--')
        if plot_ave:
            stats_set = np.array(stats_set)
            weights = stats_set[:,2]#/sum(stats_set[:,2])
            ave_mean = np.dot(stats_set[:,0], weights)
            ave_var = np.dot(stats_set[:,1]+(stats_set[:,0])**2, weights) - ave_mean**2
            ax_set[b].plot(x, norm.pdf(x, ave_mean, np.sqrt(ave_var))/float(anum),'--')
        ax_set[b].axvline(x=target_means[b])
        ax_set[b].set_title('target action number : '+str(b))
    plt.show()

def plot_posterior_all(n_means, n_vars, c_mean, c_var, rew, dis, terminal, varTH):
    # You can plot distribution from different approximation methods here.
    anum = len(n_means)
    try: 
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    except:
        colors = ['b', 'g', 'r', 'c',  'y', 'k']
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    m_numeric, v_numeric, (x, prob) = posterior_numeric(n_means, n_vars, c_mean, c_var, rew, dis, terminal,varTH=varTH)
    #m_soft, v_soft, stats_soft = posterior_soft_approx(n_means, n_vars, c_mean, c_var, rew, dis, terminal)
    m_match, v_match, stats_match = posterior_soft_approx(n_means, n_vars, c_mean, c_var, rew, dis, terminal, matching=True, varTH=varTH)
    m_approx, v_approx, stats_approx = posterior_approx(n_means, n_vars, c_mean, c_var, rew, dis, terminal, varTH=varTH)
    #m_approx_log, v_approx_log, stats_approx_log = posterior_approx_log_v2(n_means, np.log(n_vars), c_mean, np.log(c_var), rew, dis, terminal)


    f, ax_set = plt.subplots(3, sharex=True, sharey=True)
    ax_set[0].plot(x, norm.pdf(x, m_numeric, np.sqrt(v_numeric)), color = 'm', linewidth=1.5)
    
    #ax_set[2].plot(x, norm.pdf(x, m_soft, np.sqrt(v_soft)),color = 'm')
    #[ax_set[2].plot(x, stats_soft[2][b]*norm.pdf(x, stats_soft[0][b], np.sqrt(stats_soft[1][b])), color = colors[b]) for b in range(anum)]
    
    ax_set[1].plot(x, norm.pdf(x, m_match, np.sqrt(v_match)),color = 'm', linewidth=1.5)
    [ax_set[1].plot(x, stats_match[2][b]*norm.pdf(x, stats_match[0][b], np.sqrt(stats_match[1][b])), color = colors[b]) for b in range(anum)]
    
    ax_set[2].plot(x, norm.pdf(x, m_approx, np.sqrt(v_approx)),color = 'm', linewidth=1.5)
    [ax_set[2].plot(x, stats_approx[2][b]*norm.pdf(x, stats_approx[0][b], np.sqrt(stats_approx[1][b])), color = colors[b]) for b in range(anum)]
    #ax_set[3].plot(x, norm.pdf(x, m_approx_log, np.sqrt(np.exp(v_approx_log))),color = 'm')

    for ax in ax_set:
        ax.plot(x, prob, 'y--')
        [ax.axvline(x=target_means[b]) for b in range(anum)]
        ax.axvline(x=(0.8*(c_mean)+0.2*(rew+dis*max(n_means))), color='r')
    ax_set[1].legend(['approx','1','2','3','4', 'true'])
    ax_set[-1].set_xlabel(', '.join([str(v) for v in target_means]))
    plt.show()
    #plt.draw()
    #plt.pause(0.5)

def prod2Gaussian(mu1,var1,mu2,var2):
    """
    Product of Two Gaussian distributions with different mean and variance values.
    """
    var12 = 1.0/(1.0/var1+1.0/var2)
    mu12 = var12*(mu1/var1 + mu2/var2)
    C12 = (mu2-mu1)*(mu2-mu1)/(var1+var2)
    return mu12, var12, C12



def plot_IQR(T, ys, labels, save, x_name, y_name, ts = None, shadow = True, legend=(True, (1,1)), pic_name = None, colors=None,):
    # Interquartile 
    if not(colors):
        colors = ['g','k','r','b','c','m','y','burlywood','chartreuse','0.8','--', '-.', ':']
    plot_err = []
    f1, ax1 = plt.subplots()
    if len(ys.shape)==2:
        N = ys.shape[1]+1
        if ts == None :
            ts = range(0,T-1,T/N)
        xnew = np.linspace(ts[0],ts[-1],300) #300 represents number of points to make between T.min and T.max
        for (i,y) in enumerate(ys):
            power_smooth = spline(ts,[0]+list(y),xnew)
            power_smooth = [max(0,el) for el in power_smooth]
            tmp, = ax1.plot(xnew, power_smooth, colors[i], label=labels[i], linewidth=2.0)
            plot_err.append(tmp)
    else:
        if ts == None:  
            ts = range(0,T) #range(0,T-1,T/N)
        for (i,y) in enumerate(ys):
            m, ids25, ids75  = iqr(y)
            tmp, = ax1.plot(ts, m, colors[i], label=labels[i], linewidth=2.0)
            plot_err.append(tmp) 
            if shadow:       
                ax1.fill_between(ts, list(ids75), list(ids25), facecolor=colors[i], alpha=0.15)

    if legend[0]:
        ax1.legend(plot_err, labels ,loc='upper right', shadow=True,
                    prop={'family':'Times New Roman', 'size':14})#, bbox_to_anchor=legend[1])
    ax1.tick_params(axis='both',which='major',labelsize=16)
    ax1.set_xlabel(x_name,fontsize=20, fontname="Times New Roman")
    ax1.set_ylabel(y_name,fontsize=20, fontname="Times New Roman")
    ax1.set_xlim((0,ts[-1]))
    ax1.set_ylim((0,7))
    
    if save:
        f1.savefig(pic_name)
    else:
        plt.show()

def plot_smoothing(t, y, labels, x_name='x', y_name='y', save=False, legend=(True,(0.8,0.5)), window=4, colors=None):
    
    if not(colors):
        colors = COLORS
    ts = range(0, t, t/y.shape[-1] )
    f, ax = plt.subplots()
    if len(y.shape)==3:
        y = np.mean(y, axis=1)
    plot_list = []
    for i in range(y.shape[0]):
        if MARKER[i] in [':','-','--']:
            lw = 1.5
        else:
            lw = 1.5
        tmp, =ax.plot(ts, smoothing(y[i], window), MARKER[i], markeredgecolor = colors[i], markerfacecolor='None', linewidth=lw, color = colors[i])
        plot_list.append(tmp)

    for i in range(y.shape[0]):
        tmp, = ax.plot(ts, y[i], colors[i], alpha = 0.2)
    if legend[0]:
        ax.legend(plot_list, labels ,loc='lower right', shadow=True,
                    prop={'family':'Times New Roman', 'size':12}, bbox_to_anchor=legend[1])
    ax.tick_params(axis='both',which='major',labelsize=15)
    ax.set_xlabel(x_name,fontsize=16, fontname="Times New Roman")
    ax.set_ylabel(y_name,fontsize=16, fontname="Times New Roman")
    ax.set_xlim((0,ts[-1]))

    if save:
        f1.savefig(pic_name)
    else:
        plt.show()


def iqr(x):
    """
    x has to be a 2D np array. The interquantiles are computed along with the axis 1
    """
    i25 = int(0.25*x.shape[0])
    i75 = int(0.75*x.shape[0])
    x=x.T
    ids25=[]
    ids75=[]
    m = []
    for y in x:
        tmp = np.sort(y)
        ids25.append(tmp[i25])
        ids75.append(tmp[i75])
        m.append(np.mean(tmp,dtype=np.float32))
    return m, ids25, ids75

def smoothing(x, window):
    interval = int(np.floor(window*0.5))
    smoothed = list(x[:interval])
    for i in range(len(x)-window):
        smoothed.append(np.mean(x[i:i+window+1]))
    smoothed.extend(x[-interval:])
    return smoothed

def draw_maze(obj,s,rew):
    M = np.zeros((6,7))
    walls = [(0,1),(1,1),(0,4),(1,4),(3,0),(3,1),(3,5),(3,6),(5,6)]
    for (i,j) in walls:
        M[i][j] = 10
    M[0][2] = M[5][0] = M[4][6] = 3

    v_flag = obj.num2flag(s%8)
    pos = s/8
    r = pos%6
    M[r][(pos-r)/6] = 6
    imgplot = plt.imshow(M)
    plt.title(str(sum(v_flag))+'   '+str(sum(rew)))
    plt.draw()
    plt.pause(0.005)
    M[r][(pos-r)/6] = 0

def value_plot(Q_tab, obj, isMaze = True, arrow = True):
    direction={0:(0,-0.4),1:(0,0.4),2:(-0.4,0),3:(0.4,0)} #(x,y) cooridnate
    V = np.max(Q_tab,axis=1)
    best_action = np.argmax(Q_tab,axis=1)
    if isMaze:
        idx2cell = obj.idx2cell
        for i in range(8):
            f,ax = plt.subplots()
            y_mat = np.zeros(obj.dim)
            for j in range(len(idx2cell)):
                pos = idx2cell[j]
                y_mat[pos[0], pos[1]] = V[8*j+i]
                if arrow:
                    a = best_action[8*j+i]
                    ax.arrow(pos[1], pos[0], direction[a][0], direction[a][1], 
                        head_width=0.05, head_length=0.1, fc='r', ec='r')
            y_mat[obj.goal_pos] = max(V)+0.1
            ax.imshow(y_mat,cmap='gray')
    else:
        n = int(np.sqrt(len(V)))
        tab = np.zeros((n,n))
        for r in range(n):
            for c in range(n):
                if not(r==(n-1)and c==(n-1)):
                    tab[r,c] = V[n*c+r]
                    if arrow:
                        d = direction[best_action[n*c+r]]
                        plt.arrow(c,r,d[0],d[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
        tab[obj.goal_pos] = max(V[:-1])+0.1
        plt.imshow(tab,cmap='gray') 
    plt.show()

