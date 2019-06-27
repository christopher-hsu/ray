"""""
Old BRL UTIL code. Temporary Trash codes.
"""""



import sys
sys.path.insert(0,'/usr/local/lib/python2.7/site-packages')
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm
import pdb
from matplotlib import cm
from operator import itemgetter

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

class feat_vec(object):

    def __init__(self, snum):

        self.snum = snum

    def __call__(self, state):
        phi = np.zeros(self.snum,dtype = np.float32)
        phi[state] = 1.0
        return phi

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

def mean_max(mu,var,d):
    mu_bar = d*(1+d)/(1+d**2)*mu
    var_bar = d**2/(1+d**2)*var
    """
    Uncomment the following lines and compare Z and sum(p). They must be same. 
    x = np.arange(-100,100,1)
    p = norm.pdf(x,mu_bar,np.sqrt(var_bar))*norm.cdf(x,d*mu,d*np.sqrt(var))
    Z = norm.cdf(mu_bar,d*mu,v)
    """
    v = np.sqrt(var*d**2 + var_bar)
    mean =  mu_bar + var_bar*norm.pdf(mu_bar,d*mu, v)/norm.cdf(mu_bar,d*mu,v)

    return mean 

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
    #plt.figure(1)
    #plt.plot(x,max_p,"bo")
    #plt.show()
   
    max_mean = np.inner(x,max_p)*interval
    return max_mean,np.inner(x**2,max_p)*interval- max_mean**2 

def posterior_numeric_old(n_means, n_vars, c_mean, c_var, rew, dis, terminal, num_interval= 800, width = 16.0):
    if terminal:
        new_var = 1.0/(1.0/c_var + 1.0/REW_VAR)
        new_sd = np.sqrt(new_var)
        new_mean = new_var*(c_mean/c_var + rew/REW_VAR)
        interval = width*new_sd/float(num_interval)
        x = np.arange(new_mean-0.5*width*new_sd,new_mean+0.5*width*new_sd, interval)
        return new_mean, new_var, (x, norm.pdf(x, new_mean, new_sd))
    """
    Designed for ADF approach
    INPUT:
        n_means: a numpy array of Gaussian mean values of (next state, action) pairs for all available actions.
        n_sds: a numpy array of Gaussian SD values of (next state,action) pairs for all available actions.
        c_mean: a mean value of the current state and action
        c_sd: a SD value of the current state and action
    obs:
        mean and variance of the joint distribution
    """
    delta = 0.05
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    target_vars = dis*dis*np.array(n_vars, dtype = np.float32)

    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)

    c_tmp = -(c_mean-target_means)**2/2.0/(c_var + target_vars)
    c_min = np.amin(c_tmp, axis=0)
    c_max = np.amax(c_tmp, axis=0)
    B = float(c_max-c_min < 50.0)
    weights = np.exp(c_tmp - c_min*B - c_max*(1-B))/np.sqrt(c_var+target_vars)
    weights = weights/np.sum(weights, axis =0)

    mean_range = np.concatenate((target_means,[c_mean]))
    sd_range = np.sqrt(np.concatenate((dis*dis*n_vars, [c_var])))

    interval = width*max(sd_range)/float(num_interval)
    x = np.arange(min(mean_range-0.5*width*sd_range), max(mean_range+0.5*width*sd_range), interval)

    count = 0 
    done = False
    while(not(done)):
        count += 1
        eps = 1e-5*np.ones(x.shape) 
        prob = np.zeros(x.shape) 
        cdfs = []
        for i in range(len(n_means)):
            cdfs.append(np.maximum(eps,norm.cdf(x, target_means[i], np.sqrt(target_vars[i]))))
            prob += weights[i] * norm.pdf(x,bar_means[i],np.sqrt(bar_vars[i])) / cdfs[i]
        prob*=np.prod(np.array(cdfs),axis=0)

        z = np.sum(prob)*interval 
        prob = prob/z # Normalization
        if count > 50:
            delta += 0.01
            plt.plot(x,prob,'bx'); plt.show();
            pdb.set_trace()
        if (prob[0] > 0.1):
            x = np.arange(x[0]-delta, x[-1]-delta, interval)
        elif (prob[-1]>0.1):
            x = np.arange(x[0]+delta, x[-1]+delta, interval)
        else:
            done = True
        
    max_mean = np.inner(x,prob)*interval
    max_var = np.inner(x**2,prob)*interval- max_mean**2 
    return max_mean ,max_var, (x,prob)

## Moved to here on April 27th.
def posterior_numeric(n_means, n_vars, c_mean, c_var, rew, dis, terminal, num_interval=500, width = 6.0):
    # ADFQ-Numeric
    # Not for Batch
    if terminal:
        new_var = 1.0/(1.0/c_var + 1.0/REW_VAR)
        new_sd = np.sqrt(new_var)
        new_mean = new_var*(c_mean/c_var + rew/REW_VAR)
        interval = width*new_sd/float(num_interval)
        x = np.arange(new_mean-0.5*width*new_sd,new_mean+0.5*width*new_sd, interval)
        return new_mean, new_var, (x, norm.pdf(x, new_mean, new_sd))
    pdb.set_trace()
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    target_vars = dis*dis*np.array(n_vars, dtype = np.float32)

    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars) 

    add_vars = c_var+target_vars
    mean_range = np.concatenate((target_means,[c_mean]))
    sd_range = np.sqrt(np.concatenate((dis*dis*n_vars, [c_var])))

    interval = width*max(sd_range)/float(num_interval)
    x = np.arange(min(mean_range-0.5*width*sd_range), max(mean_range+0.5*width*sd_range), interval)

    eps = 1e-5*np.ones(x.shape)     
    log_prob = np.sum([np.log(np.maximum(eps,norm.cdf(x,target_means[i], np.sqrt(target_vars[i])))) for i in range(len(n_means))], axis=0) \
         + logsumexp([-0.5*np.log(add_vars[i]) \
        -0.5*(c_mean-target_means[i])**2/add_vars[i]-0.5*np.log(bar_vars[i]) \
        -0.5*(x-bar_means[i])**2/bar_vars[i] - np.log(np.maximum(eps, norm.cdf(x,target_means[i],np.sqrt(target_vars[i])))) for i in range(len(n_means))], axis=0)
    prob = np.exp(log_prob-max(log_prob))
    prob = prob/(interval*np.sum(prob))
    new_mean = interval*np.inner(x, prob)
    new_var = interval*np.inner((x-new_mean)**2, prob)
    
    return new_mean, new_var, (x, prob)
def posterior_approx(n_means, n_vars, c_mean, c_var, rew, dis, terminal, hard_approx = True, varTH = 1e-3, batch=False):
    if batch:
        batch_size = len(n_means)
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_var = np.reshape(c_var, (batch_size,1))
        rew = np.reshape(rew, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    target_vars = dis*dis*np.array(n_vars, dtype = np.float32)

    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var+target_vars
    
    log_weights = -0.5*np.log(add_vars) -(c_mean-target_means)**2/2.0/add_vars #(batch_size X anum)
    weights = np.exp(log_weights - np.max(log_weights, axis=int(batch), keepdims=batch))
    weights = weights/np.sum(weights, axis=int(batch))
    mean_new = np.sum(weights*bar_means, axis = int(batch))
    if hard_approx :
        var_new = np.sum(weights*bar_vars, axis=int(batch))
    else:
        var_new = np.dot(weights,bar_vars+bar_means**2) - mean_new**2

    var_new  = (1.-terminal)*var_new + terminal*1./(1./c_var + 1./REW_VAR)
    mean_new = (1.-terminal)*mean_new + terminal*var_new*(c_mean/c_var + rew/REW_VAR)
    if np.isnan(mean_new).any() or np.isnan(var_new).any():
        pdb.set_trace()
    return mean_new, var_new, (bar_means, bar_vars, weights)

def maxGaussian_adf(n_means, n_vars, c_mean, c_var, rew, dis, num_interval=300, width = 16.0):
    c_sd = np.sqrt(c_var)
    n_sds = np.sqrt(n_vars)
    """
    Designed for ADF approach
    INPUT:
        n_means: a numpy array of Gaussian mean values of (next state, action) pairs for all available actions.
        n_sds: a numpy array of Gaussian SD values of (next state,action) pairs for all available actions.
        c_mean: a mean value of the current state and action
        c_sd: a SD value of the current state and action
    obs:
        mean and variance of the joint distribution
    """
    delta = 0.05
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    target_vars = dis*dis*np.array(n_vars, dtype = np.float32)

    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)

    c_tmp = -(c_mean-target_means)**2/2.0/(c_var + target_vars)
    c_min = np.amin(c_tmp, axis=0)
    c_max = np.amax(c_tmp, axis=0)
    B = float(c_max-c_min < 50.0)
    weights = np.exp(c_tmp - c_min*B - c_max*(1-B))/np.sqrt(c_var+target_vars)
    weights = weights/np.sum(weights, axis =0)
    interval = width*max(np.sqrt(bar_vars))/float(num_interval)
    x = np.arange(min(bar_means-0.5*width*np.sqrt(bar_vars)),max(bar_means + 0.5*width*np.sqrt(bar_vars)),interval) # 501X1

    count = 0 
    done = False
    while(not(done)):
        count += 1
        eps = 1e-5*np.ones(x.shape) 
        prob = np.zeros(x.shape) 
        cdfs = []
        for i in range(len(n_means)):
            cdfs.append(np.maximum(eps,norm.cdf(x, target_means[i], np.sqrt(target_vars[i]))))
            prob += weights[i] * norm.pdf(x,bar_means[i],np.sqrt(bar_vars[i])) / cdfs[i]
        prob*=np.prod(np.array(cdfs),axis=0)

        z = np.sum(prob)*interval 
        prob = prob/z # Normalization
        if count > 50:
            delta += 0.01
            plt.plot(x,prob,'bx'); plt.show();
            pdb.set_trace()
        if (prob[0] > 0.1):
            x = np.arange(x[0]-delta, x[-1]-delta, interval)
        elif (prob[-1]>0.1):
            x = np.arange(x[0]+delta, x[-1]+delta, interval)
        else:
            done = True
        
    max_mean = np.inner(x,prob)*interval
    max_var = np.inner(x**2,prob)*interval- max_mean**2 
    return max_mean ,max_var, (x,prob)

def maxGaussian_smVar(n_means, n_vars, c_mean, c_var, rew, dis, bias_rate = 0.0):
    #k = np.argmax(n_means)
    #max_mean, max_var, _ = prod_of2Gaussian(c_mean, c_var, rew+dis*n_means[k], dis*dis*n_vars[k]) dt
    anum = len(n_means)
    bars = []
    w_vals = []
    for i in range(anum):
        tmp_mean = rew+dis*n_means[i]
        tmp_var = dis*dis*n_vars[i]
        m,v, _ = prod_of2Gaussian(c_mean, c_var, tmp_mean, tmp_var)
        #bias_rate = norm.cdf(c_mean,tmp_mean,np.sqrt(c_var+tmp_var))
        bars.append( (m, 1/(1/v- 0.1*bias_rate/c_var) ) )
        w_vals.append(-((c_mean-tmp_mean)**2)/2/(c_var+tmp_var))
    min_val = min(w_vals)
    max_val = max(w_vals)
    if max_val-min_val < 500:
        weights = [np.exp(w_vals[i]-min_val)/np.sqrt(c_var+dis*dis*n_vars[i]) for i in range(anum)]
    else: 
        weights = [np.exp(w_vals[i]-max_val)/np.sqrt(c_var+dis*dis*n_vars[i]) for i in range(anum)]
    
    mean_new = sum([weights[i]*bars[i][0] for i in range(anum)])/float(sum(weights))
    if np.isnan(mean_new):
        pdb.set_trace()
    return mean_new, max(0.0001,sum([weights[i]*bars[i][1] for i in range(anum)])/float(sum(weights)))

def posterior_approx(n_means, n_vars, c_mean, c_var, rew, dis, terminal, hard_approx = True, varTH = 1e-3, batch=False):
    if batch:
        batch_size = len(n_means)
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_var = np.reshape(c_var, (batch_size,1))
        rew = np.reshape(rew, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))
    target_means = rew + dis*np.array(n_means, dtype=np.float32)
    target_vars = dis*dis*np.array(n_vars, dtype = np.float32)

    bar_vars = 1.0/(1.0/c_var + 1.0/target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var+target_vars

    c_tmp = -(c_mean-target_means)**2/2.0/add_vars
    c_min = np.amin(c_tmp, axis=int(batch), keepdims = batch)
    c_max = np.amax(c_tmp, axis=int(batch), keepdims = batch)
    B = (c_max - c_min < 50.0).astype(np.float32) if batch else float(c_max-c_min < 50.0)
    weights = np.exp(c_tmp - c_min*B - c_max*(1-B))/np.sqrt(add_vars)
    weights = weights/np.sum(weights, axis = int(batch))
    #Z = np.sum(weights, axis = int(batch))
   
    log_weights = -0.5*np.log(add_vars) -(c_mean-target_means)**2/2.0/add_vars #(batch_size X anum)
    weights02 = np.exp(log_weights - np.max(log_weights, axis=int(batch), keepdims=batch))
    weights02 = weights/np.sum(weights, axis=int(batch))
    if(weights.astype(np.float16) != weights02.astype(np.float16)).any():
        print(weights)
        print(weights02)
        pdb.set_trace()

    mean_new = np.sum(weights*bar_means, axis = int(batch))
    if hard_approx :
        var_new = np.sum(weights*bar_vars, axis=int(batch))
    else:
        var_new = np.dot(weights,bar_vars+bar_means**2) - mean_new**2

    var_new  = (1.-terminal)*var_new + terminal*1./(1./c_var + 1.)
    mean_new = (1.-terminal)*mean_new + terminal*var_new*(c_mean/c_var + rew)
    if np.isnan(mean_new).any() or np.isnan(var_new).any():
        pdb.set_trace()
    return mean_new, var_new 

def maxGaussian_plot(mean_init,var_init,anum, rew, dis):
    mean = np.array([10,20,mean_init])
    var = np.array([0.01,0.01,0.01])#,0.01,0.01])
    #var = var_init*np.ones((1,anum+1))[0]
    #mean = mean_init*np.ones((1,anum+1))[0] 
    #var = var_init*np.ones((1,anum+1))[0]
    #mean = mean_init+np.random.rand(1,anum+1)[0]
    #var = var_init+np.random.rand(1,anum+1)[0]
    mean_bar = []
    for i in range(anum):
        m,_, _ = prod_of2Gaussian(mean[-1], var[-1], rew+dis*mean[i], dis*dis*var[i])
        mean_bar.append(m)

    print("mean:",mean)
    print("var:",var)
    print("estimated means", [rew+dis*x for x in mean[:-1]])
    print("mean bars", mean_bar)

    norm_m, norm_v, (x,p) = maxGaussian_adf(mean[:-1], var[:-1], mean[-1], var[-1], rew, dis,width=100.0)
    sm_approx_m, sm_approx_v = maxGaussian_smVar(mean[:-1], var[:-1], mean[-1], var[-1], rew, dis)
    k = np.argmax(mean[:-1])
    approx_m, approx_v , _ = prod_of2Gaussian(mean[-1], var[-1], rew+ dis*mean[k], dis*dis*var[k])
    print approx_m, approx_v
    f1, ax1 = plt.subplots()
    ax1.plot(x,p,'r2')
    ax1.plot(x,norm.pdf(x,norm_m, np.sqrt(norm_v)),'g1')
    ax1.plot(x,norm.pdf(x,approx_m, np.sqrt(approx_v)),'bx')
    ax1.plot(x,norm.pdf(x,sm_approx_m, np.sqrt(sm_approx_v)),'k3')
    ax1.legend(['true','numeric adf', 'approx adf','approx adf sm'])

    plt.show()


def prod_of2Gaussian(mu1,var1,mu2,var2):
    """
    Input: 
    """
    var1 = float(var1)
    var2 = float(var2)
    var12 = 1/(1/var1+1/var2)
    mu12 = var12*(mu1/var1 + mu2/var2)
    #C12 = (mu1**2)/var1 + (mu2**2)/var2 - (mu12**2)/var12
    C12 = (mu2-mu1)*(mu2-mu1)/(var1+var2)
    return mu12, var12, C12

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

def plot_V_pi(obj, Q):
    Q = Q.reshape(obj.snum, obj.anum)
    V = np.max(Q,1)
    pi = np.argmax(Q,1)
    if obj.name == ('maze' or 'minimaze'): # For Obstacles
        Pi = 4*np.ones(obj.dim)
        grid_map = obj.map_img
        for i in range(obj.snum):  
            Pi[obj.idx2cell[i]] = pi[i]
            grid_map[obj.idx2cell[i]] = V[i]
    else:
        Pi = pi.reshape(obj.dim)
        grid_map = V.reshape(obj.dim)

    plt.figure(figsize=(obj.dim[1],obj.dim[0]))
    plt.imshow(grid_map, cmap='gray', interpolation='nearest')
    ax = plt.gca()
    ax.set_xticks(np.arange(obj.dim[1]) - .5)
    ax.set_yticks(np.arange(obj.dim[0]) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #Y, X = np.mgrid[0:obj.dim[0], 0:obj.dim[1]]
    a2uv = {0: (0,1), 1: (0,-1), 2: (-1, 0), 3: (1, 0), 4:(0, 0)} # xy coordinate not matrix
    for y in range(obj.dim[0]):
        for x in range(obj.dim[1]):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y, u * .3, v * .3, color='m',
                      head_width=0.1, head_length=0.1)

    plt.text(obj.start_pos[1], obj.dim[0] - 1 - obj.start_pos[0], "S",
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.text(obj.goal_pos[1], obj.dim[0] -1 - obj.goal_pos[0], "G",
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')

    plt.show()

def plot_to_save(T, ys, labels, save, x_name, y_name, shadow = True, legend=(True, (1,1)), pic_name = None, colors=None,):
    if not(colors):
        colors = ['r','b','g','k','c','m','y','burlywood','chartreuse','0.8','--', '-.', ':']
    plot_err = []
    f1, ax1 = plt.subplots()
    if len(ys.shape)==2:
        for (i,y) in enumerate(ys):
            tmp, = ax1.plot(y, colors[i], label=labels[i], linewidth=2.0)
            plot_err.append(tmp)
    else:
        ts = range(0,T-1,T/50)
        for (i,y) in enumerate(ys):
            m, ids25, ids75  = iqr(y)
            tmp, = ax1.plot(ts[1:] ,m, colors[i], label=labels[i], linewidth=2.0)
            plot_err.append(tmp) 
            if shadow:       
                ax1.fill_between(ts[1:], ids75, ids25, facecolor=colors[i], alpha=0.15)

    if legend[0]:
        ax1.legend(plot_err, labels ,loc='lower left', bbox_to_anchor=legend[1],fontsize=25, shadow=True,
                    prop={'family':'Times New Roman'})
    ax1.tick_params(axis='both',which='major',labelsize=20)
    ax1.set_xlabel(x_name,fontsize=25, fontname="Times New Roman")
    ax1.set_ylabel(y_name,fontsize=25, fontname="Times New Roman")
    #ax1.set_ylim((0,1.2))
    
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










