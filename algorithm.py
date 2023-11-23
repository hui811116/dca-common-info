import numpy as np
import sys
import os
#import gradient_descent as gd
import utils as ut
import copy
from scipy.special import softmax, xlogy
import itertools


'''
# compared algorithms
# gradient based methods
# Sula, E.; Gastpar, M.C. Common Information Components Analysis. Entropy 2021, 23, 151.
# https://doi.org/10.3390/e23020151

# in its relaxed Wyner common information, the gradients are taken w.r.t. the (conditional) mutual information
# which can be equivalently expressed as derivative to a combination of entropy and conditional entropy functions.

def stoGradComp(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale = kwargs['ss_scale']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

	#mask_pzcx1x2 = np.ones(pzcx1x2.shape)
	itcnt =0 
	conv_flag = False
	while itcnt < maxiter:
		itcnt += 1
		# auxiliary variables
		pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		# calculate the gradient
		grad_p = ((1+gamma) * (np.log(pzcx1x2)+1)) * px1x2[None,:,:]\
					-(1-gamma) *( (np.log(pz) + 1)[:,None,None] * px1x2[None,:,:])\
					-gamma * (np.log(pzcx1) +1)[:,:,None] * px1x2[None,:,:] \
					-gamma * (np.log(np.repeat(np.expand_dims(pzcx2,axis=1),repeats=nx1,axis=1))+1)*px1x2[None,:,:]
		mean_grad_p = grad_p - np.mean(grad_p,axis=0,keepdims=True)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - mean_grad_p * ss_p

		# the compared method project the obtained encoder to the wyner setting
		#new_pz = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		#ent_z = -np.sum(new_pz*np.log(new_pz))
		#ent_pzcx1x2 = -np.sum(new_pzcx1x2 * px1x2[None,:,:] * np.log(new_pzcx1x2))
		# the convergence criterion of the reference uses
		conv_z = 0.5 * np.sum(np.fabs(new_pzcx1x2 - pzcx1x2),axis=0) # total variation
		if np.all(conv_z<convthres):
			conv_flag=True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2}
'''

# dev
# scalable splitting method based wyner common information solver.
# 
'''
def wynerAM(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	d_seed = kwargs.get("seed",None)
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		px1cz = kwargs['px1cz']
		px2cz = kwargs['px2cz']
	else:
		px1cz = rng.random((nx1,nz))
		px1cz /= np.sum(px1cz,axis=0,keepdims=True)
		px2cz = rng.random((nx2,nz))
		px2cz /= np.sum(px2cz,axis=0,keepdims=True)
	def expandProb(pzcx,adim,ndim):
		# return (nz,x1,x2)
		return np.repeat(np.expand_dims(pzcx,axis=adim),repeats=ndim,axis=adim)
	def calcPZXV(pz,px1cz,px2cz):
		expand_pzcx1 = expandProb(px1cz.T,adim=2,ndim=nx2)
		expand_pzcx2 = expandProb(px2cz.T,adim=1,ndim=nx1)
		raw_prob = expand_pzcx1 * expand_pzcx2 * pz[:,None,None]
		return raw_prob /np.sum(raw_prob,keepdims=True)
	def calcPXV(px1cz,px2cz):
		zdim = px1cz.shape[-1]
		joint_prob = calcPZXV((1/zdim)*np.ones((zdim,)).astype("float64"),px1cz,px2cz)
		raw_prob = np.sum(joint_prob,axis=0)
		return raw_prob/np.sum(raw_prob,keepdims=True) # make sure it is a valid prob
	def calcDKL(pxcz,pxcy):
		ndim = pxcy.shape[-1]
		zdim = pxcz.shape[-1]
		exp_pxycz = np.repeat(np.expand_dims(pxcz,axis=1),repeats=ndim,axis=1)
		exp_pxcyz = np.repeat(np.expand_dims(pxcy,axis=-1),repeats=zdim,axis=-1)+1e-8
		return np.sum(xlogy(exp_pxycz,exp_pxycz/exp_pxcyz),axis=0)
	def calcLoss(pzx1x2,true_px1x2,beta):
		cur_px1x2 = np.sum(pzx1x2,axis=0)
		cur_mi = ut.calcMI(cur_px1x2)
		cur_dkl = np.sum(cur_px1x2*np.log(cur_px1x2/true_px1x2))
		return cur_mi + beta * cur_dkl
	itcnt = 0
	conv_flag = False
	cur_pzx1x2 = calcPZXV((1/nz)*np.ones((nz,)).astype("float64"),px1cz,px2cz)
	cur_loss = calcLoss(cur_pzx1x2,px1x2,gamma)
	while itcnt < maxiter:
		itcnt +=1
		est_px1x2 = calcPXV(px1cz,px2cz)
		est_px1 = np.sum(est_px1x2,axis=1)
		est_px2cx1 = est_px1x2.T/est_px1[None,:]
		dkl_for_x1 = calcDKL(px2cz,px2cx1)
		dkl_est_x1 = calcDKL(px2cz,est_px2cx1)

		v1_expo = gamma * np.log(px1/est_px1)[:,None]-(1+gamma)*dkl_for_x1 + gamma * dkl_est_x1
		v1_expo -= np.amax(v1_expo,axis=0)
		new_px1cz = px1[:,None] * np.exp(v1_expo) + 1e-8
		new_px1cz /= np.sum(new_px1cz,axis=0,keepdims=True)

		est_px1x2 = calcPXV(new_px1cz,px2cz)
		est_px2 = np.sum(est_px1x2,axis=0)
		est_px1cx2 = est_px1x2 / est_px2[None,:]
		dkl_for_x2 = calcDKL(new_px1cz,px1cx2)
		dkl_est_x2 = calcDKL(new_px1cz,est_px1cx2)

		v2_expo = gamma * np.log(px2/est_px2)[:,None]-(1+gamma)*dkl_for_x2 + gamma * dkl_est_x2
		v2_expo -= np.amax(v2_expo,axis=0)
		new_px2cz = px2[:,None] * np.exp(v2_expo) + 1e-8
		new_px2cz /= np.sum(new_px2cz,axis=0,keepdims=True)

		#dtv_1 = 0.5 * np.sum(np.fabs(new_px1cz - px1cz),axis=0)
		#dtv_2 = 0.5 * np.sum(np.fabs(new_px2cz - px2cz),axis=0)
		# compute the loss value for convergence criterion
		est_pzx1x2 = calcPZXV((1/nz)*np.ones((nz,)).astype("float64"),new_px1cz,new_px2cz)
		est_loss = calcLoss(est_pzx1x2,px1x2,gamma)
		# DTV convergence criterion
		#if np.all(dtv_1 < convthres) and np.all(dtv_2 < convthres):
		if np.fabs(cur_loss - est_loss) < convthres:
			conv_flag =True
			break
		else:
			#prev_px1cz = px1cz
			#prev_px2cz = px2cz
			px1cz = new_px1cz
			px2cz = new_px2cz
			# current loss
			cur_pzx1x2 = est_pzx1x2
			cur_loss = est_loss
	# est pzx1x1
	pzx1x2 = calcPZXV((1/nz)*np.ones((nz,)).astype("float64"),px1cz,px2cz)
	pzcx1x2 = pzx1x2 / np.sum(pzx1x2,axis=0,keepdims=True)
	pz = np.sum(pzx1x2,axis=(1,2))
	pzx1 = np.sum(pzx1x2,axis=2)
	pzcx1 = pzx1 / np.sum(pzx1,axis=0,keepdims=True)
	pzx2 = np.sum(pzx1x2,axis=1)
	pzcx2 = pzx2 / np.sum(pzx2,axis=0,keepdims=True)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"est_pzx1x2":pzx1x2}
'''
# FIXME: checking the broadcast function works properly
def wynerAMnV(pxv,nz,gamma,maxiter,convthres,**kwargs):
	d_seed = kwargs.get("seed",None)
	rng = np.random.default_rng(d_seed)
	nx_shape = pxv.shape
	nview = len(pxv.shape)
	# assumption: uniformly Z
	pz = np.ones((nz,)) * (1/nz)
	px_list, p_cond = ut.computeNvPriors(pxv)
	if "init_load" in kwargs.keys():
		pxcz_list = kwargs['init_load']
	else:
		# random initialization
		pxcz_list =[]
		for vidx in range(nview):
			tmp_pxcz = rng.random((nx_shape[vidx],nz))
			pxcz_list.append(tmp_pxcz/np.sum(tmp_pxcz,axis=0,keepdims=True))
	def compute_loss(pxvz,pz,beta):
		# MILOSS
		entz = -np.sum(pz*np.log(pz))
		pzcxv_rev = pxvz / np.sum(pxvz,axis=-1,keepdims=True)
		entz_cxv = -np.sum(pxvz*np.log(pzcxv_rev))
		mi_loss = entz - entz_cxv
		# dkl
		est_pxv = np.sum(pxvz,axis=-1) # nxv
		dkl_xv = np.sum(xlogy(est_pxv,est_pxv/pxv))
		return mi_loss + beta * dkl_xv
	itcnt = 0
	conv_flag = False
	cur_pxvz = ut.calcPXVZ(pxcz_list,pz)
	cur_loss  = compute_loss(cur_pxvz,pz,gamma)
	while itcnt<maxiter:
		itcnt +=1
		for vidx in range(nview):
			tmp_px = px_list[vidx]
			tmp_pwcx = p_cond[vidx] # the last is vidx
			tmp_pwz = ut.calcPXWZ(pxcz_list,pz,vidx)# exclude vidx
			tmp_pwcz = np.squeeze(tmp_pwz/np.sum(tmp_pwz,axis=-1,keepdims=True))
			# (nx^{D-1},nz)
			# DKL exponent
			dkl_zxi = ut.computeDKL(tmp_pwcz,tmp_pwcx) # (nx,nz)
			est_pxv = ut.calcPXV(pxcz_list,pz)
			# transpose
			trs_seq = tuple([item for item in range(nview) if item != vidx]+[vidx])
			est_pwcx = np.transpose(est_pxv,axes=trs_seq)
			est_px = np.sum(est_pwcx,axis=tuple(np.arange(nview-1)))
			est_pwcx = est_pwcx / np.sum(est_pwcx,axis=tuple(np.arange(nview-1)),keepdims=True)
			# compute the second exponent
			bdiv_ratio = np.expand_dims(tmp_pwcx/est_pwcx,axis=-1)
			bdiv = np.sum(xlogy(np.expand_dims(tmp_pwcz,axis=-2),bdiv_ratio),axis=tuple(np.arange(nview-1)))
			# both (dkl_zxi and bdiv should be in (nx,nz) shape)
			# the logpx ratio should be included as well
			px_llr = np.log(tmp_px/est_px) # should be (nx,)
			new_expo = -dkl_zxi + gamma * bdiv + gamma * px_llr[:,None] # (nx,nz) for all 
			# smoothing
			new_expo -= np.amax(new_expo,axis=0)
			new_pxcz = tmp_px[:,None]* np.exp(new_expo)+1e-8 # smooth epsilon
			new_pxcz /= np.sum(new_pxcz,axis=0,keepdims=True)
			# put into the list 
			pxcz_list[vidx] = new_pxcz # update here and the subsequent steps will use new pxcz
		# pxcz_list now is a new list
		# check convergence
		new_pxvz = ut.calcPXVZ(pxcz_list,pz)
		new_loss = compute_loss(new_pxvz,pz,gamma)

		if np.fabs(new_loss - cur_loss)<convthres:
			conv_flag = True
			break
		else:
			'''
			if itcnt > 5000 and itcnt<5005:
				print("itcnt {:}".format(itcnt))
				print(cur_loss)
				print(new_loss)
				print('diff',np.fabs(new_loss - cur_loss))
				print([item for item in pxcz_list])
				#sys.exit()
			elif itcnt>5010:
				return {"error":True}
			'''
			cur_loss = new_loss
			cur_pxvz = new_pxvz
	pzcxv = ut.calcPZcXV(pxcz_list,pz)
	tr_seq = tuple([nview]+list(np.arange(0,nview)))
	est_pzxv = np.transpose(new_pxvz,axes=tr_seq)
	return {"conv":conv_flag,"niter":itcnt,"pzcxv":pzcxv,"pz":pz,"pxcz_list":pxcz_list,"est_pzxv":est_pzxv}

# VI algorithm
'''
def wynerVInV(pxv,nz,gamma,maxiter,convthres,**kwargs):
	d_seed = kwargs.get("seed",None)
	rng = np.random.default_rng(d_seed)
	nx_shape = pxv.shape
	nview = len(pxv.shape)
	# assumption: uniformly Z
	pz = np.ones((nz,)) * (1/nz)
	px_list, p_cond = ut.computeNvPriors(pxv)
	if "init_load" in kwargs.keys():
		pxcz_list = kwargs['init_load']
	else:
		# random initialization
		pxcz_list =[]
		for vidx in range(nview):
			tmp_pxcz = rng.random((nx_shape[vidx],nz))
			pxcz_list.append(tmp_pxcz/np.sum(tmp_pxcz,axis=0,keepdims=True))
	def compute_loss(pxvz,pz,beta):
		# MILOSS
		entz = -np.sum(pz*np.log(pz))
		pzcxv_rev = pxvz / np.sum(pxvz,axis=-1,keepdims=True)
		entz_cxv = -np.sum(pxvz*np.log(pzcxv_rev))
		mi_loss = entz - entz_cxv
		# dkl
		est_pxv = np.sum(pxvz,axis=-1) # nxv
		dkl_xv = np.sum(xlogy(est_pxv,est_pxv/pxv))
		return mi_loss + beta * dkl_xv
	itcnt = 0
	conv_flag = False
	cur_pxvz = ut.calcPXVZ(pxcz_list,pz)
	cur_loss  = compute_loss(cur_pxvz,pz,gamma)
	while itcnt<maxiter:
		itcnt +=1
		for vidx in range(nview):
			#tmp_px = px_list[vidx]
			#tmp_pwcx = p_cond[vidx] # the last is vidx
			tmp_pwz = ut.calcPXWZ(pxcz_list,pz,vidx)# exclude vidx
			tmp_pwcz = np.squeeze(tmp_pwz/np.sum(tmp_pwz,axis=-1,keepdims=True))
			# (nx^{D-1},nz)
			# DKL exponent
			#dkl_zxi = ut.computeDKL(tmp_pwcz,tmp_pwcx) # (nx,nz)
			est_pxv = ut.calcPXV(pxcz_list,pz)
			# transpose
			
			# smoothing
			new_expo -= np.amax(new_expo,axis=0)
			new_pxcz = np.exp(new_expo)+1e-8 # smooth epsilon
			new_pxcz /= np.sum(new_pxcz,axis=0,keepdims=True)
			# put into the list 
			pxcz_list[vidx] = new_pxcz # update here and the subsequent steps will use new pxcz
		# pxcz_list now is a new list
		# check convergence
		new_pxvz = ut.calcPXVZ(pxcz_list,pz)
		new_loss = compute_loss(new_pxvz,pz,gamma)

		if np.fabs(new_loss - cur_loss)<convthres:
			conv_flag = True
			break
		else:
			#if itcnt > 5000 and itcnt<5005:
			#	print("itcnt {:}".format(itcnt))
			#	print(cur_loss)
			#	print(new_loss)
			#	print('diff',np.fabs(new_loss - cur_loss))
			#	print([item for item in pxcz_list])
			#	#sys.exit()
			#elif itcnt>5010:
			#	return {"error":True}
			cur_loss = new_loss
			cur_pxvz = new_pxvz
	pzcxv = ut.calcPZcXV(pxcz_list,pz)
	tr_seq = tuple([nview]+list(np.arange(0,nview)))
	est_pzxv = np.transpose(new_pxvz,axes=tr_seq)
	return {"conv":conv_flag,"niter":itcnt,"pzcxv":pzcxv,"pz":pz,"pxcz_list":pxcz_list,"est_pzxv":est_pzxv}
'''
# gradient-based optimizer
# alternating minimization solver
# Proposed in: 
# T. -H. Huang and H. El Gamal, "Efficient Alternating Minimization Solvers 
# for Wyner Multi-View Unsupervised Learning," 2023 IEEE International Symposium
# on Information Theory (ISIT), Taipei, Taiwan, 2023, pp. 707-712,
# doi: 10.1109/ISIT54713.2023.10206810.
'''
def wynerDrs(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	#penalty = kwargs['penalty_coeff']
	# no penalty coefficient for this method
	# gamma is the beta in the paper
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		#pzcx1x2 = kwargs['pzcx1x2']
		px1cz = kwargs['px1cz']
		px2cz = kwargs['px2cz']
	else:
		px1cz = rng.random((nx1,nz))
		px1cz /= np.sum(px1cz,axis=0,keepdims=True)
		px2cz = rng.random((nx2,nz))
		px2cz /= np.sum(px2cz,axis=0,keepdims=True)
		# NOTE: Assume p(z) is uniformly distributed 
		# initial pzcx1x2 is the softmax of the product px1cz, px2cz
	def expandLogPxcz(log_pxxcz,adim,ndim):
		# return dim (z,x1,x2)
		return np.repeat(np.expand_dims(log_pxxcz.T,axis=adim),repeats=ndim,axis=adim)
	def calcProdProb(log_px1cz,log_px2cz,nz):
		expand_log_px1cz = expandLogPxcz(log_px1cz,adim=2,ndim=nx2)
		expand_log_px2cz = expandLogPxcz(log_px2cz,adim=1,ndim=nx1)
		return np.exp(expand_log_px1cz+expand_log_px2cz)/nz
	def calcPx1x2(log_px1cz,log_px2cz,nz):
		pzx1x2 = calcProdProb(log_px1cz,log_px2cz,nz)
		return np.sum(pzx1x2,axis=0)
	def calcDtvError(log_px1cz,log_px2cz,nz,px1x2):
		pzx1x2 = calcProdProb(log_px1cz,log_px2cz,nz)
		est_px1x2 = np.sum(pzx1x2,axis=0)
		return 0.5 * np.sum(np.fabs(est_px1x2 - px1x2))
	def calcProbSoftmax(log_px1cz,log_px2cz):
		expand_log_px1cz = expandLogPxcz(log_px1cz,adim=2,ndim=nx2)
		expand_log_px2cz = expandLogPxcz(log_px2cz,adim=1,ndim=nx1)
		return softmax(expand_log_px1cz+expand_log_px2cz,axis=0)
	def calcCondMi(log_px1cz,log_px2cz,nz):
		pzx1x2 = calcProdProb(log_px1cz,log_px2cz,nz)		
		return ut.calcMIcond(np.transpose(pzx1x2,(1,2,0)))
	log_px1cz = np.log(px1cz)
	log_px2cz = np.log(px2cz)
	# helping constants
	const_logpx1x2 = np.log(px1x2)
	itcnt =0 
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		est_px1cz = np.exp(log_px1cz)
		est_px2cz = np.exp(log_px2cz)
		est_px1x2 = calcPx1x2(log_px1cz,log_px2cz,nz)
		dtv_error = 0.5 * np.sum(np.fabs(est_px1x2 - px1x2))
		pzcx1x2 = calcProbSoftmax(log_px1cz,log_px2cz)
		# grad1
		pzx1x2  = calcProdProb(log_px1cz,log_px2cz,nz)
		grad_x1 = (1/nz) * np.exp(log_px1cz) * (1+log_px1cz) - np.sum(pzx1x2 * (const_logpx1x2[None,:,:]),axis=2).T
		grad_x1 += gamma * np.sum(-px1x2[None,:,:] * pzcx1x2,axis=2).T
		new_log_px1cz = gd.logProb2DProj(log_px1cz,grad_x1,ss_fixed)

		# update
		pzx1x2 = calcProdProb(new_log_px1cz,log_px2cz,nz)
		pzcx1x2 = calcProbSoftmax(new_log_px1cz,log_px2cz)
		est_px1x2 = calcPx1x2(new_log_px1cz,log_px2cz,nz)
		# gradient v2
		grad_x2 = (1/nz) * np.exp(log_px2cz) * (1+log_px2cz) - np.sum(pzx1x2 * (const_logpx1x2[None,:,:]),axis=1).T
		grad_x2 += gamma * np.sum(-px1x2[None,:,:]*pzcx1x2,axis=1).T

		new_log_px2cz = gd.logProb2DProj(log_px2cz,grad_x2,ss_fixed)
		# update
		new_px1cz = np.exp(new_log_px1cz)
		new_px2cz = np.exp(new_log_px2cz)
		# dtv for the log likelihoods
		dtv_px1cz = 0.5* np.sum(np.fabs(est_px1cz- new_px1cz),axis=0)
		dtv_px2cz = 0.5* np.sum(np.fabs(est_px2cz- new_px2cz),axis=0)
		dtv_error = calcDtvError(new_log_px1cz,new_log_px2cz,nz,px1x2)
		# add relaxed condition
		log_px1cz = new_log_px1cz
		log_px2cz = new_log_px2cz
		# debugging blocks
		
		if np.all(dtv_px1cz<convthres) and np.all(dtv_px2cz<convthres):
			conv_flag = True
			break
		
	# est pzx1x1
	pzx1x2 = calcProdProb(log_px1cz,log_px2cz,nz)
	pzcx1x2 = pzx1x2 / np.sum(pzx1x2,axis=0,keepdims=True)
	pz = np.sum(pzx1x2,axis=(1,2))
	pzx1 = np.sum(pzx1x2,axis=2)
	pzcx1 = pzx1 / np.sum(pzx1,axis=0,keepdims=True)
	pzx2 = np.sum(pzx1x2,axis=1)
	pzcx2 = pzx2 / np.sum(pzx2,axis=0,keepdims=True)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"est_pzx1x2":pzx1x2}
'''
# difference of convex approach
'''
def wynerDCA(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))+1e-8
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	# helper
	def expandLogPxcz(log_pxxcz,adim,ndim):
		# return dim (z,x1,x2)
		return np.repeat(np.expand_dims(log_pxxcz.T,axis=adim),repeats=ndim,axis=adim)
	def calcProbSoftmax(log_px1cz,log_px2cz,log_pz,gamma):
		expand_log_px1cz = expandLogPxcz(log_px1cz,adim=2,ndim=nx2)
		expand_log_px2cz = expandLogPxcz(log_px2cz,adim=1,ndim=nx1)
		exponent = gamma * (expand_log_px1cz+expand_log_px2cz) + log_pz[:,None,None]
		return softmax(exponent - np.amax(exponent,axis=0),axis=0)
	def calcLoss(pzx1x2,gamma):
		cur_pz = np.sum(pzx1x2,axis=(1,2))
		cur_mi = np.sum(pzx1x2*np.log(pzcx1x2/cur_pz[:,None,None]))
		cur_mix1 = ut.calcMI(np.sum(pzx1x2,axis=2))
		cur_mix2 = ut.calcMI(np.sum(pzx1x2,axis=1))
		return cur_mi - gamma * (cur_mix1+cur_mix2)
	cur_loss = calcLoss(pzcx1x2*px1x2[None,:,:],gamma)
	itcnt =0 
	conv_flag = False
	while itcnt<maxiter:
		itcnt+=1
		# compute logp(z|x1),logp(z|x2)
		pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		
		log_pz = np.log(pz)
		log_px1cz = np.log((pzcx1 * px1[None,:]/(pz[:,None])).T+1e-8)
		log_px2cz = np.log((pzcx2 * px2[None,:]/(pz[:,None])).T+1e-8)

		new_pzcx1x2 = calcProbSoftmax(log_px1cz,log_px2cz,log_pz,gamma)
		# smoothness conditions
		# NOTE: this makes things stable
		new_pzcx1x2 += 1e-8
		new_pzcx1x2 /= np.sum(new_pzcx1x2,axis=0,keepdims=True)

		# align the criterion, when loss does not change much
		#dtv_vec = 0.5 * np.sum(np.fabs(new_pzcx1x2-pzcx1x2),axis=0)
		#if np.all(dtv_vec < convthres):
		new_loss = calcLoss(new_pzcx1x2*px1x2[None,:,:],gamma)
		if np.fabs(new_loss - cur_loss) < convthres:
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			cur_loss = new_loss
	pzx1x2 = pzcx1x2 * px1x2[None,:,:]
	pz = np.sum(pzx1x2,axis=(1,2))
	pzx1 = np.sum(pzx1x2,axis=2)
	pzcx1 = pzx1 / np.sum(pzx1,axis=0,keepdims=True)
	pzx2 = np.sum(pzx1x2,axis=1)
	pzcx2 = pzx2 / np.sum(pzx2,axis=0,keepdims=True)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"est_pzx1x2":pzx1x2}
'''
# Difference of Convex Algorithm of N views
def wynerDCAnV(pxv,nz,gamma,maxiter,convthres,**kwargs):
	# assumption: single-value Lagrange multiplier
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	nx_shape = pxv.shape
	nview = len(pxv.shape)
	# assumption: uniformly Z
	if "init_load" in kwargs.keys():
		pzcxv = kwargs['pzcxv']
	else:
		# random initialization
		pzcxv = rng.random(size=(tuple([nz]+list(pxv.shape)))) + 1e-8
		pzcxv /= np.sum(pzcxv,axis=0,keepdims=True)
	pzxv = pzcxv * pxv[None,...]
	pz = np.sum(pzxv,axis=tuple(np.arange(1,nview+1))) # offset by |Z|
	pz /= np.sum(pz)
	# set sequences
	# store them to save computation time
	# precompute the priors too
	'''
	set_dict = {"uniset":set(np.arange(nview)),"tuple_list":[],'cond_list':[]}
	for widx in range(1,int(np.floor(nview/2)+1)):
		for item in itertools.combinations(set_dict['uniset'],widx):
			cmpl_tuple = tuple(set_dict['uniset']-set(item)) 
			set_dict['tuple_list'].append((item,cmpl_tuple)) # from tuple to set
			pw_wcmpl = pxv / np.sum(pxv,axis=item,keepdims=True)
			pwcmpl_w = pxv / np.sum(pxv,axis=cmpl_tuple,keepdims=True)
			set_dict['cond_list'].append((pw_wcmpl,pwcmpl_w))
	'''
	set_dict = ut.getSetDict(pxv,mi=False)
	# precompute pz
	# precompute all combinations of pzcx:
	# pzcx1, pzcx2, pzcx3,.....
	# pzcx1x2, pzcx1x3, pzcx2x3....
	def compute_loss(pzcxv,pz,pzcx_set,pxv,gamma):
		nview = len(pxv.shape)
		# pzcx_set contains pairs of (pzcx, complment_pzcxw)
		# NOTE: keep shape, so all (z,x1,....,xv)
		# Assumption: same gamma
		pzxv = pzcxv * pxv[None,...]
		ent_zcxv = -np.sum(pzxv * np.log(pzcxv))
		ent_z = -np.sum(pz * np.log(pz))
		mizxv = ent_z- ent_zcxv
		# get combinations of mi
		# computed from conditional entropy ... 
		mi_set = []
		for sidx, (pzw, pzw_cmpl) in enumerate(pzcx_set):
			pzw = np.squeeze(pzw) # for stability
			pzw_cmpl = np.squeeze(pzw_cmpl) # for stability
			pzcw = pzw/np.sum(pzw,axis=0,keepdims=True)
			pzcw_cmpl = pzw_cmpl/np.sum(pzw_cmpl,axis=0,keepdims=True)
			ent_zcw = -np.sum(pzw * np.log(pzcw))
			ent_zcw_cmpl =- np.sum(pzw_cmpl * np.log(pzcw_cmpl))
			mi_sum = ent_z - ent_zcw + ent_z - ent_zcw_cmpl
			mi_set.append(mi_sum)
		return mizxv - gamma/(1+nview*gamma) * sum(mi_set)
	itcnt =0
	conv_flag = False
	#
	pzw_set = ut.getPZWsetDCA(pzxv,set_dict) # shape (nz,nxv)<-keepdims
	pzcx_set = ut.getPZCXsetDCA(pzxv,set_dict)
	cur_loss = compute_loss(pzcxv,pz,pzw_set,pxv,gamma)
	while itcnt < maxiter:
		itcnt+=1
		# update the new pzcxv
		expo_sum = 0
		expd_pz_shape = tuple([nz]+[1]*nview)
		res_pz = np.reshape(pz,expd_pz_shape)
		for pzcx,cmpl_pzcx in pzcx_set:
			expo_sum += gamma * (np.log(pzcx/res_pz) + np.log(cmpl_pzcx/res_pz))
		# smoothing
		expo_sum -= np.amax(expo_sum,axis=0,keepdims=True)
		new_pzcxv = res_pz * np.exp(expo_sum) + 1e-8 # smoothing condition
		new_pzcxv /= np.sum(new_pzcxv,axis=0,keepdims=True)

		new_pzxv = new_pzcxv * pxv[None,...]
		new_pz = np.sum(new_pzxv,axis=tuple(np.arange(1,nview+1)))
		new_pz/=np.sum(new_pz) # for stability
		# new loss
		newpzcx_set = ut.getPZCXsetDCA(new_pzxv,set_dict)
		newpzw_set = ut.getPZWsetDCA(new_pzxv,set_dict)
		new_loss = compute_loss(new_pzcxv,new_pz,newpzw_set,pxv,gamma)
		conv_cond = np.fabs(new_loss - cur_loss)
		if conv_cond<convthres:
			conv_flag = True
			break
		else:
			pzcxv = new_pzcxv
			pz = new_pz
			pzcx_set = newpzcx_set
			cur_loss = new_loss
	# DCA has zero approximation error
	est_pzxv = pzcxv * pxv[None,...]
	return {"conv":conv_flag,"niter":itcnt,"pzcxv":pzcxv,"pz":pz,"est_pzxv":est_pzxv}
