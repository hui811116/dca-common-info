import numpy as np
import sys
import os
import gradient_descent as gd
import utils as ut
import copy

# gradient descent baseline 

# gradient based methods
# Sula, E.; Gastpar, M.C. Common Information Components Analysis. Entropy 2021, 23, 151.
# https://doi.org/10.3390/e23020151

# in its relaxed Wyner common information, the gradients are taken w.r.t. the (conditional) mutual information
# which can be equivalently expressed as derivative to a combination of entropy and conditional entropy functions.
'''
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
# modified from the Sula et. al. baseline
# Equipped with ADMM design
def stoGdDrs(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs["pzcx1x2"]
		q_prob = kwargs["q_prob"]
		dual_p = kwargs["dual_p"]
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
		# augmented variables
		q_prob = copy.deepcopy(pzcx1x2)
		# dual vars
		dual_p = np.zeros(pzcx1x2.shape)
	# gradient masking
	itcnt = 0
	conv_flag= False
	while itcnt < maxiter:
		itcnt +=1
		# auxiliary
		err_p = pzcx1x2 - q_prob
		#
		grad_p = (1+gamma)*(np.log(pzcx1x2)+1) * px1x2[None,:,:] + dual_p + penalty * err_p
		mean_grad_p = grad_p - np.mean(grad_p,axis=0,keepdims=True)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p
		# 
		err_p = new_pzcx1x2 - q_prob
		dual_p += penalty * err_p
		# auxiliary
		qz = np.sum(q_prob * px1x2[None,:,:],axis=(1,2))
		qzcx1 = np.sum(q_prob * (px2cx1.T)[None,:,:],axis=2)
		qzcx2 = np.sum(q_prob * px1cx2[None,:,:],axis=1)
		grad_q = -(1-gamma) * (np.log(qz)[:,None,None]+1) *px1x2[None,:,:] \
				 -gamma * (np.log(qzcx1)[:,:,None]+1) * px1x2[None,:,:]\
				 -gamma * (np.repeat(np.expand_dims(np.log(qzcx2),axis=1)+1,repeats=nx1,axis=1)) * px1x2[None,:,:]\
				 -dual_p - penalty* err_p
		mean_grad_q = grad_q - np.mean(grad_q,axis=0,keepdims=True)
		ss_q = gd.naiveStepSize(q_prob,-mean_grad_q,ss_init,ss_scale)
		if ss_q ==0:
			break
		new_q = q_prob - ss_q * mean_grad_q
		#
		err_p = new_pzcx1x2 - new_q
		conv_p = 0.5 * np.sum(np.fabs(err_p),axis=0)
		if np.all(conv_p<convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			q_prob = new_q
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"q_prob":q_prob,"dual_p":dual_p,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2}

# implement the multi-source version of the gradient-based algorithm
def gdnV(pxv,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	d_seed = kwargs.get("seed",None)
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
	px_list, p_cond = ut.computeNvPriors(pxv)
	cur_ci = ut.calcMInV(pzxv)
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		# each sub-objective sum \sum_{i\in[M]}{H(X_i|Z)}-H(X_1,...,X_M|Z)
		# making p(xV|z) independent
		# gradients
		res = tuple([len(pz)]+[1]*nview)
		res_pz = np.reshape(pz,res)
		grad_mizx = pxv[None,...] * np.log(pzcxv/res_pz)
		grad_ent_diff = 0
		for vidx in range(nview):
			# H(X_i|Z) = -\iint p(z,x_i)*log(p(x_i|z)) = -\iint p(z,x_i)log(p(z|xi)p(xi)/p(z))
			# this is equivalent to I(Z;X_i), since H(X_i) is a constant 
			sum_axis = tuple([item+1 for item in range(nview) if item != vidx]) # offset nz
			pzi = np.sum(pzxv,axis=sum_axis,keepdims=True)
			pzci = pzi/np.sum(pzi,axis=0,keepdims=True)
			tmp_grad = -pxv[None,:] * np.log(pzci/res_pz)
			grad_ent_diff+= tmp_grad
		# gradient of the last element -H(XV|Z)
		# NOTE: I(XV;Z) = H(XV) - H(XV|Z)
		grad_ent_diff += pxv[None,:] * np.log(pzcxv/res_pz)
		
		# main component
		grad_mi = pxv[None,:] * np.log(pzcxv/res_pz)
		full_grad = grad_mi + gamma*grad_ent_diff

		means_grad = full_grad - np.mean(full_grad,axis=0,keepdims=True)
		ss = gd.naiveStepSize(pzcxv,-means_grad,ss_init,ss_scale)
		if ss==0:
			break
		new_pzcxv = pzcxv - ss * means_grad
		# compute the CI and use this as a convergence criterion
		new_pzxv = new_pzcxv * pxv[None,...]
		new_pz = np.sum(new_pzxv,axis=tuple(np.arange(1,nview+1)))
		new_ci = ut.calcMInV(new_pzxv)
		ci_diff = np.fabs(new_ci - cur_ci)
		if ci_diff<convthres:
			conv_flag=True
			break
		else:
			pzcxv = new_pzcxv
			pz = new_pz
			cur_ci = new_ci
		est_pzxv= pzcxv * pxv[None,:]
	return {"conv":conv_flag,"niter":itcnt,"pzcxv":pzcxv,"pz":pz,"est_pzxv":est_pzxv}