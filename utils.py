import numpy as np
import sys
import os
from scipy.special import softmax, xlogy
import copy
import itertools

def getSafeSaveName(savepath,basename,extension=".pkl"):
	repeat_cnt =0
	safename = copy.copy(basename)
	while os.path.isfile(os.path.join(savepath,safename+extension)):
		repeat_cnt += 1
		safename = "{:}_{:}".format(basename,repeat_cnt)
	# return without extension
	return safename

def calcKL(px,py):
	assert px.shape == py.shape
	return np.sum(px * np.log(px/py))

def calcMI(pxy):
	return np.sum(pxy * np.log(pxy/np.sum(pxy,0,keepdims=True)/np.sum(pxy,1,keepdims=True)))

def calcMIXS(pxv,widx):
	pxw_wcmpl = pxv/np.sum(pxv,axis=widx,keepdims=True) #(nxv)
	widx_cmpl = tuple(set(np.arange(len(pxv.shape)))-set(widx))
	#pxwcmpl_w = pxv/np.sum(pxv,axis=widx_cmpl,keepdims=True) #(nxv)
	## we need pxw, and pxwcmpl...
	pxw = np.sum(pxv,axis=widx_cmpl,keepdims=True)
	#pxw_cmpl = np.sum(pxv,axis=widx,keepdims=True)
	return np.sum(xlogy(pxv,pxw_wcmpl/pxw))

def getSetDict(pxv,mi=False):
	nview = len(pxv.shape)
	set_dict = {"uniset":set(np.arange(nview)),"tuple_list":[],'cond_list':[]}
	if mi:
		set_dict['mi'] = []
	for widx in range(1,int(np.floor(nview/2)+1)):
		for item in itertools.combinations(set_dict['uniset'],widx):
			cmpl_tuple = tuple(set_dict['uniset']-set(item)) 
			set_dict['tuple_list'].append((item,cmpl_tuple)) # from tuple to set
			pw_wcmpl = pxv / np.sum(pxv,axis=item,keepdims=True)
			pwcmpl_w = pxv / np.sum(pxv,axis=cmpl_tuple,keepdims=True)
			set_dict['cond_list'].append((pw_wcmpl,pwcmpl_w))
			# compute mi
			if mi:
				set_dict['mi'].append(calcMIXS(pxv,item))
	return set_dict

def calcMInV(pzxv):
	nview = len(pzxv.shape)-1
	pz = np.sum(pzxv,axis=tuple(np.arange(1,nview+1)))
	pzcxv = pzxv / np.sum(pzxv,axis=0,keepdims=True)
	res = tuple([pz.shape[0]]+[1]*nview)
	res_pz = np.reshape(pz,res)
	return np.sum(xlogy(pzxv,pzcxv/res_pz))

def getCompMI(pzxv):
	# get all I(Z;XS), I(Z;XSc)
	nview = len(pzxv.shape)-1
	uniset = set(np.arange(nview))
	pz = np.sum(pzxv,axis=tuple(np.arange(1,nview+1)))
	compMI_list = []
	for widx in range(1,int(np.floor(nview/2)+1)):
		for item in itertools.combinations(uniset,widx):
			cmpl_tuple = tuple(uniset - set(item)) #WSc
			# get pzw, then pzcw, mi=f(pzcw,pz)
			wnew = tuple(np.array(list(item))+1) # shift 1 for Z
			pzw = np.sum(pzxv,axis=wnew)
			mizw = calcMInV(pzw)

			wcmpl_new = tuple(np.array(list(cmpl_tuple))+1)
			pz_wcmpl = np.sum(pzxv,axis=wcmpl_new)
			mizw_cmpl = calcMInV(pz_wcmpl)
			compMI_list.append([mizw,mizw_cmpl])
	return compMI_list

def computeNvPriors(pxv):
	nview = len(pxv.shape)
	# need px_list, 
	px_list = []
	for vidx in range(nview):
		sum_axis = tuple([item for item in range(nview) if item != vidx])
		px_list.append(np.sum(pxv,axis=sum_axis))
	# pvcx_list
	p_cond = []
	for vidx in range(nview):
		tr_axis = tuple([item for item in range(nview) if item != vidx]+[vidx])
		tmp_px = px_list[vidx]
		p_cond.append(np.transpose(pxv,axes=tr_axis)/tmp_px[...,:])
	return px_list, p_cond

def calcEnt(pz):
	return -np.sum(pz * np.log(pz))

def calcCondEnt(pzx):
	return -np.sum(pzx * np.log(pzx/np.sum(pzx,axis=0,keepdims=True)))

def aggregateResults(xdata,ydata,criterion,precision):
	precision_tex = "{{:.{:}f}}".format(precision)
	out_dict = {}
	for idx in range(len(xdata)):
		xtmp = xdata[idx]
		ytmp = ydata[idx]
		x_tex = precision_tex.format(xtmp)
		if not out_dict.get(x_tex,False):
			out_dict[x_tex] = {"val":ytmp,"idx":idx}
		elif criterion == "min":
			if ytmp < out_dict[x_tex]["val"]:
				out_dict[x_tex]["val"] = ytmp
				out_dict[x_tex]["idx"] = idx
		elif criterion == "max":
			if ytmp > out_dict[x_tex]["val"]:
				out_dict[x_tex]["val"] = ytmp
				out_dict[x_tex]["idx"] = idx
		else:
			sys.exit("ERROR: undefined criterion")
	
	res_list = []
	idx_list = []
	for k,v in out_dict.items():
		res_list.append([float(k),v["val"]])
		idx_list.append([float(k),v["idx"]])
	res_list = sorted(res_list,key=lambda x:x[0])
	idx_list = sorted(idx_list,key=lambda x:x[0])
	return np.array(res_list), np.array(idx_list)

def computeJointEnc(pz,pzcx1,pzcx2,px12):
	px1 = np.sum(px12,axis=1)
	px2 = np.sum(px12,axis=0)
	est_pz_x1 = np.sum(pzcx1*px1[None,:],axis=1)
	est_pz_x2 = np.sum(pzcx2*px2[None,:],axis=1)

	joint_enc= np.zeros((len(pz),len(px1),len(px2)))
	for id1 in range(len(px1)):
		for id2 in range(len(px2)):
			for iz in range(len(pz)):
				joint_enc[iz,id1,id2] = pz[iz] * px1[id1] * px2[id2] / px12[id1,id2] * pzcx1[iz,id1] * pzcx2[iz,id2] / est_pz_x1[iz] / est_pz_x2[iz]
	joint_enc /= np.sum(joint_enc,axis=0,keepdims=True)
	return joint_enc

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

def computeGlobalSolution(px1cy,px2cy,py):
	smooth_eps = 1e-9
	nx1 = px1cy.shape[0]
	nx2 = px2cy.shape[0]
	ny = py.shape[0]
	px1x2y = np.zeros((nx1,nx2,ny))
	for yy in range(ny):
		for xx1 in range(nx1):
			for xx2 in range(nx2):
				px1x2y[xx1,xx2,yy] = py[yy] * px1cy[xx1,yy] * px2cy[xx2,yy] + smooth_eps
	px1x2y /= np.sum(px1x2y)

	pycx1x2 = np.transpose(px1x2y,axes=(2,0,1)) # this is pyx1x2 actually
	pycx1x2 /= np.sum(pycx1x2,axis=0,keepdims=True)
	return pycx1x2

def expandPxcz(pxcz,nviews,xidx,reverse=False):
	# reshape to (z,1,1,1,x)
	nx = pxcz.shape[0]
	nz = pxcz.shape[1]
	res = [1] * nviews
	if reverse:
		# for (1,1,1x,z) format
		zxshape = res + [nz]
		zxshape[xidx] = nx
	else:
		zxshape = [nz]+res
		zxshape[xidx+1] = nx # offset z		
	return np.reshape(pxcz,tuple(zxshape))

def getExpandList(pxcz_list):
	nviews = len(pxcz_list)
	expand_pxcz = []
	for vidx, item in enumerate(pxcz_list):
		expd_xcz = expandPxcz(item,nviews,vidx,reverse=True)
		expand_pxcz.append(expd_xcz)
	return expand_pxcz

def calcPXVZ(pxcz_list,pz):
	# expand each pxcz
	nviews = len(pxcz_list)
	expand_pxcz = getExpandList(pxcz_list)
	mm = expand_pxcz[0]
	for vidx in range(1,nviews):
		mm = mm * expand_pxcz[vidx]
	# mm is (x1,...,xv,nz) shape now
	res_pz = np.reshape(pz,tuple([1]*nviews + [len(pz)]))
	raw_prob = mm * res_pz
	return raw_prob/np.sum(raw_prob,keepdims=True)
def calcPXV(pxcz_list,pz):
	pxvz = calcPXVZ(pxcz_list,pz)
	return np.sum(pxvz,axis=-1)
def calcPZXV(pxcz_list,pz):
	pxvz = calcPXVZ(pxcz_list,pz)
	# transpose
	xtr = np.arange(len(pxvz.shape)-1).astype("int")
	return np.transpose(pxvz,axes=tuple([-1]+list(xtr)))
def calcPZcXV(pxcz_list,pz):
	pzxv = calcPZXV(pxcz_list,pz)
	return pzxv/np.sum(pzxv,axis=0,keepdims=True)

def calcPXWZ(pxcz_list,pz,widx):
	# product by excluding widx
	nviews = len(pxcz_list)
	expand_pxcz =getExpandList(pxcz_list)
	mm = 1
	for vidx, item in enumerate(expand_pxcz):
		if vidx != widx:
			mm = mm * item
	raw_prob = mm * pz[...,:]
	return raw_prob / np.sum(raw_prob,keepdims=True)

def computeDKL(pxcz,pxcw):
	# always, pxcz as the first argument
	# pxcw as the second argument
	# the dimension is always (xxxx,nw,nz),
	# after summation, the end dimensioni is (nw,nz)
	nx = len(pxcz.shape)-1
	assert len(pxcz.shape) == len(pxcw.shape)
	# expand each
	pxcz = np.expand_dims(pxcz,axis=-2)
	sum_axis = tuple(np.arange(nx))
	return np.sum(xlogy(pxcz,pxcz/np.expand_dims(pxcw,axis=-1)),axis=sum_axis)

# For DCA solvers
def getPZWsetDCA(pzxv,set_dict):
	# should return [(pzcw,pzcw_cmpl)] as in the seq of "tuple_list","cond_list"
	out_list =[]
	# set dict
	# set_dict = {"uniset":set(np.arange(nview)),"tuple_list":[],'cond_list':[]}
	nview = len(pzxv.shape)-1
	pzcxv = pzxv/np.sum(pzxv,axis=0,keepdims=True)
	for sidx ,(idset, idset_cmpl) in enumerate(set_dict['tuple_list']):
		cp_pw, cp_pw_cmpl = set_dict['cond_list'][sidx] # pw_wcmpl
		# append index set
		add_idset = tuple(np.array(list(idset))+1)
		add_idset_cmpl = tuple(np.array(list(idset_cmpl))+1)
		#print(add_idset,add_idset_cmpl)
		pzw = np.sum(pzxv,axis=add_idset_cmpl,keepdims=True)
		pzw_cmpl = np.sum(pzxv,axis=add_idset,keepdims=True)
		out_list.append([pzw,pzw_cmpl])
	return out_list