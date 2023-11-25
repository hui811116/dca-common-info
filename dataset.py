import numpy as np
import os
import sys
import copy
import pickle

'''
def getDataset(stext,**kwargs):
	if stext == "syn2v":
		return syn2views()
	elif stext == "condindp2v":
		return syn2CondIndp()
	elif stext == "overlap2v":
		return syn2Overlap()
	elif stext == "toy2v":
		return synToy()
	elif stext == "toynv":
		return synToyNview(kwargs)
	else:
		sys.exit("ERROR: {:} does not match any defined dataset".format(stext))
'''

'''
def syn2views():
	# data
	# Statistics
	# ----------------
	# mix1x2=0.6616
	# Hx1=1.0778
	# Hx2=1.0805
	# Hx1x2=1.4968
	# ----------------
	px1cx2 = np.array([
		[0.85, 0.04, 0.06],
		[0.07, 0.88, 0.02],
		[0.08, 0.08, 0.92]])
	px2 = np.array([0.25, 0.35,0.40])
	px12 = px1cx2 * px2[None,:]
	px1 = np.sum(px12,1)
	px2cx1 = (px12/px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1]}


def syn2CondIndp():
	# data
	# Statistics
	# ----------------
	#
	# ----------------
	py = np.array([0.25,0.40,0.35])
	px1cy = np.array([
		[0.85, 0.04 , 0.06],
		[0.07, 0.40 , 0.60],
		[0.08, 0.56 , 0.34]])
	px1 = np.sum(px1cy* py[None,:],1)
	px2cy = np.array([
		[0.30, 0.08, 0.35],
		[0.40, 0.80, 0.15],
		[0.30, 0.12, 0.50]])
	px2 = np.sum(px2cy * py[None,:],1)
	px12 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
	for ix1 in range(3):
		for ix2 in range(3):
			tmp_sum = 0
			for iy in range(3):
				tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px12[ix1,ix2]= tmp_sum
	px1cx2 = px12 / px2[None,:]
	px2cx1 = (px12 / px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}

def syn2Overlap():
	# data
	# Statistics
	# ----------------
	#
	# ----------------
	py = np.array([0.25,0.40,0.35])
	px1cy = np.array([
		[0.85, 0.04 , 0.06],
		[0.07, 0.86 , 0.04],
		[0.08, 0.10 , 0.90]])
	px1 = np.sum(px1cy* py[None,:],1)
	px2cy = np.array([
		[0.06, 0.84, 0.05],
		[0.06, 0.08, 0.90],
		[0.88, 0.08, 0.05]])
	px2 = np.sum(px2cy * py[None,:],1)
	px12 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
	for ix1 in range(3):
		for ix2 in range(3):
			tmp_sum = 0
			for iy in range(3):
				tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px12[ix1,ix2]= tmp_sum
	px1cx2 = px12 / px2[None,:]
	px2cx1 = (px12 / px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}
'''
'''
def synToy():
	eps_min = 1e-2 # for smoothness
	# data
	# Statistics
	# ------------
	# 
	# ------------
	py = np.array([0.5,0.5])
	px1cy = np.array([
		[eps_min/2,0.5-eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[eps_min/2,0.5-eps_min/2]])
	px2cy = np.array([
		[eps_min/2,0.5-eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[eps_min/2,0.5-eps_min/2]])
	px1 = np.sum(px1cy * py[None,:],1)
	px2 = np.sum(px2cy * py[None,:],1)
	px12 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
	for ix1 in range(4):
		for ix2 in range(4):
			tmp_sum = 0
			for iy in range(2):
				tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px12[ix1,ix2] = tmp_sum
	px12 /= np.sum(px12)
	px1cx2 = px12 / px2[None,:]
	px2cx1 = (px12/px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}
'''
# NOTE: nview dataset
# independent copies

def datasetNView(ny,nb,corr,nv,**kwargs):
	ntest = kwargs.get("ntest",10000)
	d_seed = kwargs.get("seed",1234)
	rng = np.random.default_rng(seed=d_seed)
	dist_dict = synToyNView(ny,nb,corr,nv,**kwargs)
	# inverse transform sampling
	pxcy_list = dist_dict['pxcy_list']
	ylabel = rng.integers(ny,size=(ntest,))
	xnview = []
	xrnd = rng.random((ntest,nv))
	for v in range(nv):
		xn_tmp = []
		samp_map = np.cumsum(pxcy_list[v],axis=0)
		for ni, yi in enumerate(ylabel):
			tmp_prob = xrnd[ni,v]
			diff_mat = (samp_map[:,yi] - tmp_prob)< 0.0
			xreal = np.sum(diff_mat)
			xn_tmp.append(xreal)
		xn_tmp = np.array(xn_tmp)
		xnview.append(xn_tmp)
	return {'y_test':ylabel,'xn_test':xnview,'train':dist_dict}

def synToyNView(ny,nb,corr,nv,**kwargs):
	d_seed = kwargs.get("seed",1234)
	rng = np.random.default_rng(seed=d_seed)
	raw_eps = 1e-9
	nx = ny * nb # 
	py = np.ones((ny,))/ny
	one_pxcy = np.zeros((nx,))
	one_pxcy[:nb] = (1-corr)/ nb
	one_pxcy[nb:2*nb] = (corr)/nb
	px1cy = np.zeros((nx,ny))
	for iy in range(ny):
		pos_offset = nb *iy
		for ix in range(nb*ny):
			px1cy[ix,iy] = one_pxcy[-pos_offset+ix]
	pxcy_list = []
	for vidx in range(nv):
		pxcy_list.append(copy.deepcopy(px1cy))
	def compute_idx_(num,length,base):
		out = np.zeros((length,)).astype("int")
		cur = num
		for ii in range(length):
			out[ii] = int(cur % base)
			cur = int(cur/base)
		return out
	pxv = np.zeros(tuple([nx]*nv+[ny]))
	for iy in range(ny):
		cnt = 0
		while cnt < nx**nv:
			idx = compute_idx_(cnt,nv,nx)
			mass = 1
			for vi, vval in enumerate(idx):
				mass *= pxcy_list[vi][vval,iy]
			mass *= py[iy]
			pxv[tuple(list(idx)+[iy])] = mass
			cnt+=1
	pxv = np.sum(pxv,axis=-1)
	pxv += raw_eps
	pxv /= np.sum(pxv,keepdims=True)
	# get px
	px_list = []
	for vidx in range(nv):
		sum_axis = tuple([item for item in range(nv) if item != vidx])
		px_list.append(np.sum(pxv,axis=sum_axis))
	# p_cond ....
	p_cond =[]
	for vidx in range(nv):
		tr_axis = tuple([item for item in range(nv) if item != vidx]+[vidx])
		tmp_pvcx = np.transpose(pxv,axes=tr_axis) # put vidx to the last index
		tmp_px = px_list[vidx]
		p_cond.append(tmp_pvcx/tmp_px[...,:])

	return {"p_joint":pxv,"py":py,'px_list':px_list,"p_cond":p_cond,"pxcy_list":pxcy_list}
'''
def synExpandToy(ny,nb,corr):
	raw_eps = 1e-9
	# true cyclic correlated noise
	nx = ny * nb
	py = np.ones((ny,))/ny
	one_pxcy = np.zeros((nx,))
	one_pxcy[:nb] = (1-corr)/nb
	one_pxcy[nb:2*nb] = (corr)/nb
	px1cy = np.zeros((nx,ny))
	for iy in range(ny):
		pos_offset = nb*iy
		for ix in range(nb*ny):
			px1cy[ix,iy] = one_pxcy[-pos_offset + ix]

	px2cy = copy.deepcopy(px1cy)

	# smoothing
	px1x2 = np.sum(px1cy[:,None,:]*px2cy[None,:,:]*py[None,None,:],axis=-1)
	px1x2 += raw_eps
	px1x2 /= np.sum(px1x2,keepdims=True)

	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = px1x2.T/px1[None,:]
	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pxcy_list":[px1cy,px2cy],"py":py}
'''
'''
def synExpandToyNonUnif(ny,nb,corr):
	raw_eps = 1e-9
	# true cyclic correlated noise
	nx = ny * nb
	#py = np.ones((ny,))/ny
	py = np.arange(1,ny+1,1)
	py /= np.sum(py,keepdims=True)
	one_pxcy = np.zeros((nx,))
	one_pxcy[:nb] = (1-corr)/nb
	one_pxcy[nb:2*nb] = (corr)/nb
	px1cy = np.zeros((nx,ny))
	for iy in range(ny):
		pos_offset = nb*iy
		for ix in range(nb*ny):
			px1cy[ix,iy] = one_pxcy[-pos_offset + ix]
	#print(px1cy)
	#sys.exit()
	px2cy = copy.deepcopy(px1cy)
	px1x2 = np.zeros((nx,nx))
	for ix1 in range(nx):
		for ix2 in range(nx):
			tmpsum = 0
			for iy in range(ny):
				tmpsum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px1x2[ix1,ix2] = tmpsum
	# smoothing
	px1x2 += raw_eps
	px1x2 /= np.sum(px1x2,keepdims=True)

	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = px1x2.T/px1[None,:]
	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}
'''
'''
def synCorrUnif(ny,nb,corr):
	nx = ny * nb
	py = np.ones((ny,))/ny
	one_pxcy = np.ones((nx,)) * (corr/(nx-nb))
	one_pxcy[:nb] = (1-corr)/nb
	#print(one_pxcy)
	#print(np.sum(one_pxcy))
	px1cy = np.zeros((nx,ny))
	for iy in range(ny):
		pos_offset = nb * iy
		for ix in range(nb*ny):
			#print(one_pxcy[-pos_offset + ix])
			px1cy[ix,iy] = one_pxcy[-pos_offset + ix]
	#print(px1cy)
	#sys.exit()
	px2cy = copy.deepcopy(px1cy)
	px1x2 = np.zeros((nx,nx))
	for ix1 in range(nx):
		for ix2 in range(nx):
			tmpsum =0
			for iy in range(ny):
				tmpsum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px1x2[ix1,ix2] = tmpsum
	px1x2 /= np.sum(px1x2,keepdims=True)
	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = px1x2.T/px1[None,:]
	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}
'''
'''
def loadSampleData(dataset_path):
	with open(dataset_path,'rb') as fid:
		dataset_dict = pickle.load(fid)
	train_dict = dataset_dict['train_dict'] # ylabel, xsample, ny, nx1, nx2
	test_dict = dataset_dict['test_dict']
	# counting training set to have an estimate of px1x2,
	# send labels back too
	# the testing set simply copy what was loaded
	cnt_x1x2 = np.zeros((train_dict['nx1'],train_dict['nx2']))
	for idx in range(train_dict['xsample'].shape[0]):
		x1 = train_dict['xsample'][idx][0] # 1st view
		x2 = train_dict['xsample'][idx][1] # 2nd view
		cnt_x1x2[x1,x2] += 1
	est_px1x2 = cnt_x1x2 / np.sum(cnt_x1x2,keepdims=True)
	return {
		'ytrain':train_dict['ylabel'],'xtrain':train_dict['xsample'],
		"ytest":test_dict['ylabel'],"xtest":test_dict['xsample'],
		"ny":train_dict['ny'],'nx1':train_dict['nx1'],"nx2":train_dict['nx2'],
		"p_joint":est_px1x2,
	}
'''