import numpy as np
import sys
import os
import pickle
from scipy.io import savemat

argv = sys.argv

rec_dict = {}
for item in os.listdir(argv[1]):
	#print(item)
	if "_config" in item:
		#"skip for now"
		#print("has config")
		pass
	else:
		full_path = os.path.join(argv[1],item)
		cfg_name = ".".join(item.split(".")[:-1])+"_config.pkl"
		cfg_path = os.path.join(argv[1],cfg_name)
		#print(item,cfg_name)
		with open(full_path,'rb') as fid:
			rec = pickle.load(fid)
		# config
		with open(cfg_path,'rb') as fid:
			cfg = pickle.load(fid)
		#print(cfg.keys())
		#dict_keys(['method', 'maxiter', 'convthres', 'nrun', 'ss_init', 'ss_scale',
		#'output_dir', 'seed', 'ny', 'nb', 'corr', 'betamin', 'betamax', 'numbeta',
		#'nview', 'patience'])
		method = cfg['method']
		nview = cfg['nview']
		ny = cfg['ny']
		#nz = cfg.get("nz",ny)
		nx = ny * cfg['nb']
		corr = cfg['corr']
		nrun = cfg['nrun']
		beta_range = np.geomspace(cfg['betamin'],cfg['betamax'],cfg['numbeta'])

		#print(len(rec))
		if not method in rec_dict.keys():
			rec_dict[method] = {}
		if not nview in rec_dict[method].keys():
			rec_dict[method][nview] = {}
		if not ny in rec_dict[method][nview].keys():
			rec_dict[method][nview][ny] = {}
		if not corr in rec_dict[method][nview][ny]:
			rec_dict[method][nview][ny][corr] = {}
		# 
		for dd in rec:
			#"nz":nz,"beta":beta,"conv":out_dict['conv'],'niter':out_dict['niter'],
			#	"VIzx":vi_mi,"VHz":vi_entz,"WIzx":wy_mi,"WHz":entz,'DKL':dkl_error,
			#	"runtime":rt_dt,'acc':ev_dict['acc'],'acc_cnt':ev_dict['acc_cnt'],
			#	'total_cnt':ev_dict['total_cnt']
			## sv_dict
			#	sv_dict["CI_{:}".format(cidx)] = cmi
			#	sv_dict["IXX_{:}".format(cidx)] = mixx
			#	sv_dict['Izw_{:}'.format(cidx)] = mizw
			#	sv_dict['Izwcmpl_{:}'.format(cidx)] = mizw_cmpl

			nz = dd['nz']
			beta = dd['beta']
			if not nz in rec_dict[method][nview][ny][corr].keys():
				rec_dict[method][nview][ny][corr][nz] = {}
			if not beta in rec_dict[method][nview][ny][corr][nz].keys():
				rec_dict[method][nview][ny][corr][nz][beta] = []
			# metrics required
			# x: sum I(XS;Z)
			# y: I(Z;X^V)
			# ref: sum I(XS;XSc)
			# dkl: approximation error
			conv = int(dd['conv'])
			wy_mi = dd['WIzx']
			dkl = dd['DKL']
			rt = dd.get('runtime',0.)
			acc = dd.get("acc",0.)
			niter = dd['niter']
			ci_list = []
			ccnt = 0
			while True:
				tmp_key = "CI_{:}".format(ccnt)
				if tmp_key in dd.keys():
					ci_list.append(dd[tmp_key])
				else:
					break
				ccnt +=1 
			ci_sum = np.sum(np.array(ci_list).astype("float"))
			stats = np.array([conv,wy_mi,ci_sum,dkl,rt,niter,acc]).astype("float")
			rec_dict[method][nview][ny][corr][nz][beta].append(stats)

# collected, print out
hdr = "Nview,ny,corr,nz,beta,conv,wy_mi,ci_sum,dkl,rt,niter,acc"
print("Method,Nview,ny,corr,nz,beta,conv,wy_mi,ci_sum,dkl,rt,niter,acc")
mat_dict = {"header":hdr,"array":[]}
for method,m_d in rec_dict.items():
	for nview, v_d in m_d.items():
		for ny, y_d in v_d.items():
			for corr, c_d in y_d.items():
				for nz, z_d in c_d.items():
					for beta, b_l in z_d.items():
						for bi in b_l:
							data_arr = [nview,ny,corr,nz,beta] + list(bi)
							stats_list =["{:}".format(item) for item in [method,nview,ny,corr,nz,beta]]
							kk_list = ["{:.5e}".format(item) for item in bi]
							print(",".join(stats_list+kk_list))
							mat_dict['array'].append(data_arr)

mat_name = argv[1] + "_{:}.mat".format(method)
savemat(mat_name,mat_dict)