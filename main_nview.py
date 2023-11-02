import numpy as np
import os
import sys
import pickle
import utils as ut
import algorithm as alg
import baseline as bs
import dataset as dt
import datetime
import copy
import argparse
import evaluation as ev
from scipy.io import savemat
from scipy.special import xlogy
import time


parser = argparse.ArgumentParser()
parser.add_argument("method",choices=['wynerdca','wyneram','gd'])
parser.add_argument("--maxiter",type=int,default=50000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=10,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=1e-1,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output_dir",type=str,default="wynerCI_results",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--ny",type=int,default=2,help="number of uniform hidden labels")
parser.add_argument("--nb",type=int,default=2,help="number of blocks for observations")
parser.add_argument("--corr",type=float,default=0,help="cyclic observation uncertainty given a label")
parser.add_argument("--betamin",type=float,default=0.1,help="minimum trade-off parameter for search")
parser.add_argument("--betamax",type=float,default=2.5,help="maximum trade-off parameter for search")
parser.add_argument("--numbeta",type=int,default=20,help="number of search points")
parser.add_argument("--nview",type=int,default=3,help="number of views")


args = parser.parse_args()
argsdict = vars(args)

print(argsdict)

data_dict = dt.synToyNView(args.ny,args.nb,args.corr,args.nview)
prob_joint = data_dict["p_joint"] # pxv
# I(Z;XV) = I(XS;XSc|Z) - I(XS;Z) - I(XSc;Z) + I(XS;XSc) \forall S\in\Pi_V
# shoud compute all combinations of XS...
# required for relaxed Wyner
# not needed for Wyner VI
set_dict = ut.getSetDict(prob_joint,mi=True) # obtain the prior I(XS;XSc)

gamma_range = np.geomspace(args.betamin,args.betamax,num=args.numbeta)

alg_dict = {
'ss_init':args.ss_init,
'ss_scale':args.ss_scale,
"seed":args.seed,
}

if args.method == "wynerdca":
	algrun = alg.wynerDCAnV
elif args.method == "wyneram":
	algrun = alg.wynerAMnV
elif args.method == "gd":
	algrun = bs.gdnV
else:
	sys.exit("undefined method {:}".format(args.method))

nz_set = np.array([args.ny]) # FIXME: assume knowing the cardinality of 
#header = ["nz",'beta','conv','niter','IZX1X2','IZX1','IZX2','IX1X2_Z','DKL_X12','HZ','IX1X2','V_IZ12','V_IZX1','V_IZX2','V_IX1X2_Z',"V_HZ",'V_IX1X2']
res_record =[]
#nz_set = np.arange(max(2,args.startz),args.endz+1,args.stepz)
for beta in gamma_range:
	for nz in nz_set:
		for nn in range(args.nrun):
			out_dict = algrun(prob_joint,nz,beta,args.maxiter,args.convthres,**alg_dict)
			# calculate the mutual informations
			pz = out_dict["pz"]
			
			pzcxv = out_dict['pzcxv'] # this might not be a valid prob, but is projected to a valid one
			est_pzxv = out_dict['est_pzxv']
			# Wyner problem evaluation # note: given p(z|x^V), 
			wy_pzxv = pzcxv * prob_joint[None,...]

			# conditional MI
			# I(XS;XSc|Z) = I(Z;XV) - I(XS;Z)-I(XSc;Z) + I(XS;XSc)
			comp_mizw = ut.getCompMI(wy_pzxv) # get a list of mizw, all combinations

			# KL distance
			est_pxv = np.sum(est_pzxv,axis=0)
			dkl_error = ut.calcKL(prob_joint,est_pxv)

			# wyner MI
			entz = ut.calcEnt(np.sum(wy_pzxv,axis=tuple(np.arange(1,args.nview+1))))
			entzcxv = np.sum(-wy_pzxv * np.log(pzcxv))
			wy_mi = entz - entzcxv

			# vi loss
			vi_entz = ut.calcEnt(np.sum(est_pzxv,axis=tuple(np.arange(1,args.nview+1))))
			vi_entzcxv = np.sum(-xlogy(est_pzxv,pzcxv))
			vi_mi = vi_entz - vi_entzcxv
			
			# what to save? # list of dictionaries
			sv_dict = {
				"nz":nz,"beta":beta,"conv":out_dict['conv'],'niter':out_dict['niter'],
				"VIzx":vi_mi,"VHz":vi_entz,"WIzx":wy_mi,"WHz":entz,'DKL':dkl_error}
			# compute conditional MI
			status_tex = ["nz,{:},beta,{:.3f},conv,{:},nit,{:},DKL,{:.3e},IW,{:.3e},IV:{:.3e}".format(
							nz,beta,out_dict['conv'],out_dict['niter'],dkl_error,wy_mi,vi_mi)]
			condmi_list =[]
			for cidx, item in enumerate(comp_mizw):
				mixx = set_dict['mi'][cidx]
				mizw, mizw_cmpl = item[0], item[1]
				cmi = wy_mi - mizw - mizw_cmpl + mixx
				condmi_list.append(cmi) # by definition
				# sv_dict
				sv_dict["CI_{:}".format(cidx)] = cmi
				sv_dict["IXX_{:}".format(cidx)] = mixx
				sv_dict['Izw_{:}'.format(cidx)] = mizw
				sv_dict['Izwcmpl_{:}'.format(cidx)] = mizw_cmpl
				status_tex.append("CI{:},{:.3e}".format(cidx,cmi))
			# reporting			
			print(",".join(status_tex))
			# saving
			res_record.append(sv_dict)
			#res_cnt +=1

timenow= datetime.datetime.now()
# result directory
d_cwd = os.getcwd()
d_save_dir = os.path.join(d_cwd,args.output_dir)
os.makedirs(d_save_dir,exist_ok=True)
savename_base = "wynerCI_{:}_V{:}_y{:}b{:}_cr{:.4f}_{:}".format(args.method,args.nview,args.ny,args.nb,args.corr,timenow.strftime("%Y%m%d"))
safe_savename = ut.getSafeSaveName(args.output_dir,savename_base,'.npy')

with open(os.path.join(d_save_dir,safe_savename+"_config.pkl"),"wb") as fid:
	pickle.dump(argsdict,fid)
with open(os.path.join(d_save_dir,safe_savename+".pkl"),'wb') as fid:
	pickle.dump(res_record,fid)
print("Saving the results to:{:}".format(os.path.join(d_save_dir,safe_savename+".pkl")))