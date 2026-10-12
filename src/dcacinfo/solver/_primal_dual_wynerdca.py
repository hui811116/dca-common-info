import numpy as np
import itertools
import dcacinfo.utils as ut

def _init_variables(pxv,nview,nz,seed,load_pzcxv=None):
    rng = np.random.default_rng(seed)
    pzcxv = rng.random(size=(tuple([nz]+list(pxv.shape)))) + 1e-8 if load_pzcxv is None else load_pzcxv
    #pzcxv = rng.random(size=(tuple([nz]+list(pxv.shape)))) + 1e-8
    pzcxv /= np.sum(pzcxv,axis=0,keepdims=True)
    pzxv = pzcxv * pxv[None,...]
    pz = np.sum(pzxv,axis=tuple(np.arange(1,nview+1))) # offset by |Z|
    pz /= np.sum(pz)
    set_dict = ut.getSetDict(pxv,mi=False)
    return pzcxv, pz, set_dict

def _cmi_subset(pzxv,set_dict):
    cmi_dict = np.zeros(len(set_dict["tuple_list"]))
    tiny = np.finfo(float).tiny
    nview = pzxv.ndim - 1
    pz = np.sum(pzxv, axis=tuple(range(1, nview + 1)), keepdims=True)
    pxv_given_z = np.divide(
        pzxv,
        pz,
        out=np.zeros_like(pzxv, dtype=float),
        where=pz > 0,
    )
    for sidx, (item, cmpl_item) in enumerate(set_dict["tuple_list"]):
        item_axes = tuple(index + 1 for index in item)
        cmpl_axes = tuple(index + 1 for index in cmpl_item)
        pz_xs = np.sum(pzxv, axis=cmpl_axes, keepdims=True)
        pz_xsc = np.sum(pzxv, axis=item_axes, keepdims=True)
        pxs_given_z = np.divide(
            pz_xs,
            pz,
            out=np.zeros_like(pz_xs, dtype=float),
            where=pz > 0,
        )
        pxsc_given_z = np.divide(
            pz_xsc,
            pz,
            out=np.zeros_like(pz_xsc, dtype=float),
            where=pz > 0,
        )
        cmi_dict[sidx] = np.sum(
            pzxv
            * (
                np.log(np.maximum(pxv_given_z, tiny))
                - np.log(np.maximum(pxs_given_z, tiny))
                - np.log(np.maximum(pxsc_given_z, tiny))
            )
        )
    return cmi_dict

def primal_dual_wyner_dca(pxv,nz,epsilon,maxiter,convthres,**kwargs):
    d_seed = kwargs.get("seed",None)
    nview = len(pxv.shape)

    # assumption: uniformly Z
    pzcxv, pz, set_dict = _init_variables(pxv,nview,nz,d_seed,load_pzcxv=kwargs.get("load_pzcxv",None))
    tuple_list = set_dict["tuple_list"]
    dual_vars = np.ones(len(tuple_list))/len(tuple_list)

    # primal step
    niter= 0
    conv_flag = False
    pzxv = pzcxv * pxv[None,...]
    pzcx_set = ut.getPZCXsetDCA(pzxv,set_dict)
    mi_zxv = ut.calcMInV(pzxv)
    loss = mi_zxv - np.sum(dual_vars * _cmi_subset(pzxv,set_dict))
    while niter < maxiter:
        niter += 1
        # Primal step
        # L = I(Z;X_1,...,X_n) - \sum_{S\in set_dict} \dual_vars[S] * [I(X_S;Z)+I(X_{S^c};Z)]
        # p_{z|x^V}^{k+1} (raw) = \exp{\sum_{S\in set_dict} \dual_vars[S] * [\log{p_{z|x^S}^{k}} + \log{p_{z|x^{S^c}}^{k}}]}
        # normalize p_{z|x^V}^{k+1} (raw) to get p_{z|x^V}^{k+1}
        # update the new pzcxv
        expo_sum = 0
        expd_pz_shape = tuple([nz]+[1]*nview)
        res_pz = np.reshape(pz,expd_pz_shape)
        for sidx, (pzcx,cmpl_pzcx) in enumerate(pzcx_set):
            #expo_sum += gamma_arr[sidx] * (np.log(pzcx/res_pz) + np.log(cmpl_pzcx/res_pz))
            expo_sum += dual_vars[sidx] * (np.log(pzcx/res_pz) + np.log(cmpl_pzcx/res_pz))
        # smoothing
        expo_sum -= np.amax(expo_sum,axis=0,keepdims=True)
        new_pzcxv = res_pz * np.exp(expo_sum) + 1e-8 # smoothing condition
        new_pzcxv /= np.sum(new_pzcxv,axis=0,keepdims=True)

        new_pzxv = new_pzcxv * pxv[None,...]
        new_pz = np.sum(new_pzxv,axis=tuple(np.arange(1,nview+1)))
        new_pz/=np.sum(new_pz) # for stability

        # dual step
        # compute conditional mutual information I(X_S;X_{S^c}|Z) for each S in set_dict
        # dual problem corresponds to assigning mass to maximum I(X_S;X_{S^c}|Z)
        # use online mirror descent to update dual_vars smoothly

        cmi_dict = _cmi_subset(new_pzxv,set_dict)
        # update dual_vars
        omd_step_size = 1.0/(niter**0.5)
        log_dual = np.log(dual_vars) + omd_step_size * (cmi_dict - epsilon)
        log_dual -= np.max(log_dual)
        dual_vars = np.exp(log_dual)
        dual_vars /= np.sum(dual_vars)
        # compute primal dual gap
        # compute joint mutual information I(Z;X_1,...,X_n)
        mi_zxv = ut.calcMInV(new_pzxv)
        new_loss = mi_zxv - np.sum(dual_vars * cmi_dict)
        pzcxv = new_pzcxv
        pz = new_pz
        pzxv = new_pzxv
        if np.abs(new_loss - loss) < convthres:
            conv_flag = True
            break
        loss = new_loss
        pzcx_set = ut.getPZCXsetDCA(pzxv, set_dict)
    # DCA has zero approximation error
    est_pzxv = pzcxv * pxv[None,...]
    return {"conv":conv_flag,"niter":niter,"pzcxv":pzcxv,"pz":pz,"est_pzxv":est_pzxv}