##########################################################################
# box_DFL
# Copyright (C)     2018 G.Liuzzi,  S.Lucidi,   F.Rinaldi:
#
# Python version    2020 D.M.Pinto, F.Dominici
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
##########################################################################

import numpy as np
try:
    import ghalton
except ModuleNotFoundError:  # HDB: SciPy Halton shim
    from core.ioh_core.algorithms.dfl import halton as ghalton

def DFLINT(alg, mm, M, x_initial_in, lb, ub, max_fun, outlev):
    #
    # Function DFLINT solves the integer problem:
    #
    #                   min      f(x)
    #                   s.t. lb <= x <= ub
    #                            g_i(x) <= 0,  i=1,...,mm
    #                              x \in Z^n
    #
    # The user must provide func_f to evaluate the function f, lower and upper
    # bounds (lb and ub vectors)
    #
    # Inputs:
    #
    #         alg      : which algorithm to use. One of BBOA, DCS, FS
    #
    #         func_f   : handle of the function to be minimized, i.e. f(x).
    #
    #         mm       : number of constraints. If problem is unconstrained,
    #                    then mm = 0
    #
    #         M        : dimension of the memory for non-monotone search (must be >= 1) 
    #
    #         x_initial: the initial point to start the optimizer.
    #
    #         lb, ub   : lower and upper bounds.
    #
    #         max_fun  : maximum number of allowed function evaluations
    #
    #         outlev   : 0 for no output, >0 otherwise
    #
    # Outputs:
    #
    #          x       : best point
    # 
    #          f       : o.f. value related to x
    #
    #          stopfl  : 1 if code stopped for alpha< threshold
    #                    2 if code stopped for nf >= max_fun
    #                   99 if initial point does not satisfy the bound
    #                         constraints
    #          Dused   : set of directions used in the last iteration
    #
    # Functions called: either funct or functpen (provided by the optimizer),
    #                   nm_discrete_linesearch (provided by the optimizer),
    #                   nm_discrete_search (provided by the optimizer),
    #                   generate_dirs (provided by optimizer).
    #
    # Written by G. Liuzzi, S. Lucidi, F. Rinaldi, 2016.
    #

    
    x_initial = np.round(x_initial_in)
    
    ubint = np.round(ub)
    lbint = np.round(lb)
    
    if (np.sum(abs(ub-ubint)) != 0) or (np.sum(abs(lb-lbint)) != 0):
        print('\n\nERROR: upper and/or lower bound on some variable is NOT integer.')
        print('\n       Please correct and resubmit.\n\n')
        x       = x_initial
        f       = np.inf
        stopfl  = 99
        Dused   = []
        return x,f, stopfl, Dused
    
    m = max(mm,0)
    if M < 1:
        M = 1
    
    if alg!='DCS' and alg!='FS':
        alg = 'BBOA'
    
    iter        = 0                 # iteration counter
    alpha_start = 1                 # the starting stepsize 
    nf          = 0                 # number of function evaluations performed
    cache_hits  = 0                 # number of times point found in cache
    n = len(x_initial)              # dimension of the problem
    stop        = 0                 # stopping condition
    W           = np.empty(M)       # memory of function value for NM linesearch
    xW          = np.empty((n,M))
    
    nnf=0

    xf=np.inf*np.ones((max_fun,n+m+1))
    
    sequencer   = ghalton.Halton(n)       ## (input) la dimensione dello spazio
    Phalton     = sequencer.get(1000000)   ## (input) il numero di punti da generare nello spazio
    ihalton     = 7
    
    eta         = 1.5
    allones     = 0
    
    rho         = 1.0
    eps         = 0.1*np.ones((m,1))
    
    chk_feas = (x_initial >= lb) & (x_initial <= ub)
    if min(chk_feas) == 0:
        print('\n\nInitial point does not satisfy the bound constraints!\n\n')
        x       = x_initial
        f       = np.inf
        stopfl  = 99
        Dused   = []
        return x,f, stopfl, Dused
    
    # D           denotes the set of search direction (one per each column)
    # alpha_tilde is a row vector of stepsizes (one per each direction in D)
    
    
    x  = x_initial
    if m > 0:
        f, nnf, xf  = functpen(x,nnf,xf,eps)
    else:
        f, nnf, xf  = funct(x,nnf,xf)
    
    nf = nf+1
    
    if m > 0:    
        g = xf[0,(n+1):]
        rho = max(1.0,np.sum(np.where(g<0,0,g)))
        eps[(g > 0)<1.0] = 1e-3
        #eps[g[g > 0]<1.0] = 1e-3
        f, nnf, xf   = functpen(x,nnf,xf,eps)
    
    W[0] = f
    xW[:,0] = x
    bestf = f
    bestx = x
    
    D = np.identity(n)
    successes = np.zeros(n) # components of this vector counts number of successes each direction has had
                            # where "success" means that alpha > 0 is returned
                            # by LS along the direction itself
    
    
    if alg == 'BBOA':
        alpha_tilde   = np.round((ub+lb)/2.0)
    else:
        alpha_tilde   = np.ones(n)
    
    old_maxalpha  = np.inf
    ndir          = np.shape(D)[1]
    
    
    
    # HDB: avoid writing DFL_INT_log.txt into CWD when outlev == 0
    import os as _hdb_os
    if outlev > 0:
        log = open('DFL_INT_log.txt', 'w')
    else:
        log = open(_hdb_os.devnull, 'w')
    try:
    
        # print("%04d-%02d-%02d" % (year, month, day), file=log)
    
    
        if outlev > 0:
            if (m > 0):
                print_format = '| %05d | %05d | %05d | %+13.8e | %+13.8e | %+13.8e | %+13.8e | %5d |   \n'
                print('\n', file = log)
                print('|  iter |    nf | cache |        fpen     |        f_ref    |         viol    |    max_alpha    |  ndir |\n', file = log)
                print('|  iter |    nf | cache |        fpen     |        f_ref    |         viol    |    max_alpha    |  ndir |\n')
            else:
                print_format = '| %5d | %5d | %5d | %+13.8e | %+13.8e | %+13.8e | %5d |   \n'
                print('\n', file = log)
                print('|  iter |    nf | cache |        f        |        f_ref    |    max_alpha    |  ndir |\n', file = log)
                print('|  iter |    nf | cache |        f        |        f_ref    |    max_alpha    |  ndir |\n')
        else:
            # HDB: keep silent on stdout when outlev == 0 (log goes to devnull)
            print('   fun.evals =      ', file = log)
    
    
        while stop != 1:
     
            iter += 1
        
            y = x 
            fy = f
            cache_hits = 0
            for idir in range(ndir):

    #########################################################
    #########################################################
                if nf >= max_fun:
                    return x, f, stop, D
    #########################################################
    #########################################################

    
                d = D[:,idir]
    
                if iter == 1:
                    f_ref = W[0]
                else:
                    f_ref = max(W)
            
                if (alg == "BBOA") or (alg == "DCS"):        
                    alpha, x_trial, f_trial, xf, cache_hits, nnf, nf = nm_discrete_linesearch(y,d,alpha_tilde[idir],lb,ub,f_ref,xf,eps,cache_hits,nnf,m,nf)
                else:
                    alpha, x_trial, f_trial, xf, cache_hits, nnf, nf = nm_discrete_search(y,d,alpha_tilde[idir],lb,ub,f_ref,xf,eps,cache_hits,nnf,m,nf)
                
    #########################################################
    #########################################################
                if nf >= max_fun:
                    return x, f, stop, D
    #########################################################
    #########################################################

            
                if alpha <= 0:
                    d = -d
                    if (alg == "BBOA") or (alg == "DCS"):        
                        alpha, x_trial, f_trial, xf, cache_hits, nnf, nf = nm_discrete_linesearch(y,d,alpha_tilde[idir],lb,ub,f_ref,xf,eps,cache_hits,nnf,m,nf)
                    else:
                        alpha, x_trial, f_trial, xf, cache_hits, nnf, nf = nm_discrete_search(y,d,alpha_tilde[idir],lb,ub,f_ref,xf,eps,cache_hits,nnf,m,nf)
                    if alpha > 0:
                        successes[idir] = successes[idir]+1
                        if allones >= 1:
                            allones = 0
                        D[:,idir] = d
                        y  = x_trial
                        fy = f_trial
                    
                        alpha_tilde[idir] = alpha
                        W = np.roll(W,1)
                        xW = np.roll(xW,1,axis=1)
                        W[0] = fy
                        xW[:,0] = y
                        if(fy < bestf):
                            bestf = fy
                            bestx = y
                    else:
                        alpha_tilde[idir] = max(1,np.floor(alpha_tilde[idir]/2))
                else:        
                    successes[idir] = successes[idir]+1
                    if allones >= 1:
                        allones = 0
                
                    y  = x_trial
                    fy = f_trial
                    alpha_tilde[idir] = alpha
                    W = np.roll(W,1)
                    xW = np.roll(xW,1,axis=1)
                    W[0] = fy
                    xW[:,0] = y
                    if(fy < bestf):
                        bestf = fy
                        bestx = y
                if m > 0:
                    if (allones >= 1):
                       break
                else:
                    if (allones > 1):
                       break
    
            if(m > 0):
                sxf = np.shape(xf)[0]
                diff = np.square(xf[:,:n]-np.tile(y,(sxf,1)))
                mn = min(np.sum(diff,axis=1))
                ind = np.argmin(np.sum(diff,axis=1))
                g = xf[ind,(n+1):]
    
            if (np.linalg.norm(y-x) <= 1e-14) and (max(alpha_tilde) == 1) and (old_maxalpha == 1):
            
                allones=allones+1
    
                if (alg == 'DCS') or (alg == 'FS'):
                    if(bestf < fy):
                        y  = bestx
                        fy = bestf
                    else:
                        stopfl = 1
                        stop   = 1
                        Dused  = D
                    iexit = 1
                else:
                    iexit = 0
                
                
                while iexit == 0:
                    # enrich set D
                    D, successes, alpha_tilde, iexit, ihalton = generate_dirs(n,D,successes,alpha_tilde,eta,0,Phalton,ihalton)
                    if iexit == 0:
                        eta = eta + 0.5
                    if eta >= 0.5*(np.linalg.norm(ub -lb)/2):
                        #stop execution
                        if (bestf < fy):
                            y  = bestx
                            fy = bestf
                        else:
                            stopfl = 1
                            stop   = 1
                            Dused  = D
                        iexit = 1
                    
                ndir = np.shape(D)[1]
            
                if m > 0:
                    #check on the penalty parameters eps
                    ind_change = np.where(g<0,0,g) > rho
                    nchg = np.shape(ind_change)[0]
                    eps_changed = 0
                    for i in range(nchg):
                        if (ind_change[i] == True) and (eps[i] > 1e-10): 
                            eps[i] = eps[i]/2                             
                            allones = 0
                            eps_changed = 1
    
                    if eps_changed == 1:
                        sxf= np.shape(xf)[0]
                        diff=np.square(xf[:,:n]-np.tile(x,(sxf,1)))
                        mn=min(np.sum(diff,axis=1))
                        ind=np.argmin(np.sum(diff,axis=1))
                        if mn <= 1e-16:    
                            fval = xf[ind,n]
                            gval = xf[ind,n+1:]
                            m = np.shape(gval)[0]
                            f = fval + np.sum(np.divide(np.where(gval<0,0,gval),eps))
                            cache_hits += 1
                        else:
                            if m > 0:
                                f, nnf, xf = functpen(x,nnf,xf,eps)
                            else:
                                f, nnf, xf = funct(x,nnf,xf)
                            nf += 1
                        
                        # HDB: keep W shape (M,), not (1,M). A row vector makes
                        # W[0] / max(W) multi-element arrays and later triggers
                        # ValueError: truth value of an array is ambiguous in
                        # linesearch (f_trial < f_ref).
                        W           = np.nan*np.ones(M)
                        xW          = np.nan*np.ones((n,M))
                        W[0]    = f
                        xW[:,0] = x
                        bestf   = f
                        bestx   = x
        
            if m > 0:
                rho = max(1e-8,rho*0.5)
            else:
                rho = rho/2.0
            
            if (np.linalg.norm(y-x) <= 1e-14) and (ndir >=5000):
                stopfl = 1
                stop   = 1
                Dused  = D
                x      = bestx
                f      = bestf
            
            x = y
            f = fy
        
            old_maxalpha = max(alpha_tilde)
        
            if outlev > 0:
                if (m > 0):            
                    print(print_format %(iter, nf, cache_hits, f, f_ref, np.sum(np.where(g<0,0,g)), max(alpha_tilde), ndir), file = log)
                    print(print_format %(iter, nf, cache_hits, f, f_ref, np.sum(np.where(g<0,0,g)), max(alpha_tilde), ndir))
                else:
                    print(print_format %(iter, nf, cache_hits, f, f_ref, max(alpha_tilde), ndir), file = log)
                    print(print_format %(iter, nf, cache_hits, f, f_ref, max(alpha_tilde), ndir))
            else:
                # HDB: no stdout spam when outlev == 0
                print('                    %5d' %nf, file = log)
        
            if nf >= max_fun:
                stopfl = 2
                stop   = 1
                Dused  = D
                x      = bestx
                f      = bestf
    
        if outlev == 0:
            print('\n', file = log)

    finally:
        # HDB: always close the log handle
        try:
            log.close()
        except Exception:
            pass

    return x, f, stopfl, Dused
##########################################################################
# END OF CODE DFLINT
##########################################################################


##########################################################################
# funct
##########################################################################
def funct(xint,nnf,xf):
    
    floc,_ = func_f(xint)
        
    nx = np.shape(xint)[0]
    xf[nnf,:(nx)] = xint
    xf[nnf, (nx)] = floc 
        
    nnf += 1
    return floc, nnf, xf
##########################################################################
# END OF CODE funct
##########################################################################


##########################################################################
# functpen
########################################################################## 
def functpen(xint,nnf,xf,eps):
    
    floc, gloc = func_f(xint)
    
    nx = np.shape(xint)[0]
    ng = np.shape(gloc)[0]
    
    fpen = floc + np.sum(np.divide(np.where(gloc<0,0,gloc),eps))
    
    xf[nnf,:(nx)] = xint
    xf[nnf, (nx)] = floc 
    xf[nnf,(nx+1):] = gloc
    

    nnf += 1
    return fpen, nnf, xf
##########################################################################
# END OF CODE functpen
##########################################################################


def nm_discrete_linesearch(y,d,alpha_tilde,lb,ub,f_ref,xf,eps,cache_hits,nnf,m,nf):

    #
    # Function nm_discrete_linesearch
    #
    # Purpose:
    #
    # This function performs a nonmonotone discrete linesearch
    # along a given direction d (d \in Z^n)
    #
    # Inputs:
    #
    # y            : starting point for the linesearch
    #
    # d            : search direction
    #
    # alpha_tilde  : starting stepsize
    #
    # lb, ub       : lower and upper bounds
    #
    # f_ref        : reference o.f. value 
    #
    # Output:
    #
    #
    # alpha        : 1) alpha > 0 if linesearch finds a point guaranteeing 
    #                simple decrease: f(y+alpha d)<f_ref
    #                2) alpha = 0 failure
    #
    # x            : best point found in the linesearch 
    #
    # f            : o.f. value related to x 
    #

    # calculate dimension of the problem
    n = len(d)

    # initialize vector alpha_max
    alpha_max = np.inf * np.ones(n)

    # calculate max alpha
    indices = ( d > 0 )

    alpha_max[indices]=np.divide(ub[indices] - y[indices],d[indices])
    

    indices = ( d < 0 )

    alpha_max[indices]=np.divide(lb[indices] - y[indices],d[indices])

    #compute starting alpha
    alpha_bar  = np.floor( min(alpha_max) )
    alpha_init = min(alpha_tilde, alpha_bar)

    #Build first point for starting linesearch
    if (alpha_init > 0):
        y_trial = y + alpha_init * d
        sxf=np.shape(xf)[0]
        diff=np.square(xf[:,:n]-np.tile(y_trial,(sxf,1)))
        mn=min(np.sum(diff,axis=1))
        ind=np.argmin(np.sum(diff,axis=1))
        #diff
        #keyboard
        if (mn<=1e-16):
            fval = xf[ind,n]
            gval = xf[ind,n+1:]
            if m > 0:
                f_trial = fval + np.sum(np.divide(np.where(gval<0,0,gval),eps))
            else:
                f_trial = fval
            cache_hits += 1
        else:
            if m > 0:
                f_trial, nnf, xf = functpen(y_trial,nnf,xf,eps)
            else:
                f_trial, nnf, xf = funct(y_trial,nnf,xf)
            nf += 1
    else:
        f_trial = np.inf
        

    # cicle for updating alpha
    if (alpha_init > 0) and (f_trial < f_ref):
        
        # initialize alpha and best point
        alpha=alpha_init
        x = y_trial
        f = f_trial
        
        #calculate trial point
        if alpha < alpha_bar:
            y_trial = y + min(alpha_bar,2*alpha)* d
            sxf = np.shape(xf)[0]
            diff = np.square(xf[:,:n]-np.tile(y_trial,(sxf,1)))
            mn = min(np.sum(diff,axis=1))
            ind = np.argmin(np.sum(diff,axis=1))
            #diff
            #keyboard
            if (mn<=1e-16):
                fval = xf[ind,n]
                gval = xf[ind,n+1:]
                if m > 0:
                    f_trial = fval + np.sum(np.divide(np.where(gval<0,0,gval),eps))
                else:
                    f_trial = fval
                cache_hits += 1
            else:
                if m > 0:
                    f_trial, nnf, xf = functpen(y_trial,nnf,xf,eps)
                else:
                    f_trial, nnf, xf = funct(y_trial,nnf,xf)
                nf += 1
        else:
            f_trial = np.inf
            
                
        # expansion step (increase stepsize)
        while (alpha<alpha_bar) and (f_trial < f_ref):
            
            # alpha calulation and best point updatingd
            alpha=min(alpha_bar, 2*alpha)

            # best point updating
            x = y_trial
            f = f_trial

            #next point to be tested
            if(alpha < alpha_bar):
                y_trial = y + min(alpha_bar, 2* alpha) * d
                sxf = np.shape(xf)[0]
                diff = np.square(xf[:,:n]-np.tile(y_trial,(sxf,1)))
                mn = min(np.sum(diff,axis=1))
                ind = np.argmin(np.sum(diff,axis=1))
                #diff
                #keyboard
                if (mn<=1e-16):
                    fval = xf[ind,n]
                    gval = xf[ind,n+1:]
                    if m > 0:
                        f_trial = fval + np.sum(np.divide(np.where(gval<0,0,gval),eps))
                    else:
                        f_trial = fval
                    cache_hits += 1
                else:
                    if m > 0:
                        f_trial, nnf, xf = functpen(y_trial,nnf,xf,eps)
                    else:
                        f_trial, nnf, xf = funct(y_trial,nnf,xf)
                    nf += 1
            else:
                f_trial = np.inf
                
    
    else:
        alpha = 0
        x = y
        f = np.inf
        
    return alpha, x, f , xf, cache_hits, nnf, nf

 
##########################################################################
# END OF CODE nm_discrete_linesearch
##########################################################################

    

def nm_discrete_search(y,d,alpha_tilde,lb,ub, f_ref, xf,eps,cache_hits,nnf,m,nf):
    #
    # Function nm_discrete_search
    #
    # Purpose:
    #
    # This function performs a nonmonotone discrete linesearch
    # along a given direction d (d \in Z^n)
    #
    # Inputs:
    #
    # y            : starting point for the linesearch
    #
    # d            : search direction
    #
    # alpha_tilde  : starting stepsize
    #
    # lb, ub       : lower and upper bounds
    #
    # f_ref        : reference o.f. value 
    #
    # Output:
    #
    #
    # alpha        : 1) alpha > 0 if linesearch finds a point guaranteeing 
    #                simple decrease: f(y+alpha d)<f_ref
    #                2) alpha = 0 failure
    #
    # x            : best point found in the search 
    #
    # f            : o.f. value related to x 
    #

    # calculate dimension of the problem
    n = len(d)

    # initialize vector alpha_max
    alpha_max = np.inf * np.ones(n)

    # calculate max alpha
    indices = ( d > 0 )

    alpha_max[indices]=np.divide(ub[indices] - y[indices],d[indices])
    

    indices = ( d < 0 )

    alpha_max[indices]=np.divide(lb[indices] - y[indices],d[indices])

    #compute starting alpha
    alpha_bar  = np.floor( min(alpha_max) )
    alpha_init = min(alpha_tilde, alpha_bar)
    
    #Build first point for starting search
    if (alpha_init > 0):
        y_trial = y + alpha_init * d
        sxf=np.shape(xf)[0]
        diff=np.square(xf[:,:n]-np.tile(y_trial,(sxf,1)))
        mn=min(np.sum(diff,axis=1))
        ind=np.argmin(np.sum(diff,axis=1))
        #diff
        #keyboard
        if (mn<=1e-16):
            fval = xf[ind,n]
            gval = xf[ind,n+1:]
            if m > 0:
                f_trial = fval + np.sum(np.divide(np.where(gval<0,0,gval),eps))
            else:
                f_trial = fval
            cache_hits += 1
        else:
            if m > 0:
                f_trial, nnf, xf = functpen(y_trial,nnf,xf,eps)
            else:
                f_trial, nnf, xf = funct(y_trial,nnf,xf)
            nf += 1
        if (f_trial < f_ref):
            x = y_trial
            f = f_trial
            alpha = alpha_init
        else:
            f_trial = np.inf
            alpha = 0
            x = y
            f = np.inf
    else:
        f_trial = np.inf
        alpha = 0
        x = y
        f = np.inf

    return alpha, x, f , xf, cache_hits, nnf, nf
##########################################################################
# END OF CODE nm_discrete_search
##########################################################################


def prime_vector(d):
    n = len(d)
    flag = 0
    if(n==1):
        flag = True
        return flag
    temp = np.gcd(np.array(abs(d[0]),dtype=int),np.array(abs(d[1]),dtype=int))
    if(n==2):
        flag = (temp == 1)
        return flag
    for i in np.arange(2,n,1):
        temp = np.gcd(temp,np.array(abs(d[i]),dtype=int))
        #temp = numpy_gcd(temp,abs(d[i]))
        if temp == 1:
            flag = True
            return flag
    if temp != 1:
        flag = False
        return flag


##########################################################################
# END OF CODE prime_vector
##########################################################################


def generate_dirs(n,D,succ,alpha_tilde,eta,betaLS,Phalton,ihalton):
    #
    # Function generate_dirs
    #
    # Purpose:
    #
    # This function generate new integer directions which are added to set D 
    #
    # Inputs:
    #
    # n            : dimension of the problem
    #
    # D            : matrix of current directions (one per each column) 
    #
    # alpha_tilde  : array of stepsizes along direction already in D
    #
    # Output:
    #
    # Dout         : [new_direction D] 
    #
    # succout      : [0 succ] 
    #
    # alpha        : array of stepsizes along the directions in Dout
    #                alpha = [new_step_sizes alpha_tilde]
    #

    # d = rand(n,1)
    # d = d./norm(d)
    # 
    # Q = [null(d') d]
    # 
    # Dout = [Q D]
    # alpha = [ones(1,n) alpha_tilde]
    
    
    mD = np.shape(D)[1]
    
    for j in range(1000):
        #keyboard
        v = 2*np.asarray(Phalton[ihalton-1], dtype = np.float64) - np.ones(n)
        ihalton += 1
        v = eta*(v/np.linalg.norm(v))
        
        if (np.linalg.norm(v) < 1e-16):
            break
        
        #d = abs(round(v)) good if H=norm(d)^2*eye(n,n) - 2*d*d' used
        d = np.round(v)    
    
        #now check whether d is a prime vector
        if prime_vector(d) == True:
            trovato = False
            #check whether d is already in D
            d = np.reshape(d,(len(d),1))
            DIFF1 = D - np.tile(d,(1,mD))
            DIFF2 = D + np.tile(d,(1,mD))
            if( min ( np.sum(abs(DIFF1),axis=0)) == 0 ) or ( min ( np.sum(abs(DIFF2),axis=0)) == 0 ):
                trovato = True
                
            if trovato == False:
                H       = d.copy() #norm(d)^2*eye(n,n) - 2*d*d'
                Dout    = np.hstack((H,D))
                succout = np.hstack((np.array(0),succ))
                alpha   = np.hstack((np.array(max(betaLS,max(alpha_tilde))),alpha_tilde))
                iexit   = 1              
                return Dout, succout, alpha, iexit, ihalton

    Dout    = D
    succout = succ
    alpha   = alpha_tilde
    iexit   = 0
    
    return Dout, succout, alpha, iexit, ihalton

##########################################################################
# END OF CODE generate_dirs
##########################################################################    

def kowalik(xint):  # FUNCTION NAME: kowalik
    # 
    # Nonsmooth KOWALIK-OSBORNE function
    # Matlab Code by G.Liuzzi (October 4, 2017).
    # The number of variables n should be adjusted below.
    # The default value of n = 2.
    # 
    
    pdim      = 4
    lbint     = np.zeros(pdim)
    ubint     = 100.0*np.ones(pdim)
    startp    = np.array([0.25, 0.39, 0.415, 0.39])
    lb        = startp-10.0
    ub        = startp+10.0
    
    x = lb + ((ub - lb)/(ubint - lbint))*xint
    
    z = np.array([[0.1957, 4.0],
                 [0.1947, 2.0],
                 [0.1735, 1.0],
                 [0.1600, 0.5],
                 [0.0844, 0.25],
                 [0.0627, 0.1670],
                 [0.0456, 0.1250],
                 [0.0342, 0.1],
                 [0.0323, 0.0833],
                 [0.0235, 0.0714],
                 [0.0246, 0.0625]])
    u = z[:,1]
    
    f = (x[0]*(np.square(u) + x[1]*u))/(np.square(u) + x[2]*u + x[3]) - z[:,0]
    
    y = max(abs(f))
    
    return y, 'no costraint'



def davidon2_b(xint):
    #----------------------------
    # Function Davidon 2
    #----------------------------
    
    pdim      = 4 # problem has 4 variables
    lbint     = np.zeros(pdim)
    ubint     = 100.0*np.ones(pdim)
    startp    = np.array([25.0,5.0,-5.0,-1.0])
    lb        = startp-10.0
    ub        = startp+10.0
    
    x = lb + ((ub - lb)/(ubint - lbint))*xint
    
    t = np.zeros(21)
    f = np.zeros(21)    
    
    for i in range(21):
        t[i] = 0.25 + 0.75*((i+1)-1)/20.0
        f[i] = x[3] - np.square(x[0]*np.square(t[i]) + x[1]*t[i] + x[2]) - np.sqrt(t[i])
    
    y = max(abs(f))
    J = np.arange(len(x)-2)
    g = (3-2*x[J+1])*x[J+1] - x[J] - 2*x[J+2] + 2.5
    
    return y, g

# HDB: do not execute the demo call at import time
# y , g = davidon2_b((50,50,50,50))


if __name__ == '__main__':
    y, g = davidon2_b((50, 50, 50, 50))
    
    #choose which algorithm to run
    # alg = {BBOA, DCS, FS}
    
    
    alg = 'BBOA'
     
    #set nonmonotone memory size M (=1 for monotone)
    M = 4
    
    #example = "box"
    example = "con"
     
    if example == "box":
        func_f = kowalik
        #set the problem to be solved
        pname     = 'kowalik-osborne'
        pdim      = 4 # problem has 4 variables
        m         = 0 # problem has 0 general constraints
        lbint     = np.zeros(pdim)
        ubint     = 100.0*np.ones(pdim)
        x_initial = (ubint+lbint)/2
        startp    = np.array([0.25, 0.39, 0.415, 0.39])
        lb        = startp-10.0
        ub        = startp+10.0
        max_fun   = 5000
        outlev    = 1
        
        outline = ['Solving problem ', pname, ' using algorithm ']
        if M > 0:
            outline = [outline, 'NM-', alg, ': ']
        else:
            outline = [outline, ' M-', alg, ': ']
        print('\n\n %s' %(outline))
        
        #call the optimizer
        x,f, stopfl, Dused = DFLINT(alg, m, M, x_initial, lbint, ubint, max_fun, outlev)
        
    elif example == "con":
        func_f = davidon2_b

        #set nonmonotone memory size M (=1 for monotone)
        M = 4
        
        #set the problem to be solved
        pname     = 'davidon 2 (b)'
        pdim      = 4 # problem has 4 variables
        m         = 2 # problem has 2 general constraints
        lbint     = np.zeros(pdim)
        ubint     = 100.0*np.ones(pdim)
        x_initial = 50.0*np.ones(pdim)
        startp    = np.array([25.0,5.0,-5.0,-1.0])
        lb        = startp-10.0
        ub        = startp+10.0
        max_fun   = 5000
        outlev    = 1
        
        outline = ['Solving problem ', pname, ' using algorithm ']
        if M > 0:
            outline = [outline, 'NM-', alg, ': ']
        else:
            outline = [outline, ' M-', alg, ': ']
        print('\n\n %s' %(outline))
        
        #call the optimizer
        x,f, stopfl, Dused = DFLINT(alg, m, M, x_initial, lbint, ubint, max_fun, outlev)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        



