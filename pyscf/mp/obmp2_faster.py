#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
OB-MP2
'''

import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is None or mo_coeff is None:
        if mp.mo_energy is None or mp.mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')
        mo_coeff = None
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
    else:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen is 0 or mp.frozen is None)
    
    nuc = mp._scf.energy_nuc()
    ene_hf = mp._scf.energy_tot()

    nmo = mp.nmo
    nocc = mp.nocc

    niter = mp.niter
    ene_old = 0.
    #eri_ao = mp.mol.intor('int2e_sph')

    print()
    print("shift = ", mp.shift)
    print ("thresh = ", mp.thresh)
    print()
    

    for it in range(niter):

        #h2mo = [] #numpy.zeros((nmo,nmo,nmo,nmo)) #int_transform(eri_ao, mp.mo_coeff)
        #print(h1ao.shape)
        h1ao = mp._scf.get_hcore(mp.mol)
        h1mo = numpy.matmul(mp.mo_coeff.T,numpy.matmul(h1ao, mp.mo_coeff))
        h1mo_vqe = 0
        h1mo_vqe += h1mo

        cg = numpy.asarray(mp.mo_coeff[:,:])
        h2mo = ao2mo.general(mp._scf._eri, (cg,cg,cg,cg), compact=False)
        h2mo = h2mo.reshape(nmo,nmo,nmo,nmo)
        
        #####################
        ### Hartree-Fock

        fock_hf = h1mo
        veff, c0_hf = make_veff(mp)
        fock_hf += veff

        #initializing w/ HF
        fock = fock_hf
        c0 = c0_hf

        if  mp.second_order:
            mp.ampf = 1.0

        #####################
        ### MP1 amplitude
        tmp1, tmp1_bar = make_amp(mp)
        
        #####################
        ### BCH 1st order  
        c0, c1 = first_BCH(mp, fock_hf, tmp1, tmp1_bar, c0)

        # symmetrize c1
        fock += 0.5 * (c1 + c1.T)

        
        #####################
        ### BCH 2nd order  
        if mp.second_order:

            c0, c1 = second_BCH(mp, fock_hf, tmp1, tmp1_bar, c0)
            # symmetrize c1
            fock += 0.5 * (c1 + c1.T)

        ene = c0
        for i in range(nocc):
            ene += 2. * fock[i,i]
        
        ene_tot = ene + nuc
        de = abs(ene_tot - ene_old)
        ene_old = ene_tot
        print('iter = %d'%it, ' energy = %8.6f'%ene_tot, ' energy diff = %8.6f'%de, flush=True)

        if de < mp.thresh:
            break

        ## diagonalizing correlated Fock 
        mo_energy, U = scipy.linalg.eigh(fock)
        mo_coeff = numpy.matmul(mp.mo_coeff, U)
        mp.mo_energy = mo_energy
        mp.mo_coeff  = mo_coeff


    return ene_tot - ene_hf, tmp1, h1mo_vqe, h2mo, fock_hf

#################################################################################################################

def int_transform(eri_ao, mo_coeff):
    nao = mo_coeff.shape[0]
    nmo = mo_coeff.shape[1]
    eri_mo = numpy.dot(mo_coeff.T, eri_ao.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nao,nao,nmo).transpose(1,0,3,2)
    eri_mo = numpy.dot(mo_coeff.T, eri_mo.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    return eri_mo

def make_veff(mp):
    nmo  = mp.nmo
    nocc = mp.nocc
    mo_coeff  = mp.mo_coeff

    co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    cg = numpy.asarray(mo_coeff[:,:], order='F')
    
    
    h2mo_ggoo = ao2mo.general(mp._scf._eri, (cg,cg,co,co), compact=False)
    h2mo_ggoo = h2mo_ggoo.reshape(nmo,nmo,nocc,nocc)

    h2mo_goog = ao2mo.general(mp._scf._eri, (cg,co,co,cg), compact=False)
    h2mo_goog = h2mo_goog.reshape(nmo,nocc,nocc,nmo)

    veff = numpy.zeros((nmo,nmo))
    veff += 2.*numpy.einsum('ijkk -> ij',h2mo_ggoo) - numpy.einsum('ijjk -> ik',h2mo_goog)

    c0_hf = 0.
    for i in range(nocc):
        for j in range(nocc):
            c0_hf -= 2.*h2mo_ggoo[i,i,j,j] - h2mo_ggoo[i,j,j,i]

    return veff, c0_hf


def make_amp(mp):
    nmo  = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    mo_energy = mp.mo_energy
    mo_coeff  = mp.mo_coeff

    co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
    h2mo = ao2mo.general(mp._scf._eri, (co,cv,co,cv))
    h2mo = h2mo.reshape(nocc,nvir,nocc,nvir)

    tmp1 = numpy.zeros((nocc,nvir,nocc,nvir))
    x = numpy.tile(mo_energy[:nocc,None] - mo_energy[None,nocc:],(nocc,nvir,1,1))
    x += numpy.einsum('ijkl -> klij', x) - mp.shift
    tmp1 = mp.ampf * h2mo/x
    
    tmp1_bar = numpy.zeros((nocc,nvir,nocc,nvir))
    tmp1_bar = tmp1 - 0.5*numpy.einsum('ijkl -> ilkj', tmp1)         
    return tmp1, tmp1_bar

def first_BCH(mp, fock_hf, tmp1, tmp1_bar, c0):
    mo_coeff  = mp.mo_coeff
    nmo  = mp.nmo
    nocc = mp.nocc
    nvir = mp.nmo - nocc

    c1 = numpy.zeros((nmo,nmo), dtype=fock_hf.dtype)

    co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
    cg = numpy.asarray(mo_coeff[:,:], order='F')

    h2mo_ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv))
    h2mo_ovov = h2mo_ovov.reshape(nocc,nvir,nocc,nvir)

    h2mo_ovgv = ao2mo.general(mp._scf._eri, (co,cv,cg,cv))
    h2mo_ovgv = h2mo_ovgv.reshape(nocc,nvir,nmo,nvir)

    h2mo_ovog = ao2mo.general(mp._scf._eri, (co,cv,co,cg))
    h2mo_ovog = h2mo_ovog.reshape(nocc,nvir,nocc,nmo)
    
    c0 -= 4.*numpy.sum(h2mo_ovov * tmp1_bar)
    ####################### c1[j,B] #########################

    c1_jb = 4.*numpy.einsum('ijkl -> ij',numpy.einsum('ijkl -> klij',tmp1_bar)\
            *numpy.tile(fock_hf[:nocc,nocc:],(nocc,nvir,1,1)))
    c1_jb = numpy.pad(c1_jb, [(0, nvir), (nocc, 0)], mode='constant')
    ####################### c1[p,j] #########################
    for j in range(nocc):
        c1[:,j] = 4*numpy.einsum('ijkl -> k',h2mo_ovgv*\
                numpy.einsum('ijkl -> jkil',numpy.tile(tmp1_bar[:,:,j,:],(nmo,1,1,1))))
    
    ####################### c1[p,B] #########################
    for b in range(nvir):
        c1[:,b+nocc] = -4*numpy.einsum('ijkl -> l',h2mo_ovog*\
                numpy.einsum('ijkl -> jkli',numpy.tile(tmp1_bar[:,:,:,b],(nmo,1,1,1))))

    c1 += c1_jb
    return c0, c1

def second_BCH(mp, fock_hf, tmp1, tmp1_bar, c0):
    nmo  = mp.nmo
    nocc = mp.nocc
    nvir = mp.nmo - nocc

    c1 = numpy.zeros((nmo,nmo), dtype=fock_hf.dtype)
    #[1]
    y1 = numpy.zeros((nocc,nvir), dtype=fock_hf.dtype)
    y1 = numpy.einsum('ijkl -> kl', numpy.einsum('ijkl -> klij',\
        numpy.tile(fock_hf[:nocc,nocc:],(nocc,nvir,1,1))) * tmp1_bar)

    c1[:nocc,nocc:] += 4.*numpy.einsum('ijkl -> ij',\
        numpy.tile(y1,(nocc,nvir,1,1)) * tmp1_bar)

    #[2] [3] [8] [11]
    y1 = numpy.zeros((nocc,nvir,nocc,nvir), dtype=fock_hf.dtype)

    for c in range(nvir):
        y1 += numpy.einsum('ijkl -> klij',numpy.tile(fock_hf[nocc:,c-nvir].T,(nocc,nvir,nocc,1))) \
        *numpy.einsum('ijkl ->jikl',numpy.tile(tmp1_bar[:,c,:,:],(nvir,1,1,1)))

    for k in range(nocc):
        c1[:nocc,k] += 2.*(numpy.einsum('ijkl -> k',tmp1 \
                            * numpy.einsum('ijkl -> jkil',numpy.tile(y1[:,:,k,:],(nocc,1,1,1)))) \
                            + numpy.einsum('ijkl -> i',tmp1 * numpy.tile(y1[k,:,:,:],(nocc,1,1,1))))

    for b in range(nvir):    
        c1[b+nocc,nocc:] -= 2. * numpy.einsum('ijkl -> l',y1 * \
            numpy.einsum('ijkl -> jkli',numpy.tile(tmp1[:,:,:,b],(nvir,1,1,1))))
                
    c0 -= 4.*numpy.sum(tmp1 * y1)

    # [6] [7] [4] [10]
    y1 = numpy.zeros((nocc,nvir,nocc,nvir), dtype=fock_hf.dtype)

    for k in range(nocc):
        y1 += numpy.einsum('ijkl -> ljik',numpy.tile(fock_hf[:nocc,k],(nocc,nvir,nvir,1))) \
        * numpy.tile(tmp1_bar[k,:,:,:],(nocc,1,1,1))

    for c in range (nvir):
        c1[nocc:,c+nocc] += 2. * (numpy.einsum('ijkl -> l',tmp1 * \
            numpy.einsum('ijkl -> jkli',numpy.tile(y1[:,:,:,c],(nvir,1,1,1)))) \
             +  numpy.einsum('ijkl -> j',tmp1 * \
            numpy.einsum('ijkl -> jikl',numpy.tile(y1[:,c,:,:],(nvir,1,1,1)))))          

    for k in range(nocc):
        c1[:nocc,k] -= 2.*(numpy.einsum('ijkl -> k',tmp1 \
                        * numpy.einsum('ijkl -> jkil',numpy.tile(y1[:,:,k,:],(nocc,1,1,1)))))    

    c0 += 4.*numpy.sum(tmp1 * y1)    
    #[5]
    y1 = numpy.zeros((nocc,nocc), dtype=fock_hf.dtype)

    for k in range(nocc):
        y1[:,k] += numpy.einsum('ijkl -> i',tmp1 * numpy.tile(tmp1_bar[k,:,:,:],(nocc,1,1,1)))

    for k in range(nocc):
        c1[:,k] -= 2. * numpy.einsum('ij -> i', \
                    fock_hf[:nocc,:].T * numpy.tile(y1[:,k],(nmo,1)))

    #[9]
    y1 = numpy.zeros((nvir,nvir), dtype=fock_hf.dtype)

    for c in range(nvir):                
        y1[:,c] += numpy.einsum('ijkl -> j',tmp1 * \
            numpy.einsum('ijkl -> jikl',numpy.tile(tmp1_bar[:,c,:,:],(nvir,1,1,1))))     
                 
    for c in range(nvir):
        c1[:,c+nocc] -= 2. * numpy.einsum('ij -> i', \
                    fock_hf[nocc:,:].T * numpy.tile(y1[:,c],(nmo,1)))
    return c0, c1

#def eval_IP_EA():
    ### evaluating IPs and EAs
    #ip_obmp2 = []
    #for h in range(nocc):
    #    tmp2 = 0.
    #    for i in range(nocc-1):
    #        for j in range(nocc-1):
    #            for a in range(nvir):
    #                A = a+nocc
    #                if i != h and j != h:
    #                    tmp2 +=  tmp1_bar[i,a,j,h] * h2mo[i,A,j,h]
    #
    #    ip_obmp2.append(27.2114*(-mo_energy[h] + 2.*tmp2))
    #
    #
    #tmp1_new = numpy.zeros((nocc,nvir,nvir,nvir), dtype=fock_hf.dtype)
    #for i in range(nocc):
    #    for a in range(nvir):
    #        for l in range(nvir):
    #            for b in range(nvir):
    #                A = a+nocc
    #                B = b+nocc
    #                L = l+nocc
    #                x = mo_energy[i] + mo_energy[L] - mo_energy[A] - mo_energy[B]
    #                tmp1_new[i,a,l,b] = 1. * h2mo[i,A,L,B]/x
    #                
    #tmp1_bar_new = numpy.zeros((nocc,nvir,nvir,nvir), dtype=tmp1.dtype)
    #for i in range(nocc):
    #    for a in range(nvir):
    #        for l in range(nvir):
    #            for b in range(nvir):
    #                tmp1_bar_new[i,a,l,b] = tmp1_new[i,a,l,b] - 0.5 * tmp1_new[i,b,l,a]
    #                
    #ea_obmp2 = []
    #for l in range(nvir):
    #    L = l+nocc
    #    tmp2 = 0.
    #    for a in range(nvir):
    #        for b in range(nvir):
    #            for i in range(nocc):
    #                A = a+nocc
    #                B = b+nocc
    #                if a != l and b != l:
    #                    tmp2 +=  tmp1_bar_new[i,a,l,b] * h2mo[i,A,L,B]
    #
    #    ea_obmp2.append(27.2114*(-mo_energy[L] - 2.*tmp2))
    #
    ##print("ip_obmp2 (in eV)", flush=True)
    ##print(ip_obmp2[nocc-1], flush=True)
    #print("ea_obmp2 (in eV)", flush=True)
    #print(ea_obmp2, flush=True)



def make_rdm1(mp): # , t2=None, eris=None, verbose=logger.NOTE, ao_repr=False):
    '''Spin-traced one-particle density matrix.
    The occupied-virtual orbital response is not included.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Kwargs:
        ao_repr : boolean
            Whether to transfrom 1-particle density matrix to AO
            representation.
    '''
    from pyscf.cc import ccsd_rdm

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mp.mo_energy[:nocc,None] - mp.mo_energy[None,nocc:] 
    eris = mp.ao2mo(mp.mo_coeff)

    t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        t2[i] = t2i

    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = numpy.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T
    return ccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                               ao_repr=False)

def _gamma1_intermediates(mp, t2=None, eris=None):
    if t2 is None: t2 = mp.t2
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None: eris = mp.ao2mo()
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
        dtype = eris.ovov.dtype
    else:
        dtype = t2.dtype

    dm1occ = numpy.zeros((nocc,nocc), dtype=dtype)
    dm1vir = numpy.zeros((nvir,nvir), dtype=dtype)
    for i in range(nocc):
        if t2 is None:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
        l2i = t2i.conj()
        dm1vir += numpy.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                - numpy.einsum('jca,jbc->ba', l2i, t2i)
        dm1occ += numpy.einsum('iab,jab->ij', l2i, t2i) * 2 \
                - numpy.einsum('iab,jba->ij', l2i, t2i)
    return -dm1occ, dm1vir


#def make_rdm2(mp, t2=None, eris=None, verbose=logger.NOTE):
#    r'''
#    Spin-traced two-particle density matrix in MO basis
#
#    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>
#
#    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
#    E = einsum('pqrs,pqrs', eri, rdm2)
#    '''
#    if t2 is None: t2 = mp.t2
#    nmo = nmo0 = mp.nmo
#    nocc = nocc0 = mp.nocc
#    nvir = nmo - nocc
#    if t2 is None:
#        if eris is None: eris = mp.ao2mo()
#        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
#        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
#
#    if not (mp.frozen is 0 or mp.frozen is None):
#        nmo0 = mp.mo_occ.size
#        nocc0 = numpy.count_nonzero(mp.mo_occ > 0)
#        moidx = get_frozen_mask(mp)
#        oidx = numpy.where(moidx & (mp.mo_occ > 0))[0]
#        vidx = numpy.where(moidx & (mp.mo_occ ==0))[0]
#    else:
#        moidx = oidx = vidx = None
#
#    dm1 = make_rdm1(mp, t2, eris, verbose)
#    dm1[numpy.diag_indices(nocc0)] -= 2
#
#    dm2 = numpy.zeros((nmo0,nmo0,nmo0,nmo0), dtype=dm1.dtype) # Chemist notation
#    #dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,3,1,2)*2 - t2.transpose(0,2,1,3)
#    #dm2[nocc:,:nocc,nocc:,:nocc] = t2.transpose(3,0,2,1)*2 - t2.transpose(2,0,3,1)
#    for i in range(nocc):
#        if t2 is None:
#            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
#            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
#            t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
#        else:
#            t2i = t2[i]
#        # dm2 was computed as dm2[p,q,r,s] = < p^\dagger r^\dagger s q > in the
#        # above. Transposing it so that it be contracted with ERIs (in Chemist's
#        # notation):
#        #   E = einsum('pqrs,pqrs', eri, rdm2)
#        dovov = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
#        dovov *= 2
#        if moidx is None:
#            dm2[i,nocc:,:nocc,nocc:] = dovov
#            dm2[nocc:,i,nocc:,:nocc] = dovov.conj().transpose(0,2,1)
#        else:
#            dm2[oidx[i],vidx[:,None,None],oidx[:,None],vidx] = dovov
#            dm2[vidx[:,None,None],oidx[i],vidx[:,None],oidx] = dovov.conj().transpose(0,2,1)
#
#    # Be careful with convention of dm1 and dm2
#    #   dm1[q,p] = <p^\dagger q>
#    #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
#    #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
#    # When adding dm1 contribution, dm1 subscripts need to be flipped
#    for i in range(nocc0):
#        dm2[i,i,:,:] += dm1.T * 2
#        dm2[:,:,i,i] += dm1.T * 2
#        dm2[:,i,i,:] -= dm1.T
#        dm2[i,:,:,i] -= dm1
#
#    for i in range(nocc0):
#        for j in range(nocc0):
#            dm2[i,i,j,j] += 4
#            dm2[i,j,j,i] -= 2
#
#    return dm2#.transpose(1,0,3,2)


def get_nocc(mp):
    if mp._nocc is not None:
        return mp._nocc
    elif mp.frozen is None:
        nocc = numpy.count_nonzero(mp.mo_occ > 0)
        assert(nocc > 0)
        return nocc
    elif isinstance(mp.frozen, (int, numpy.integer)):
        nocc = numpy.count_nonzero(mp.mo_occ > 0) - mp.frozen
        assert(nocc > 0)
        return nocc
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        occ_idx = mp.mo_occ > 0
        occ_idx[list(mp.frozen)] = False
        nocc = numpy.count_nonzero(occ_idx)
        assert(nocc > 0)
        return nocc
    else:
        raise NotImplementedError

def get_nmo(mp):
    if mp._nmo is not None:
        return mp._nmo
    elif mp.frozen is None:
        return len(mp.mo_occ)
    elif isinstance(mp.frozen, (int, numpy.integer)):
        return len(mp.mo_occ) - mp.frozen
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        return len(mp.mo_occ) - len(set(mp.frozen))
    else:
        raise NotImplementedError

def get_frozen_mask(mp):
    '''Get boolean mask for the restricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresonds to the frozen orbital.
    '''
    moidx = numpy.ones(mp.mo_occ.size, dtype=numpy.bool)
    if mp._nmo is not None:
        moidx[mp._nmo:] = False
    elif mp.frozen is None:
        pass
    elif isinstance(mp.frozen, (int, numpy.integer)):
        moidx[:mp.frozen] = False
    elif len(mp.frozen) > 0:
        moidx[list(mp.frozen)] = False
    else:
        raise NotImplementedError
    return moidx


class OBMP2(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.thresh = 1e-08
        self.shift = 0.0
        self.niter = 100
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

        self.mom = False
        self.occ_exc = [None, None]
        self.vir_exc = [None, None]

        self.second_order = False
        self.ampf = 0.5

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    int_transform = int_transform

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot


    def kernel(self, shift=0.0, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
               _kern=kernel):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_corr,self.tmp1, self.h1mo_vqe, self.h2mo, self.fock_hf = _kern(self, mo_energy, mo_coeff,
                                     eris, with_t2, self.verbose)
        self._finalize()
        return self.e_corr,self.tmp1, self.h1mo_vqe, self.h2mo, self.fock_hf

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E(%s) = %.15g',
                    self.__class__.__name__, self.e_tot)
        return self

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_veff = make_veff
    make_amp  = make_amp
    first_BCH = first_BCH
    second_BCH = second_BCH
    make_rdm1 = make_rdm1
    #make_rdm2 = make_rdm2

    #as_scanner = as_scanner

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfmp2
        mymp = dfmp2.DFMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = copy.copy(mymp.with_df)
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        from pyscf.grad import mp2
        return mp2.Gradients(self)

#RMP2 = MP2

#from pyscf import scf
#scf.hf.RHF.MP2 = lib.class_as_method(MP2)
#scf.rohf.ROHF.MP2 = None


def _mo_energy_without_core(mp, mo_energy):
    return mo_energy[get_frozen_mask(mp)]

def _mo_without_core(mp, mo):
    return mo[:,get_frozen_mask(mp)]

def _mem_usage(nocc, nvir):
    nmo = nocc + nvir
    basic = ((nocc*nvir)**2 + nocc*nvir**2*2)*8 / 1e6
    incore = nocc*nvir*nmo**2/2*8 / 1e6 + basic
    outcore = basic
    return incore, outcore, basic

class _ChemistsERIs:
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        self.mo_coeff = _mo_without_core(mp, mo_coeff)

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (time.clock(), time.time())
    eris = _ChemistsERIs(mp, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory - mem_now)
    if max_memory < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 max_memory, mem_basic)

    co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore < max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((co,cv,co,cv)).reshape(nocc*nvir,nocc*nvir)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv))

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
                 '3-tensor integrals.\n'
                 'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        #ao2mo.outcore.general(mp.mol, (co,cv,co,cv), eris.feri,
        #                      max_memory=max_memory, verbose=log)
        #eris.ovov = eris.feri['eri_mo']
        eris.ovov = _ao2mo_ovov(mp, co, cv, eris.feri, max(2000, max_memory), log)

    time1 = log.timer('Integral transformation', *time0)
    return eris

#
# the MO integral for MP2 is (ov|ov). This is the efficient integral
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)
#
def _ao2mo_ovov(mp, orbo, orbv, feri, max_memory=2000, verbose=None):
    time0 = (time.clock(), time.time())
    log = logger.new_logger(mp, verbose)

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]
    nbas = mol.nbas
    assert(nvir <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocc)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, nocc**2*(nao*(nao+dmax)/2+nvir**2)*8/1e6)

    buf_i = numpy.empty((nocc*dmax**2*nao))
    buf_li = numpy.empty((nocc**2*dmax**2))
    buf1 = numpy.empty_like(buf_li)

    fint = gto.moleintor.getints4c
    jk_blk_slices = []
    count = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip+1]:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                jk_blk_slices.append((i0,i1,j0,j1))

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = numpy.ndarray((nocc,(i1-i0)*(j1-j0)*nao), buffer=buf_i)
                tmp_li = numpy.ndarray((nocc,nocc*(i1-i0)*(j1-j0)), buffer=buf_li)
                lib.ddot(orbo.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao), c=tmp_i)
                lib.ddot(orbo.T, tmp_i.reshape(nocc*(i1-i0)*(j1-j0),nao).T, c=tmp_li)
                tmp_li = tmp_li.reshape(nocc,nocc,(i1-i0),(j1-j0))
                save(str(count), tmp_li.transpose(1,0,2,3))
                buf_li, buf1 = buf1, buf_li
                count += 1
                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = buf_i = buf_li = buf1 = None

    h5dat = feri.create_dataset('ovov', (nocc*nvir,nocc*nvir), 'f8',
                                chunks=(nvir,nvir))
    occblk = int(min(nocc, max(4, 250/nocc, max_memory*.9e6/8/(nao**2*nocc)/5)))
    def load(i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(jk_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = ftmp[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(ftmp[str(k)][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def save(i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,nocc*nvir)

    orbv = numpy.asarray(orbv, order='F')
    buf_prefecth = numpy.empty((occblk,nocc,nao,nao))
    buf = numpy.empty_like(buf_prefecth)
    bufw = numpy.empty((occblk*nocc,nvir**2))
    bufw1 = numpy.empty_like(bufw)
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save) as bsave:
            load(0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocc, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*nocc,nao,nao)

                dat = _ao2mo.nr_e2(eri, orbv, (0,nvir,0,nvir), 's1', 's1', out=bufw)
                bsave(i0, i1, dat.reshape(i1-i0,nocc,nvir,nvir).transpose(0,2,1,3))
                bufw, bufw1 = bufw1, bufw
                time1 = log.timer_debug1('pass2 ao2mo [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)
    return h5dat


del(WITH_T2)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    mp = OBMP2(mf)
    mp.verbose = 5

    #pt = OBMP2(mf)
    #emp2, t2 = pt.kernel()
    #print(emp2 - -0.204019967288338)
    #pt.max_memory = 1
    #emp2, t2 = pt.kernel()
    #print(emp2 - -0.204019967288338)
    #
    #pt = MP2(scf.density_fit(mf, 'weigend'))
    #print(pt.kernel()[0] - -0.204254500454)
