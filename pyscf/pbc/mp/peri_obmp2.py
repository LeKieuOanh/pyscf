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
Periodic OB-MP2
'''

import time, logging, tracemalloc
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf import __config__
from pyscf.pbc.mp import kmp2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)
LARGE_DENOM = getattr(__config__, 'LARGE_DENOM', 1e14)


def kernel(mp, mo_energy, mo_coeff, mo_occ, with_t2=WITH_T2,
           verbose=logger.NOTE):
    
    nuc = mp._scf.energy_nuc()
    
    nmo = mp.nmo
    nkpts = numpy.shape(mo_energy)[0]
    nocc = mp.nocc
    niter = mp.niter
    ene_old = 0.
    dm = mp._scf.make_rdm1(mo_coeff, mo_occ)
    #print('===========dm========')
    #print(dm)
    """
    h1ao = mp._scf.get_hcore()
    veffao = mp._scf.get_veff(mp._scf.cell, dm)
    veff= [reduce(numpy.dot, (mo.T.conj(), veffao[k], mo))
                            for k, mo in enumerate(mo_coeff)]
    fock_hf = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
    fock_hf += veff
    fock_hf += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo))
                            for k, mo in enumerate(mo_coeff)]
    c0_hf = 0
    for kp in range(nkpts):
        for i in range(nocc):
            c0_hf -=  veff[kp][i,i].real
    c0_hf/= nkpts
    """
    print()
    print('**********************************')
    print('************** OBMP2 *************')
    sort_idx = numpy.argsort(mo_energy)
    for it in range(niter):
        h1ao = mp._scf.get_hcore()
        veffao = mp._scf.get_veff(mp._scf.cell, dm)
        #
        #h1mo = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
        #for k in range(nkpts):
        #    h1mo[k] = numpy.matmul(mo_coeff[k].T.conj(),numpy.matmul(h1ao[k], mo_coeff[k]))

        
        #####################
        ### Hartree-Fock
        #veff_ao, veff, c0_hf = make_veff(mp, mo_coeff, mo_energy)
        veff= [reduce(numpy.dot, (mo.T.conj(), veffao[k], mo))
                                for k, mo in enumerate(mo_coeff)]
        
        c0_hf = 0
        for kp in range(nkpts):
            for i in range(nocc):
                c0_hf -=  veff[kp][i,i].real
        c0_hf/= nkpts
        #fock_hf = h1mo
        #fock_hf += veff
        fock_hf = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
        fock_hf += veff
        fock_hf += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo))
                                for k, mo in enumerate(mo_coeff)]
        numpy.set_printoptions(precision=6)

        #initializing w/ HF
        fock = 0
        fock += fock_hf
        c0 = c0_hf 

        ene_hf = 0
        for k in range(nkpts):
            for i in range(nocc):
                ene_hf += 2*fock[k][i,i].real/nkpts

        ene_hf +=c0_hf + nuc

        if  mp.second_order:
            mp.ampf = 1.0

        #####################
        ### MP1 amplitude
        #tmp1, tmp1_bar, h2mo_ovgg = (mp, mo_energy, mo_coeff)
        
        #####################
        ### BCH 1st order  
        c0_1st, c1 = first_BCH(mp, mo_energy, mo_coeff, fock_hf)
        
        for k in range(nkpts):
            fock[k] += (c1[k] + c1[k].T.conj())
            #print(abs(fock[k] - fock[k].T.conj()) < 1e-15)
            
        #####################
        ### BCH 2nd order  
        #if mp.second_order:

        #    c0_2nd, c1 = second_BCH(mp, mo_coeff, mo_energy, fock_hf)
            # symmetrize c1
        #    for k in range(nkpts):
        #        fock[k] += (c1[k] + c1[k].T.conj())
        ene = 0
        for k in range(nkpts):
            for i in range(nocc):
                ene += 2*fock[k][i,i].real/nkpts

        ene_tot = ene + c0 + c0_1st + nuc #+ c0_2nd
        #ene_tot = ene + c0 + nuc
        print('e_corr = ',ene_tot - ene_hf) 
        de = abs(ene_tot - ene_old)
        ene_old = ene_tot

        tracemalloc.start(25)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        stat = top_stats[:10]
        total_mem = sum(stat.size for stat in top_stats)
        print()
        print('iter = %d'%it, ' ene = %8.8f'%ene_tot, ' ene diff = %8.8f'%de, flush=True)
        #print("Total allocated size: %.3f Mb" % (total_mem / 1024**2))
        print(mo_energy[0])
        print()
        #print(mo_energy[1])
        if de < mp.thresh:
            break

        ## diagonalizing correlated Fock 
        #old_mo_coeff = mo_coeff
        new_mo_coeff = numpy.empty_like(mo_coeff)
        #new_mo_energy = numpy.empty_like(mo_energy)
        new_mo_energy = numpy.empty_like(mo_energy, dtype=complex)
        
        for k in range(nkpts):
            #new_mo_energy[k], U = scipy.linalg.eigvalsh(fock[k])
            new_mo_energy[k], U = scipy.linalg.eigh(fock[k])
            new_mo_coeff[k] = numpy.dot(mo_coeff[k], U)

            mo_energy[k] = new_mo_energy[k][sort_idx[k]].real
            mo_coeff[k] = new_mo_coeff[k][:,sort_idx[k]]
            
            #mp.mo_energy = mo_energy
            #mp.mo_coeff  = mo_coeff

    return ene_tot, mo_energy

#################################################################################################################


def make_veff(mp, mo_coeff, mo_energy):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    kpts = mp.kpts
    dm = mp._scf.make_rdm1()
    veff_ao = mp._scf.get_veff(mp._scf.cell, dm)


    veff = numpy.zeros((nkpts, nmo,nmo), dtype=complex)
    
    for kp in range(nkpts):
        veff[kp] = numpy.matmul(mo_coeff[kp].T.conj(),numpy.matmul(veff_ao[kp], mo_coeff[kp]))
    
    c0_hf = 0
    for kp in range(nkpts):
        for i in range(nocc):
            c0_hf -=  veff[kp][i,i].real
    c0_hf/= nkpts
    
    return veff_ao, veff, c0_hf


def ene_denom(mp, mo_energy, ki, ka, kj, kb):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")
    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    eia = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_ia = numpy.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
    eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

    ejb = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_jb = numpy.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
    ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
    e_iajb = lib.direct_sum('ia,jb -> iajb', eia, ejb)

    return e_iajb
"""
def make_amp(mp, mo_energy, mo_coeff):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]
    kpts = mp.kpts
    kconserv = mp.khelper.kconserv
    fao2mo = mp._scf.with_df.ao2mo

    tmp1 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    tmp1_bar = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    h2mo_ovgg = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nmo,nmo), dtype=complex)
    h2mo_ovov = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                kp = kj
                kq = kb
                o_i = mo_coeff[ki][:,:nocc]
                o_a = mo_coeff[ka][:,nocc:]
                o_p = mo_coeff[kp]
                o_q = mo_coeff[kq]
                h2mo_ovgg[ki,kj,ka] = fao2mo((o_i,o_a,o_p,o_q),
                                (kpts[ki],kpts[ka],kpts[kp],kpts[kq]),
                                compact=False).reshape(nocc,nvir,nmo,nmo)/nkpts
                h2mo_ovov[ki,kj,ka] = h2mo_ovgg[ki,kj,ka][:,:,:nocc, nocc:]
                
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e_iajb = ene_denom(mp, mo_energy, ki, ka, kj, kb)
                w_iajb = (h2mo_ovov[ki,kj,ka]
                            -0.5*h2mo_ovov[ki,kj,kb].transpose(0,3,2,1))
                
                tmp1[ki,kj,ka]  = (h2mo_ovov[ki,kj,ka]/e_iajb).conj()
                tmp1_bar[ki,kj,ka]  =  (w_iajb/e_iajb).conj()
                
    return tmp1, tmp1_bar, h2mo_ovgg
"""
def first_BCH(mp, mo_energy, mo_coeff, fock_hf):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]
    kpts = mp.kpts
    #kconserv = mp.khelper.kconserv
    kconserv = kpts_helper.get_kconserv(mp._scf.cell, kpts)
    fao2mo = mp._scf.with_df.ao2mo
    
    tmp1 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    tmp1_bar = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    h2mo_ovgg = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nmo,nmo), dtype=complex)
    h2mo_ovov = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    h2mo_ovog = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nmo), dtype=complex)
    h2mo_ovgv = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nmo,nvir), dtype=complex)
    
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                kp = kj
                kq = kb
                o_i = mo_coeff[ki][:,:nocc]
                o_a = mo_coeff[ka][:,nocc:]
                o_p = mo_coeff[kp]
                o_q = mo_coeff[kq]
                h2mo_ovgg[ki,kj,ka] = fao2mo((o_i,o_a,o_p,o_q),
                                (kpts[ki],kpts[ka],kpts[kp],kpts[kq]),
                                compact=False).reshape(nocc,nvir,nmo,nmo)/nkpts
                h2mo_ovov[ki,kj,ka] = h2mo_ovgg[ki,kj,ka][:,:,:nocc, nocc:]
                h2mo_ovgv[ki,kj,ka] = h2mo_ovgg[ki,kj,ka][:,:,:, nocc:]
                h2mo_ovog[ki,kj,ka] = h2mo_ovgg[ki,kj,ka][:,:,:nocc, :]
                
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e_iajb = ene_denom(mp, mo_energy, ki, ka, kj, kb)
                w_iajb = (h2mo_ovov[ki,kj,ka]
                            -0.5*h2mo_ovov[ki,kj,kb].transpose(0,3,2,1))
                
                tmp1[ki,kj,ka]  = (h2mo_ovov[ki,kj,ka]/e_iajb).conj()
                tmp1_bar[ki,kj,ka]  =  (w_iajb/e_iajb).conj()
    
    tmp1 *= mp.ampf
    tmp1_bar *= mp.ampf
    c1 = numpy.zeros((nkpts,nkpts,nmo,nmo), dtype=complex)
    c1_test = numpy.zeros((nkpts,nmo,nmo), dtype=complex)
    

    ### Tb_iajb * g_iapb -> c1_pj
    c1[:,:,:,:nocc] += 2*numpy.einsum('qweiajb, rweiapb -> qrpj',tmp1_bar,h2mo_ovgv)
    
    ### Tb_iajb * g_iajp -> c1_bp
    for kr in range(nkpts):
        c1[kr,kr,nocc:,:] -= 2*numpy.einsum('qweiajb, qweiajp -> bp',tmp1_bar,h2mo_ovog)
        
    ### f_ai * Tb_iajb -> c1_bj
    c1[:,:,nocc:,:nocc] += 2*numpy.einsum('qai, qwqiajb -> wbj'
                                      ,fock_hf[:,nocc:,:nocc],tmp1_bar)
    
        
    c1 = numpy.einsum('wwps -> wps', c1)
    
    c0_1st = -4*numpy.einsum('qweiajb, qweiajb',tmp1_bar, h2mo_ovov)
    c0_1st = c0_1st.real/nkpts

    c0_2nd =0
    if mp.second_order:
        c2 = numpy.zeros((nkpts, nkpts, nmo,nmo), dtype=complex)
        
        buf = tmp1_bar.transpose(2,0,1,3,4,5,6)
        buf_h2 = tmp1.transpose(2,0,1,3,4,5,6)
        #buf = buf.conj()
        #tmp1_bar = tmp1_bar.conj()
        #tmp1 = tmp1.conj()
        ### BCH second order ###
        ### [1] f_ia * Tb_iajb * Tb_jkbc -> E_kc
        
        for kj in range(nkpts):
            y1 = 2*numpy.einsum('eia, eeiajb -> jb',fock_hf[:,:nocc,nocc:], buf[kj])
            c2[:,:,:nocc,nocc:] += numpy.einsum('jb, wejbkc -> wekc', y1, tmp1_bar[kj].conj())
        
        
        ### [2] f_ac * Tb_ickb * T_iajb -> E_kj
        ### [3] f_ac * Tb_kcjb * T_iajb -> E_ki
        """
        for kj in range(nkpts):
            y1 = numpy.einsum('ac, qwickb -> qwiakb',
                            fock_hf[kj,nocc:,nocc:], buf[kj])
            c2[:,:,:nocc,:nocc] += numpy.einsum('qwiakb, qeiajb -> wekj', 
                                y1, buf_h2[kj])
            c2[:,:,:nocc,:nocc] += numpy.einsum('qwkajb, ewiajb -> qeki', 
                                y1, buf_h2[kj])
            c0_2nd -= 4*numpy.einsum('qwiajb,qwiajb ->', y1, buf_h2[kj]).real
            ### [8] f_ac * Tb_icjd * T_iajb -> E_bd
            #for kb in range(nkpts):
            #    c2[kb,kb,nocc:,nocc:] -= numpy.einsum('qwiajd, qwiajb -> bd',
            #                            y1, tmp1[kj])
        """ 
        for kj in range(nkpts):
            y1 = numpy.einsum('ac, qwickb -> qwiakb',
                            fock_hf[kj,nocc:,nocc:], tmp1_bar[:,:,kj,:,:,:,:].conj())
            c2[:,:,:nocc,:nocc] += numpy.einsum('qwiakb, qeiajb -> wekj', 
                                y1, tmp1[:,:,kj,:,:,:,:])
            c2[:,:,:nocc,:nocc] += numpy.einsum('qwkajb, ewiajb -> qeki', 
                                y1, tmp1[:,:,kj,:,:,:,:])
            c0_2nd -= 4*numpy.einsum('qwiajb,qwiajb ->', y1, tmp1[:,:,kj,:,:,:,:]).real
            ### [8] f_ac * Tb_icjd * T_iajb -> E_bd
            for kb in range(nkpts):
                c2[kb,kb,nocc:,nocc:] -= numpy.einsum('qwiajd, qwiajb -> bd',
                                        y1, tmp1[kj])
        
        ### [4] f_ik * Tb_kalb * T_iajb -> E_jl
        ### [7] f_ik * Tb_kcjb * T_iajb -> E_ca
        for kj in range(nkpts):
            y1 = numpy.einsum('ik, qwkalb -> qwialb', 
                                fock_hf[kj,:nocc, :nocc], tmp1_bar[kj].conj())
            c2[:,:,:nocc,:nocc] -= numpy.einsum('qwialb, ewiajb -> qelj', y1, tmp1[kj])
            c2[:,:,nocc:,nocc:] += numpy.einsum('qwicjb, qeiajb -> ca', y1, tmp1[kj])
            c0_2nd += 4*numpy.einsum('qwiajb, qwiajb ->', y1, tmp1[kj]).real
            ### [6] f_ik * Tb_kajd * T_iajb -> E_bd
            for kb in range(nkpts):
                c2[kb,kb,nocc:,nocc:] += numpy.einsum('qwiajd, qwiajb -> bd', y1, tmp1[kj])
        c0_2nd /= nkpts

        ### [5] f_ip * T_iajb * Tb_kajb -> E_pk
        y1 = numpy.einsum('qweiajb, rwekajb -> qrik', tmp1, tmp1_bar.conj())
        c2[:,:,:,:nocc] -= numpy.einsum('qip, qrik -> qrpk', fock_hf[:,:nocc,:], y1)
        ### [9] f_ap * T_iajb * Tb_icjb -> E_cp
        y1 = numpy.einsum('qweiajb, qwricjb -> erac', tmp1, tmp1_bar.conj())
        c2[:,:,:,nocc:] -= numpy.einsum('eap, erac -> erpc',fock_hf[:,nocc:,:], y1)
        c1 += numpy.einsum('wwps -> wps', c2)
        
    return c0_1st+c0_2nd, c1

"""
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
    
    t2 = mp.tmp_dip
    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = numpy.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T
    return ccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                               ao_repr=False)

def _gamma1_intermediates(mp, t2):
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
"""
def _add_padding(mp, mo_coeff, mo_energy, mo_occ):
    from pyscf.pbc import tools
    from pyscf.pbc.cc.ccsd import _adjust_occ
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    # Check if these are padded mo coefficients and energies
    if not numpy.all([x.shape[0] == nmo for x in mo_coeff]):
        mo_coeff = padded_mo_coeff(mp, mo_coeff)

    if not numpy.all([x.shape[0] == nmo for x in mo_energy]):
        mo_energy = padded_mo_energy(mp, mo_energy)
    
    if not numpy.all([x.shape[0] == nmo for x in mo_occ]):
        mo_occ = padded_mo_occ(mp, mo_occ)

    return mo_coeff, mo_energy, mo_occ

def _padding_k_idx(nmo, nocc, kind="split"):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.
    Args:
        nmo (Iterable): k-dependent orbital number;
        nocc (Iterable): k-dependent occupation numbers;
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    if kind not in ("split", "joint"):
        raise ValueError("The 'kind' argument must be one of 'split', 'joint'")

    if kind == "split":
        indexes_o = []
        indexes_v = []
    else:
        indexes = []

    nocc = numpy.array(nocc)
    nmo = numpy.array(nmo)
    nvirt = nmo - nocc
    dense_o = numpy.amax(nocc)
    dense_v = numpy.amax(nvirt)
    dense_nmo = dense_o + dense_v

    for k_o, k_nmo in zip(nocc, nmo):
        k_v = k_nmo - k_o
        if kind == "split":
            indexes_o.append(numpy.arange(k_o))
            indexes_v.append(numpy.arange(dense_v - k_v, dense_v))
        else:
            indexes.append(numpy.concatenate((
                numpy.arange(k_o),
                numpy.arange(dense_nmo - k_v, dense_nmo),
            )))

    if kind == "split":
        return indexes_o, indexes_v

    else:
        return indexes


def padding_k_idx(mp, kind="split"):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.

    This implementation stores k-dependent Fock and other matrix in dense arrays with additional dimensions
    corresponding to k-point indexes. In case when the occupation numbers depend on the k-point index (i.e. a metal) or
    when some k-points have more Bloch basis functions than others the corresponding data structure has to be padded
    with entries that are not used (fictitious occupied and virtual degrees of freedom). Current convention stores these
    states at the Fermi level as shown in the following example.

    +----+--------+--------+--------+
    |    |  k=0   |  k=1   |  k=2   |
    |    +--------+--------+--------+
    |    | nocc=2 | nocc=3 | nocc=2 |
    |    | nvir=4 | nvir=3 | nvir=3 |
    +====+========+========+========+
    | v3 |  k0v3  |  k1v2  |  k2v2  |
    +----+--------+--------+--------+
    | v2 |  k0v2  |  k1v1  |  k2v1  |
    +----+--------+--------+--------+
    | v1 |  k0v1  |  k1v0  |  k2v0  |
    +----+--------+--------+--------+
    | v0 |  k0v0  |        |        |
    +====+========+========+========+
    |          Fermi level          |
    +====+========+========+========+
    | o2 |        |  k1o2  |        |
    +----+--------+--------+--------+
    | o1 |  k0o1  |  k1o1  |  k2o1  |
    +----+--------+--------+--------+
    | o0 |  k0o0  |  k1o0  |  k2o0  |
    +----+--------+--------+--------+

    In the above example, `get_nmo(mp, per_kpoint=True) == (6, 6, 5)`, `get_nocc(mp, per_kpoint) == (2, 3, 2)`. The
    resulting dense `get_nmo(mp) == 7` and `get_nocc(mp) == 3` correspond to padded dimensions. This function will
    return the following indexes corresponding to the filled entries of the above table:

    >>> padding_k_idx(mp, kind="split")
    ([(0, 1), (0, 1, 2), (0, 1)], [(0, 1, 2, 3), (1, 2, 3), (1, 2, 3)])

    >>> padding_k_idx(mp, kind="joint")
    [(0, 1, 3, 4, 5, 6), (0, 1, 2, 4, 5, 6), (0, 1, 4, 5, 6)]

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    return _padding_k_idx(mp.get_nmo(per_kpoint=True), mp.get_nocc(per_kpoint=True), kind=kind)

def padded_mo_occ(mp, mo_occ):
    """
    Pads occupancy of active MOs.

    Returns:
        Padded molecular occupancy.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mp.nmo), dtype=mo_occ[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], padding_convention[k])] = mo_occ[k][frozen_mask[k]]

    return result


def padded_mo_energy(mp, mo_energy):
    """
    Pads energies of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_energy (ndarray): original non-padded molecular energies;

    Returns:
        Padded molecular energies.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mp.nmo), dtype=mo_energy[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], padding_convention[k])] = mo_energy[k][frozen_mask[k]]

    return result


def padded_mo_coeff(mp, mo_coeff):
    """
    Pads coefficients of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_coeff (ndarray): original non-padded molecular coefficients;

    Returns:
        Padded molecular coefficients.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mo_coeff[0].shape[0], mp.nmo), dtype=mo_coeff[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], numpy.arange(result.shape[1]), padding_convention[k])] = mo_coeff[k][:, frozen_mask[k]]

    return result


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    '''Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupuation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    '''
    frozen = numpy.array(frozen)
    nocc = numpy.count_nonzero(mo_occ > 0)
    nvir = len(mo_occ) - nocc
    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(numpy.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\n'
                           'frozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and numpy.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError('Freezing orbital not in MO list!\n\nkpt_idx %s\n'
                           'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1))


def get_nocc(mp, per_kpoint=False):
    '''Number of occupied orbitals for k-point calculations.

    Number of occupied orbitals for use in a calculation with k-points, taking into
    account frozen orbitals.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of occupied
            orbitals at each k-point.  False gives the max of this list.

    Returns:
        nocc (int, list of int): Number of occupied orbitals. For return type, see description of arg
            `per_kpoint`.

    '''
    for i, moocc in enumerate(mp.mo_occ):
        if numpy.any(moocc % 1 != 0):
            raise RuntimeError("Fractional occupation numbers encountered @ kp={:d}: {}. This may have been caused by "
                               "smearing of occupation numbers in the mean-field calculation. If so, consider "
                               "executing mf.smearing_method = False; mf.mo_occ = mf.get_occ() prior to calling "
                               "this".format(i, moocc))
    if mp._nocc is not None:
        return mp._nocc
    if isinstance(mp.frozen, (int, numpy.integer)):
        nocc = [(numpy.count_nonzero(mp.mo_occ[ikpt]) - mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nocc = []
        for ikpt in range(mp.nkpts):
            max_occ_idx = numpy.max(numpy.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = numpy.sum(numpy.array(mp.frozen) <= max_occ_idx)
            nocc.append(numpy.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    elif isinstance(mp.frozen[0], (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(frozen, mo_occ, ikpt) for ikpt, frozen, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nocc = []
        for ikpt, frozen in enumerate(mp.frozen):
            max_occ_idx = numpy.max(numpy.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = numpy.sum(numpy.array(frozen) <= max_occ_idx)
            nocc.append(numpy.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    else:
        raise NotImplementedError

    assert any(numpy.array(nocc) > 0), ('Must have occupied orbitals! \n\nnocc %s\nfrozen %s\nmo_occ %s' %
           (nocc, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        nocc = numpy.amax(nocc)

    return nocc


def get_nmo(mp, per_kpoint=False):
    '''Number of orbitals for k-point calculations.

    Number of orbitals for use in a calculation with k-points, taking into account
    frozen orbitals.

    Note:
        If `per_kpoint` is False, then the number of orbitals here is equal to max(nocc) + max(nvir),
        where each max is done over all k-points.  Otherwise the number of orbitals is returned
        as a list of number of orbitals at each k-point.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of orbitals at each k-point.
            For a description of False, see Note.

    Returns:
        nmo (int, list of int): Number of orbitals. For return type, see description of arg
            `per_kpoint`.

    '''
    if mp._nmo is not None:
        return mp._nmo

    if isinstance(mp.frozen, (int, numpy.integer)):
        nmo = [len(mp.mo_occ[ikpt]) - mp.frozen for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen[ikpt]) for ikpt in range(nkpts)]
    else:
        raise NotImplementedError

    assert all(numpy.array(nmo) > 0), ('Must have a positive number of orbitals!\n\nnmo %s\nfrozen %s\nmo_occ %s' %
           (nmo, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        # Depending on whether there are more occupied bands, we want to make sure that
        # nmo has enough room for max(nocc) + max(nvir) number of orbitals for occupied
        # and virtual space
        nocc = mp.get_nocc(per_kpoint=True)
        nmo = numpy.max(nocc) + numpy.max(numpy.array(nmo) - numpy.array(nocc))

    return nmo


def get_frozen_mask(mp):
    '''Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `numpy.bool`): Boolean mask of orbitals to include.

    '''
    moidx = [numpy.ones(x.size, dtype=numpy.bool) for x in mp.mo_occ]
    if isinstance(mp.frozen, (int, numpy.integer)):
        for idx in moidx:
            idx[:mp.frozen] = False
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        frozen = list(mp.frozen)
        for idx in moidx:
            idx[frozen] = False
    elif isinstance(mp.frozen[0], (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]
        for ikpt, kpt_occ in enumerate(moidx):
            kpt_occ[mp.frozen[ikpt]] = False
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
        self.kpts = mf.kpts
        self.nkpts = numpy.shape(mf.mo_energy)[0]
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
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
    #int_transform = int_transform

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
        return self.ene_tot #+ self._scf.e_tot


    def kernel(self, shift=0.0, mo_energy=None, mo_coeff=None, mo_occ=None, with_t2=WITH_T2,
               _kern=kernel):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.     
        '''
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None or mo_occ is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')

        mo_coeff, mo_energy, mo_occ = _add_padding(self, mo_coeff, mo_energy, mo_occ)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        #self.dump_flags()
        #_kern(self, mo_energy, mo_coeff, eris, with_t2, self.verbose)
        self.ene_tot, self.mo_energy= _kern(self, mo_energy, mo_coeff, mo_occ, with_t2, self.verbose)
        self._finalize()
        return self.ene_tot, self.mo_energy

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E(%s) = %.15g',
                    self.__class__.__name__, self.e_tot)
        return self

    make_veff = make_veff
    #make_amp  = make_amp
    first_BCH = first_BCH
    #second_BCH = second_BCH
    #make_rdm1 = make_rdm1
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


del(WITH_T2)