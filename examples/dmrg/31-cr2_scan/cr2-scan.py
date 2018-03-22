#!/usr/bin/env python

'''
Scanning Cr2 molecule dissociation curve with regular CASSCF module (see
examples/mcscf/31-cr2_scan) is not a difficult task.  The calculation becomes
challenge when large active space is required in the DMRG-CASSCF (or
FCIQMC-CASSCF, SHCI-CASSCF) methods.  In this example, we demonstrated
how to solve the following problems when running a CAS(12,42) curve scanning:
1. How to increase the size of active space from a small CASSCF calculation.
2. How to project the active space from one point to another
'''

import os
import numpy as np
from shutil import copyfile
from pyscf import gto, scf, mcscf, lib, symm
from pyscf.dmrgscf import DMRGSCF, settings

settings.MPIPREFIX = 'mpirun -np 8'

def run(b, dm_guess, mo_guess, ci=None):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'cr2-%3.2f.out' % b
    mol.atom = [
        ['Cr',(  0.,  0., -b/2)],
        ['Cr',(  0.,  0.,  b/2)],
    ]
    mol.basis = 'ccpvdzdk'
    mol.symmetry = True
    mol.build()

    mf = scf.sfx2c1e(scf.RHF(mol))
    mf.max_cycle = 100
    mf.conv_tol = 1e-9
    mf.kernel(dm_guess)

#---------------------
# CAS(12,12)
#
# The active space of CAS(12,42) can be generated by function sort_mo_by_irrep,
# or AVAS, dmet_cas methods.  Here we first run a CAS(12,12) calculation and
# using the lowest CASSCF canonicalized orbitals in the CAS(12,42) calculation
# as the core and valence orbitals.
    mc = mcscf.CASSCF(mf, 12, 12)
    if mo_guess is None:
# the initial guess for first calculation
        ncas = {'A1g' : 2, 'E1gx' : 1, 'E1gy' : 1, 'E2gx' : 1, 'E2gy' : 1,
                'A1u' : 2, 'E1ux' : 1, 'E1uy' : 1, 'E2ux' : 1, 'E2uy' : 1}
        mo_guess = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    else:
        mo_guess = mcscf.project_init_guess(mc, mo_guess)

# Projection may destroy the spatial symmetry of the MCSCF orbitals.
        try:
            print('Irreps of the projected CAS(12,12) orbitals',
                  symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_guess))
        except:
            print('Projected CAS(12,12) orbitals does not have right symmetry')

# FCI solver with multi-threads is not stable enough for this sytem
    mc.fcisolver.threads = 1
# To avoid spin contamination
    mc.fix_spin_()
# By default, canonicalization as well as the CASSCF optimization do not
# change the orbital symmetry labels. "Orbital energy" mc.mo_energy, the
# diagonal elements of the general fock matrix, may have wrong ordering.
# To use the lowest CASSCF orbitals as the core + active orbitals in the next
# step, we have to sort the orbitals based on the "orbital energy". This
# operation will change the orbital symmetry labels.
    mc.sorting_mo_energy = True
# "mc.natorb = True" will transform the active space, to its natural orbital
# representation.  By default, the active space orbitals are NOT reordered
# even the option sorting_mo_energy is enabled.  Transforming the active space
# orbitals is a dangerous operation because it needs also to update the CI
# wavefunction. For DMRG solver (or other approximate FCI solver), the CI
# wavefunction may not be able to consistently converged and it leads to
# inconsistency between the orbital space and the CI wavefunction
# representations.  Unless required by the following CAS calculation, setting
# mc.natorb=True should be avoided
#    mc.natorb = True

    mc.kernel(mo_guess, ci)
    mc.analyze()

#---------------------
# CAS(12,42)
#
# Using the lowest CASSCF canonicalized orbitals in the CAS(12,42) DMRG-CASSCF
# calculation.
    norb = 42
    nelec = 12
    mc1 = DMRGSCF(mf, norb, nelec)
    mc1.fcisolver.maxM = 4000

# Enable internal rotation since the bond dimension of DMRG calculation is
# small, the active space energy can be optimized wrt to the orbital rotation
# within the active space.
    mc1.internal_rotation = True
# Sorting the orbital energy is not a step of must here.
#    mc1.sorting_mo_energy = True
#    mc1.natorb = True

    mc1.kernel()

# Passing the results as an initial guess to the next point.
    return mf.make_rdm1(), mc.mo_coeff, mc.ci

dm = mo = ci = None
for b in np.arange(1.6, 3.5, 0.2):
    dm, mo, ci = run(b, dm, mo, ci)
