import unittest

import numpy as np

from tacs import TACS, constitutive, elements


class ConstitutiveTest(unittest.TestCase):
    def setUp(self):
        # fd/cs step size
        if TACS.dtype is complex:
            self.dh = 1e-50
            self.rtol = 1e-11
        else:
            self.dh = 1e-6
            self.rtol = 1e-2
        self.dtype = TACS.dtype

        # Basically, only check relative tolerance
        self.atol = 1e99
        self.print_level = 0

        # Set element index
        self.elem_index = 0

        # Set the variable arrays
        self.x = np.ones(3, dtype=self.dtype)
        self.pt = np.zeros(3)
        # This constituitive class has no dvs
        self.dvs = np.array([], dtype=self.dtype)

        # 6 dof stiffness matrix
        """
        kx  0  0   0   0   0
         0 ky  0   0   0   0
         0  0 kz   0   0   0
         0  0  0 krx   0   0
         0  0  0   0 kry   0
         0  0  0   0   0 krz
        """
        k = np.arange(6, dtype=self.dtype)

        # Create stiffness (need class)
        self.con = constitutive.DOFSpringConstitutive(k=k)

        # Seed random number generator in tacs for consistent test results
        elements.SeedRandomGenerator(0)

    def test_constitutive_density(self):
        # Test density dv sensitivity
        fail = constitutive.TestConstitutiveDensity(
            self.con,
            self.elem_index,
            self.pt,
            self.x,
            self.dvs,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_constitutive_specific_heat(self):
        # Test specific heat dv sensitivity
        fail = constitutive.TestConstitutiveSpecificHeat(
            self.con,
            self.elem_index,
            self.pt,
            self.x,
            self.dvs,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_constitutive_heat_flux(self):
        # Test heat flux dv sensitivity
        fail = constitutive.TestConstitutiveHeatFlux(
            self.con,
            self.elem_index,
            self.pt,
            self.x,
            self.dvs,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_constitutive_stress(self):
        # Test stress dv sensitivity
        fail = constitutive.TestConstitutiveStress(
            self.con,
            self.elem_index,
            self.pt,
            self.x,
            self.dvs,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_constitutive_thermal_strain(self):
        # Test thermal strain dv sensitivity
        fail = constitutive.TestConstitutiveThermalStrain(
            self.con,
            self.elem_index,
            self.pt,
            self.x,
            self.dvs,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_constitutive_failure(self):
        # Test failure dv sensitivity
        fail = constitutive.TestConstitutiveFailure(
            self.con,
            self.elem_index,
            self.pt,
            self.x,
            self.dvs,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_constitutive_failure_strain_sens(self):
        # Test failure dv sensitivity
        fail = constitutive.TestConstitutiveFailureStrainSens(
            self.con,
            self.elem_index,
            self.pt,
            self.x,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)
