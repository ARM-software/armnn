# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest
import pyarmnn as ann


def test_activation_function():
    assert 0 == ann.ActivationFunction_Sigmoid
    assert 1 == ann.ActivationFunction_TanH
    assert 2 == ann.ActivationFunction_Linear
    assert 3 == ann.ActivationFunction_ReLu
    assert 4 == ann.ActivationFunction_BoundedReLu
    assert 5 == ann.ActivationFunction_SoftReLu
    assert 6 == ann.ActivationFunction_LeakyReLu
    assert 7 == ann.ActivationFunction_Abs
    assert 8 == ann.ActivationFunction_Sqrt
    assert 9 == ann.ActivationFunction_Square


def test_permutation_vector():
    pv = ann.PermutationVector((0, 2, 3, 1))
    assert pv[0] == 0
    assert pv[2] == 3

    pv2 = ann.PermutationVector((0, 2, 3, 1))
    assert pv == pv2

    pv4 = ann.PermutationVector((0, 3, 1, 2))
    assert pv.IsInverse(pv4)

    with pytest.raises(ValueError) as err:
        pv4[4]

    assert err.type is ValueError
