# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest

from pyarmnn import BackendOptions, BackendOption, BackendId, OptimizerOptions, ShapeInferenceMethod_InferAndValidate


@pytest.mark.parametrize("data", (True, -100, 128, 0.12345, 'string'))
def test_backend_option_ctor(data):
    bo = BackendOption("name", data)
    assert "name" == bo.GetName()


def test_backend_options_ctor():
    backend_id = BackendId('a')
    bos = BackendOptions(backend_id)

    assert 'a' == str(bos.GetBackendId())

    another_bos = BackendOptions(bos)
    assert 'a' == str(another_bos.GetBackendId())


def test_backend_options_add():
    backend_id = BackendId('a')
    bos = BackendOptions(backend_id)
    bo = BackendOption("name", 1)
    bos.AddOption(bo)

    assert 1 == bos.GetOptionCount()
    assert 1 == len(bos)

    assert 'name' == bos[0].GetName()
    assert 'name' == bos.GetOption(0).GetName()
    for option in bos:
        assert 'name' == option.GetName()

    bos.AddOption(BackendOption("name2", 2))

    assert 2 == bos.GetOptionCount()
    assert 2 == len(bos)


def test_backend_option_ownership():
    backend_id = BackendId('b')
    bos = BackendOptions(backend_id)
    bo = BackendOption('option', True)
    bos.AddOption(bo)

    assert bo.thisown

    del bo

    assert 1 == bos.GetOptionCount()
    option = bos[0]
    assert not option.thisown
    assert 'option' == option.GetName()

    del option

    option_again = bos[0]
    assert not option_again.thisown
    assert 'option' == option_again.GetName()


def test_optimizer_options_with_model_opt():
    a = BackendOptions(BackendId('a'))

    oo = OptimizerOptions(True,
                          False,
                          False,
                          ShapeInferenceMethod_InferAndValidate,
                          True,
                          [a],
                          True)

    mo = oo.m_ModelOptions

    assert 1 == len(mo)
    assert 'a' == str(mo[0].GetBackendId())

    b = BackendOptions(BackendId('b'))

    c = BackendOptions(BackendId('c'))

    oo.m_ModelOptions = (a, b, c)

    mo = oo.m_ModelOptions

    assert 3 == len(oo.m_ModelOptions)

    assert 'a' == str(mo[0].GetBackendId())
    assert 'b' == str(mo[1].GetBackendId())
    assert 'c' == str(mo[2].GetBackendId())


def test_optimizer_option_default():
    oo = OptimizerOptions(True,
                          False,
                          False,
                          ShapeInferenceMethod_InferAndValidate,
                          True)

    assert 0 == len(oo.m_ModelOptions)


def test_optimizer_options_fail():
    a = BackendOptions(BackendId('a'))

    with pytest.raises(TypeError) as err:
        OptimizerOptions(True,
                         False,
                         False,
                         ShapeInferenceMethod_InferAndValidate,
                         True,
                         a,
                         True)

    assert "Wrong number or type of arguments" in str(err.value)

    with pytest.raises(TypeError) as err:
        oo = OptimizerOptions(True,
                              False,
                              False,
                              ShapeInferenceMethod_InferAndValidate,
                              True)

        oo.m_ModelOptions = 'nonsense'

    assert "in method 'OptimizerOptions_m_ModelOptions_set', argument 2" in str(err.value)

    with pytest.raises(TypeError) as err:
        oo = OptimizerOptions(True,
                              False,
                              False,
                              ShapeInferenceMethod_InferAndValidate,
                              True)

        oo.m_ModelOptions = ['nonsense', a]

    assert "in method 'OptimizerOptions_m_ModelOptions_set', argument 2" in str(err.value)
