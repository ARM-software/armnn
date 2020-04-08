# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
from copy import copy

import pytest
import numpy as np
import pyarmnn as ann


def __get_tensor_info(dt):
    tensor_info = ann.TensorInfo(ann.TensorShape((2, 3)), dt)

    return tensor_info


@pytest.mark.parametrize("dt", [ann.DataType_Float32, ann.DataType_Float16,
                                ann.DataType_QAsymmU8, ann.DataType_QSymmS8,
                                ann.DataType_QAsymmS8])
def test_create_tensor_with_info(dt):
    tensor_info = __get_tensor_info(dt)
    elements = tensor_info.GetNumElements()
    num_bytes = tensor_info.GetNumBytes()
    d_type = dt

    tensor = ann.Tensor(tensor_info)

    assert tensor_info != tensor.GetInfo(), "Different objects"
    assert elements == tensor.GetNumElements()
    assert num_bytes == tensor.GetNumBytes()
    assert d_type == tensor.GetDataType()


def test_create_tensor_undefined_datatype():
    tensor_info = ann.TensorInfo()
    tensor_info.SetDataType(99)

    with pytest.raises(ValueError) as err:
        ann.Tensor(tensor_info)

    assert 'The data type provided for this Tensor is not supported.' in str(err.value)


@pytest.mark.parametrize("dt", [ann.DataType_Float32])
def test_tensor_memory_output(dt):
    tensor_info = __get_tensor_info(dt)
    tensor = ann.Tensor(tensor_info)

    # empty memory area because inference has not yet been run.
    assert tensor.get_memory_area().tolist()  # has random stuff
    assert 4 == tensor.get_memory_area().itemsize, "it is float32"


@pytest.mark.parametrize("dt", [ann.DataType_Float32, ann.DataType_Float16,
                                ann.DataType_QAsymmU8, ann.DataType_QSymmS8,
                                ann.DataType_QAsymmS8])
def test_tensor__str__(dt):
    tensor_info = __get_tensor_info(dt)
    elements = tensor_info.GetNumElements()
    num_bytes = tensor_info.GetNumBytes()
    d_type = dt
    dimensions = tensor_info.GetNumDimensions()

    tensor = ann.Tensor(tensor_info)

    assert str(tensor) == "Tensor{{DataType: {}, NumBytes: {}, NumDimensions: " \
                               "{}, NumElements: {}}}".format(d_type, num_bytes, dimensions, elements)


def test_create_empty_tensor():
    tensor = ann.Tensor()

    assert 0 == tensor.GetNumElements()
    assert 0 == tensor.GetNumBytes()
    assert tensor.get_memory_area() is None


@pytest.mark.parametrize("dt", [ann.DataType_Float32, ann.DataType_Float16,
                                ann.DataType_QAsymmU8, ann.DataType_QSymmS8,
                                ann.DataType_QAsymmS8])
def test_create_tensor_from_tensor(dt):
    tensor_info = __get_tensor_info(dt)
    tensor = ann.Tensor(tensor_info)
    copied_tensor = ann.Tensor(tensor)

    assert copied_tensor != tensor, "Different objects"
    assert copied_tensor.GetInfo() != tensor.GetInfo(), "Different objects"
    assert copied_tensor.get_memory_area().ctypes.data == tensor.get_memory_area().ctypes.data,  "Same memory area"
    assert copied_tensor.GetNumElements() == tensor.GetNumElements()
    assert copied_tensor.GetNumBytes() == tensor.GetNumBytes()
    assert copied_tensor.GetDataType() == tensor.GetDataType()


@pytest.mark.parametrize("dt", [ann.DataType_Float32, ann.DataType_Float16,
                                ann.DataType_QAsymmU8, ann.DataType_QSymmS8,
                                ann.DataType_QAsymmS8])
def test_copy_tensor(dt):
    tensor = ann.Tensor(__get_tensor_info(dt))
    copied_tensor = copy(tensor)

    assert copied_tensor != tensor, "Different objects"
    assert copied_tensor.GetInfo() != tensor.GetInfo(), "Different objects"
    assert copied_tensor.get_memory_area().ctypes.data == tensor.get_memory_area().ctypes.data,  "Same memory area"
    assert copied_tensor.GetNumElements() == tensor.GetNumElements()
    assert copied_tensor.GetNumBytes() == tensor.GetNumBytes()
    assert copied_tensor.GetDataType() == tensor.GetDataType()


@pytest.mark.parametrize("dt", [ann.DataType_Float32, ann.DataType_Float16,
                                ann.DataType_QAsymmU8, ann.DataType_QSymmS8,
                                ann.DataType_QAsymmS8])
def test_copied_tensor_has_memory_area_access_after_deletion_of_original_tensor(dt):

    tensor = ann.Tensor(__get_tensor_info(dt))

    tensor.get_memory_area()[0] = 100

    initial_mem_copy = np.array(tensor.get_memory_area())

    assert 100 == initial_mem_copy[0]

    copied_tensor = ann.Tensor(tensor)

    del tensor
    np.testing.assert_array_equal(copied_tensor.get_memory_area(), initial_mem_copy)
    assert 100 == copied_tensor.get_memory_area()[0]


def test_create_const_tensor_incorrect_args():
    with pytest.raises(ValueError) as err:
        ann.Tensor('something', 'something')

    expected_error_message = "Incorrect number of arguments or type of arguments provided to create Tensor."
    assert expected_error_message in str(err.value)


@pytest.mark.parametrize("dt", [ann.DataType_Float16])
def test_tensor_memory_output_fp16(dt):
    # Check Tensor with float16
    tensor_info = __get_tensor_info(dt)
    tensor = ann.Tensor(tensor_info)

    assert tensor.GetNumElements() == 6
    assert tensor.GetNumBytes() == 12
    assert tensor.GetDataType() == ann.DataType_Float16
