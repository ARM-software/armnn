# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest
import numpy as np

import pyarmnn as ann


def _get_const_tensor_info(dt):
    tensor_info = ann.TensorInfo(ann.TensorShape((2, 3)), dt, 0.0, 0, True)

    return tensor_info


@pytest.mark.parametrize("dt, data",
                         [
                             (ann.DataType_Float32, np.random.randint(1, size=(2, 4)).astype(np.float32)),
                             (ann.DataType_Float16, np.random.randint(1, size=(2, 4)).astype(np.float16)),
                             (ann.DataType_QAsymmU8, np.random.randint(1, size=(2, 4)).astype(np.uint8)),
                             (ann.DataType_QAsymmS8, np.random.randint(1, size=(2, 4)).astype(np.int8)),
                             (ann.DataType_QSymmS8, np.random.randint(1, size=(2, 4)).astype(np.int8)),
                             (ann.DataType_Signed32, np.random.randint(1, size=(2, 4)).astype(np.int32)),
                             (ann.DataType_QSymmS16, np.random.randint(1, size=(2, 4)).astype(np.int16))
                         ], ids=['float32', 'float16', 'unsigned int8', 'signed int8', 'signed int8', 'int32', 'int16'])
def test_const_tensor_too_many_elements(dt, data):
    tensor_info = _get_const_tensor_info(dt)
    num_bytes = tensor_info.GetNumBytes()

    with pytest.raises(ValueError) as err:
        ann.ConstTensor(tensor_info, data)

    assert 'ConstTensor requires {} bytes, {} provided.'.format(num_bytes, data.nbytes) in str(err.value)


@pytest.mark.parametrize("dt, data",
                         [
                             (ann.DataType_Float32, np.random.randint(1, size=(2, 2)).astype(np.float32)),
                             (ann.DataType_Float16, np.random.randint(1, size=(2, 2)).astype(np.float16)),
                             (ann.DataType_QAsymmU8, np.random.randint(1, size=(2, 2)).astype(np.uint8)),
                             (ann.DataType_QAsymmS8, np.random.randint(1, size=(2, 2)).astype(np.int8)),
                             (ann.DataType_QSymmS8, np.random.randint(1, size=(2, 2)).astype(np.int8)),
                             (ann.DataType_Signed32, np.random.randint(1, size=(2, 2)).astype(np.int32)),
                             (ann.DataType_QSymmS16, np.random.randint(1, size=(2, 2)).astype(np.int16))
                         ], ids=['float32', 'float16', 'unsigned int8', 'signed int8', 'signed int8', 'int32', 'int16'])
def test_const_tensor_too_little_elements(dt, data):
    tensor_info = _get_const_tensor_info(dt)
    num_bytes = tensor_info.GetNumBytes()

    with pytest.raises(ValueError) as err:
        ann.ConstTensor(tensor_info, data)

    assert 'ConstTensor requires {} bytes, {} provided.'.format(num_bytes, data.nbytes) in str(err.value)


@pytest.mark.parametrize("dt, data",
                         [
                             (ann.DataType_Float32, np.random.randint(1, size=(2, 2, 3, 3)).astype(np.float32)),
                             (ann.DataType_Float16, np.random.randint(1, size=(2, 2, 3, 3)).astype(np.float16)),
                             (ann.DataType_QAsymmU8, np.random.randint(1, size=(2, 2, 3, 3)).astype(np.uint8)),
                             (ann.DataType_QAsymmS8, np.random.randint(1, size=(2, 2, 3, 3)).astype(np.int8)),
                             (ann.DataType_QSymmS8, np.random.randint(1, size=(2, 2, 3, 3)).astype(np.int8)),
                             (ann.DataType_Signed32, np.random.randint(1, size=(2, 2, 3, 3)).astype(np.int32)),
                             (ann.DataType_QSymmS16, np.random.randint(1, size=(2, 2, 3, 3)).astype(np.int16))
                         ], ids=['float32', 'float16', 'unsigned int8', 'signed int8', 'signed int8', 'int32', 'int16'])
def test_const_tensor_multi_dimensional_input(dt, data):
    tensor = ann.ConstTensor(ann.TensorInfo(ann.TensorShape((2, 2, 3, 3)), dt, 0.0, 0, True), data)

    assert data.size == tensor.GetNumElements()
    assert data.nbytes == tensor.GetNumBytes()
    assert dt == tensor.GetDataType()
    assert tensor.get_memory_area().data


def test_create_const_tensor_from_tensor():
    tensor_info = ann.TensorInfo(ann.TensorShape((2, 3)), ann.DataType_Float32, 0.0, 0, True)
    tensor = ann.Tensor(tensor_info)
    copied_tensor = ann.ConstTensor(tensor)

    assert copied_tensor != tensor, "Different objects"
    assert copied_tensor.GetInfo() != tensor.GetInfo(), "Different objects"
    assert copied_tensor.get_memory_area().ctypes.data == tensor.get_memory_area().ctypes.data, "Same memory area"
    assert copied_tensor.GetNumElements() == tensor.GetNumElements()
    assert copied_tensor.GetNumBytes() == tensor.GetNumBytes()
    assert copied_tensor.GetDataType() == tensor.GetDataType()


def test_const_tensor_from_tensor_has_memory_area_access_after_deletion_of_original_tensor():
    tensor_info = ann.TensorInfo(ann.TensorShape((2, 3)), ann.DataType_Float32, 0.0, 0, True)
    tensor = ann.Tensor(tensor_info)

    tensor.get_memory_area()[0] = 100

    copied_mem = tensor.get_memory_area().copy()

    assert 100 == copied_mem[0], "Memory was copied correctly"

    copied_tensor = ann.ConstTensor(tensor)

    tensor.get_memory_area()[0] = 200

    assert 200 == tensor.get_memory_area()[0], "Tensor and copied Tensor point to the same memory"
    assert 200 == copied_tensor.get_memory_area()[0], "Tensor and copied Tensor point to the same memory"

    assert 100 == copied_mem[0], "Copied test memory not affected"

    copied_mem[0] = 200  # modify test memory to equal copied Tensor

    del tensor
    np.testing.assert_array_equal(copied_tensor.get_memory_area(), copied_mem), "After initial tensor was deleted, " \
                                                                                "copied Tensor still has " \
                                                                                "its memory as expected"


def test_create_const_tensor_incorrect_args():
    with pytest.raises(ValueError) as err:
        ann.ConstTensor('something', 'something')

    expected_error_message = "Incorrect number of arguments or type of arguments provided to create Const Tensor."
    assert expected_error_message in str(err.value)


@pytest.mark.parametrize("dt, data",
                         [
                             # -1 not in data type enum
                             (-1, np.random.randint(1, size=(2, 3)).astype(np.float32)),
                         ], ids=['unknown'])
def test_const_tensor_unsupported_datatype(dt, data):
    tensor_info = _get_const_tensor_info(dt)

    with pytest.raises(ValueError) as err:
        ann.ConstTensor(tensor_info, data)

    assert 'The data type provided for this Tensor is not supported: -1' in str(err.value)


@pytest.mark.parametrize("dt, data",
                         [
                             (ann.DataType_Float32, [[1, 1, 1], [1, 1, 1]]),
                             (ann.DataType_Float16, [[1, 1, 1], [1, 1, 1]]),
                             (ann.DataType_QAsymmU8, [[1, 1, 1], [1, 1, 1]]),
                             (ann.DataType_QAsymmS8, [[1, 1, 1], [1, 1, 1]]),
                             (ann.DataType_QSymmS8, [[1, 1, 1], [1, 1, 1]])
                         ], ids=['float32', 'float16', 'unsigned int8', 'signed int8', 'signed int8'])
def test_const_tensor_incorrect_input_datatype(dt, data):
    tensor_info = _get_const_tensor_info(dt)

    with pytest.raises(TypeError) as err:
        ann.ConstTensor(tensor_info, data)

    assert 'Data must be provided as a numpy array.' in str(err.value)


@pytest.mark.parametrize("dt, data",
                         [
                             (ann.DataType_Float32, np.random.randint(1, size=(2, 3)).astype(np.float32)),
                             (ann.DataType_Float16, np.random.randint(1, size=(2, 3)).astype(np.float16)),
                             (ann.DataType_QAsymmU8, np.random.randint(1, size=(2, 3)).astype(np.uint8)),
                             (ann.DataType_QAsymmS8, np.random.randint(1, size=(2, 3)).astype(np.int8)),
                             (ann.DataType_QSymmS8, np.random.randint(1, size=(2, 3)).astype(np.int8)),
                             (ann.DataType_Signed32, np.random.randint(1, size=(2, 3)).astype(np.int32)),
                             (ann.DataType_QSymmS16, np.random.randint(1, size=(2, 3)).astype(np.int16))
                         ], ids=['float32', 'float16', 'unsigned int8', 'signed int8', 'signed int8', 'int32', 'int16'])
class TestNumpyDataTypes:

    def test_copy_const_tensor(self, dt, data):
        tensor_info = _get_const_tensor_info(dt)
        tensor = ann.ConstTensor(tensor_info, data)
        copied_tensor = ann.ConstTensor(tensor)

        assert copied_tensor != tensor, "Different objects"
        assert copied_tensor.GetInfo() != tensor.GetInfo(), "Different objects"
        assert copied_tensor.get_memory_area().ctypes.data == tensor.get_memory_area().ctypes.data, "Same memory area"
        assert copied_tensor.GetNumElements() == tensor.GetNumElements()
        assert copied_tensor.GetNumBytes() == tensor.GetNumBytes()
        assert copied_tensor.GetDataType() == tensor.GetDataType()

    def test_const_tensor__str__(self, dt, data):
        tensor_info = _get_const_tensor_info(dt)
        d_type = tensor_info.GetDataType()
        num_dimensions = tensor_info.GetNumDimensions()
        num_bytes = tensor_info.GetNumBytes()
        num_elements = tensor_info.GetNumElements()
        tensor = ann.ConstTensor(tensor_info, data)

        assert str(tensor) == "ConstTensor{{DataType: {}, NumBytes: {}, NumDimensions: " \
                              "{}, NumElements: {}}}".format(d_type, num_bytes, num_dimensions, num_elements)

    def test_const_tensor_with_info(self, dt, data):
        tensor_info = _get_const_tensor_info(dt)
        elements = tensor_info.GetNumElements()
        num_bytes = tensor_info.GetNumBytes()
        d_type = dt

        tensor = ann.ConstTensor(tensor_info, data)

        assert tensor_info != tensor.GetInfo(), "Different objects"
        assert elements == tensor.GetNumElements()
        assert num_bytes == tensor.GetNumBytes()
        assert d_type == tensor.GetDataType()

    def test_immutable_memory(self, dt, data):
        tensor_info = _get_const_tensor_info(dt)

        tensor = ann.ConstTensor(tensor_info, data)

        with pytest.raises(ValueError) as err:
            tensor.get_memory_area()[0] = 0

        assert 'is read-only' in str(err.value)

    def test_numpy_dtype_matches_ann_dtype(self, dt, data):
        np_data_type_mapping = {ann.DataType_QAsymmU8: np.uint8,
                                ann.DataType_QAsymmS8: np.int8,
                                ann.DataType_QSymmS8: np.int8,
                                ann.DataType_Float32: np.float32,
                                ann.DataType_QSymmS16: np.int16,
                                ann.DataType_Signed32: np.int32,
                                ann.DataType_Float16: np.float16}

        tensor_info = _get_const_tensor_info(dt)
        tensor = ann.ConstTensor(tensor_info, data)
        assert np_data_type_mapping[tensor.GetDataType()] == data.dtype


# This test checks that mismatched numpy and PyArmNN datatypes with same number of bits raises correct error.
@pytest.mark.parametrize("dt, data",
                         [
                             (ann.DataType_Float32, np.random.randint(1, size=(2, 3)).astype(np.int32)),
                             (ann.DataType_Float16, np.random.randint(1, size=(2, 3)).astype(np.int16)),
                             (ann.DataType_QAsymmU8, np.random.randint(1, size=(2, 3)).astype(np.int8)),
                             (ann.DataType_QAsymmS8, np.random.randint(1, size=(2, 3)).astype(np.uint8)),
                             (ann.DataType_QSymmS8, np.random.randint(1, size=(2, 3)).astype(np.uint8)),
                             (ann.DataType_Signed32, np.random.randint(1, size=(2, 3)).astype(np.float32)),
                             (ann.DataType_QSymmS16, np.random.randint(1, size=(2, 3)).astype(np.float16))
                         ], ids=['float32', 'float16', 'unsigned int8', 'signed int8', 'signed int8', 'int32', 'int16'])
def test_numpy_dtype_mismatch_ann_dtype(dt, data):
    np_data_type_mapping = {ann.DataType_QAsymmU8: np.uint8,
                            ann.DataType_QAsymmS8: np.int8,
                            ann.DataType_QSymmS8: np.int8,
                            ann.DataType_Float32: np.float32,
                            ann.DataType_QSymmS16: np.int16,
                            ann.DataType_Signed32: np.int32,
                            ann.DataType_Float16: np.float16}

    tensor_info = _get_const_tensor_info(dt)
    with pytest.raises(TypeError) as err:
        ann.ConstTensor(tensor_info, data)

    assert str(err.value) == "Expected data to have type {} for type {} but instead got numpy.{}".format(
        np_data_type_mapping[dt], dt, data.dtype)


@pytest.mark.parametrize("dt, data",
                         [
                             (ann.DataType_Float32, np.random.randint(1, size=(2, 3)).astype(np.float32)),
                             (ann.DataType_Float16, np.random.randint(1, size=(2, 3)).astype(np.float16)),
                             (ann.DataType_QAsymmU8, np.random.randint(1, size=(2, 3)).astype(np.uint8)),
                             (ann.DataType_QAsymmS8, np.random.randint(1, size=(2, 3)).astype(np.int8)),
                             (ann.DataType_QSymmS8, np.random.randint(1, size=(2, 3)).astype(np.int8)),
                             (ann.DataType_Signed32, np.random.randint(1, size=(2, 3)).astype(np.int32)),
                             (ann.DataType_QSymmS16, np.random.randint(1, size=(2, 3)).astype(np.int16))
                         ], ids=['float32', 'float16', 'unsigned int8', 'signed int8', 'signed int8', 'int32', 'int16'])
class TestConstTensorConstructorErrors:

    def test_tensorinfo_isconstant_not_set(self, dt, data):
        with pytest.raises(ValueError) as err:
            ann.ConstTensor(ann.TensorInfo(ann.TensorShape((2, 2, 3, 3)), dt, 0.0, 0, False), data)

        assert str(err.value) == "TensorInfo when initializing ConstTensor must be set to constant."

    def test_tensor_tensorinfo_isconstant_not_set(self, dt, data):
        with pytest.raises(ValueError) as err:
            ann.ConstTensor(ann.Tensor(ann.TensorInfo(ann.TensorShape((2, 2, 3, 3)), dt, 0.0, 0, False), data))

        assert str(err.value) ==  "TensorInfo of Tensor when initializing ConstTensor must be set to constant."