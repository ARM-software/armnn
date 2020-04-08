# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest
import pyarmnn as ann


def test_tensor_shape_tuple():
    tensor_shape = ann.TensorShape((1, 2, 3))

    assert 3 == tensor_shape.GetNumDimensions()
    assert 6 == tensor_shape.GetNumElements()


def test_tensor_shape_one():
    tensor_shape = ann.TensorShape((10,))
    assert 1 == tensor_shape.GetNumDimensions()
    assert 10 == tensor_shape.GetNumElements()


def test_tensor_shape_empty():
    with pytest.raises(RuntimeError) as err:
        ann.TensorShape(())

    assert "Tensor numDimensions must be greater than 0" in str(err.value)


def test_tensor_shape_tuple_mess():
    tensor_shape = ann.TensorShape((1, "2", 3.0))

    assert 3 == tensor_shape.GetNumDimensions()
    assert 6 == tensor_shape.GetNumElements()


def test_tensor_shape_list():

    with pytest.raises(TypeError) as err:
        ann.TensorShape([1, 2, 3])

    assert "Argument is not a tuple" in str(err.value)


def test_tensor_shape_tuple_mess_fail():

    with pytest.raises(TypeError) as err:
        ann.TensorShape((1, "two", 3.0))

    assert "All elements must be numbers" in str(err.value)


def test_tensor_shape_varags():
    with pytest.raises(TypeError) as err:
        ann.TensorShape(1, 2, 3)

    assert "__init__() takes 2 positional arguments but 4 were given" in str(err.value)


def test_tensor_shape__get_item_out_of_bounds():
    tensor_shape = ann.TensorShape((1, 2, 3))
    with pytest.raises(ValueError) as err:
        for i in range(4):
            tensor_shape[i]

    assert "Invalid dimension index: 3 (number of dimensions is 3)" in str(err.value)


def test_tensor_shape__set_item_out_of_bounds():
    tensor_shape = ann.TensorShape((1, 2, 3))
    with pytest.raises(ValueError) as err:
        for i in range(4):
            tensor_shape[i] = 1

    assert "Invalid dimension index: 3 (number of dimensions is 3)" in str(err.value)


def test_tensor_shape___str__():
    tensor_shape = ann.TensorShape((1, 2, 3))

    assert str(tensor_shape) == "TensorShape{Shape(1, 2, 3), NumDimensions: 3, NumElements: 6}"
