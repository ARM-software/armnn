# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import pyarmnn as ann


def test_tensor_info_ctor_shape():
    tensor_shape = ann.TensorShape((1, 1, 2))

    tensor_info = ann.TensorInfo(tensor_shape, ann.DataType_QAsymmU8, 0.5, 1)

    assert 2 == tensor_info.GetNumElements()
    assert 3 == tensor_info.GetNumDimensions()
    assert ann.DataType_QAsymmU8 == tensor_info.GetDataType()
    assert 0.5 == tensor_info.GetQuantizationScale()
    assert 1 == tensor_info.GetQuantizationOffset()

    shape = tensor_info.GetShape()

    assert 2 == shape.GetNumElements()
    assert 3 == shape.GetNumDimensions()


def test_tensor_info__str__():
    tensor_info = ann.TensorInfo(ann.TensorShape((2, 3)), ann.DataType_QAsymmU8, 0.5, 1, True)

    assert tensor_info.__str__() == "TensorInfo{DataType: 2, IsQuantized: 1, QuantizationScale: 0.500000, " \
                                    "QuantizationOffset: 1, IsConstant: 1, NumDimensions: 2, NumElements: 6}"
