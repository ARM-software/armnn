# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
import inspect

import pytest

import pyarmnn as ann
import numpy as np
import pyarmnn._generated.pyarmnn as generated


def test_activation_descriptor_default_values():
    desc = ann.ActivationDescriptor()
    assert desc.m_Function == ann.ActivationFunction_Sigmoid
    assert desc.m_A == 0
    assert desc.m_B == 0


def test_argminmax_descriptor_default_values():
    desc = ann.ArgMinMaxDescriptor()
    assert desc.m_Function == ann.ArgMinMaxFunction_Min
    assert desc.m_Axis == -1


def test_batchnormalization_descriptor_default_values():
    desc = ann.BatchNormalizationDescriptor()
    assert desc.m_DataLayout == ann.DataLayout_NCHW
    np.allclose(0.0001, desc.m_Eps)


def test_batchtospacend_descriptor_default_values():
    desc = ann.BatchToSpaceNdDescriptor()
    assert desc.m_DataLayout == ann.DataLayout_NCHW
    assert [1, 1] == desc.m_BlockShape
    assert [(0, 0), (0, 0)] == desc.m_Crops


def test_batchtospacend_descriptor_assignment():
    desc = ann.BatchToSpaceNdDescriptor()
    desc.m_BlockShape = (1, 2, 3)

    ololo = [(1, 2), (3, 4)]
    size_1 = len(ololo)
    desc.m_Crops = ololo

    assert size_1 == len(ololo)
    desc.m_DataLayout = ann.DataLayout_NHWC
    assert ann.DataLayout_NHWC == desc.m_DataLayout
    assert [1, 2, 3] == desc.m_BlockShape
    assert [(1, 2), (3, 4)] == desc.m_Crops


@pytest.mark.parametrize("input_shape, value, vtype", [([-1], -1, 'int'), (("one", "two"), "'one'", 'str'),
                                                       ([1.33, 4.55], 1.33, 'float'),
                                                       ([{1: "one"}], "{1: 'one'}", 'dict')], ids=lambda x: str(x))
def test_batchtospacend_descriptor_rubbish_assignment_shape(input_shape, value, vtype):
    desc = ann.BatchToSpaceNdDescriptor()
    with pytest.raises(TypeError) as err:
        desc.m_BlockShape = input_shape

    assert "Failed to convert python input value {} of type '{}' to C type 'j'".format(value, vtype) in str(err.value)


@pytest.mark.parametrize("input_crops, value, vtype", [([(1, 2), (3, 4, 5)], '(3, 4, 5)', 'tuple'),
                                                       ([(1, 'one')], "(1, 'one')", 'tuple'),
                                                       ([-1], -1, 'int'),
                                                       ([(1, (1, 2))], '(1, (1, 2))', 'tuple'),
                                                       ([[1, [1, 2]]], '[1, [1, 2]]', 'list')
                                                       ], ids=lambda x: str(x))
def test_batchtospacend_descriptor_rubbish_assignment_crops(input_crops, value, vtype):
    desc = ann.BatchToSpaceNdDescriptor()
    with pytest.raises(TypeError) as err:
        desc.m_Crops = input_crops

    assert "Failed to convert python input value {} of type '{}' to C type".format(value, vtype) in str(err.value)


def test_batchtospacend_descriptor_empty_assignment():
    desc = ann.BatchToSpaceNdDescriptor()
    desc.m_BlockShape = []
    assert [] == desc.m_BlockShape


def test_batchtospacend_descriptor_ctor():
    desc = ann.BatchToSpaceNdDescriptor([1, 2, 3], [(4, 5), (6, 7)])
    assert desc.m_DataLayout == ann.DataLayout_NCHW
    assert [1, 2, 3] == desc.m_BlockShape
    assert [(4, 5), (6, 7)] == desc.m_Crops


def test_channelshuffle_descriptor_default_values():
    desc = ann.ChannelShuffleDescriptor()
    assert desc.m_Axis == 0
    assert desc.m_NumGroups == 0

def test_convolution2d_descriptor_default_values():
    desc = ann.Convolution2dDescriptor()
    assert desc.m_PadLeft == 0
    assert desc.m_PadTop == 0
    assert desc.m_PadRight == 0
    assert desc.m_PadBottom == 0
    assert desc.m_StrideX == 1
    assert desc.m_StrideY == 1
    assert desc.m_DilationX == 1
    assert desc.m_DilationY == 1
    assert desc.m_BiasEnabled == False
    assert desc.m_DataLayout == ann.DataLayout_NCHW

def test_convolution3d_descriptor_default_values():
    desc = ann.Convolution3dDescriptor()
    assert desc.m_PadLeft == 0
    assert desc.m_PadTop == 0
    assert desc.m_PadRight == 0
    assert desc.m_PadBottom == 0
    assert desc.m_PadFront == 0
    assert desc.m_PadBack == 0
    assert desc.m_StrideX == 1
    assert desc.m_StrideY == 1
    assert desc.m_StrideZ == 1
    assert desc.m_DilationX == 1
    assert desc.m_DilationY == 1
    assert desc.m_DilationZ == 1
    assert desc.m_BiasEnabled == False
    assert desc.m_DataLayout == ann.DataLayout_NDHWC


def test_depthtospace_descriptor_default_values():
    desc = ann.DepthToSpaceDescriptor()
    assert desc.m_BlockSize == 1
    assert desc.m_DataLayout == ann.DataLayout_NHWC


def test_depthwise_convolution2d_descriptor_default_values():
    desc = ann.DepthwiseConvolution2dDescriptor()
    assert desc.m_PadLeft == 0
    assert desc.m_PadTop == 0
    assert desc.m_PadRight == 0
    assert desc.m_PadBottom == 0
    assert desc.m_StrideX == 1
    assert desc.m_StrideY == 1
    assert desc.m_DilationX == 1
    assert desc.m_DilationY == 1
    assert desc.m_BiasEnabled == False
    assert desc.m_DataLayout == ann.DataLayout_NCHW


def test_detectionpostprocess_descriptor_default_values():
    desc = ann.DetectionPostProcessDescriptor()
    assert desc.m_MaxDetections == 0
    assert desc.m_MaxClassesPerDetection == 1
    assert desc.m_DetectionsPerClass == 1
    assert desc.m_NmsScoreThreshold == 0
    assert desc.m_NmsIouThreshold == 0
    assert desc.m_NumClasses == 0
    assert desc.m_UseRegularNms == False
    assert desc.m_ScaleH == 0
    assert desc.m_ScaleW == 0
    assert desc.m_ScaleX == 0
    assert desc.m_ScaleY == 0


def test_fakequantization_descriptor_default_values():
    desc = ann.FakeQuantizationDescriptor()
    np.allclose(6, desc.m_Max)
    np.allclose(-6, desc.m_Min)


def test_fill_descriptor_default_values():
    desc = ann.FillDescriptor()
    np.allclose(0, desc.m_Value)


def test_gather_descriptor_default_values():
    desc = ann.GatherDescriptor()
    assert desc.m_Axis == 0


def test_fully_connected_descriptor_default_values():
    desc = ann.FullyConnectedDescriptor()
    assert desc.m_BiasEnabled == False
    assert desc.m_TransposeWeightMatrix == False


def test_instancenormalization_descriptor_default_values():
    desc = ann.InstanceNormalizationDescriptor()
    assert desc.m_Gamma == 1
    assert desc.m_Beta == 0
    assert desc.m_DataLayout == ann.DataLayout_NCHW
    np.allclose(1e-12, desc.m_Eps)


def test_lstm_descriptor_default_values():
    desc = ann.LstmDescriptor()
    assert desc.m_ActivationFunc == 1
    assert desc.m_ClippingThresCell == 0
    assert desc.m_ClippingThresProj == 0
    assert desc.m_CifgEnabled == True
    assert desc.m_PeepholeEnabled == False
    assert desc.m_ProjectionEnabled == False
    assert desc.m_LayerNormEnabled == False


def test_l2normalization_descriptor_default_values():
    desc = ann.L2NormalizationDescriptor()
    assert desc.m_DataLayout == ann.DataLayout_NCHW
    np.allclose(1e-12, desc.m_Eps)


def test_mean_descriptor_default_values():
    desc = ann.MeanDescriptor()
    assert desc.m_KeepDims == False


def test_normalization_descriptor_default_values():
    desc = ann.NormalizationDescriptor()
    assert desc.m_NormChannelType == ann.NormalizationAlgorithmChannel_Across
    assert desc.m_NormMethodType == ann.NormalizationAlgorithmMethod_LocalBrightness
    assert desc.m_NormSize == 0
    assert desc.m_Alpha == 0
    assert desc.m_Beta == 0
    assert desc.m_K == 0
    assert desc.m_DataLayout == ann.DataLayout_NCHW


def test_origin_descriptor_default_values():
    desc = ann.ConcatDescriptor()
    assert 0 == desc.GetNumViews()
    assert 0 == desc.GetNumDimensions()
    assert 1 == desc.GetConcatAxis()


def test_origin_descriptor_incorrect_views():
    desc = ann.ConcatDescriptor(2, 2)
    with pytest.raises(RuntimeError) as err:
        desc.SetViewOriginCoord(1000, 100, 1000)
    assert "Failed to set view origin coordinates." in str(err.value)


def test_origin_descriptor_ctor():
    desc = ann.ConcatDescriptor(2, 2)
    value = 5
    for i in range(desc.GetNumViews()):
        for j in range(desc.GetNumDimensions()):
            desc.SetViewOriginCoord(i, j, value+i)
    desc.SetConcatAxis(1)

    assert 2 == desc.GetNumViews()
    assert 2 == desc.GetNumDimensions()
    assert [5, 5] == desc.GetViewOrigin(0)
    assert [6, 6] == desc.GetViewOrigin(1)
    assert 1 == desc.GetConcatAxis()


def test_pad_descriptor_default_values():
    desc = ann.PadDescriptor()
    assert desc.m_PadValue == 0
    assert desc.m_PaddingMode == ann.PaddingMode_Constant


def test_permute_descriptor_default_values():
    pv = ann.PermutationVector((0, 2, 3, 1))
    desc = ann.PermuteDescriptor(pv)
    assert desc.m_DimMappings.GetSize() == 4
    assert desc.m_DimMappings[0] == 0
    assert desc.m_DimMappings[1] == 2
    assert desc.m_DimMappings[2] == 3
    assert desc.m_DimMappings[3] == 1


def test_pooling_descriptor_default_values():
    desc = ann.Pooling2dDescriptor()
    assert desc.m_PoolType == ann.PoolingAlgorithm_Max
    assert desc.m_PadLeft == 0
    assert desc.m_PadTop == 0
    assert desc.m_PadRight == 0
    assert desc.m_PadBottom == 0
    assert desc.m_PoolHeight == 0
    assert desc.m_PoolWidth == 0
    assert desc.m_StrideX == 0
    assert desc.m_StrideY == 0
    assert desc.m_OutputShapeRounding == ann.OutputShapeRounding_Floor
    assert desc.m_PaddingMethod == ann.PaddingMethod_Exclude
    assert desc.m_DataLayout == ann.DataLayout_NCHW

def test_pooling_3d_descriptor_default_values():
    desc = ann.Pooling3dDescriptor()
    assert desc.m_PoolType == ann.PoolingAlgorithm_Max
    assert desc.m_PadLeft == 0
    assert desc.m_PadTop == 0
    assert desc.m_PadRight == 0
    assert desc.m_PadBottom == 0
    assert desc.m_PadFront == 0
    assert desc.m_PadBack == 0
    assert desc.m_PoolHeight == 0
    assert desc.m_PoolWidth == 0
    assert desc.m_StrideX == 0
    assert desc.m_StrideY == 0
    assert desc.m_StrideZ == 0
    assert desc.m_OutputShapeRounding == ann.OutputShapeRounding_Floor
    assert desc.m_PaddingMethod == ann.PaddingMethod_Exclude
    assert desc.m_DataLayout == ann.DataLayout_NCDHW


def test_reshape_descriptor_default_values():
    desc = ann.ReshapeDescriptor()
    # check the empty Targetshape
    assert desc.m_TargetShape.GetNumDimensions() == 0

def test_reduce_descriptor_default_values():
    desc = ann.ReduceDescriptor()
    assert desc.m_KeepDims == False
    assert desc.m_vAxis == []
    assert desc.m_ReduceOperation == ann.ReduceOperation_Sum

def test_slice_descriptor_default_values():
    desc = ann.SliceDescriptor()
    assert desc.m_TargetWidth == 0
    assert desc.m_TargetHeight == 0
    assert desc.m_Method == ann.ResizeMethod_NearestNeighbor
    assert desc.m_DataLayout == ann.DataLayout_NCHW


def test_resize_descriptor_default_values():
    desc = ann.ResizeDescriptor()
    assert desc.m_TargetWidth == 0
    assert desc.m_TargetHeight == 0
    assert desc.m_Method == ann.ResizeMethod_NearestNeighbor
    assert desc.m_DataLayout == ann.DataLayout_NCHW
    assert desc.m_AlignCorners == False


def test_spacetobatchnd_descriptor_default_values():
    desc = ann.SpaceToBatchNdDescriptor()
    assert desc.m_DataLayout == ann.DataLayout_NCHW


def test_spacetodepth_descriptor_default_values():
    desc = ann.SpaceToDepthDescriptor()
    assert desc.m_BlockSize == 1
    assert desc.m_DataLayout == ann.DataLayout_NHWC


def test_stack_descriptor_default_values():
    desc = ann.StackDescriptor()
    assert desc.m_Axis == 0
    assert desc.m_NumInputs == 0
    # check the empty Inputshape
    assert desc.m_InputShape.GetNumDimensions() == 0


def test_slice_descriptor_default_values():
    desc = ann.SliceDescriptor()
    desc.m_Begin = [1, 2, 3, 4, 5]
    desc.m_Size = (1, 2, 3, 4)

    assert [1, 2, 3, 4, 5] == desc.m_Begin
    assert [1, 2, 3, 4] == desc.m_Size


def test_slice_descriptor_ctor():
    desc = ann.SliceDescriptor([1, 2, 3, 4, 5], (1, 2, 3, 4))

    assert [1, 2, 3, 4, 5] == desc.m_Begin
    assert [1, 2, 3, 4] == desc.m_Size


def test_strided_slice_descriptor_default_values():
    desc = ann.StridedSliceDescriptor()
    desc.m_Begin = [1, 2, 3, 4, 5]
    desc.m_End = [6, 7, 8, 9, 10]
    desc.m_Stride = (10, 10)
    desc.m_BeginMask = 1
    desc.m_EndMask = 2
    desc.m_ShrinkAxisMask = 3
    desc.m_EllipsisMask = 4
    desc.m_NewAxisMask = 5

    assert [1, 2, 3, 4, 5] == desc.m_Begin
    assert [6, 7, 8, 9, 10] == desc.m_End
    assert [10, 10] == desc.m_Stride
    assert 1 == desc.m_BeginMask
    assert 2 == desc.m_EndMask
    assert 3 == desc.m_ShrinkAxisMask
    assert 4 == desc.m_EllipsisMask
    assert 5 == desc.m_NewAxisMask


def test_strided_slice_descriptor_ctor():
    desc = ann.StridedSliceDescriptor([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], (10, 10))
    desc.m_Begin = [1, 2, 3, 4, 5]
    desc.m_End = [6, 7, 8, 9, 10]
    desc.m_Stride = (10, 10)

    assert [1, 2, 3, 4, 5] == desc.m_Begin
    assert [6, 7, 8, 9, 10] == desc.m_End
    assert [10, 10] == desc.m_Stride


def test_softmax_descriptor_default_values():
    desc = ann.SoftmaxDescriptor()
    assert desc.m_Axis == -1
    np.allclose(1.0, desc.m_Beta)


def test_space_to_batch_nd_descriptor_default_values():
    desc = ann.SpaceToBatchNdDescriptor()
    assert [1, 1] == desc.m_BlockShape
    assert [(0, 0), (0, 0)] == desc.m_PadList
    assert ann.DataLayout_NCHW == desc.m_DataLayout


def test_space_to_batch_nd_descriptor_assigned_values():
    desc = ann.SpaceToBatchNdDescriptor()
    desc.m_BlockShape = (90, 100)
    desc.m_PadList = [(1, 2), (3, 4)]
    assert [90, 100] == desc.m_BlockShape
    assert [(1, 2), (3, 4)] == desc.m_PadList
    assert ann.DataLayout_NCHW == desc.m_DataLayout


def test_space_to_batch_nd_descriptor_ctor():
    desc = ann.SpaceToBatchNdDescriptor((1, 2, 3), [(1, 2), (3, 4)])
    assert [1, 2, 3] == desc.m_BlockShape
    assert [(1, 2), (3, 4)] == desc.m_PadList
    assert ann.DataLayout_NCHW == desc.m_DataLayout


def test_transpose_convolution2d_descriptor_default_values():
    desc = ann.TransposeConvolution2dDescriptor()
    assert desc.m_PadLeft == 0
    assert desc.m_PadTop == 0
    assert desc.m_PadRight == 0
    assert desc.m_PadBottom == 0
    assert desc.m_StrideX == 0
    assert desc.m_StrideY == 0
    assert desc.m_BiasEnabled == False
    assert desc.m_DataLayout == ann.DataLayout_NCHW
    assert desc.m_OutputShapeEnabled == False

def test_transpose_descriptor_default_values():
    pv = ann.PermutationVector((0, 3, 2, 1, 4))
    desc = ann.TransposeDescriptor(pv)
    assert desc.m_DimMappings.GetSize() == 5
    assert desc.m_DimMappings[0] == 0
    assert desc.m_DimMappings[1] == 3
    assert desc.m_DimMappings[2] == 2
    assert desc.m_DimMappings[3] == 1
    assert desc.m_DimMappings[4] == 4

def test_view_descriptor_default_values():
    desc = ann.SplitterDescriptor()
    assert 0 == desc.GetNumViews()
    assert 0 == desc.GetNumDimensions()


def test_elementwise_unary_descriptor_default_values():
    desc = ann.ElementwiseUnaryDescriptor()
    assert desc.m_Operation == ann.UnaryOperation_Abs


def test_logical_binary_descriptor_default_values():
    desc = ann.LogicalBinaryDescriptor()
    assert desc.m_Operation == ann.LogicalBinaryOperation_LogicalAnd

def test_view_descriptor_incorrect_input():
    desc = ann.SplitterDescriptor(2, 3)
    with pytest.raises(RuntimeError) as err:
        desc.SetViewOriginCoord(1000, 100, 1000)
    assert "Failed to set view origin coordinates." in str(err.value)

    with pytest.raises(RuntimeError) as err:
        desc.SetViewSize(1000, 100, 1000)
    assert "Failed to set view size." in str(err.value)


def test_view_descriptor_ctor():
    desc = ann.SplitterDescriptor(2, 3)
    value_size = 1
    value_orig_coord = 5
    for i in range(desc.GetNumViews()):
        for j in range(desc.GetNumDimensions()):
            desc.SetViewOriginCoord(i, j, value_orig_coord+i)
            desc.SetViewSize(i, j, value_size+i)

    assert 2 == desc.GetNumViews()
    assert 3 == desc.GetNumDimensions()
    assert [5, 5] == desc.GetViewOrigin(0)
    assert [6, 6] == desc.GetViewOrigin(1)
    assert [1, 1] == desc.GetViewSizes(0)
    assert [2, 2] == desc.GetViewSizes(1)


def test_createdescriptorforconcatenation_ctor():
    input_shape_vector = [ann.TensorShape((2, 1)), ann.TensorShape((3, 1)), ann.TensorShape((4, 1))]
    desc = ann.CreateDescriptorForConcatenation(input_shape_vector, 0)
    assert 3 == desc.GetNumViews()
    assert 0 == desc.GetConcatAxis()
    assert 2 == desc.GetNumDimensions()
    c = desc.GetViewOrigin(1)
    d = desc.GetViewOrigin(0)


def test_createdescriptorforconcatenation_wrong_shape_for_axis():
    input_shape_vector = [ann.TensorShape((1, 2)), ann.TensorShape((3, 4)), ann.TensorShape((5, 6))]
    with pytest.raises(RuntimeError) as err:
        desc = ann.CreateDescriptorForConcatenation(input_shape_vector, 0)

    assert "All inputs to concatenation must be the same size along all dimensions  except the concatenation dimension" in str(
        err.value)


@pytest.mark.parametrize("input_shape_vector", [([-1, "one"]),
                                                ([1.33, 4.55]),
                                                ([{1: "one"}])], ids=lambda x: str(x))
def test_createdescriptorforconcatenation_rubbish_assignment_shape_vector(input_shape_vector):
    with pytest.raises(TypeError) as err:
        desc = ann.CreateDescriptorForConcatenation(input_shape_vector, 0)

    assert "in method 'CreateDescriptorForConcatenation', argument 1 of type 'std::vector< armnn::TensorShape,std::allocator< armnn::TensorShape > >'" in str(
        err.value)


generated_classes = inspect.getmembers(generated, inspect.isclass)
generated_classes_names = list(map(lambda x: x[0], generated_classes))
@pytest.mark.parametrize("desc_name", ['ActivationDescriptor',
                                       'ArgMinMaxDescriptor',
                                       'PermuteDescriptor',
                                       'SoftmaxDescriptor',
                                       'ConcatDescriptor',
                                       'SplitterDescriptor',
                                       'Pooling2dDescriptor',
                                       'FullyConnectedDescriptor',
                                       'Convolution2dDescriptor',
                                       'Convolution3dDescriptor',
                                       'DepthwiseConvolution2dDescriptor',
                                       'DetectionPostProcessDescriptor',
                                       'NormalizationDescriptor',
                                       'L2NormalizationDescriptor',
                                       'BatchNormalizationDescriptor',
                                       'InstanceNormalizationDescriptor',
                                       'BatchToSpaceNdDescriptor',
                                       'FakeQuantizationDescriptor',
                                       'ReduceDescriptor',
                                       'ResizeDescriptor',
                                       'ReshapeDescriptor',
                                       'SpaceToBatchNdDescriptor',
                                       'SpaceToDepthDescriptor',
                                       'LstmDescriptor',
                                       'MeanDescriptor',
                                       'PadDescriptor',
                                       'SliceDescriptor',
                                       'StackDescriptor',
                                       'StridedSliceDescriptor',
                                       'TransposeConvolution2dDescriptor',
                                       'TransposeDescriptor',
                                       'ElementwiseUnaryDescriptor',
                                       'FillDescriptor',
                                       'GatherDescriptor',
                                       'LogicalBinaryDescriptor',
                                       'ChannelShuffleDescriptor'])
class TestDescriptorMassChecks:

    def test_desc_implemented(self, desc_name):
        assert desc_name in generated_classes_names

    def test_desc_equal(self, desc_name):
        desc_class = next(filter(lambda x: x[0] == desc_name, generated_classes))[1]

        assert desc_class() == desc_class()


generated_classes = inspect.getmembers(generated, inspect.isclass)
generated_classes_names = list(map(lambda x: x[0], generated_classes))
@pytest.mark.parametrize("desc_name", ['ActivationDescriptor',
                                       'ArgMinMaxDescriptor',
                                       'PermuteDescriptor',
                                       'SoftmaxDescriptor',
                                       'ConcatDescriptor',
                                       'SplitterDescriptor',
                                       'Pooling2dDescriptor',
                                       'FullyConnectedDescriptor',
                                       'Convolution2dDescriptor',
                                       'Convolution3dDescriptor',
                                       'DepthwiseConvolution2dDescriptor',
                                       'DetectionPostProcessDescriptor',
                                       'NormalizationDescriptor',
                                       'L2NormalizationDescriptor',
                                       'BatchNormalizationDescriptor',
                                       'InstanceNormalizationDescriptor',
                                       'BatchToSpaceNdDescriptor',
                                       'FakeQuantizationDescriptor',
                                       'ReduceDescriptor',
                                       'ResizeDescriptor',
                                       'ReshapeDescriptor',
                                       'SpaceToBatchNdDescriptor',
                                       'SpaceToDepthDescriptor',
                                       'LstmDescriptor',
                                       'MeanDescriptor',
                                       'PadDescriptor',
                                       'SliceDescriptor',
                                       'StackDescriptor',
                                       'StridedSliceDescriptor',
                                       'TransposeConvolution2dDescriptor',
                                       'TransposeDescriptor',
                                       'ElementwiseUnaryDescriptor',
                                       'FillDescriptor',
                                       'GatherDescriptor',
                                       'LogicalBinaryDescriptor',
                                       'ChannelShuffleDescriptor'])
class TestDescriptorMassChecks:

    def test_desc_implemented(self, desc_name):
        assert desc_name in generated_classes_names

    def test_desc_equal(self, desc_name):
        desc_class = next(filter(lambda x: x[0] == desc_name, generated_classes))[1]

        assert desc_class() == desc_class()

