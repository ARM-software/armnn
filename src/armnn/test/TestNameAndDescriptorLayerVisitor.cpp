//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TestNameAndDescriptorLayerVisitor.hpp"
#include "Network.hpp"

namespace armnn
{

void Set2dDataValues(SplitterDescriptor descriptor, uint32_t value)
{
    for (unsigned int i = 0; i < descriptor.GetNumViews(); ++i)
    {
        for (unsigned int j = 0; j < descriptor.GetNumDimensions(); ++j)
        {
            descriptor.SetViewOriginCoord(i, j, value);
            descriptor.SetViewSize(i, j, value);
        }
    }
}

void Set2dDataValues(OriginsDescriptor& descriptor, uint32_t value)
{
    for (unsigned int i = 0; i < descriptor.GetNumViews(); ++i)
    {
        for (unsigned int j = 0; j < descriptor.GetNumDimensions(); ++j)
        {
            descriptor.SetViewOriginCoord(i, j, value);
        }
    }
}

BOOST_AUTO_TEST_SUITE(TestNameAndDescriptorLayerVisitor)

BOOST_AUTO_TEST_CASE(CheckPermuteLayerVisitorNameAndDescriptor)
{
    const char* layerName = "PermuteLayer";
    PermuteDescriptor descriptor({0, 1, 2, 3});
    TestPermuteLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddPermuteLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckPermuteLayerVisitorNameNullAndDescriptor)
{
    PermuteDescriptor descriptor({0, 1, 2, 3});
    TestPermuteLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddPermuteLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckBatchToSpaceNdLayerVisitorNameAndDescriptor)
{
    const char* layerName = "BatchToSpaceNdLayer";
    BatchToSpaceNdDescriptor descriptor({1, 1}, {{0, 0}, {0, 0}});
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    TestBatchToSpaceNdLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddBatchToSpaceNdLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckBatchToSpaceNdLayerVisitorNameNullAndDescriptor)
{
    BatchToSpaceNdDescriptor descriptor({1, 1}, {{0, 0}, {0, 0}});
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    TestBatchToSpaceNdLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddBatchToSpaceNdLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckPooling2dLayerVisitorNameAndDescriptor)
{
    const char* layerName = "Pooling2dLayer";
    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType            = PoolingAlgorithm::Max;
    descriptor.m_PadLeft             = 1;
    descriptor.m_PadRight            = 1;
    descriptor.m_PadTop              = 1;
    descriptor.m_PadBottom           = 1;
    descriptor.m_PoolWidth           = 1;
    descriptor.m_PoolHeight          = 1;
    descriptor.m_StrideX             = 1;
    descriptor.m_StrideY             = 1;
    descriptor.m_OutputShapeRounding = OutputShapeRounding::Ceiling;
    descriptor.m_PaddingMethod       = PaddingMethod::IgnoreValue;
    descriptor.m_DataLayout          = DataLayout::NHWC;
    TestPooling2dLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddPooling2dLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckPooling2dLayerVisitorNameNullAndDescriptor)
{
    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType            = PoolingAlgorithm::Max;
    descriptor.m_PadLeft             = 1;
    descriptor.m_PadRight            = 1;
    descriptor.m_PadTop              = 1;
    descriptor.m_PadBottom           = 1;
    descriptor.m_PoolWidth           = 1;
    descriptor.m_PoolHeight          = 1;
    descriptor.m_StrideX             = 1;
    descriptor.m_StrideY             = 1;
    descriptor.m_OutputShapeRounding = OutputShapeRounding::Ceiling;
    descriptor.m_PaddingMethod       = PaddingMethod::IgnoreValue;
    descriptor.m_DataLayout          = DataLayout::NHWC;
    TestPooling2dLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddPooling2dLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckActivationLayerVisitorNameAndDescriptor)
{
    const char* layerName = "ActivationLayer";
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Linear;
    descriptor.m_A = 2;
    descriptor.m_B = 2;
    TestActivationLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddActivationLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckActivationLayerVisitorNameNullAndDescriptor)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Linear;
    descriptor.m_A = 2;
    descriptor.m_B = 2;
    TestActivationLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddActivationLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNormalizationLayerVisitorNameAndDescriptor)
{
    const char* layerName = "NormalizationLayer";
    NormalizationDescriptor descriptor;
    descriptor.m_NormChannelType = NormalizationAlgorithmChannel::Within;
    descriptor.m_NormMethodType  = NormalizationAlgorithmMethod::LocalContrast;
    descriptor.m_NormSize        = 1;
    descriptor.m_Alpha           = 1;
    descriptor.m_Beta            = 1;
    descriptor.m_K               = 1;
    descriptor.m_DataLayout      = DataLayout::NHWC;
    TestNormalizationLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddNormalizationLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNormalizationLayerVisitorNameNullAndDescriptor)
{
    NormalizationDescriptor descriptor;
    descriptor.m_NormChannelType = NormalizationAlgorithmChannel::Within;
    descriptor.m_NormMethodType  = NormalizationAlgorithmMethod::LocalContrast;
    descriptor.m_NormSize        = 1;
    descriptor.m_Alpha           = 1;
    descriptor.m_Beta            = 1;
    descriptor.m_K               = 1;
    descriptor.m_DataLayout      = DataLayout::NHWC;
    TestNormalizationLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddNormalizationLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSoftmaxLayerVisitorNameAndDescriptor)
{
    const char* layerName = "SoftmaxLayer";
    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 2;
    TestSoftmaxLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddSoftmaxLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSoftmaxLayerVisitorNameNullAndDescriptor)
{
    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 2;
    TestSoftmaxLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddSoftmaxLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSplitterLayerVisitorNameAndDescriptor)
{
    const char* layerName = "SplitterLayer";
    SplitterDescriptor descriptor(2, 2);
    Set2dDataValues(descriptor, 1);
    TestSplitterLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddSplitterLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSplitterLayerVisitorNameNullAndDescriptor)
{
    SplitterDescriptor descriptor(2, 2);
    Set2dDataValues(descriptor, 1);
    TestSplitterLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddSplitterLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckConcatLayerVisitorNameAndDescriptor)
{
    const char* layerName = "ConcatLayer";
    OriginsDescriptor descriptor(2, 2);
    Set2dDataValues(descriptor, 1);
    descriptor.SetConcatAxis(1);
    TestConcatLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddConcatLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckConcatLayerVisitorNameNullAndDescriptor)
{
    OriginsDescriptor descriptor(2, 2);
    Set2dDataValues(descriptor, 1);
    descriptor.SetConcatAxis(1);
    TestConcatLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddConcatLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckResizeLayerVisitorNameAndDescriptor)
{
    const char* layerName = "ResizeLayer";
    ResizeDescriptor descriptor;
    descriptor.m_TargetHeight = 1;
    descriptor.m_TargetWidth  = 1;
    descriptor.m_DataLayout   = DataLayout::NHWC;
    TestResizeLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddResizeLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckResizeLayerVisitorNameNullAndDescriptor)
{
    ResizeDescriptor descriptor;
    descriptor.m_TargetHeight = 1;
    descriptor.m_TargetWidth  = 1;
    descriptor.m_DataLayout   = DataLayout::NHWC;
    TestResizeLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddResizeLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckL2NormalizationLayerVisitorNameAndDescriptor)
{
    const char* layerName = "L2NormalizationLayer";
    L2NormalizationDescriptor descriptor;
    descriptor.m_DataLayout = DataLayout::NHWC;
    TestL2NormalizationLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddL2NormalizationLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckL2NormalizationLayerVisitorNameNullAndDescriptor)
{
    L2NormalizationDescriptor descriptor;
    descriptor.m_DataLayout = DataLayout::NHWC;
    TestL2NormalizationLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddL2NormalizationLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckReshapeLayerVisitorNameAndDescriptor)
{
    const char* layerName = "ReshapeLayer";
    ReshapeDescriptor descriptor({1, 2, 3, 4});
    TestReshapeLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddReshapeLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckReshapeLayerVisitorNameNullAndDescriptor)
{
    ReshapeDescriptor descriptor({1, 2, 3, 4});
    TestReshapeLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddReshapeLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSpaceToBatchNdLayerVisitorNameAndDescriptor)
{
    const char* layerName = "SpaceToBatchNdLayer";
    SpaceToBatchNdDescriptor descriptor({2, 2}, {{1, 1}, {1, 1}});
    TestSpaceToBatchNdLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddSpaceToBatchNdLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSpaceToBatchNdLayerVisitorNameNullAndDescriptor)
{
    SpaceToBatchNdDescriptor descriptor({2, 2}, {{1, 1}, {1, 1}});
    TestSpaceToBatchNdLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddSpaceToBatchNdLayer(descriptor);
    layer->Accept(visitor);
}


BOOST_AUTO_TEST_CASE(CheckMeanLayerVisitorNameAndDescriptor)
{
    const char* layerName = "MeanLayer";
    MeanDescriptor descriptor({1, 2}, false);
    TestMeanLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddMeanLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckMeanLayerVisitorNameNullAndDescriptor)
{
    MeanDescriptor descriptor({1, 2}, false);
    TestMeanLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddMeanLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckPadLayerVisitorNameAndDescriptor)
{
    const char* layerName = "PadLayer";
    PadDescriptor descriptor({{1, 2}, {3, 4}});
    TestPadLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddPadLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckPadLayerVisitorNameNullAndDescriptor)
{
    PadDescriptor descriptor({{1, 2}, {3, 4}});
    TestPadLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddPadLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckStridedSliceLayerVisitorNameAndDescriptor)
{
    const char* layerName = "StridedSliceLayer";
    StridedSliceDescriptor descriptor({1, 2}, {3, 4}, {3, 4});
    descriptor.m_BeginMask      = 1;
    descriptor.m_EndMask        = 1;
    descriptor.m_ShrinkAxisMask = 1;
    descriptor.m_EllipsisMask   = 1;
    descriptor.m_NewAxisMask    = 1;
    descriptor.m_DataLayout     = DataLayout::NHWC;
    TestStridedSliceLayerVisitor visitor(descriptor, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddStridedSliceLayer(descriptor, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckStridedSliceLayerVisitorNameNullAndDescriptor)
{
    StridedSliceDescriptor descriptor({1, 2}, {3, 4}, {3, 4});
    descriptor.m_BeginMask      = 1;
    descriptor.m_EndMask        = 1;
    descriptor.m_ShrinkAxisMask = 1;
    descriptor.m_EllipsisMask   = 1;
    descriptor.m_NewAxisMask    = 1;
    descriptor.m_DataLayout     = DataLayout::NHWC;
    TestStridedSliceLayerVisitor visitor(descriptor);
    Network net;

    IConnectableLayer *const layer = net.AddStridedSliceLayer(descriptor);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_SUITE_END()

} //namespace armnn
