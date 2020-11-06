//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TestNameAndDescriptorLayerVisitor.hpp"
#include "Network.hpp"

#include <armnn/Exceptions.hpp>

namespace
{

#define TEST_CASE_CHECK_LAYER_VISITOR_NAME_AND_DESCRIPTOR(name) \
BOOST_AUTO_TEST_CASE(Check##name##LayerVisitorNameAndDescriptor) \
{ \
    const char* layerName = "name##Layer"; \
    armnn::name##Descriptor descriptor = GetDescriptor<armnn::name##Descriptor>(); \
    Test##name##LayerVisitor visitor(descriptor, layerName); \
    armnn::Network net; \
    armnn::IConnectableLayer *const layer = net.Add##name##Layer(descriptor, layerName); \
    layer->Accept(visitor); \
}

#define TEST_CASE_CHECK_LAYER_VISITOR_NAME_NULLPTR_AND_DESCRIPTOR(name) \
BOOST_AUTO_TEST_CASE(Check##name##LayerVisitorNameNullptrAndDescriptor) \
{ \
    armnn::name##Descriptor descriptor = GetDescriptor<armnn::name##Descriptor>(); \
    Test##name##LayerVisitor visitor(descriptor); \
    armnn::Network net; \
    armnn::IConnectableLayer *const layer = net.Add##name##Layer(descriptor); \
    layer->Accept(visitor); \
}

#define TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(name) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME_AND_DESCRIPTOR(name) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME_NULLPTR_AND_DESCRIPTOR(name)

template<typename Descriptor> Descriptor GetDescriptor();

template<>
armnn::ActivationDescriptor GetDescriptor<armnn::ActivationDescriptor>()
{
    armnn::ActivationDescriptor descriptor;
    descriptor.m_Function = armnn::ActivationFunction::Linear;
    descriptor.m_A        = 2.0f;
    descriptor.m_B        = 2.0f;

    return descriptor;
}

template<>
armnn::ArgMinMaxDescriptor GetDescriptor<armnn::ArgMinMaxDescriptor>()
{
    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Function = armnn::ArgMinMaxFunction::Max;
    descriptor.m_Axis     = 1;

    return descriptor;
}

template<>
armnn::BatchToSpaceNdDescriptor GetDescriptor<armnn::BatchToSpaceNdDescriptor>()
{
    return armnn::BatchToSpaceNdDescriptor({ 1, 1 }, {{ 0, 0 }, { 0, 0 }});
}

template<>
armnn::ComparisonDescriptor GetDescriptor<armnn::ComparisonDescriptor>()
{
    return armnn::ComparisonDescriptor(armnn::ComparisonOperation::GreaterOrEqual);
}

template<>
armnn::ConcatDescriptor GetDescriptor<armnn::ConcatDescriptor>()
{
    armnn::ConcatDescriptor descriptor(2, 2);
    for (unsigned int i = 0u; i < descriptor.GetNumViews(); ++i)
    {
        for (unsigned int j = 0u; j < descriptor.GetNumDimensions(); ++j)
        {
            descriptor.SetViewOriginCoord(i, j, i);
        }
    }

    return descriptor;
}

template<>
armnn::ElementwiseUnaryDescriptor GetDescriptor<armnn::ElementwiseUnaryDescriptor>()
{
    return armnn::ElementwiseUnaryDescriptor(armnn::UnaryOperation::Abs);
}

template<>
armnn::FillDescriptor GetDescriptor<armnn::FillDescriptor>()
{
    return armnn::FillDescriptor(1);
}

template<>
armnn::GatherDescriptor GetDescriptor<armnn::GatherDescriptor>()
{
    return armnn::GatherDescriptor();
}

template<>
armnn::InstanceNormalizationDescriptor GetDescriptor<armnn::InstanceNormalizationDescriptor>()
{
    armnn::InstanceNormalizationDescriptor descriptor;
    descriptor.m_Gamma      = 1.0f;
    descriptor.m_Beta       = 2.0f;
    descriptor.m_Eps        = 0.0001f;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    return descriptor;
}

template<>
armnn::L2NormalizationDescriptor GetDescriptor<armnn::L2NormalizationDescriptor>()
{
    armnn::L2NormalizationDescriptor descriptor;
    descriptor.m_Eps        = 0.0001f;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    return descriptor;
}

template<>
armnn::LogicalBinaryDescriptor GetDescriptor<armnn::LogicalBinaryDescriptor>()
{
    return armnn::LogicalBinaryDescriptor(armnn::LogicalBinaryOperation::LogicalOr);
}

template<>
armnn::MeanDescriptor GetDescriptor<armnn::MeanDescriptor>()
{
    return armnn::MeanDescriptor({ 1, 2, }, true);
}

template<>
armnn::NormalizationDescriptor GetDescriptor<armnn::NormalizationDescriptor>()
{
    armnn::NormalizationDescriptor descriptor;
    descriptor.m_NormChannelType = armnn::NormalizationAlgorithmChannel::Within;
    descriptor.m_NormMethodType  = armnn::NormalizationAlgorithmMethod::LocalContrast;
    descriptor.m_NormSize        = 1u;
    descriptor.m_Alpha           = 1.0f;
    descriptor.m_Beta            = 1.0f;
    descriptor.m_K               = 1.0f;
    descriptor.m_DataLayout      = armnn::DataLayout::NHWC;

    return descriptor;
}

template<>
armnn::PadDescriptor GetDescriptor<armnn::PadDescriptor>()
{
    return armnn::PadDescriptor({{ 1, 2 }, { 3, 4 }});
}

template<>
armnn::PermuteDescriptor GetDescriptor<armnn::PermuteDescriptor>()
{
    return armnn::PermuteDescriptor({ 0, 1, 2, 3 });
}

template<>
armnn::Pooling2dDescriptor GetDescriptor<armnn::Pooling2dDescriptor>()
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType            = armnn::PoolingAlgorithm::Max;
    descriptor.m_PadLeft             = 1u;
    descriptor.m_PadRight            = 1u;
    descriptor.m_PadTop              = 1u;
    descriptor.m_PadBottom           = 1u;
    descriptor.m_PoolWidth           = 1u;
    descriptor.m_PoolHeight          = 1u;
    descriptor.m_StrideX             = 1u;
    descriptor.m_StrideY             = 1u;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Ceiling;
    descriptor.m_PaddingMethod       = armnn::PaddingMethod::IgnoreValue;
    descriptor.m_DataLayout          = armnn::DataLayout::NHWC;

    return descriptor;
}

template<>
armnn::ReshapeDescriptor GetDescriptor<armnn::ReshapeDescriptor>()
{
    return armnn::ReshapeDescriptor({ 1, 2, 3, 4 });
}

template<>
armnn::ResizeDescriptor GetDescriptor<armnn::ResizeDescriptor>()
{
    armnn::ResizeDescriptor descriptor;
    descriptor.m_TargetHeight = 1u;
    descriptor.m_TargetWidth  = 1u;
    descriptor.m_DataLayout   = armnn::DataLayout::NHWC;

    return descriptor;
}

template<>
armnn::SliceDescriptor GetDescriptor<armnn::SliceDescriptor>()
{
    return armnn::SliceDescriptor({ 1, 1 }, { 2, 2 });
}

template<>
armnn::SoftmaxDescriptor GetDescriptor<armnn::SoftmaxDescriptor>()
{
    armnn::SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 2.0f;
    descriptor.m_Axis = -1;

    return descriptor;
}

template<>
armnn::SpaceToBatchNdDescriptor GetDescriptor<armnn::SpaceToBatchNdDescriptor>()
{
    return armnn::SpaceToBatchNdDescriptor({ 2, 2 }, {{ 1, 1 } , { 1, 1 }});
}

template<>
armnn::SpaceToDepthDescriptor GetDescriptor<armnn::SpaceToDepthDescriptor>()
{
    return armnn::SpaceToDepthDescriptor(2, armnn::DataLayout::NHWC);
}

template<>
armnn::SplitterDescriptor GetDescriptor<armnn::SplitterDescriptor>()
{
    armnn::SplitterDescriptor descriptor(2, 2);
    for (unsigned int i = 0u; i < descriptor.GetNumViews(); ++i)
    {
        for (unsigned int j = 0u; j < descriptor.GetNumDimensions(); ++j)
        {
            descriptor.SetViewOriginCoord(i, j, i);
            descriptor.SetViewSize(i, j, 1);
        }
    }

    return descriptor;
}

template<>
armnn::StackDescriptor GetDescriptor<armnn::StackDescriptor>()
{
    return armnn::StackDescriptor(1, 2, { 2, 2 });
}

template<>
armnn::StridedSliceDescriptor GetDescriptor<armnn::StridedSliceDescriptor>()
{
    armnn::StridedSliceDescriptor descriptor({ 1, 2 }, { 3, 4 }, { 3, 4 });
    descriptor.m_BeginMask      = 1;
    descriptor.m_EndMask        = 1;
    descriptor.m_ShrinkAxisMask = 1;
    descriptor.m_EllipsisMask   = 1;
    descriptor.m_NewAxisMask    = 1;
    descriptor.m_DataLayout     = armnn::DataLayout::NHWC;

    return descriptor;
}

template<>
armnn::TransposeDescriptor GetDescriptor<armnn::TransposeDescriptor>()
{
    return armnn::TransposeDescriptor({ 0, 1, 2, 3 });
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(TestNameAndDescriptorLayerVisitor)

TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Activation)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(ArgMinMax)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(DepthToSpace)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(BatchToSpaceNd)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Comparison)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Concat)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(ElementwiseUnary)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Fill)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Gather)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(InstanceNormalization)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(L2Normalization)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(LogicalBinary)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(LogSoftmax)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Mean)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Normalization)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Pad)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Permute)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Pooling2d)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Reshape)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Resize)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Slice)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Softmax)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(SpaceToBatchNd)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(SpaceToDepth)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Splitter)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Stack)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(StridedSlice)
TEST_SUITE_NAME_AND_DESCRIPTOR_LAYER_VISITOR(Transpose)

BOOST_AUTO_TEST_SUITE_END()
