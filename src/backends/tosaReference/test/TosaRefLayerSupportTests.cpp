//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include <armnn/Optional.hpp>
#include <armnn/Types.hpp>
#include <tosaReference/TosaRefLayerSupport.hpp>

#include <doctest/doctest.h>

#include <string>

using namespace armnn;

TEST_SUITE("TosaRefLayerSupported")
{

TEST_CASE("IsLayerSupportedTosaReferenceAddition")
{
    TensorShape shape0 = {1,1,3,4};
    TensorShape shape1 = {1,1,3,4};
    TensorShape outShape = {1,1,3,4};
    TensorInfo in0(shape0, DataType::Float32);
    TensorInfo in1(shape1, DataType::Float32);
    TensorInfo out(outShape, DataType::Float32);

    BaseDescriptor desc;
    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Addition,
                                                     {in0, in1, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceAdditionUnsupported")
{
    TensorShape shape0 = {1,1,3,4};
    TensorShape shape1 = {4};
    TensorShape outShape = {1,1,3,4};
    TensorInfo in0(shape0, DataType::Signed64);
    TensorInfo in1(shape1, DataType::Signed64);
    TensorInfo out(outShape, DataType::Signed64);

    BaseDescriptor desc;
    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Addition,
                                                     {in0, in1, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceConstant")
{
    TensorInfo outputInfo({1,1,3,4}, DataType::Float32);

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Constant,
                                                     {outputInfo},
                                                     BaseDescriptor(),
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceConstantUnsupported")
{
    TensorInfo outputInfo({1,1,3,4}, DataType::Signed64);

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Constant,
                                                     {outputInfo},
                                                     BaseDescriptor(),
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceConv2d")
{
    TensorInfo inputInfo ({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 3, 3, 1 }, DataType::Float32);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32);
    TensorInfo biasesInfo ({ 1 }, DataType::Float32);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = true;

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Convolution2d,
                                                     {inputInfo, outputInfo, weightsInfo, biasesInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceConv2dUnsupported")
{
    // If inputs and weights are Fp32, output must match.
    TensorInfo inputInfo ({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 3, 3, 1 }, DataType::Signed64);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32, 0.0f, 0, true);
    TensorInfo biasesInfo ({ 1 }, DataType::Float32, 0.0f, 0, true);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = true;

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Convolution2d,
                                                     {inputInfo, outputInfo, weightsInfo, biasesInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceMaxPooling2d")
{
    TensorShape inShape = {1,1,3,4};
    TensorShape outShape = {1,1,3,4};
    TensorInfo in(inShape, DataType::Float32);
    TensorInfo out(outShape, DataType::Float32);

    Pooling2dDescriptor desc;
    desc.m_PoolHeight = 1;
    desc.m_PoolWidth = 1;
    desc.m_StrideX = 1;
    desc.m_StrideY = 1;
    desc.m_PoolType = armnn::PoolingAlgorithm::Max;

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceAvgPooling2d_IgnoreValue")
{
    TensorShape inShape = {1,1,3,4};
    TensorShape outShape = {1,1,3,4};
    TensorInfo in(inShape, DataType::Float32);
    TensorInfo out(outShape, DataType::Float32);

    Pooling2dDescriptor desc;
    desc.m_PoolHeight = 1;
    desc.m_PoolWidth = 1;
    desc.m_StrideX = 1;
    desc.m_StrideY = 1;
    desc.m_PaddingMethod = PaddingMethod::IgnoreValue;
    desc.m_PoolType = PoolingAlgorithm::Average;

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceMaxPooling2dUnsupported")
{
    TensorShape inShape = {1,1,3,4};
    TensorShape outShape = {1,1,3,4};
    TensorInfo in(inShape, DataType::Signed64);
    TensorInfo out(outShape, DataType::Signed64);

    Pooling2dDescriptor desc;
    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceAvgPooling2dUnsupported_InputOutputDatatypeDifferent")
{
    TensorShape inShape = {1,1,3,4};
    TensorShape outShape = {1,1,3,4};
    TensorInfo in(inShape, DataType::Float32);
    TensorInfo out(outShape, DataType::Float16);

    Pooling2dDescriptor desc;
    desc.m_PaddingMethod = PaddingMethod::IgnoreValue;
    desc.m_PoolType = PoolingAlgorithm::Average;

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceReshape")
{
    TensorShape inShape = {3,4};
    TensorShape outShape = {12};
    TensorInfo in(inShape, DataType::Float32);
    TensorInfo out(outShape, DataType::Float32);

    ReshapeDescriptor desc;
    desc.m_TargetShape = {12};

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Reshape,
                                                     {in, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceReshapeUnsupported")
{
    TensorShape inShape = {3,4};
    TensorShape outShape = {12};
    TensorInfo in(inShape, DataType::Signed64);
    TensorInfo out(outShape, DataType::Signed64);

    ReshapeDescriptor desc;
    desc.m_TargetShape = {12};

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Reshape,
                                                     {in, out},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceSlice")
{
    TensorShape inShape = {3,2,3};
    TensorShape outShape = {2,1,3};
    TensorInfo in(inShape, DataType::Float32);
    TensorInfo out(outShape, DataType::Float32);

    SliceDescriptor descriptor;
    descriptor.m_Begin = {1,0,0 };
    descriptor.m_Size  = {2,1,3 };

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Slice,
                                                     {in, out},
                                                     descriptor,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceSliceUnsupported")
{
    TensorShape inShape = {3,2,3};
    TensorShape outShape = {2,1,3};
    TensorInfo in(inShape, DataType::Signed64);
    TensorInfo out(outShape, DataType::Signed64);

    SliceDescriptor descriptor;
    descriptor.m_Begin = {1,0,0};
    descriptor.m_Size  = {2,1,3};

    TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Slice,
                                                     {in, out},
                                                     descriptor,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

}
