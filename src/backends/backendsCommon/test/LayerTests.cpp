//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "LayerTests.hpp"
#include "WorkloadTestUtils.hpp"
#include "TensorUtils.hpp"
#include <ResolveType.hpp>

#include "test/TensorHelpers.hpp"
#include "TensorCopyUtils.hpp"
#include "Permute.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/assert.hpp>

#include <armnn/LayerSupport.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <algorithm>
#include <boost/cast.hpp>

#include "WorkloadTestUtils.hpp"
#include "Conv2dTestImpl.hpp"
#include "BatchNormTestImpl.hpp"
#include "ActivationTestImpl.hpp"
#include "Pooling2dTestImpl.hpp"
#include "FullyConnectedTestImpl.hpp"
#include "GatherTestImpl.hpp"
#include "SpaceToBatchNdTestImpl.hpp"
#include "SpaceToDepthTestImpl.hpp"
#include "SplitterTestImpl.hpp"
#include "SoftmaxTestImpl.hpp"
#include "StridedSliceTestImpl.hpp"
#include "NormTestImpl.hpp"
#include "LstmTestImpl.hpp"
#include "ConvertFp16ToFp32TestImpl.hpp"
#include "ConvertFp32ToFp16TestImpl.hpp"
#include "DebugTestImpl.hpp"
#include "DequantizeTestImpl.hpp"
#include "QuantizeTestImpl.hpp"
#include "TransposeConvolution2dTestImpl.hpp"

// 3-channel 16x8 image used as common input data for a number of Conv2d tests.
static std::vector<float> ConvInput3x8x16({
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
});

// 2-channel bias used by a number of Conv2d tests.
static std::vector<float> Bias2({0, 2});

static std::vector<float> Bias4({1, 2, 3, 4});

static std::vector<float> Bias8({1, 2, 3, 4, 1, 2, 3, 4});

struct Simple3dSoftmaxOutputData
{
    const std::vector<float> outputData =
            {
                0.0964599f, 0.26220518f, 0.0964599f, 0.0964599f,
                0.15903549f, 0.0964599f, 0.0964599f, 0.0964599f
            };

    const armnn::TensorShape inputShape{ 1, 8, 1 };

    const std::vector<float> inputData =
            {
                    0.f, 1.f, 0.f, 0.f,
                    .5f, 0.f, 0.f, 0.f,
            };
};

struct Simple4dSoftmaxData
{
    const armnn::TensorShape inputShape{ 1, 8, 1, 1 };

    const std::vector<float> outputData = { 0.0964599f, 0.26220518f, 0.0964599f, 0.0964599f,
                                            0.15903549f, 0.0964599f, 0.0964599f, 0.0964599f };
    const std::vector<float> inputData =
            {
                    0.f, 1.f, 0.f, 0.f,
                    .5f, 0.f, 0.f, 0.f
            };
};

// Helper function that returns either Bias2 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
boost::multi_array<T, 1> GetBias2(bool biasEnabled, float qScale)
{
    if(biasEnabled)
    {
        armnn::TensorInfo biasDesc({static_cast<unsigned int>(Bias2.size())}, ArmnnType);
        boost::multi_array<T, 1> bias = MakeTensor<T, 1>(biasDesc, QuantizedVector<T>(qScale, 0.0f, Bias2));
        return bias;
    }
    else
    {
        return boost::multi_array<T, 1>();
    }
}

// Helper function that returns either Bias4 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
boost::multi_array<T, 1> GetBias4(bool biasEnabled, float qScale)
{
    if(biasEnabled)
    {
        armnn::TensorInfo biasDesc({static_cast<unsigned int>(Bias4.size())}, ArmnnType);
        boost::multi_array<T, 1> bias = MakeTensor<T, 1>(biasDesc, QuantizedVector<T>(qScale, 0.0f, Bias4));
        return bias;
    }
    else
    {
        return boost::multi_array<T, 1>();
    }
}

// Helper function that returns either Bias8 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
boost::multi_array<T, 1> GetBias8(bool biasEnabled, float qScale)
{
    if(biasEnabled)
    {
        armnn::TensorInfo biasDesc({static_cast<unsigned int>(Bias4.size())}, ArmnnType);
        boost::multi_array<T, 1> bias = MakeTensor<T, 1>(biasDesc, QuantizedVector<T>(qScale, 0.0f, Bias8));
        return bias;
    }
    else
    {
        return boost::multi_array<T, 1>();
    }
}

// Helper function that returns either Bias4 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
boost::multi_array<T, 1> GetBias(bool biasEnabled, float qScale, armnn::TensorInfo outputInfo, armnn::DataLayout layout)
{
    const armnnUtils::DataLayoutIndexed dataLayoutIndexed(layout);
    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int outputChannels = outputInfo.GetShape()[channelsIndex];

    switch (outputChannels)
    {
        case 2:
        default:
        {
            return GetBias2<ArmnnType>(biasEnabled, qScale);
        }
        case 4:
        {
            return GetBias4<ArmnnType>(biasEnabled, qScale);
        }
        case 8:
        {
            return GetBias8<ArmnnType>(biasEnabled, qScale);
        }
    }
}


template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x5TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use common single-batch 3-channel 16x8 image.
    armnn::TensorInfo inputDesc({1, 3, 8, 16}, ArmnnType);
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, QuantizedVector<T>(qScale, qOffset, ConvInput3x8x16));

    // Use a 2-element batch with 3-channel 3x5 kernels.
    armnn::TensorInfo kernelDesc({2, 3, 5, 3}, ArmnnType);
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            1, 1, 1,
            1, -1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,

            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,

            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,


            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,

            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,

            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        })));

    // Expected output is 2 batch elements of a 1-channel 14x4 image.
    armnn::TensorInfo outputDesc({1, 2, 4, 14}, ArmnnType);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24,
            -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25,
            -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f,
            -23.5f, -23.5f, -23.5f,
            -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f,
            -23.5f, -23.5f, -23.5f,

            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        })));

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        qScale,
        qOffset,
        layout);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x3TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use a 3x3 kernel, which exercises ArmCompute's direct convolution path.

    // Use common single-batch 3-channel 16x8 image.
    armnn::TensorInfo inputDesc({1, 3, 8, 16}, ArmnnType);
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, QuantizedVector<T>(qScale, qOffset, ConvInput3x8x16));

    // Use a 2-element batch of 3-channel 3x3 kernels.
    armnn::TensorInfo kernelDesc({2, 3, 3, 3}, ArmnnType);
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            1, 1, 1,
            1, -1, 1,
            1, 1, 1,

            0, 0, 0,
            0, 0, 0,
            0, 0, 0,

            2, 2, 2,
            2, 2, 2,
            2, 2, 2,


            0, 0, 0,
            0, 0, 0,
            0, 0, 0,

            1, 1, 1,
            1, 1, 1,
            1, 1, 1,

            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        })));

    // Expected output is 1 batch of a 2-channel 14x6 image.
    armnn::TensorInfo outputDesc({1, 2, 6, 14}, ArmnnType);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15,
            -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,

            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        })));

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        qScale,
        qOffset,
        layout);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x3NhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    armnn::DataLayout dataLayout)
{
    // Use common single-batch 5x5 image.

    armnn::TensorInfo inputDesc({1, 3, 4, 1}, ArmnnType);
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc,
                                                      {
                                                       1, 5, 2, 3,
                                                       8, 7, 3, 6,
                                                       3, 3, 9, 1
                                                       });


    // Use a 2-element batch of 3-channel 3x3 kernels.
    armnn::TensorInfo kernelDesc({1, 3, 3, 1}, ArmnnType);
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc, {
                                                                    4, 5, 6,
                                                                    0, 0, 0,
                                                                    3, 2, 1
                                                                    });

    // Expected output is 1 batch of a 5x5 image.
    armnn::TensorInfo outputDesc({1, 3, 4, 1}, ArmnnType);

    const std::vector<float> outputData =
            {
                    23, 41, 33, 21,
                    44, 65, 76, 52,
                    82, 85, 79, 42
            };

    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, outputData);

    return SimpleConvolution2dNhwcTestImpl<ArmnnType, ArmnnType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        boost::multi_array<T, 1>(),
        expectedOutput,
        dataLayout,
        qScale,
        qOffset);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x3Stride2x2TestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float qScale,
        int32_t qOffset,
        bool biasEnabled,
        const armnn::DataLayout& dataLayout)
{
    // Input is a single-batch, 1 channel, 5x5 image.
    armnn::TensorInfo inputDesc({1, 5, 5, 1}, ArmnnType);
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc,
            {
                1, 5, 2, 3, 5,
                8, 7, 3, 6, 3,
                3, 3, 9, 1, 9,
                4, 1, 8, 1, 3,
                6, 8, 1, 9, 2
            });

    // Use a 3x3 kernel.
    armnn::TensorInfo kernelDesc({1, 3, 3, 1}, ArmnnType);
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc,
            {
                4, 5, 6,
                0, 0, 0,
                3, 2, 1
            });

    // Expected output is a single-batch, 1 channel, 3x3 image.
    armnn::TensorInfo outputDesc({1, 3, 3, 1}, ArmnnType);

    const std::vector<T> outputData =
            {
                23, 33, 24,
                91, 99, 48,
                26, 50, 19
            };

    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, outputData);

    uint32_t padLeft = 1;
    uint32_t padTop = 1;
    uint32_t padRight = 1;
    uint32_t padBottom = 1;
    uint32_t strideX  = 2;
    uint32_t strideY  = 2;

    return SimpleConvolution2dNhwcTestImpl<ArmnnType, ArmnnType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        boost::multi_array<T, 1>(),
        expectedOutput,
        dataLayout,
        qScale,
        qOffset,
        padLeft,
        padTop,
        padRight,
        padBottom,
        strideX,
        strideY);
}

LayerTestResult<float, 4> SimpleConvolution2d3x5Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x5TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.f, 0, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x5Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x5TestCommon<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return SimpleConvolution2d3x3NhwcTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        0.f,
        0,
        biasEnabled,
        armnn::DataLayout::NHWC);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3Stride2x2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3Stride2x2TestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        0.f,
        0,
        biasEnabled,
        layout);
}

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<int16_t, 4> SimpleConvolution2d3x5QSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
return SimpleConvolution2d3x5TestCommon<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<int16_t, 4> SimpleConvolution2d3x3QSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout,
    float qScale,
    int32_t qOffset)
{
    // Use a single-batch 1-channel 3x3 image as input.
    armnn::TensorInfo inputDesc({1, 1, 3, 3}, ArmnnType);
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            11,21,31,
            12,22,32,
            13,23,33
        })));

    // Use 1 batch of a 1-channel 2x2 kernel.
    armnn::TensorInfo kernelDesc({1, 1, 2, 2}, ArmnnType);
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            -11,-21,
            -12,-22,
        })));

// Expected output is 1 batch of a 1-channel 6x8 image.
// Manually calculated like this:
//[-11*0 -21*0  -12*0 -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0 -12*0  -22*0 ..]
//[-11*0 -21*0  -12*0 -22*11 ; -11*0  -21*0  -12*11 -22*21 ; -11*0  -21*0  -12*21 -22*31 ; -11*0  -21*0 -12*31 -22*0 ..]
//[-11*0 -21*11 -12*0 -22*12 ; -11*11 -21*21 -12*12 -22*22 ; -11*21 -21*31 -12*22 -22*32 ; -11*31 -21*0 -12*32 -22*0 ..]
//[-11*0 -21*12 -12*0 -22*13 ; -11*12 -21*22 -12*13 -22*23 ; -11*22 -21*32 -12*23 -22*33 ; -11*32 -21*0 -12*33 -22*0 ..]
//[-11*0 -21*13 -12*0 -22*0  ; -11*13 -21*23 -12*0  -22*0  ; -11*23 -21*33 -12*0  -22*0  ; -11*33 -21*0 -12*0  -22*0 ..]
//[-11*0 -21*0  -12*0 -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0 -12*0  -22*0 ..]
//[..... .....  ..... .....  ; .....  .....  .....  .....  ; .....  .....  .....  .....  ; .....  ..... .....  ..... ..]
    armnn::TensorInfo outputDesc({1, 1, 8, 6}, ArmnnType);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
               0,    0,      0,    0,    0,    0,
            -242,  -594,  -934, -372,    0,    0,
            -495, -1190, -1850, -725,    0,    0,
            -538, -1256, -1916, -748,    0,    0,
            -273, -626,  -946,  -363,    0,    0,
               0,    0,     0,     0,    0,    0,
               0,    0,     0,     0,    0,    0,
               0,    0,     0,     0,    0,    0
        })));

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<ArmnnBType>(false, qScale * qScale),
        expectedOutput,
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        2,  // Padding top.
        3,  // Padding right.
        4); // Padding bottom.
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2dAsymmetricPaddingTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout,
    float qScale,
    int32_t qOffset)
{
    // Use a single-batch 1-channel 5x5 image as input.
    armnn::TensorInfo inputDesc({ 1, 1, 5, 5 }, ArmnnType);
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            11,21,31,41,51,
            12,22,32,42,52,
            13,23,33,43,53,
            14,24,34,44,54,
            15,25,35,45,55,
        })));

    // Use 1 batch of a 1-channel 4x4 kernel.
    armnn::TensorInfo kernelDesc({ 1, 1, 4, 4 }, ArmnnType);
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            -11,-21,-31,-41,
            -12,-22,-32,-42,
            -13,-23,-33,-43,
            -14,-24,-34,-44,
        })));

    // Expected output is 1 batch of a 1-channel 5x5 image.
    armnn::TensorInfo outputDesc({ 1, 1, 5, 5 }, ArmnnType);
    std::vector<T> myVec(outputDesc.GetNumElements(), 0);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            -7140, -10580, -13940,  -9300, -5230,
            -9590, -14120, -18520, -12290, -6860,
            -9980, -14560, -18960, -12560, -7000,
            -7518, -10904, -14144,  -9318, -5152,
            -5032,  -7256,  -9376,  -6142, -3368,
        })));

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<ArmnnBType>(false, qScale * qScale),
        expectedOutput,
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2); // Padding bottom.
}

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout layout)
{
    return SimpleConvolution2dAsymmetricPaddingTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, layout, 0.0f, 0);
}

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout layout)
{
    return Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTestCommon
            <armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, layout, 0.0f, 0);
}

LayerTestResult<float, 4> Convolution1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return Convolution1dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, 0.0f, 0, biasEnabled);
}

LayerTestResult<uint8_t, 4> Convolution1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return Convolution1dTestImpl<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, 0.1f, 128, biasEnabled);
}

LayerTestResult<float,4> CompareConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory)
{
    return CompareConvolution2dTestImpl<armnn::DataType::Float32>(
            workloadFactory, memoryManager, refWorkloadFactory);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d3x3DilationTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const std::vector<float>& inputNoQuantizedValues,
    armnn::TensorInfo& inputTensorInfo,
    const std::vector<float>& kernelNoQuantizedValues,
    armnn::TensorInfo& kernelTensorInfo,
    const std::vector<float>& outputExpectedNoQuantizedValues,
    armnn::TensorInfo& outputTensorInfo,
    uint32_t dilationX,
    uint32_t dilationY,
    armnn::DataLayout layout = armnn::DataLayout::NCHW,
    uint32_t padLeft = 0,
    uint32_t padTop = 0,
    uint32_t padRight = 0,
    uint32_t padBottom = 0,
    uint32_t strideX  = 1,
    uint32_t strideY  = 1,
    bool biasEnabled = false
)
{
    float qScale;
    int32_t qOffset;
    switch (ArmnnType)
    {
        case armnn::DataType::QuantisedAsymm8:
        {
            qScale = 0.1f;
            qOffset = 128;
            break;
        }
        case armnn::DataType::QuantisedSymm16:
        {
            qScale = 0.1f;
            qOffset = 0;
            break;
        }
        case armnn::DataType::Float32:
        default:
        {
            qScale = 0.f;
            qOffset = 0;
            break;
        }
    }

    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);
    kernelTensorInfo.SetQuantizationScale(qScale);
    kernelTensorInfo.SetQuantizationOffset(qOffset);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    auto input = MakeTensor<T, 4>(inputTensorInfo,
                                  std::vector<T>(QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                    inputTensorInfo.GetQuantizationOffset(),
                                                                    inputNoQuantizedValues)));
    auto kernel = MakeTensor<T, 4>(kernelTensorInfo,
                                  std::vector<T>(QuantizedVector<T>(kernelTensorInfo.GetQuantizationScale(),
                                                                    kernelTensorInfo.GetQuantizationOffset(),
                                                                    kernelNoQuantizedValues)));
    auto expectedOutput = MakeTensor<T, 4>(outputTensorInfo,
                                           std::vector<T>(QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                             outputTensorInfo.GetQuantizationOffset(),
                                                                             outputExpectedNoQuantizedValues)));

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            input,
            kernel,
            GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
            expectedOutput,
            qScale,
            qOffset,
            layout,
            padLeft,
            padTop,
            padRight,
            padBottom,
            strideX,
            strideY,
            dilationX,
            dilationY);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> Convolution2d3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 1, 10, 10}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    armnn::TensorInfo kernelTensorInfo({ 1, 1, 3, 3}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
    {
        6., 5., 5., 5.,
        6., 5., 5., 5.,
        6., 5., 5., 5.,
        3., 2., 2., 2.
    };

    return Convolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> Convolution2d2x3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 2, 10, 10}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    armnn::TensorInfo kernelTensorInfo({ 1, 2, 3, 3}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
    {
        12., 10., 10., 10.,
        12., 10., 10., 10.,
        12., 10., 10., 10.,
         6.,  4.,  4.,  4.
    };

    return Convolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 1, 10, 10}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    armnn::TensorInfo kernelTensorInfo({ 1, 1, 2, 2}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
    {
        1, 2,
        3, 4
    };

    // Since the dilation rate is 2 this will dilate the kernel to be like 3x3: d(K-1)+1 --> 2 x (2-1) + 1 = 3,
    // therefore the output will be 4x4: (I − K + 2P)/S +1 => trunc ( (10 - 3 + 2x2 ) / 3 + 1 )
    // where, dilation size = d = 2; kernel size = K = 2; input size = I = 10; padding size = P = 2; stride = S = 3
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
    {
        4,  7,  7, 3,
        6, 10, 10, 4,
        6, 10, 10, 4,
        2,  3,  3, 1
    };
    uint32_t padLeft = 1;
    uint32_t padTop = 1;
    uint32_t padRight = 1;
    uint32_t padBottom = 1;

    return Convolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            2,
            2,
            layout,
            padLeft,
            padTop,
            padRight,
            padBottom,
            3,
            3,
            biasEnabled
            );
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedAsymm8>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedAsymm8>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedAsymm8>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dAsymmetricTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use a single-batch 2-channel 5x5 image as input.
    armnn::TensorInfo inputTensorInfo({ 1, 2, 5, 5 }, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset(),
        {
             0,  1,  2,  3,  4,
             5,  6,  7,  8,  9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,

            25, 26, 27, 28, 29,
            30, 31, 32, 33, 34,
            35, 36, 37, 38, 39,
            40, 41, 42, 43, 44,
            45, 46, 47, 48, 49
        })));

    // Use a depth multiplier of 1 on a 2-channel 4x4 kernel.
    armnn::TensorInfo kernelTensorInfo({ 1, 2, 4, 4 }, ArmnnType);
    auto kernel = MakeTensor<T, 4>(kernelTensorInfo, std::vector<T>(
        QuantizedVector<T>(kernelTensorInfo.GetQuantizationScale(), kernelTensorInfo.GetQuantizationOffset(),
        {
            32, 31, 30, 29,
            28, 27, 26, 25,
            24, 23, 22, 21,
            20, 19, 18, 17,

            16, 15, 14, 13,
            12, 11, 10,  9,
             8,  7,  6,  5,
             4,  3,  2,  1
        })));

    // Expected output is 1 batch of a 2-channel 5x5 image.
    // Calculated using the python tensorflow library with strideX=1, strideY=1.
    armnn::TensorInfo outputTensorInfo({ 1, 2, 5, 5 }, ArmnnType);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputTensorInfo, std::vector<T>(
        QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
        {
            1062, 1580, 1850, 1530, 1117,
            2140, 3108, 3500, 2842, 2042,
            3580, 5068, 5460, 4342, 3062,
            3618, 5072, 5390, 4248, 2971,
            3074, 4282, 4510, 3533, 2457,

            1550, 2284, 2362, 1955, 1428,
            2910, 4206, 4342, 3528, 2536,
            3390, 4886, 5022, 4068, 2916,
            3566, 5056, 5182, 4133, 2922,
            3100, 4352, 4452, 3517, 2465
        })));

    return DepthwiseConvolution2dAsymmetricTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2,  // Padding bottom.
        1,  // strideX
        1); // strideY
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dNhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled)
{
    auto layout = armnn::DataLayout::NHWC;

    armnn::TensorInfo inputTensorInfo({ 1, 2, 5, 5}, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset(),
        {
             0,  1,  2,  3,  4,
             5,  6,  7,  8,  9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,

            25, 26, 27, 28, 29,
            30, 31, 32, 33, 34,
            35, 36, 37, 38, 39,
            40, 41, 42, 43, 44,
            45, 46, 47, 48, 49
        })));

    armnn::TensorInfo kernelTensorInfo({ 1, 2, 4, 4 }, ArmnnType);
    auto kernel = MakeTensor<T, 4>(kernelTensorInfo, std::vector<T>(
        QuantizedVector<T>(kernelTensorInfo.GetQuantizationScale(), kernelTensorInfo.GetQuantizationOffset(),
        {
             32, 31, 30, 29,
             28, 27, 26, 25,
             24, 23, 22, 21,
             20, 19, 18, 17,

             16, 15, 14, 13,
             12, 11, 10,  9,
              8,  7,  6,  5,
              4,  3,  2,  1
        })));

    armnn::TensorInfo outputTensorInfo({ 1, 2, 5, 5}, ArmnnType);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputTensorInfo, std::vector<T>(
        QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
        {
            1062, 1580, 1850, 1530, 1117,
            2140, 3108, 3500, 2842, 2042,
            3580, 5068, 5460, 4342, 3062,
            3618, 5072, 5390, 4248, 2971,
            3074, 4282, 4510, 3533, 2457,

            1550, 2284, 2362, 1955, 1428,
            2910, 4206, 4342, 3528, 2536,
            3390, 4886, 5022, 4068, 2916,
            3566, 5056, 5182, 4133, 2922,
            3100, 4352, 4452, 3517, 2465
        })));

    return DepthwiseConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2,  // Padding bottom.
        1,  // strideX
        1);  // strideY
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled)
{
    auto layout = armnn::DataLayout::NHWC;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 9, 9}, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset(),
        {
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0
        })));

    armnn::TensorInfo kernelTensorInfo({ 1, 1, 3, 3}, ArmnnType);
    auto kernel = MakeTensor<T, 4>(kernelTensorInfo, std::vector<T>(
        QuantizedVector<T>(kernelTensorInfo.GetQuantizationScale(), kernelTensorInfo.GetQuantizationOffset(),
        {
             1, 2, 3,
             4, 5, 6,
             7, 8, 9
        })));

    uint32_t padLeft = 0;
    uint32_t padTop = 0;
    uint32_t padRight = 0;
    uint32_t padBottom = 0;
    uint32_t strideX  = 1;
    uint32_t strideY  = 1;
    uint32_t dilationX  = 3;
    uint32_t dilationY  = 3;

    // Since the dilation rate is 3 this will reduce the size of the output from 9x9 to 3x3 of all 5s.
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3}, ArmnnType);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputTensorInfo, std::vector<T>(
        QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
        {
             5, 5, 5,
             5, 5, 5,
             5, 5, 5
        })));

    return DepthwiseConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        qScale,
        qOffset,
        layout,
        padLeft,
        padTop,
        padRight,
        padBottom,
        strideX,
        strideY,
        dilationX,
        dilationY);
}


template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2d3x3DilationTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const std::vector<float>& inputNoQuantizedValues,
        armnn::TensorInfo& inputTensorInfo,
        const std::vector<float>& kernelNoQuantizedValues,
        armnn::TensorInfo& kernelTensorInfo,
        const std::vector<float>& outputExpectedNoQuantizedValues,
        armnn::TensorInfo& outputTensorInfo,
        uint32_t dilationX,
        uint32_t dilationY,
        armnn::DataLayout layout = armnn::DataLayout::NCHW,
        bool biasEnabled = false)
{
    float qScale;
    int32_t qOffset;
    switch (ArmnnType)
    {
        case armnn::DataType::QuantisedAsymm8:
        {
            qScale = 0.1f;
            qOffset = 128;
            break;
        }
        case armnn::DataType::QuantisedSymm16:
        {
            qScale = 0.1f;
            qOffset = 0;
            break;
        }
        case armnn::DataType::Float32:
        default:
        {
            qScale = 0.f;
            qOffset = 0;
            break;
        }
    }

    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);
    kernelTensorInfo.SetQuantizationScale(qScale);
    kernelTensorInfo.SetQuantizationOffset(qOffset);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    auto input = MakeTensor<T, 4>(inputTensorInfo,
                                  std::vector<T>(QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                    inputTensorInfo.GetQuantizationOffset(),
                                                                    inputNoQuantizedValues)));
    auto kernel = MakeTensor<T, 4>(kernelTensorInfo,
                                   std::vector<T>(QuantizedVector<T>(kernelTensorInfo.GetQuantizationScale(),
                                                                     kernelTensorInfo.GetQuantizationOffset(),
                                                                     kernelNoQuantizedValues)));
    auto expectedOutput = MakeTensor<T, 4>(outputTensorInfo,
                                           std::vector<T>(QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                             outputTensorInfo.GetQuantizationOffset(),
                                                                             outputExpectedNoQuantizedValues)));

    uint32_t padLeft = 0;
    uint32_t padTop = 0;
    uint32_t padRight = 0;
    uint32_t padBottom = 0;
    uint32_t strideX  = 1;
    uint32_t strideY  = 1;

    return DepthwiseConvolution2dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            input,
            kernel,
            GetBias<ArmnnBType>(biasEnabled, qScale * qScale, outputTensorInfo, layout),
            expectedOutput,
            qScale,
            qOffset,
            layout,
            padLeft,
            padTop,
            padRight,
            padBottom,
            strideX,
            strideY,
            dilationX,
            dilationY);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2d3x3Dilation3x3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 1, 10, 10}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };

    armnn::TensorInfo kernelTensorInfo({ 1, 1, 3, 3}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
            {
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9
            };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    3., 2., 2., 2.
            };

    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2d2x3x3Dilation3x3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 2, 10, 10}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };

    armnn::TensorInfo kernelTensorInfo({ 1, 2, 3, 3}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
            {
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,

                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9
            };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 2x4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 2, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    3., 2., 2., 2.,

                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    3., 2., 2., 2.
            };

    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2dMult4Test(
            armnn::IWorkloadFactory& workloadFactory,
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
            bool biasEnabled,
            const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 2, 3, 3}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,

                    21.0, 22.0, 23.0,
                    24.0, 25.0, 26.0,
                    27.0, 28.0, 29.0
            };

    armnn::TensorInfo kernelTensorInfo({ 4, 2, 2, 2}, ArmnnType);

    std::vector<float> kernelNoQuantizedValues =
            {
                    0.25f, 0.25f,
                    0.25f, 0.25f,

                    0.25f, 0.25f,
                    0.25f, 0.25f,

                    0.0f , 0.0f,
                    0.0f , 0.1f,

                    0.0f , 0.0f,
                    0.0f , 0.1f,

                    0.2f , 0.0f,
                    0.0f , 0.0f,

                    0.2f , 0.0f,
                    0.0f , 0.0f,

                    0.0f , 0.3f,
                    0.0f , 0.0f,

                    0.0f , 0.3f,
                    0.0f , 0.0f
            };

    armnn::TensorInfo outputTensorInfo({ 1, 8, 2, 2}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                    10.f, 10.f,
                    10.f, 10.f,

                    1.f, 1.f,
                    1.f, 1.f,

                    2.f, 2.f,
                    2.f, 2.f,

                    3.f, 3.f,
                    3.f, 3.f,

                    23.f, 24.f,
                    26.f, 27.f,

                    2.5f, 2.6000001f,
                    2.8f, 2.9f,

                    4.2000003f, 4.4f,
                    4.8f, 5.f,

                    6.6000004f, 6.9f,
                    7.5000005f, 7.8f
            };


    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            1,
            1,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2dMult2Test(
            armnn::IWorkloadFactory& workloadFactory,
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
            bool biasEnabled,
            const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 2, 3, 3}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,

                    21.0, 22.0, 23.0,
                    24.0, 25.0, 26.0,
                    27.0, 28.0, 29.0
            };

    armnn::TensorInfo kernelTensorInfo({ 2, 2, 2, 2}, ArmnnType);

    std::vector<float> kernelNoQuantizedValues =
            {
                    0.25f, 0.25f,
                    0.25f, 0.25f,

                    0.2f , 0.0f,
                    0.0f , 0.0f,

                    0.0f , 0.0f,
                    0.0f , 0.1f,

                    0.0f , 0.3f,
                    0.0f , 0.0f

            };

    armnn::TensorInfo outputTensorInfo({ 1, 4, 2, 2}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                    10.f, 10.f,
                    10.f, 10.f,

                    1.f, 1.f,
                    1.f, 1.f,

                    4.2000003f, 4.4f,
                    4.8f, 5.f,

                    6.6000004f, 6.9f,
                    7.5000005f, 7.8f
            };


    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            1,
            1,
            layout,
            biasEnabled);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedAsymm8>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedAsymm8>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2dMult4Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2dMult2Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout);

LayerTestResult<float, 4> DepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return DepthwiseConvolution2dNhwcTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, biasEnabled);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul64Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo({ 1, 1, 2, 2 }, armnn::DataType::Float32);
    auto input = MakeTensor<float, 4>(inputTensorInfo, { 1.f, 2.f, 3.f, 4.f });

    std::vector<float> kernelData;
    std::vector<float> singleDepthKernel{ 1.f, -1.f, -1.f, 1.f };
    for (unsigned int i = 0; i < 64; ++i)
    {
        kernelData.insert(kernelData.end(), singleDepthKernel.begin(), singleDepthKernel.end());
    }
    armnn::TensorInfo kernelTensorInfo({ 64, 1, 2, 2 }, armnn::DataType::Float32);
    auto kernel = MakeTensor<float, 4>(kernelTensorInfo, kernelData);

    std::vector<float> expectedOutputData(64, 0.f);
    armnn::TensorInfo outputTensorInfo({ 1, 64, 1, 1 }, armnn::DataType::Float32);
    auto expectedOutput = MakeTensor<float, 4>(outputTensorInfo, expectedOutputData);

    return DepthwiseConvolution2dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            input,
            kernel,
            boost::multi_array<float, 1>(),
            expectedOutput,
            0.f,
            0,
            armnn::DataLayout::NCHW);
}

LayerTestResult<float, 4> DepthwiseConvolution2dAsymmetricTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dAsymmetricTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dDepthMul1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            0.f,
            0,
            false);
}

LayerTestResult<int16_t, 4> DepthwiseConvolution2dInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<int16_t, 4> DepthwiseConvolution2dDepthMul1Int16Test(
                armnn::IWorkloadFactory& workloadFactory,
                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                bool biasEnabled,
                const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> CompareDepthwiseConvolution2dFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::DataLayout layout)
{
    return CompareDepthwiseConvolution2dTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, refWorkloadFactory, layout);
}

LayerTestResult<uint8_t, 4> CompareDepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::DataLayout layout)
{
    return CompareDepthwiseConvolution2dTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, refWorkloadFactory, layout);
}

LayerTestResult<float,4> SimpleNormalizationAcrossTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    auto normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    auto normChannel = armnn::NormalizationAlgorithmChannel::Across;
    return SimpleNormalizationTestImpl(workloadFactory, memoryManager, normChannel, normMethod);
}

LayerTestResult<float,4> SimpleNormalizationWithinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    auto normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    auto normChannel = armnn::NormalizationAlgorithmChannel::Within;
    return SimpleNormalizationTestImpl(workloadFactory, memoryManager, normChannel, normMethod);
}

LayerTestResult<float,4> SimpleNormalizationAcrossNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    auto normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    auto normChannel = armnn::NormalizationAlgorithmChannel::Across;
    return SimpleNormalizationNhwcTestImpl(workloadFactory, memoryManager, normChannel, normMethod);
}

LayerTestResult<float,2> SimpleSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, beta);
}

LayerTestResult<float,2> SimpleAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta,
        int axis)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, beta, axis);
}

LayerTestResult<float,3> Simple3dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta)
{
    Simple3dSoftmaxOutputData data;
    return Simple3dSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, beta,
                                                             data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<float,3> Simple3dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta,
        int axis)
{
    armnn::TensorShape inputShape;
    std::vector<float> inputData;
    std::vector<float> outputData;
    switch (axis)
    {
    case -3:
    case 0:
        {
            inputShape = {5, 2, 2};

            inputData =
                    {
                            17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f,

                            15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.087144312427294f,
                            0.087144312427294f,

                            0.087144312427294f, 0.087144312427294f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f
                    };
            break;
        }
    case -2:
    case 1:
        {
            inputShape = {2, 5, 2};

            inputData =
                    {
                            17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,

                            17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                            0.087144312427294f,
                            0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                            7.246299848982885e-08f,

                            0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                            0.087144312427294f,
                            0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                            7.246299848982885e-08f
                    };
        break;
        }
    case -1:
    case 2:
        {
            inputShape = {2, 2, 5};

            inputData =
                    {
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,

                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f
                    };
            break;
        }
    }

    return Simple3dSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, beta,
                                                             inputShape, outputData, inputData, axis);
}

LayerTestResult<float,4> Simple4dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta)
{
    Simple4dSoftmaxData data;
    return Simple4dSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, beta, data.inputShape,
                                                             data.outputData, data.inputData);
}

LayerTestResult<float,4> Simple4dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta,
        int axis)
{
    armnn::TensorShape inputShape;
    std::vector<float> inputData;
    std::vector<float> outputData;
    switch (axis)
    {
    case -4:
    case 0:
        {
            inputShape = {5, 2, 2, 2};

            inputData =
                    {
                            17.0f, -1.0f, 17.0f, -1.0f, 17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f,
                            16.0f, -2.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f, 15.0f, -3.0f,
                            15.0f, -3.0f, 15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 14.0f, -4.0f,
                            14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f, 1.0f, -17.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.643914213228014f,
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.236882800924671f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.236882800924671f,
                            0.236882800924671f,
                            0.236882800924671f, 0.087144312427294f, 0.087144312427294f, 0.087144312427294f,
                            0.087144312427294f,

                            0.087144312427294f, 0.087144312427294f, 0.087144312427294f, 0.087144312427294f,
                            0.032058600957022f,
                            0.032058600957022f, 0.032058600957022f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f,
                            7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f, 7.246299848982885e-08f
                    };
            break;
        }
    case -3:
    case 1:
        {
            inputShape = {2, 5, 2, 2};

            inputData =
                    {
                            17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f,
                            15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f,
                            17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f,
                            15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.087144312427294f,
                            0.087144312427294f,
                            0.087144312427294f, 0.087144312427294f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f,


                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.087144312427294f,
                            0.087144312427294f,
                            0.087144312427294f, 0.087144312427294f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f
                    };
            break;
        }
    case -2:
    case 2:
        {
        inputShape = {2, 2, 5, 2};

        inputData =
                {
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f
                };

        outputData =
                {
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f,
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f,

                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f,
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f
                };
        break;
        }
    case -1:
    case 3:
        {
            inputShape = {2, 2, 2, 5};

            inputData =
                    {
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,

                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f
                    };
            break;
        }
    }

    return Simple4dSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, beta, inputShape,
                                                             outputData, inputData, axis);
}

LayerTestResult<uint8_t,2> SimpleSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, beta);
}

LayerTestResult<uint8_t,3> Simple3dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta)
{
    Simple3dSoftmaxOutputData data;
    return Simple3dSoftmaxTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, beta,
                                                                     data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<uint8_t,4> Simple4dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta)
{
    Simple4dSoftmaxData data;

    return Simple4dSoftmaxTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, beta,
                                                                     data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<int16_t,2> SimpleSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, beta);
}

LayerTestResult<int16_t,3> Simple3dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta)
{
    Simple3dSoftmaxOutputData data;
    return Simple3dSoftmaxTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, beta,
                                                                     data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<int16_t,4> Simple4dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta)
{
    Simple4dSoftmaxData data;

    return Simple4dSoftmaxTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, beta,
                                                                     data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<float,4> CompareNormalizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::NormalizationAlgorithmChannel normChannel,
    armnn::NormalizationAlgorithmMethod normMethod)
{
    return CompareNormalizationTestImpl(workloadFactory, memoryManager, refWorkloadFactory, normChannel, normMethod);
}

LayerTestResult<float,2> CompareSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float beta)
{
    return CompareSoftmaxTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, refWorkloadFactory, beta);
}

LayerTestResult<uint8_t,2> CompareSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float beta)
{
    return CompareSoftmaxTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, refWorkloadFactory, beta);
}

std::vector<LayerTestResult<float,3>> SplitterTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SplitterTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

std::vector<LayerTestResult<uint8_t,3>> SplitterUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SplitterTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.0f, 0);
}

std::vector<LayerTestResult<int16_t,3>> SplitterInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SplitterTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<float, 3> CopyViaSplitterTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return CopyViaSplitterTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<uint8_t, 3> CopyViaSplitterUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return CopyViaSplitterTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<int16_t, 3> CopyViaSplitterInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return CopyViaSplitterTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 1.0f, 0);
}

#if defined(ARMNNREF_ENABLED)

// The LSTM test units are run only for the reference backend at the moment

void LstmUtilsZeroVectorTest()
{
    armnn::TensorInfo inputDesc({4}, armnn::DataType::Float32);
    boost::multi_array<float, 1> input = MakeTensor<float, 1>(inputDesc, std::vector<float>(
            {2., 3., 3., 4.}));

    boost::multi_array<float, 1> expectedOutput = MakeTensor<float, 1>(inputDesc, std::vector<float>(
            {0., 0., 0., 0.}));

    return LstmUtilsZeroVectorTestImpl<armnn::DataType::Float32>(input, 4, expectedOutput);
}

void LstmUtilsMeanStddevNormalizationNoneZeroInputTest()
{
    uint32_t batchSize = 2;
    uint32_t vecSize = 4;
    armnn::TensorInfo inputDesc({batchSize, vecSize}, armnn::DataType::Float32);
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            { 0.1f, 0.2f, 0.3f, 0.4f,      //batch 0
              0.9f, 1.0f, 1.1f, 1.2f }));  //batch 1

    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            { -1.34164071f, -0.447213531f, 0.44721365f,  1.34164071f,      //batch 0
              -1.34163153f, -0.447210163f, 0.447211236f, 1.3416326f  }));  //batch 1

    return LstmUtilsMeanStddevNormalizationTestImpl<armnn::DataType::Float32>(input,
            vecSize, batchSize, expectedOutput);
}

void LstmUtilsMeanStddevNormalizationAllZeroInputTest()
{
    uint32_t batchSize = 2;
    uint32_t vecSize = 4;
    armnn::TensorInfo inputDesc({batchSize, vecSize}, armnn::DataType::Float32);
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            { 0.0f, 0.0f, 0.0f, 0.0f,      //batch 0
              0.0f, 0.0f, 0.0f, 0.0f }));  //batch 1

    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            { 0.0f, 0.0f, 0.0f, 0.0f,      //batch 0
              0.0f, 0.0f, 0.0f, 0.0f }));  //batch 1

    return LstmUtilsMeanStddevNormalizationTestImpl<armnn::DataType::Float32>(input,
            vecSize, batchSize, expectedOutput);
}

void LstmUtilsMeanStddevNormalizationMixedZeroInputTest()
{
    uint32_t batchSize = 2;
    uint32_t vecSize = 4;
    armnn::TensorInfo inputDesc({batchSize, vecSize}, armnn::DataType::Float32);
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            { 0.0f, 0.0f, 0.0f, 0.0f,      //batch 0
              0.1f, 0.2f, 0.3f, 0.4f }));  //batch 1

    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            {         0.0f,          0.0f,        0.0f,        0.0f,      //batch 0
              -1.34164071f, -0.447213531f, 0.44721365f, 1.34164071f }));  //batch 1

    return LstmUtilsMeanStddevNormalizationTestImpl<armnn::DataType::Float32>(input,
            vecSize, batchSize, expectedOutput);
}


void LstmUtilsVectorBatchVectorCwiseProductTest()
{
    uint32_t batchSize = 4;
    uint32_t vecSize = 29;
    armnn::TensorInfo vecDesc({vecSize}, armnn::DataType::Float32);
    boost::multi_array<float, 1> vector = MakeTensor<float, 1>(vecDesc, std::vector<float>(
            {   1.1f,   2.2f,   3.3f,   4.4f,   5.5f,   6.6f,   7.7f,   8.8f,   9.9f, 10.1f,
              11.11f, 12.12f, 13.13f, 14.14f, 15.15f, 16.16f, 17.17f, 18.18f, 19.19f, 20.2f,
              21.21f, 22.22f, 23.23f, 24.24f, 25.25f, 26.26f, 27.27f, 28.28f,     0.0f}));

    armnn::TensorInfo batchVecDesc({batchSize, vecSize}, armnn::DataType::Float32);
    boost::multi_array<float, 2> batchVector = MakeTensor<float, 2>(batchVecDesc, std::vector<float>(
            { /* batch 0 */
                1.1f,   2.2f,   3.3f,   4.4f,   5.5f,   6.6f,   7.7f,   8.8f,   9.9f,  10.1f,
              11.11f, 12.12f, 13.13f, 14.14f, 15.15f, 16.16f, 17.17f, 18.18f, 19.19f,  20.2f,
              21.21f, 22.22f, 23.23f, 24.24f, 25.25f, 26.26f, 27.27f, 28.28f,   0.0f,
              /* batch 1 */
                -1.1f,   -2.2f,   -3.3f,   -4.4f,   -5.5f,   -6.6f,   -7.7f,   -8.8f,   -9.9f, -10.1f,
              -11.11f, -12.12f, -13.13f, -14.14f, -15.15f, -16.16f, -17.17f, -18.18f, -19.19f, -20.2f,
              -21.21f, -22.22f, -23.23f, -24.24f, -25.25f, -26.26f, -27.27f, -28.28f,    0.0f,
              /* batch 2 */
                1.1f,   -2.2f,   3.3f,   -4.4f,   5.5f,   -6.6f,   7.7f,   -8.8f,   9.9f, -10.1f,
              11.11f, -12.12f, 13.13f, -14.14f, 15.15f, -16.16f, 17.17f, -18.18f, 19.19f, -20.2f,
              21.21f, -22.22f, 23.23f, -24.24f, 25.25f, -26.26f, 27.27f, -28.28f,   0.0f,
              /* batch 3 */
                -1.1f,   2.2f,   -3.3f,   4.4f,   -5.5f,   6.6f,   -7.7f,   8.8f,   -9.9f, 10.1f,
              -11.11f, 12.12f, -13.13f, 14.14f, -15.15f, 16.16f, -17.17f, 18.18f, -19.19f, 20.2f,
              -21.21f, 22.22f, -23.23f, 24.24f, -25.25f, 26.26f, -27.27f, 28.28f,    0.0f}));

    // Expect output = input * output + output.
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(batchVecDesc, std::vector<float>(
            { /* batch 0 */
                 1.210000f,    4.840000f,   10.889999f,   19.360001f,   30.250000f,   43.559998f,
                59.289997f,   77.440002f,   98.009995f,  102.010010f,  123.432091f,  146.894394f,
               172.396896f,  199.939606f,  229.522491f,  261.145599f,  294.808899f,  330.512421f,
               368.256134f,  408.040039f,  449.864075f,  493.728363f,  539.632874f,  587.577576f,
               637.562500f,  689.587585f,  743.652954f,  799.758423f,    0.000000f,
              /* batch 1 */
                -1.210000f,   -4.840000f,  -10.889999f,  -19.360001f,  -30.250000f,  -43.559998f,
               -59.289997f,  -77.440002f,  -98.009995f, -102.010010f, -123.432091f, -146.894394f,
              -172.396896f, -199.939606f, -229.522491f, -261.145599f, -294.808899f, -330.512421f,
              -368.256134f, -408.040039f, -449.864075f, -493.728363f, -539.632874f, -587.577576f,
              -637.562500f, -689.587585f, -743.652954f, -799.758423f,    0.000000f,
              /* batch 2 */
                 1.210000f,   -4.840000f,  10.889999f,   -19.360001f,   30.250000f,  -43.559998f,
                59.289997f,  -77.440002f,  98.009995f,  -102.010010f,  123.432091f, -146.894394f,
               172.396896f, -199.939606f, 229.522491f,  -261.145599f,  294.808899f, -330.512421f,
               368.256134f, -408.040039f, 449.864075f,  -493.728363f,  539.632874f, -587.577576f,
               637.562500f, -689.587585f, 743.652954f,  -799.758423f,    0.000000f,
              /* batch 3 */
                -1.210000f,    4.840000f,  -10.889999f,   19.360001f,  -30.250000f,   43.559998f,
               -59.289997f,   77.440002f,  -98.009995f,  102.010010f, -123.432091f,  146.894394f,
              -172.396896f,  199.939606f, -229.522491f,  261.145599f, -294.808899f,  330.512421f,
              -368.256134f,  408.040039f, -449.864075f,  493.728363f, -539.632874f,  587.577576f,
              -637.562500f,  689.587585f, -743.652954f,  799.758423f,    0.000000f}));

    return LstmUtilsVectorBatchVectorCwiseProductTestImpl<armnn::DataType::Float32>(vector, batchVector,
            vecSize, batchSize, expectedOutput);
}


void LstmUtilsVectorBatchVectorAddTest()
{
    uint32_t batchSize = 2;
    uint32_t vecSize = 3;
    armnn::TensorInfo vecDesc({vecSize}, armnn::DataType::Float32);
    boost::multi_array<float, 1> vector = MakeTensor<float, 1>(vecDesc, std::vector<float>(
            { 0.0f, -0.5f, 1.0f}));

    armnn::TensorInfo batchVecDesc({batchSize, vecSize}, armnn::DataType::Float32);
    boost::multi_array<float, 2> batchVector = MakeTensor<float, 2>(batchVecDesc, std::vector<float>(
            { 1.0f, 2.0f, 3.0f,    //batch 0
              4.0f, 5.0f, 6.0f})); //batch 1

    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(batchVecDesc, std::vector<float>(
            { 1.0f, 1.5f, 4.0f,
              4.0f, 4.5f, 7.0f}));

    return LstmUtilsVectorBatchVectorAddTestImpl<armnn::DataType::Float32>(vector, batchVector,
            vecSize, batchSize, expectedOutput);
}

#endif

LayerTestResult<float, 2> LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({ 2, 2 }, armnn::DataType::Float32);
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            { 2., 3., 3., 4. }));

    armnn::TensorInfo outputDesc({ 2, 4 }, armnn::DataType::Float32);
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(outputDesc, std::vector<float>(
            {-0.36444446f, -0.00352185f, 0.12886585f, -0.05163646f,
             -0.42734814f, -0.00478661f,  0.13455015f, -0.03560682f}));
    return LstmLayerWithCifgWithPeepholeNoProjectionTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, input, expectedOutput);
}

LayerTestResult<float, 2> LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({ 2, 5 }, armnn::DataType::Float32);
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            {0.787926f, 0.151646f, 0.071352f, 0.118426f, 0.458058f,
             0.295743f, 0.544053f, 0.690064f, 0.858138f, 0.497181f}));

    armnn::TensorInfo outputDesc({ 2, 16 }, armnn::DataType::Float32);
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(outputDesc, std::vector<float>(
            {-0.00396806f, 0.029352f,     -0.00279226f, 0.0159977f,   -0.00835576f,
             -0.0211779f,  0.0283512f,    -0.0114597f,  0.00907307f,  -0.0244004f,
             -0.0152191f,  -0.0259063f,   0.00914318f,  0.00415118f,  0.017147f,
             0.0134203f, -0.013869f,    0.0287268f,   -0.00334693f, 0.00733398f,  -0.0287926f,
             -0.0186926f,   0.0193662f,   -0.0115437f,  0.00422612f,  -0.0345232f,
             0.00223253f,   -0.00957321f, 0.0210624f,   0.013331f,    0.0150954f,
             0.02168f}));
    return LstmLayerNoCifgWithPeepholeWithProjectionTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, input, expectedOutput);
}

LayerTestResult<float, 2> LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({2, 2}, armnn::DataType::Float32);
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            {2., 3., 3., 4.}));


    armnn::TensorInfo outputDesc({2, 4}, armnn::DataType::Float32);
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(outputDesc, std::vector<float>(
            {{-0.02973187f, 0.1229473f,   0.20885126f, -0.15358765f,
              -0.0185422f,   0.11281417f,  0.24466537f, -0.1826292f}}));

    return LstmNoCifgNoPeepholeNoProjectionTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, input, expectedOutput);
}


LayerTestResult<float, 2> LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({ 2, 5 }, armnn::DataType::Float32);
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            {0.7f, 0.8f, 0.1f, 0.2f, 0.3f,     //batch 0
             0.3f, 0.2f, 0.9f, 0.8f, 0.1f}));  //batch 1

    armnn::TensorInfo outputDesc({ 2, 3 }, armnn::DataType::Float32);
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(outputDesc, std::vector<float>(
            {  0.0244077f,  0.128027f, -0.00170918f,    //batch 0
             -0.00692428f, 0.0848741f,    0.063445f})); //batch 1
    return LstmLayerNoCifgWithPeepholeWithProjectionWithLayerNormTestImpl<armnn::DataType::Float32>(
            workloadFactory, memoryManager, input, expectedOutput);
}


LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgNoPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    const armnn::DataType datatype = armnn::DataType::QuantisedSymm16;
    const armnn::DataType constantDatatype = armnn::DataType::QuantisedAsymm8;

    armnn::TensorInfo inputDesc({2, 2}, datatype);
    boost::multi_array<int16_t , 2> input = MakeTensor<int16_t , 2>(inputDesc, QuantizedVector<int16_t>(qScale, qOffset,
            std::vector<float>{2., 3., 3., 4.}));

    armnn::TensorInfo outputDesc({2, 4}, datatype);
    boost::multi_array<int16_t, 2> expectedOutput = MakeTensor<int16_t, 2>(outputDesc, QuantizedVector<int16_t>(qScale,
            qOffset, std::vector<float>({{-0.02973187f, 0.1229473f,   0.20885126f, -0.15358765f,
                                          -0.0185422f,  0.11281417f,  0.24466537f, -0.1826292f}})));

    return LstmNoCifgNoPeepholeNoProjectionTestImpl<datatype>(
        workloadFactory, memoryManager, input, expectedOutput, qScale, qOffset, constantDatatype);

}

LayerTestResult<int16_t, 2> LstmLayerInt16WithCifgWithPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    const armnn::DataType datatype = armnn::DataType::QuantisedSymm16;
    const armnn::DataType constantDatatype = armnn::DataType::QuantisedAsymm8;

    armnn::TensorInfo inputDesc({ 2, 2 }, datatype);
    boost::multi_array<int16_t, 2> input = MakeTensor<int16_t, 2>(inputDesc, QuantizedVector<int16_t>(qScale, qOffset,
            std::vector<float>({ 2., 3., 3., 4. })));

    armnn::TensorInfo outputDesc({ 2, 4 }, datatype);
    boost::multi_array<int16_t, 2> expectedOutput = MakeTensor<int16_t, 2>(outputDesc, QuantizedVector<int16_t>(qScale,
            qOffset, std::vector<float>(
            {-0.36444446f, -0.00352185f, 0.12886585f, -0.05163646f,
             -0.42734814f, -0.00478661f, 0.13455015f, -0.03560682f})));

    return LstmLayerWithCifgWithPeepholeNoProjectionTestImpl<datatype>(
        workloadFactory, memoryManager, input, expectedOutput, qScale, qOffset, constantDatatype);
}

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgWithPeepholeWithProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const float qScale = 2.0f;
    const int32_t qOffset = 0;

    const armnn::DataType datatype = armnn::DataType::QuantisedSymm16;
    const armnn::DataType constantDatatype = armnn::DataType::QuantisedAsymm8;

    armnn::TensorInfo inputDesc({ 2, 5 }, datatype);
    boost::multi_array<int16_t, 2> input = MakeTensor<int16_t, 2>(inputDesc, QuantizedVector<int16_t>(qScale,
            qOffset, std::vector<float>(
            {0.787926f, 0.151646f, 0.071352f, 0.118426f, 0.458058f,
             0.295743f, 0.544053f, 0.690064f, 0.858138f, 0.497181f})));

    armnn::TensorInfo outputDesc({ 2, 16 }, datatype);
    boost::multi_array<int16_t, 2> expectedOutput = MakeTensor<int16_t, 2>(outputDesc, QuantizedVector<int16_t>(qScale,
            qOffset, std::vector<float>(
            {-0.00396806f,  0.029352f,   -0.00279226f, 0.0159977f,  -0.00835576f,
             -0.0211779f,   0.0283512f,  -0.0114597f,  0.00907307f, -0.0244004f,
             -0.0152191f,  -0.0259063f,   0.00914318f, 0.00415118f,  0.017147f,
              0.0134203f,  -0.013869f,    0.0287268f, -0.00334693f,  0.00733398f, -0.0287926f,
             -0.0186926f,   0.0193662f,  -0.0115437f,  0.00422612f, -0.0345232f,
              0.00223253f, -0.00957321f,  0.0210624f,  0.013331f,    0.0150954f,   0.02168f})));

    return LstmLayerNoCifgWithPeepholeWithProjectionTestImpl<datatype>(
        workloadFactory, memoryManager, input, expectedOutput, qScale, qOffset, constantDatatype);
}

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16ConstantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    const armnn::DataType datatype = armnn::DataType::QuantisedSymm16; // datatype & constants set to QSymm16

    armnn::TensorInfo inputDesc({2, 2}, datatype);
    boost::multi_array<int16_t , 2> input = MakeTensor<int16_t , 2>(inputDesc, QuantizedVector<int16_t>(qScale,
            qOffset, std::vector<float>{2., 3., 3., 4.}));

    armnn::TensorInfo outputDesc({2, 4}, datatype);
    boost::multi_array<int16_t, 2> expectedOutput = MakeTensor<int16_t, 2>(outputDesc, QuantizedVector<int16_t>(qScale,
            qOffset, std::vector<float>({{-0.02973187f, 0.1229473f,   0.20885126f, -0.15358765f,
                                          -0.0185422f,  0.11281417f,  0.24466537f, -0.1826292f}})));

    return LstmNoCifgNoPeepholeNoProjectionTestImpl<datatype>(
        workloadFactory, memoryManager, input, expectedOutput, qScale, qOffset, datatype);
}

// QuantizedLstm
LayerTestResult<uint8_t, 2> QuantizedLstmTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({2, 2}, armnn::DataType::QuantisedAsymm8);
    boost::multi_array<uint8_t, 2> input = MakeTensor<uint8_t, 2>(inputDesc, std::vector<uint8_t>(
        {166, 179, 50, 150}));

    armnn::TensorInfo outputDesc({2, 4}, armnn::DataType::QuantisedAsymm8);
    boost::multi_array<uint8_t, 2> expectedOutput = MakeTensor<uint8_t, 2>(outputDesc, std::vector<uint8_t>(
        {140, 151, 146, 112, 136, 156, 142, 112 }));

    return QuantizedLstmTestImpl(workloadFactory, memoryManager, input, expectedOutput);
}

LayerTestResult<float,3> ConcatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int outputWidth = 3;
    unsigned int outputHeight = 6;
    unsigned int outputChannels = 3;

    unsigned int inputWidth1 = 3;
    unsigned int inputHeight1 = 6;
    unsigned int inputChannels1 = 2;

    unsigned int inputWidth2 = 3;
    unsigned int inputHeight2 = 6;
    unsigned int inputChannels2 = 1;

    // Define the tensor descriptors.
    armnn::TensorInfo outputTensorInfo({ outputChannels, outputHeight, outputWidth }, armnn::DataType::Float32);
    armnn::TensorInfo inputTensorInfo1({ inputChannels1, inputHeight1, inputWidth1 }, armnn::DataType::Float32);
    armnn::TensorInfo inputTensorInfo2({ inputChannels2, inputHeight2, inputWidth2 }, armnn::DataType::Float32);

    LayerTestResult<float,3> ret(outputTensorInfo);

    ret.outputExpected = MakeTensor<float, 3>(outputTensorInfo, std::vector<float>(
    {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,

            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f,

            37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f,
            43.0f, 44.0f, 45.0f,
            46.0f, 47.0f, 48.0f,
            49.0f, 50.0f, 51.0f,
            52.0f, 53.0f, 54.0f,
        })
    );

    auto input1 = MakeTensor<float, 3>(inputTensorInfo1, std::vector<float>(
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f,

            19.0f, 20.0f, 21.0f,
            22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f,
            28.0f, 29.0f, 30.0f,
            31.0f, 32.0f, 33.0f,
            34.0f, 35.0f, 36.0f,
        })
    );

    auto input2 = MakeTensor<float, 3>(inputTensorInfo2, std::vector<float>(
        {
            37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f,
            43.0f, 44.0f, 45.0f,
            46.0f, 47.0f, 48.0f,
            49.0f, 50.0f, 51.0f,
            52.0f, 53.0f, 54.0f,
        })
    );

    std::vector<unsigned int> wOrigin1 = {0, 0, 0}; //Extent of the window is defined by size of input[0].
    armnn::ConcatQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = {2, 0, 0}; //Extent of the window is defined by size of input[1].
    armnn::ConcatQueueDescriptor::ViewOrigin window2(wOrigin2);

    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    bool subTensorsSupported = workloadFactory.SupportsSubTensors();

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo1.GetShape(), wOrigin1.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo1);

    std::unique_ptr<armnn::ITensorHandle> inputHandle2  =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo2.GetShape(), wOrigin2.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo2);

    armnn::ConcatQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConcat(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<float,4> AdditionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int batchSize = 2;
    unsigned int channels  = 2;
    unsigned int height    = 2;
    unsigned int width     = 3;

    armnn::TensorInfo inputTensorInfo1, inputTensorInfo2;
    armnn::TensorInfo outputTensorInfo;

    unsigned int shape[] = {batchSize, channels, height, width};

    inputTensorInfo1 = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    inputTensorInfo2 = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);


    auto input1 = MakeTensor<float, 4>(inputTensorInfo1, std::vector<float>(
        {
            0.0f, 2.0f, 1.0f,
            0.2f, 1.0f, 2.0f,

            1.0f, 2.0f, 1.0f,
            0.2f, 1.0f, 2.0f,

            0.0f, 2.0f, 1.0f,
            4.2f, 1.0f, 2.0f,

            0.0f, 0.0f, 1.0f,
            0.2f, 1.0f, 2.0f,
        }));

    auto input2 = MakeTensor<float, 4>(inputTensorInfo2, std::vector<float>(
        {
            1.0f, 2.0f, 1.0f,
            0.0f, 1.0f, 2.0f,

            1.0f, 2.0f, -2.0f,
            0.2f, 1.0f, 2.0f,

            0.0f, 2.0f, 1.0f,
            4.2f, 0.0f, -3.0f,

            0.0f, 0.0f, 1.0f,
            0.7f, 1.0f, 5.0f,
        }));

    LayerTestResult<float,4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<float, 4>(outputTensorInfo, std::vector<float>(
        {
            1.0f, 4.0f, 2.0f,
            0.2f, 2.0f, 4.0f,

            2.0f, 4.0f, -1.0f,
            0.4f, 2.0f, 4.0f,

            0.0f, 4.0f, 2.0f,
            8.4f, 1.0f, -1.0f,

            0.0f, 0.0f, 2.0f,
            0.9f, 2.0f, 7.0f,
        }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = workloadFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateAddition(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<float, 5> Addition5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int depth     = 2;
    unsigned int batchSize = 2;
    unsigned int channels  = 2;
    unsigned int height    = 2;
    unsigned int width     = 3;

    armnn::TensorInfo inputTensorInfo1, inputTensorInfo2;
    armnn::TensorInfo outputTensorInfo;

    unsigned int shape[] = {depth, batchSize, channels, height, width};

    inputTensorInfo1 = armnn::TensorInfo(5, shape, armnn::DataType::Float32);
    inputTensorInfo2 = armnn::TensorInfo(5, shape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(5, shape, armnn::DataType::Float32);


    auto input1 = MakeTensor<float, 5>(inputTensorInfo1, std::vector<float>(
        {
            2.6f, 4.0f, 4.4f,  2.7f, 4.6f, 2.8f,
            2.3f, 1.9f, 3.4f,  2.9f, 2.2f, 4.5f,

            2.8f, 1.9f, 2.3f,  2.6f, 4.7f, 3.5f,
            0.4f, 1.5f, 2.1f,  0.7f, 5.0f, 1.1f,


            1.0f, 2.7f, 0.0f,  0.6f, 0.8f, 0.9f,
            1.0f, 2.6f, 0.4f,  3.8f, 0.4f, 0.8f,

            0.5f, 4.3f, 3.1f,  4.4f, 0.7f, 1.4f,
            0.4f, 4.4f, 0.7f,  0.6f, 4.7f, 1.2f,

        }));

    auto input2 = MakeTensor<float, 5>(inputTensorInfo2, std::vector<float>(
        {
            4.4f, 3.0f, 1.0f,  0.0f, 3.9f, 3.1f,
            1.7f, 2.9f, 1.3f,  0.4f, 0.4f, 4.3f,

            4.5f, 0.2f, 2.2f,  4.1f, 3.9f, 3.0f,
            0.1f, 2.5f, 4.1f,  4.6f, 1.5f, 0.0f,


            0.5f, 4.9f, 2.5f,  1.5f, 3.4f, 4.5f,
            2.0f, 3.0f, 4.9f,  1.6f, 2.4f, 3.4f,

            3.6f, 1.8f, 1.3f,  2.6f, 2.1f, 4.8f,
            2.0f, 4.3f, 4.0f,  0.2f, 0.6f, 4.4f,
        }));

    LayerTestResult<float, 5> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<float, 5>(outputTensorInfo, std::vector<float>(
        {
            7.0f, 7.0f, 5.4f,  2.7f, 8.5f, 5.9f,
            4.0f, 4.8f, 4.7f,  3.3f, 2.6f, 8.8f,

            7.3f, 2.1f, 4.5f,  6.7f, 8.6f, 6.5f,
            0.5f, 4.0f, 6.2f,  5.3f, 6.5f, 1.1f,


            1.5f, 7.6f, 2.5f,  2.1f, 4.2f, 5.4f,
            3.0f, 5.6f, 5.3f,  5.4f, 2.8f, 4.2f,

            4.1f, 6.1f, 4.4f,  7.0f, 2.8f, 6.2f,
            2.4f, 8.7f, 4.7f,  0.8f, 5.3f, 5.6f,
        }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = workloadFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateAddition(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0][0], outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> AdditionBroadcastTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo1 = armnn::TensorInfo({1, 3, 2, 1}, ArmnnType);
    armnn::TensorInfo inputTensorInfo2 = armnn::TensorInfo({1, 1, 2, 3}, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 2, 3}, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo1.SetQuantizationScale(qScale);
        inputTensorInfo1.SetQuantizationOffset(qOffset);
        inputTensorInfo2.SetQuantizationScale(qScale);
        inputTensorInfo2.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset,
        {
            0.0f,
            1.0f,

            2.0f,
            3.0f,

            4.0f,
            5.0f,
        }));

    auto input2 = MakeTensor<T, 4>(inputTensorInfo2, QuantizedVector<T>(qScale, qOffset,
        {
            0.5f, 1.5f, 2.5f,
            3.5f, 4.5f, 5.5f,
        }));

    LayerTestResult<T,4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset,
        {
            0.5f, 1.5f, 2.5f,
            4.5f, 5.5f, 6.5f,

            2.5f, 3.5f, 4.5f,
            6.5f, 7.5f, 8.5f,

            4.5f, 5.5f, 6.5f,
            8.5f, 9.5f, 10.5f,
        }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = workloadFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateAddition(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> AdditionBroadcast1ElementTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo1 = armnn::TensorInfo({1, 3, 2, 3}, ArmnnType);
    armnn::TensorInfo inputTensorInfo2 = armnn::TensorInfo({1, 1, 1, 1}, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 2, 3}, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo1.SetQuantizationScale(qScale);
        inputTensorInfo1.SetQuantizationOffset(qOffset);
        inputTensorInfo2.SetQuantizationScale(qScale);
        inputTensorInfo2.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset,
        {
             0.0f,  1.0f,  2.0f,
             3.0f,  4.0f,  5.0f,
             6.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 11.0f,
            12.0f, 13.0f, 14.0f,
            15.0f, 16.0f, 17.0f,
        }));

    auto input2 = MakeTensor<T, 4>(inputTensorInfo2, QuantizedVector<T>(qScale, qOffset,
        {
            0.5f,
        }));

    LayerTestResult<T,4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset,
        {
             0.5f,  1.5f,  2.5f,
             3.5f,  4.5f,  5.5f,
             6.5f,  7.5f,  8.5f,
             9.5f, 10.5f, 11.5f,
            12.5f, 13.5f, 14.5f,
            15.5f, 16.5f, 17.5f,
        }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = workloadFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateAddition(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<float, 4> AdditionBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcastTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<uint8_t, 4> AdditionBroadcastUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcastTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 2.f, 0);
}

LayerTestResult<int16_t, 4> AdditionBroadcastInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcastTestImpl<armnn::DataType::QuantisedSymm16>(
        workloadFactory, memoryManager, 2.f, 0);
}

LayerTestResult<float, 4> AdditionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcast1ElementTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<uint8_t, 4> AdditionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcast1ElementTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.1333333f, 128);
}

LayerTestResult<int16_t, 4> AdditionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcast1ElementTestImpl<armnn::DataType::QuantisedSymm16>(
        workloadFactory, memoryManager, 0.1333333f, 0);
}

LayerTestResult<float,4> CompareAdditionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory)
{
    unsigned int batchSize = 4;
    unsigned int channels  = 1;
    unsigned int height    = 2;
    unsigned int width     = 3;

    armnn::TensorInfo inputTensorInfo1, inputTensorInfo2;
    armnn::TensorInfo outputTensorInfo;

    unsigned int shape[] = {batchSize, channels, height, width};

    inputTensorInfo1 = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    inputTensorInfo2 = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);

    auto input1 = MakeRandomTensor<float, 4>(inputTensorInfo1, 1232);
    auto input2 = MakeRandomTensor<float, 4>(inputTensorInfo2, 456);

    LayerTestResult<float,4> ret(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = workloadFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle1Ref = refWorkloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2Ref = refWorkloadFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refWorkloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    armnn::AdditionQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo1, inputHandle1Ref.get());
    SetWorkloadInput(refData, refInfo, 1, inputTensorInfo2, inputHandle2Ref.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateAddition(data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateAddition(refData, refInfo);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();
    inputHandle1Ref->Allocate();
    inputHandle2Ref->Allocate();
    outputHandleRef->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1Ref.get(), &input1[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle2Ref.get(), &input2[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();
    workloadRef->PostAllocationConfigure();
    workloadRef->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());
    CopyDataFromITensorHandle(&ret.outputExpected[0][0][0][0], outputHandleRef.get());

    return ret;
}

namespace {
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DivisionTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int shape0[4],
    const std::vector<T>& values0,
    float scale0,
    int32_t offset0,
    const unsigned int shape1[4],
    const std::vector<T> & values1,
    float scale1,
    int32_t offset1,
    const unsigned int outShape[4],
    const std::vector<T> & outValues,
    float outScale,
    int32_t outOffset)
{
    armnn::TensorInfo inputTensorInfo0(4, shape0, ArmnnType);
    armnn::TensorInfo inputTensorInfo1(4, shape1, ArmnnType);
    armnn::TensorInfo outputTensorInfo(4, outShape, ArmnnType);

    inputTensorInfo0.SetQuantizationScale(scale0);
    inputTensorInfo0.SetQuantizationOffset(offset0);

    inputTensorInfo1.SetQuantizationScale(scale1);
    inputTensorInfo1.SetQuantizationOffset(offset1);

    outputTensorInfo.SetQuantizationScale(outScale);
    outputTensorInfo.SetQuantizationOffset(outOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, values1);

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outValues);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DivisionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(data,  info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDivision(data, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}
} // anonymous namespace

LayerTestResult<float,4> DivisionByZeroTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 2;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0({
                                1.f,  1.f,  1.f,  1.f,  0.f, 0.f, 0.f, 0.f,
                               -1.f, -1.f, -1.f, -1.f,  5.f, 5.f, 5.f, 5.f });

    std::vector<float> input1({
                               0.f, 0.f, -0.f, -0.f,  0.f, 0.f, -0.f, -0.f,
                               0.f, 0.f, -0.f, -0.f,  5.f, 5.f,  5.f,  5.f });

    std::vector<float> output({
                               INFINITY, INFINITY, -INFINITY, -INFINITY,  NAN, NAN, -NAN, -NAN,
                               -INFINITY, -INFINITY, INFINITY, INFINITY,  1, 1, 1, 1 });

    return DivisionTestHelper<armnn::DataType::Float32>(workloadFactory,
                                                        memoryManager,
                                                        shape, input0, 1.0f, 0,
                                                        shape, input1, 1.0f, 0,
                                                        shape, output, 1.0f, 0);
}

LayerTestResult<float,4> DivisionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 2;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0({
                                      2,  2,  2,  2,    3,  3,  3,  3,
                                      4,  4,  4,  4,    5,  5,  5,  5 });

    std::vector<float> input1({
                                      1,  1,  1,  1,    2,  2,  2,  2,
                                      4,  4,  4,  4,    4,  4,  4,  4 });

    std::vector<float> output({
                                      2,  2,  2,  2,    1.5,  1.5,  1.5,  1.5,
                                      1, 1, 1, 1,  1.25, 1.25, 1.25, 1.25 });


    return DivisionTestHelper<armnn::DataType::Float32>(workloadFactory,
                                                        memoryManager,
                                                        shape, input0, 1.0f, 0,
                                                        shape, input1, 1.0f, 0,
                                                        shape, output, 1.0f, 0);
}

LayerTestResult<float, 4> DivisionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 2, 4, 6, 8, 10, 12, 14, 16});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 2 });

    std::vector<float> output({ 1, 2, 3, 4, 5, 6, 7, 8});


    return DivisionTestHelper<armnn::DataType::Float32>(workloadFactory,
                                                        memoryManager,
                                                        shape0, input0, 1.0f, 0,
                                                        shape1, input1, 1.0f, 0,
                                                        shape0, output, 1.0f, 0);
}

LayerTestResult<float, 4> DivisionBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    std::vector<float> input0({
                                      1,   4,       3,  8,      5, 12,
                                      7,   16,      9, 20,     11, 24,
                                      13,  28,     15, 32,     17, 36});

    unsigned int shape1[] = { 1, 1, 1, 2 };
    std::vector<float> input1({ 1, 2 });

    std::vector<float> output({
                                      1,   2,      3,  4,      5,  6,
                                      7,   8,      9, 10,     11, 12,
                                      13, 14,     15, 16,     17, 18});

    return DivisionTestHelper<armnn::DataType::Float32>(workloadFactory,
                                                        memoryManager,
                                                        shape0, input0, 1.0f, 0,
                                                        shape1, input1, 1.0f, 0,
                                                        shape0, output, 1.0f, 0);
}

LayerTestResult<uint8_t,4> DivisionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 2;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<uint8_t> input0({2,  2,  2,  2,    3,  3,  3,  3,
                                 4,  4,  4,  4,    5,  5,  5,  5 });

    std::vector<uint8_t> input1({1,  1,  1,  1,    2,  2,  2,  2,
                                 4,  4,  4,  4,    4,  4,  4,  4 });

    std::vector<uint8_t> output({8,  8,  8,  8,    6,  6,  6,  6,
                                 4,  4,  4,  4,    5,  5,  5,  5});


    return DivisionTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                memoryManager,
                                                                shape, input0, 1.0f,  0,
                                                                shape, input1, 1.0f,  0,
                                                                shape, output, 0.25f, 0);
}

LayerTestResult<uint8_t, 4> DivisionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<uint8_t> input0({ 2, 4, 6, 8, 10, 12, 14, 16});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<uint8_t> input1({ 2 });

    std::vector<uint8_t> output({ 1, 2, 3, 4, 5, 6, 7, 8});

    return DivisionTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                memoryManager,
                                                                shape0, input0, 1.0f, 0,
                                                                shape1, input1, 1.0f, 0,
                                                                shape0, output, 1.0f, 0);
}

LayerTestResult<uint8_t, 4> DivisionBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    std::vector<uint8_t> input0({1,   4,     3,  8,      5,  12,
                                 7,   16,    9,  20,     11, 24,
                                 13,  28,    15, 32,     17, 36});

    unsigned int shape1[] = { 1, 1, 1, 2 };
    std::vector<uint8_t> input1({ 1, 2 });

    std::vector<uint8_t> output({1,   2,      3,  4,      5,  6,
                                 7,   8,      9, 10,     11, 12,
                                 13, 14,     15, 16,     17, 18});

    return DivisionTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                memoryManager,
                                                                shape0, input0, 1.0f, 0,
                                                                shape1, input1, 1.0f, 0,
                                                                shape0, output, 1.0f, 0);
}

LayerTestResult<int16_t,4> DivisionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int16_t> input0({2,  2,  2,  2,    3,  3,  3,  3,
                                 4,  4,  4,  4,    5,  5,  5,  5 });

    std::vector<int16_t> input1({1,  1,  1,  1,    2,  2,  2,  2,
                                 4,  4,  4,  4,    4,  4,  4,  4 });

    std::vector<int16_t> output({8,  8,  8,  8,    6,  6,  6,  6,
                                 4,  4,  4,  4,    5,  5,  5,  5});


    return DivisionTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                memoryManager,
                                                                shape, input0, 1.0f,  0,
                                                                shape, input1, 1.0f,  0,
                                                                shape, output, 0.25f, 0);
}

LayerTestResult<int16_t, 4> DivisionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<int16_t> input0({ 2, 4, 6, 8, 10, 12, 14, 16});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<int16_t> input1({ 2 });

    std::vector<int16_t> output({ 1, 2, 3, 4, 5, 6, 7, 8});

    return DivisionTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                memoryManager,
                                                                shape0, input0, 1.0f, 0,
                                                                shape1, input1, 1.0f, 0,
                                                                shape0, output, 1.0f, 0);
}

LayerTestResult<int16_t, 4> DivisionBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    std::vector<int16_t> input0({1,   4,     3,  8,      5,  12,
                                 7,   16,    9,  20,     11, 24,
                                 13,  28,    15, 32,     17, 36});

    unsigned int shape1[] = { 1, 1, 1, 2 };
    std::vector<int16_t> input1({ 1, 2 });

    std::vector<int16_t> output({1,   2,      3,  4,      5,  6,
                                 7,   8,      9, 10,     11, 12,
                                 13, 14,     15, 16,     17, 18});

    return DivisionTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                memoryManager,
                                                                shape0, input0, 1.0f, 0,
                                                                shape1, input1, 1.0f, 0,
                                                                shape0, output, 1.0f, 0);
}

template<typename DescriptorType>
std::unique_ptr<armnn::IWorkload> CreateWorkload(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const DescriptorType& descriptor)
{
    return CreateWorkload(workloadFactory, info, descriptor);
};

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::MaximumQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::MaximumQueueDescriptor& descriptor)
{
    return workloadFactory.CreateMaximum(descriptor, info);
}

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::MinimumQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::MinimumQueueDescriptor& descriptor)
{
    return workloadFactory.CreateMinimum(descriptor, info);
}

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::EqualQueueDescriptor>(
        const armnn::IWorkloadFactory& workloadFactory,
        const armnn::WorkloadInfo& info,
        const armnn::EqualQueueDescriptor& descriptor)
{
    return workloadFactory.CreateEqual(descriptor, info);
}

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::GreaterQueueDescriptor>(
        const armnn::IWorkloadFactory& workloadFactory,
        const armnn::WorkloadInfo& info,
        const armnn::GreaterQueueDescriptor& descriptor)
{
    return workloadFactory.CreateGreater(descriptor, info);
}

namespace {

template <typename Descriptor,
          armnn::DataType ArmnnTypeInput,
          armnn::DataType ArmnnTypeOutput,
          typename TInput = armnn::ResolveType<ArmnnTypeInput>,
          typename TOutput = armnn::ResolveType<ArmnnTypeOutput>>
LayerTestResult<TOutput, 4> ElementwiseTestHelper(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
    const unsigned int shape0[4], std::vector<TInput> values0,
    const unsigned int shape1[4], std::vector<TInput> values1,
    const unsigned int outShape[4], std::vector<TOutput> outValues,
    float qScale = 0.0f, int qOffset = 0)
{
    const uint32_t dimensionCount = 4;
    armnn::TensorInfo inputTensorInfo0{dimensionCount, shape0, ArmnnTypeInput};
    armnn::TensorInfo inputTensorInfo1{dimensionCount, shape1, ArmnnTypeInput};
    armnn::TensorInfo outputTensorInfo{dimensionCount, outShape, ArmnnTypeOutput};

    auto input0 = MakeTensor<TInput, 4>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<TInput, 4>(inputTensorInfo1, values1);

    if (armnn::IsQuantizedType<TInput>())
    {
        inputTensorInfo0.SetQuantizationScale(qScale);
        inputTensorInfo0.SetQuantizationOffset(qOffset);

        inputTensorInfo1.SetQuantizationScale(qScale);
        inputTensorInfo1.SetQuantizationOffset(qOffset);

        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    LayerTestResult<TOutput,4> ret(outputTensorInfo);

    if(ArmnnTypeOutput == armnn::DataType::Boolean)
    {
        ret.compareBoolean = true;
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    Descriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    auto workload = CreateWorkload<Descriptor>(workloadFactory, info, data);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    ret.outputExpected = MakeTensor<TOutput, 4>(outputTensorInfo, outValues);
    return ret;
}

template <typename Descriptor, armnn::DataType ArmnnT, typename T = armnn::ResolveType<ArmnnT>>
LayerTestResult<T, 4> ElementwiseTestHelper(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
    const unsigned int shape0[4], std::vector<T> values0,
    const unsigned int shape1[4], std::vector<T> values1,
    const unsigned int outShape[4], std::vector<T> outValues,
    float qScale = 0.0f, int qOffset = 0)
{
    return ElementwiseTestHelper<Descriptor, ArmnnT, ArmnnT>
        (workloadFactory,
         memoryManager,
         shape0,
         values0,
         shape1,
         values1,
         outShape,
         outValues,
         qScale,
         qOffset);
}
}

LayerTestResult<uint8_t, 4> EqualSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 2;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0({ 1, 1, 1, 1,  5, 5, 5, 5,
                                3, 3, 3, 3,  4, 4, 4, 4 });

    std::vector<float> input1({ 1, 1, 1, 1,  3, 3, 3, 3,
                                5, 5, 5, 5,  4, 4, 4, 4 });

    std::vector<uint8_t> output({ 1, 1, 1, 1,  0, 0, 0, 0,
                                  0, 0, 0, 0,  1, 1, 1, 1 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 2, 3, 4, 5, 6, 7, 8});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 1 });

    std::vector<uint8_t> output({ 1, 0, 0, 0, 0, 0, 0, 0});

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1DVectorTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0({ 1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12 });

    std::vector<float> input1({ 1, 2, 3});

    std::vector<uint8_t> output({ 1, 1, 1, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> EqualUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    // See dequantized values to the right.
    std::vector<uint8_t> input0({ 1, 1, 1, 1, 6, 6, 6, 6,
                                  3, 3, 3, 3, 7, 7, 7, 7 });

    std::vector<uint8_t> input1({ 2, 2, 2, 2, 6, 6, 6, 6,
                                  3, 3, 3, 3, 5, 5, 5, 5 });

    std::vector<uint8_t> output({ 0, 0, 0, 0, 1, 1, 1, 1,
                                  1, 1, 1, 1, 0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        1.0f,
        0);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> input1({ 1 });

    std::vector<uint8_t> output({ 1, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1DVectorUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> input1({ 1, 1, 3});

    std::vector<uint8_t> output({ 1, 0, 1, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<uint8_t, 4> GreaterSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 2;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0({ 1, 1, 1, 1,  5, 5, 5, 5,
                                3, 3, 3, 3,  4, 4, 4, 4 });

    std::vector<float> input1({ 1, 1, 1, 1,  3, 3, 3, 3,
                                5, 5, 5, 5,  4, 4, 4, 4 });

    std::vector<uint8_t> output({ 0, 0, 0, 0,  1, 1, 1, 1,
                                  0, 0, 0, 0,  0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 2, 3, 4, 5, 6, 7, 8});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 1 });

    std::vector<uint8_t> output({ 0, 1, 1, 1, 1, 1, 1, 1});

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1DVectorTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0({ 1, 2.9f, 2.1f, 4, 5, 6,
                                7, 8, 9, 10, 11, 12 });

    std::vector<float> input1({ 1, 3, 2});

    std::vector<uint8_t> output({ 0, 0, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1 });

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> GreaterUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    // See dequantized values to the right.
    std::vector<uint8_t> input0({ 1, 1, 1, 1, 6, 6, 6, 6,
                                  3, 3, 3, 3, 5, 5, 5, 5 });

    std::vector<uint8_t> input1({ 2, 2, 2, 2, 6, 6, 6, 6,
                                  2, 2, 2, 2, 5, 5, 5, 5 });

    std::vector<uint8_t> output({ 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        1.0f,
        0);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> input1({ 1 });

    std::vector<uint8_t> output({ 0, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1 });

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1DVectorUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> input1({ 1, 1, 3});

    std::vector<uint8_t> output({ 0, 1, 0, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1 });

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<float, 4> MaximumSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                           const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 2;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0({ 1, 1, 1, 1,  5, 5, 5, 5,
                                3, 3, 3, 3,  4, 4, 4, 4 });

    std::vector<float> input1({ 2, 2, 2, 2,  3, 3, 3, 3,
                                4, 4, 4, 4,  5, 5, 5, 5 });

    std::vector<float> output({ 2, 2, 2, 2,  5, 5, 5, 5,
                                4, 4, 4, 4,  5, 5, 5, 5 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
}

LayerTestResult<float, 4> MaximumBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 2, 3, 4, 5, 6, 7, 8});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 2 });

    std::vector<float> output({ 2, 2, 3, 4, 5, 6, 7, 8});

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<float, 4> MaximumBroadcast1DVectorTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<float> input1({ 1, 2, 3});

    std::vector<float> output({ 1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> MaximumUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    // See dequantized values to the right.
    std::vector<uint8_t> input0({ 1, 1, 1, 1, 6, 6, 6, 6,
                                  3, 3, 3, 3, 4, 4, 4, 4 });

    std::vector<uint8_t> input1({ 2, 2, 2, 2, 3, 3, 3, 3,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    std::vector<uint8_t> output({ 2, 2, 2, 2, 6, 6, 6, 6,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        1.0f,
        0);
}

LayerTestResult<uint8_t, 4> MaximumBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0({ 1, 2, 3, 4,  5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> input1({2});

    std::vector<uint8_t> output({ 2, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<uint8_t, 4> MaximumBroadcast1DVectorUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> input1({ 1, 10, 3});

    std::vector<uint8_t> output({ 1, 10, 3, 4, 10, 6,
                                  7, 10, 9, 10, 11, 12 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<int16_t, 4> MaximumInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int16_t> input0({ 1, 1, 1, 1, 6, 6, 6, 6,
                                  3, 3, 3, 3, 4, 4, 4, 4 });

    std::vector<int16_t> input1({ 2, 2, 2, 2, 3, 3, 3, 3,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    std::vector<int16_t> output({ 2, 2, 2, 2, 6, 6, 6, 6,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        1.0f,
        0);
}

LayerTestResult<int16_t, 4> MaximumBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0({ 1, 2, 3, 4,  5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<int16_t> input1({2});

    std::vector<int16_t> output({ 2, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<int16_t, 4> MaximumBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int16_t> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<int16_t> input1({ 1, 10, 3});

    std::vector<int16_t> output({ 1, 10, 3, 4, 10, 6,
                                  7, 10, 9, 10, 11, 12 });

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<float, 4> MinimumBroadcast1ElementTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 2, 3, 4, 5, 6, 7, 8});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 2 });

    std::vector<float> output({ 1, 2, 2, 2, 2, 2, 2, 2});

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}


LayerTestResult<float, 4> MinimumBroadcast1ElementTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 6, 3, 2, 8, 9, 1, 10});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 5 });

    std::vector<float> output({ 1, 5, 3, 2, 5, 5, 1, 5});

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> MinimumBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0({ 1, 2, 3, 3, 2, 1,
                                  7, 1, 2, 3, 4, 5 });

    std::vector<uint8_t> input1({ 1, 2, 3});

    std::vector<uint8_t> output({ 1, 2, 3, 1, 2, 1,
                                  1, 1, 2, 1, 2, 3 });

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<int16_t, 4> MinimumInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int16_t> input0({ 1, 1, 1, 1, 6, 6, 6, 6,
                                  3, 3, 3, 3, 4, 4, 4, 4 });

    std::vector<int16_t> input1({ 2, 2, 2, 2, 3, 3, 3, 3,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    std::vector<int16_t> output({ 1, 1, 1, 1, 3, 3, 3, 3,
                                  3, 3, 3, 3, 4, 4, 4, 4 });

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        1.0f,
        0);
}

LayerTestResult<int16_t, 4> MinimumBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0({ 1, 2, 3, 4,  5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<int16_t> input1({2});

    std::vector<int16_t> output({ 1, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2 });

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

LayerTestResult<int16_t, 4> MinimumBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int16_t> input0({ 1, 2, 3, 4, 5, 6,
                                  7, 8, 9, 10, 11, 12 });

    std::vector<int16_t> input1({ 1, 10, 3});

    std::vector<int16_t> output({ 1, 2, 3, 1, 5, 3,
                                  1, 8, 3, 1, 10, 3 });

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        1.0f,
        0);
}

namespace {
template<std::size_t NumDims>
LayerTestResult<float,NumDims> MultiplicationTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int shape0[NumDims],
    const std::vector<float> & values0,
    const unsigned int shape1[NumDims],
    const std::vector<float> & values1,
    const unsigned int outShape[NumDims],
    const std::vector<float> & outValues)
{
    armnn::TensorInfo inputTensorInfo0{NumDims, shape0, armnn::DataType::Float32};
    armnn::TensorInfo inputTensorInfo1{NumDims, shape1, armnn::DataType::Float32};
    armnn::TensorInfo outputTensorInfo{NumDims, outShape, armnn::DataType::Float32};

    auto input0 = MakeTensor<float, NumDims>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<float, NumDims>(inputTensorInfo1, values1);

    LayerTestResult<float,NumDims> ret(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::MultiplicationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMultiplication(data, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), input0.origin());
    CopyDataToITensorHandle(inputHandle1.get(), input1.origin());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(ret.output.origin(), outputHandle.get());

    ret.outputExpected = MakeTensor<float, NumDims>(outputTensorInfo, outValues);
    return ret;
}
} // anonymous namespace


LayerTestResult<float,4> MultiplicationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 2;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0({
        1,  1,  1,  1,    2,  2,  2,  2,
        3,  3,  3,  3,    4,  4,  4,  4 });

    std::vector<float> input1({
        2,  2,  2,  2,    3,  3,  3,  3,
        4,  4,  4,  4,    5,  5,  5,  5 });

    std::vector<float> output({
        2,  2,  2,  2,    6,  6,  6,  6,
        12, 12, 12, 12,  20, 20, 20, 20 });

    return MultiplicationTestHelper<4>(workloadFactory,
                                       memoryManager,
                                       shape,
                                       input0,
                                       shape,
                                       input1,
                                       shape,
                                       output);
}

LayerTestResult<float,5> Multiplication5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int width = 3;
    const unsigned int height = 2;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 2;
    const unsigned int depth = 2;

    unsigned int shape[] = { depth, batchSize, channelCount, height, width };

    std::vector<float> input0({
        1.80f, 0.20f, 2.30f,  1.30f, 2.10f, 1.00f,
        2.60f, 0.60f, 2.10f,  2.30f, 2.30f, 2.00f,

        2.50f, 1.00f, 2.90f,  3.10f, 1.50f, 2.40f,
        2.80f, 1.10f, 1.00f,  3.20f, 1.00f, 2.30f,


        0.30f, 2.20f, 1.00f,  0.20f, 1.60f, 1.40f,
        0.80f, 3.20f, 0.10f,  0.10f, 3.10f, 2.10f,

        1.50f, 2.40f, 1.40f,  0.70f, 2.40f, 1.40f,
        1.60f, 1.20f, 1.90f,  0.80f, 0.00f, 0.10f,
    });

    std::vector<float> input1({
        0.70f, 1.00f, 2.90f,  2.20f, 3.10f, 2.80f,
        1.80f, 2.00f, 0.50f,  2.30f, 1.20f, 2.70f,

        2.40f, 0.20f, 3.20f,  1.60f, 0.20f, 2.50f,
        2.30f, 0.70f, 2.70f,  1.80f, 2.90f, 2.70f,


        3.20f, 3.20f, 0.70f,  1.90f, 2.70f, 2.50f,
        2.40f, 0.90f, 2.30f,  1.80f, 2.50f, 2.00f,

        1.60f, 2.20f, 1.60f,  2.00f, 0.30f, 3.20f,
        0.40f, 3.00f, 2.60f,  0.30f, 0.00f, 2.50f,
    });

    std::vector<float> output({
        1.26f, 0.20f, 6.67f,  2.86f, 6.51f, 2.80f,
        4.68f, 1.20f, 1.05f,  5.29f, 2.76f, 5.40f,

        6.00f, 0.20f, 9.28f,  4.96f, 0.30f, 6.00f,
        6.44f, 0.77f, 2.70f,  5.76f, 2.90f, 6.21f,


        0.96f, 7.04f, 0.70f,  0.38f, 4.32f, 3.50f,
        1.92f, 2.88f, 0.23f,  0.18f, 7.75f, 4.20f,

        2.40f, 5.28f, 2.24f,  1.40f, 0.72f, 4.48f,
        0.64f, 3.60f, 4.94f,  0.24f, 0.00f, 0.25f,
    });

    return MultiplicationTestHelper<5>(workloadFactory,
                                       memoryManager,
                                       shape,
                                       input0,
                                       shape,
                                       input1,
                                       shape,
                                       output);
}

LayerTestResult<float, 4> MultiplicationBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 2, 3, 4, 5, 6, 7, 8});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 2 });

    std::vector<float> output({ 2, 4, 6, 8, 10, 12, 14, 16});

    return MultiplicationTestHelper<4>(workloadFactory,
                                       memoryManager,
                                       shape0,
                                       input0,
                                       shape1,
                                       input1,
                                       shape0,
                                       output);
}

LayerTestResult<float, 4> MultiplicationBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    std::vector<float> input0({
        1,   2,      3,  4,      5,  6,
        7,   8,      9, 10,     11, 12,
        13, 14,     15, 16,     17, 18});

    unsigned int shape1[] = { 1, 1, 1, 2 };
    std::vector<float> input1({ 1, 2 });

    std::vector<float> output({
        1,   4,       3,  8,      5, 12,
        7,   16,      9, 20,     11, 24,
        13,  28,     15, 32,     17, 36});

    return MultiplicationTestHelper<4>(workloadFactory,
                                       memoryManager,
                                       shape0,
                                       input0,
                                       shape1,
                                       input1,
                                       shape0,
                                       output);
}

LayerTestResult<float,4> CompareMultiplicationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory)
{
    const unsigned int width = 16;
    const unsigned int height = 32;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 5;

    armnn::TensorInfo inputTensorInfo0;
    armnn::TensorInfo inputTensorInfo1;
    armnn::TensorInfo outputTensorInfo;

    constexpr unsigned int shape[] = { batchSize, channelCount, height, width };

    inputTensorInfo0 = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    inputTensorInfo1 = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);

    LayerTestResult<float,4> comparisonResult(outputTensorInfo);

    auto input0 = MakeRandomTensor<float, 4>(inputTensorInfo0, 803506992);
    auto input1 = MakeRandomTensor<float, 4>(inputTensorInfo1, 54902257);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0Ref = refWorkloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1Ref = refWorkloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refWorkloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::MultiplicationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    armnn::MultiplicationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo0, inputHandle0Ref.get());
    SetWorkloadInput(refData, refInfo, 1, inputTensorInfo1, inputHandle1Ref.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMultiplication(data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateMultiplication(refData, refInfo);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();
    inputHandle0Ref->Allocate();
    inputHandle1Ref->Allocate();
    outputHandleRef->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle0Ref.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1Ref.get(), &input1[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();
    workloadRef->PostAllocationConfigure();
    workloadRef->Execute();
    CopyDataFromITensorHandle(&comparisonResult.output[0][0][0][0], outputHandle.get());
    CopyDataFromITensorHandle(&comparisonResult.outputExpected[0][0][0][0], outputHandleRef.get());

    return comparisonResult;
}

LayerTestResult<float,4> CompareBatchNormTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory)
{
    const unsigned int width     = 2;
    const unsigned int height    = 3;
    const unsigned int channels  = 5;
    const unsigned int batchSize = 3;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;
    armnn::TensorInfo tensorInfo;

    constexpr unsigned int shape[]       = {batchSize, channels, height, width};
    constexpr unsigned int tensorShape[] = {channels};

    inputTensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::Float32);
    tensorInfo = armnn::TensorInfo(1, tensorShape, armnn::DataType::Float32);

    auto input = MakeRandomTensor<float, 4>(inputTensorInfo, 21312);

    auto mean     = MakeRandomTensor<float, 1>(tensorInfo, 123);
    auto variance = MakeRandomTensor<float, 1>(tensorInfo, 234, 0.0f);
    auto beta     = MakeRandomTensor<float, 1>(tensorInfo, 123);
    auto gamma    = MakeRandomTensor<float, 1>(tensorInfo, 345);

    LayerTestResult<float,4> ret(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandleRef  = refWorkloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refWorkloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::BatchNormalizationQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle meanTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle varianceTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle betaTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle gammaTensor(tensorInfo);

    AllocateAndCopyDataToITensorHandle(&meanTensor, &mean[0]);
    AllocateAndCopyDataToITensorHandle(&varianceTensor, &variance[0]);
    AllocateAndCopyDataToITensorHandle(&betaTensor, &beta[0]);
    AllocateAndCopyDataToITensorHandle(&gammaTensor, &gamma[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Mean             = &meanTensor;
    data.m_Variance         = &varianceTensor;
    data.m_Beta             = &betaTensor;
    data.m_Gamma            = &gammaTensor;
    data.m_Parameters.m_Eps = 0.01f;

    armnn::BatchNormalizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateBatchNormalization(data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateBatchNormalization(refData, refInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();
    inputHandleRef->Allocate();
    outputHandleRef->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);
    CopyDataToITensorHandle(inputHandleRef.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();
    workloadRef->PostAllocationConfigure();
    workloadRef->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());
    CopyDataFromITensorHandle(&ret.outputExpected[0][0][0][0], outputHandleRef.get());

    return ret;
}

template<typename T>
void PermuteTensorData(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::PermutationVector& mappings,
        armnn::TensorInfo & inputTensorInfo,
        const T * inputData,
        std::vector<T>& outputData)
{
    BOOST_ASSERT_MSG(inputData != nullptr, "inputData must not be null");
    if (inputData == nullptr)
    {
        // Nullptr is an error in the test. By returning without doing the concatenation
        // I expect the caller to fail the test. It still makes sense to report this as
        // an assert for Debug builds.
        return;
    }

    armnn::TensorInfo outputTensorInfo = armnnUtils::Permuted(inputTensorInfo, mappings);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PermuteQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = armnn::PermuteDescriptor{mappings};
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePermute(queueDescriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData);

    workload->PostAllocationConfigure();
    workload->Execute();

    outputData.resize(outputTensorInfo.GetNumElements());
    CopyDataFromITensorHandle(&outputData[0], outputHandle.get());
    inputTensorInfo = outputTensorInfo;
}

armnn::OriginsDescriptor CreateDescriptorForConcatenation(
        const std::vector<armnn::TensorInfo> & inputTensorInfos,
        unsigned int concatDim)
{
    std::vector<armnn::TensorShape> shapes;
    shapes.reserve(inputTensorInfos.size());
    for (const armnn::TensorInfo& it: inputTensorInfos)
    {
        shapes.push_back(it.GetShape());
    }

    return armnn::CreateDescriptorForConcatenation(shapes.begin(),
                                                   shapes.end(),
                                                   concatDim);
}

//
// Concatenation is only supported for N and C dimensions for NCHW and the inner most dimension
// In case of <4 dimensions we need to make sure that the concat dimensions are at least
// the 3rd slowest iterating one or the inner most dimension.
//

bool NeedPermuteForConcat(
        const std::vector<armnn::TensorInfo> & inputTensorInfos,
        unsigned int concatDim)
{
    // See note above. Additionally we expect the input shapes to have the
    // same number of dimensions.
    unsigned int nDimensions = 0;

    // Determine the number of dimensions as well as sanity check them
    // agains test implementation issues.
    for (auto && tensorInfo : inputTensorInfos)
    {
        if (!nDimensions)
        {
            nDimensions = tensorInfo.GetShape().GetNumDimensions();
        }
        else
        {
            BOOST_ASSERT_MSG(nDimensions == tensorInfo.GetShape().GetNumDimensions(),
                "Input shapes must have the same number of dimensions");
        }
    }

    return (nDimensions < 3 || (nDimensions == 3 && (nDimensions-concatDim) < 3 && (nDimensions-concatDim) != 1));
}

armnn::TensorShape ExpandTensorShapeTo3dForPermute(const armnn::TensorShape & inputShape)
{
    unsigned int numDims = inputShape.GetNumDimensions();
    if (numDims >= 3)
    {
        // Nothing to do if the inputShape has at least 3 dimensions.
        return inputShape;
    }

    std::vector<unsigned int> newDims(size_t(3), 1u);
    unsigned int expandedBy = 3 - numDims;
    for (unsigned int i=0; i<numDims; ++i)
    {
        newDims[expandedBy+i] = inputShape[i];
    }
    return armnn::TensorShape(3u, &newDims[0]);
}

void Generate3dPermuteVectorForConcat(
        unsigned int numDimensions,
        unsigned int & concatDim,
        std::pair<armnn::PermutationVector, armnn::PermutationVector> & permutations)
{
    BOOST_ASSERT_MSG(numDimensions <= 3,
       "Only dimensions 1,2 and 3 are supported by this helper");
    unsigned int expandedBy = 3 - numDimensions;
    unsigned int expandedConcatAxis = concatDim + expandedBy;

    if (expandedConcatAxis == 2)
    {
        concatDim = 0;
        armnn::PermutationVector forwardPermutation({1, 2, 0});
        armnn::PermutationVector reversePermutation({2, 0, 1});
        permutations = std::make_pair(forwardPermutation, reversePermutation);
    }
    else if (expandedConcatAxis == 1)
    {
        concatDim = 0;
        armnn::PermutationVector forwardPermutation({2, 0, 1});
        armnn::PermutationVector reversePermutation({1, 2, 0});
        permutations = std::make_pair(forwardPermutation, reversePermutation);
    }
    else
    {
        BOOST_ASSERT(expandedConcatAxis == 0);
        concatDim = 0;
    }
}

//
// Permute the input tensors so we can do a supported concatenation.
// Also treat lower than 3d tensors as 3d by adding dummy 1 dimensions
// at the front. Finally this function tells what the output shape
// of the permuted concatenated tensor is going to be.
//
template <typename T>
void PermuteInputsForConcat(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        std::vector<armnn::TensorInfo> & inputTensorInfos,
        std::vector<T *> & inputData,
        std::vector<std::vector<T>> & inputDataStorage,
        armnn::PermutationVector & permuteVector,
        unsigned int & concatDim,
        armnn::TensorInfo & outputTensorInfo)
{
    BOOST_ASSERT_MSG(inputTensorInfos.size() > 1,
        "Expecting more than one tensor to be concatenated here");

    unsigned int numDims = 0;
    unsigned int nthInput = 0;
    const armnn::PermutationVector identity({0, 1, 2});

    std::pair<armnn::PermutationVector, armnn::PermutationVector> permutations =
        std::make_pair(identity, identity);

    inputDataStorage.resize(inputData.size());

    for (auto && tensorInfo : inputTensorInfos)
    {
        if (numDims == 0)
        {
            numDims = tensorInfo.GetShape().GetNumDimensions();
            Generate3dPermuteVectorForConcat(numDims, concatDim, permutations);

            // Store the reverese permutation.
            permuteVector = permutations.second;
            BOOST_ASSERT_MSG(!permuteVector.IsEqual(identity),
                "Test logic error, we don't need permutation, so we shouldn't arrive here");
        }
        else
        {
            BOOST_ASSERT_MSG(numDims == tensorInfo.GetShape().GetNumDimensions(),
                "All inputs must have the same number of dimensions");
        }

        armnn::TensorInfo newTensorInfo = tensorInfo;
        newTensorInfo.SetShape(ExpandTensorShapeTo3dForPermute(tensorInfo.GetShape()));

        PermuteTensorData<T>(workloadFactory,
                             memoryManager,
                             permutations.first,
                             newTensorInfo,
                             inputData[nthInput],
                             inputDataStorage[nthInput]);

        inputData[nthInput] = inputDataStorage[nthInput].data();
        inputTensorInfos[nthInput] = newTensorInfo;

        ++nthInput;
    }

    outputTensorInfo.SetShape(
        armnnUtils::Permuted(
            ExpandTensorShapeTo3dForPermute(outputTensorInfo.GetShape()),
            permutations.first));
}


//
// This is the pair of PermuteInputsForConcat(...) which permutes back
// the output of the concatenation so we can check it against an expected
// output.
//
template <typename T>
void PermuteOutputForConcat(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::TensorInfo & tensorInfo,
        const armnn::PermutationVector & permuteVector,
        std::unique_ptr<armnn::ITensorHandle> && inputDataHandle,
        T * data)
{
    BOOST_ASSERT_MSG(data != nullptr, "data must not be null");
    if (data == nullptr)
    {
        // Nullptr is an error in the test. By returning without doing the permutation
        // I expect the caller to fail the test. It still makes sense to report this as
        // an assert for Debug builds.
        return;
    }

    armnn::TensorInfo resultTensorInfo = tensorInfo;
    std::vector<T> inputData(tensorInfo.GetNumElements());
    std::vector<T> outputData;

    CopyDataFromITensorHandle(&inputData[0], inputDataHandle.get());

    PermuteTensorData<T>(workloadFactory,
                         memoryManager,
                         permuteVector,
                         resultTensorInfo,
                         &inputData[0],
                         outputData);

    ::memcpy(data, &outputData[0], sizeof(T)*outputData.size());
}

template <typename T>
void Concatenate(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    std::initializer_list<const armnn::TensorInfo> inputTensorInfosOrig,
    std::initializer_list<T *> inputsOrig,
    const armnn::TensorInfo& outputTensorInfoOrig,
    T * output,
    unsigned int concatDim,
    bool useSubtensor)
{
    BOOST_ASSERT_MSG(output != nullptr, "output must not be null");
    if (output == nullptr)
    {
        // Nullptr is an error in the test. By returning without doing the permutation
        // I expect the caller to fail the test. It still makes sense to report this as
        // an assert for Debug builds.
        return;
    }

    // Saves a copy of the parameters which we might need to change.
    std::vector<armnn::TensorInfo> inputTensorInfos(inputTensorInfosOrig.begin(), inputTensorInfosOrig.end());
    std::vector<T *> inputs            = inputsOrig;
    armnn::TensorInfo outputTensorInfo = outputTensorInfoOrig;

    armnn::PermutationVector permuteVector{0, 1, 2};

    // Holds and automatically releases memory for the reshaped input data.
    std::vector<std::vector<T>> tmpInputDataStorage;

    const size_t inputCount = inputTensorInfos.size();

    bool needPermuteForConcat = NeedPermuteForConcat(inputTensorInfos, concatDim);

    if (needPermuteForConcat)
    {
        //
        // We need to permute the inputs, because concatenation along
        // the requested axis is not supported.
        //
        PermuteInputsForConcat<T>(workloadFactory,
                                  memoryManager,
                                  inputTensorInfos,
                                  inputs,
                                  tmpInputDataStorage,
                                  permuteVector,
                                  concatDim,
                                  outputTensorInfo);
    }

    armnn::WorkloadInfo workloadInfo;

    std::vector<std::unique_ptr<armnn::ITensorHandle>> inputHandles;
    inputHandles.reserve(inputCount);

    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ConcatQueueDescriptor queueDescriptor;
    armnn::OriginsDescriptor viewsDescriptor = CreateDescriptorForConcatenation(inputTensorInfos, concatDim);
    queueDescriptor.m_Parameters = viewsDescriptor;

    if (useSubtensor)
    {
        queueDescriptor.m_ViewOrigins.reserve(viewsDescriptor.GetNumViews());
        for (unsigned int i = 0; i < viewsDescriptor.GetNumViews(); ++i)
        {
            queueDescriptor.m_ViewOrigins.emplace_back(std::vector<unsigned int>(viewsDescriptor.GetViewOrigin(i),
                viewsDescriptor.GetViewOrigin(i) + viewsDescriptor.GetNumDimensions()));
        }

        outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

        const bool subTensorsSupported = workloadFactory.SupportsSubTensors();
        for (unsigned int i = 0; i < inputCount; ++i)
        {
            const armnn::TensorInfo& inputTensorInfo = inputTensorInfos[i];
            std::unique_ptr<armnn::ITensorHandle> inputHandle =
                subTensorsSupported ?
                    workloadFactory.CreateSubTensorHandle(*outputHandle,
                                                          inputTensorInfo.GetShape(),
                                                          queueDescriptor.m_ViewOrigins[i].m_Origin.data()) :
                    workloadFactory.CreateTensorHandle(inputTensorInfo);

            inputHandles.emplace_back(std::move(inputHandle));
        }

    }
    else
    {
        for (unsigned int i = 0; i < inputCount; ++i)
        {
            std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfos[i]);
            inputHandles.emplace_back(std::move(inputHandle));
        }
    }

    for (unsigned int i = 0; i < inputCount; ++i)
    {
        AddInputToWorkload(queueDescriptor, workloadInfo, inputTensorInfos[i], inputHandles[i].get());
    }

    AddOutputToWorkload(queueDescriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConcat(queueDescriptor, workloadInfo);

    for (auto& inputHandle : inputHandles)
    {
        inputHandle->Allocate();
    }

    outputHandle->Allocate();

    unsigned int nextInputId = 0;
    for (auto& inputHandle : inputHandles)
    {
        CopyDataToITensorHandle(inputHandle.get(), inputs[nextInputId]);
        ++nextInputId;
    }

    workload->PostAllocationConfigure();
    workload->Execute();

    if (needPermuteForConcat)
    {
        PermuteOutputForConcat<T>(workloadFactory,
                                  memoryManager,
                                  outputTensorInfo,
                                  permuteVector,
                                  std::move(outputHandle),
                                  output);
    }
    else
    {
        CopyDataFromITensorHandle(output, outputHandle.get());
    }
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> Concatenation1dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 3 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 1>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, { 1.0f, 2.0f, 3.0f }));
    auto input1 = MakeTensor<T, 1>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, { 4.0f, 5.0f, 6.0f }));
    auto input2 = MakeTensor<T, 1>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, { 7.0f, 8.0f, 9.0f }));

    armnn::TensorInfo outputTensorInfo({ 9 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 1> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { inputTensorInfo, inputTensorInfo, inputTensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   0,
                   true);

    result.output = MakeTensor<T, 1>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 1>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f
    }));

    return result;
}

LayerTestResult<float, 1> Concatenation1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation1dTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Concatenation2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& outputTensorInfo,
    unsigned int dimension,
    const float qScale,
    const int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 2, 3 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 2>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f,
    }));

    auto input1 = MakeTensor<T, 2>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        4.0f, 5.0f, 6.0f,

        // Batch 1
        13.0f, 14.0f, 15.0f,
    }));

    auto input2 = MakeTensor<T, 2>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        7.0f, 8.0f, 9.0f,

        // Batch 1
        16.0f, 17.0f, 18.0f,
    }));

    LayerTestResult<T, 2> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { inputTensorInfo, inputTensorInfo, inputTensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   dimension,
                   true);

    result.output = MakeTensor<T, 2>(outputTensorInfo, output);
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Concatenation2dDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 6, 3 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 2> result = Concatenation2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 0, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f,

        // Batch 2
        4.0f, 5.0f, 6.0f,

        // Batch 3
        13.0f, 14.0f, 15.0f,

        // Batch 4
        7.0f, 8.0f, 9.0f,

        // Batch 5
        16.0f, 17.0f, 18.0f,
    }));

    return result;
}

LayerTestResult<float, 2> Concatenation2dDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim0TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Concatenation2dDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 2, 9 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 2> result = Concatenation2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 1, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
    }));

    return result;
}

LayerTestResult<float, 2> Concatenation2dDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim1TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Concatenation2dDim0DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3 }, ArmnnType, qScale, qOffset);
    auto input0 = MakeTensor<T, 2>(input0TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f,
    }));

    armnn::TensorInfo input1TensorInfo({ 3, 3 }, ArmnnType, qScale, qOffset);
    auto input1 = MakeTensor<T, 2>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        4.0f, 5.0f, 6.0f,

        // Batch 1
        13.0f, 14.0f, 15.0f,

        // Batch 0
        7.0f, 8.0f, 9.0f,
    }));

    armnn::TensorInfo input2TensorInfo({ 1, 3 }, ArmnnType, qScale, qOffset);
    auto input2 = MakeTensor<T, 2>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 1
        16.0f, 17.0f, 18.0f,
    }));

    armnn::TensorInfo outputTensorInfo({ 6, 3 }, ArmnnType, qScale, qOffset);
    LayerTestResult<T, 2> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { input0TensorInfo, input1TensorInfo, input2TensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   0,
                   true);

    result.output = MakeTensor<T, 2>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f,

        // Batch 2
        4.0f, 5.0f, 6.0f,

        // Batch 3
        13.0f, 14.0f, 15.0f,

        // Batch 4
        7.0f, 8.0f, 9.0f,

        // Batch 5
        16.0f, 17.0f, 18.0f,
    }));

    return result;
}

LayerTestResult<float, 2> Concatenation2dDim0DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim0DiffInputDimsTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Concatenation2dDim1DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3 }, ArmnnType, qScale, qOffset);
    auto input0 = MakeTensor<T, 2>(input0TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f,
    }));

    armnn::TensorInfo input1TensorInfo({ 2, 5 }, ArmnnType, qScale, qOffset);
    auto input1 = MakeTensor<T, 2>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,

        // Batch 1
        13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
    }));

    armnn::TensorInfo input2TensorInfo({ 2, 1 }, ArmnnType, qScale, qOffset);
    auto input2 = MakeTensor<T, 2>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        9.0f,

        // Batch 1
        18.0f
    }));

    armnn::TensorInfo outputTensorInfo({ 2, 9 }, ArmnnType, qScale, qOffset);
    LayerTestResult<T, 2> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { input0TensorInfo, input1TensorInfo, input2TensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   1,
                   true);

    result.output = MakeTensor<T, 2>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
    }));

    return result;
}

LayerTestResult<float, 2> Concatenation2dDim1DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim1DiffInputDimsTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Concatenation3dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& outputTensorInfo,
    unsigned int dimension,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 2, 3, 2 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 3>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f
    }));

    auto input1 = MakeTensor<T, 3>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        7.0f, 8.0f,

        // Batch 0, Channel 1
        9.0f, 10.0f,

        // Batch 0, Channel 2
        11.0f, 12.0f,

        // Batch 1, Channel 0
        25.0f, 26.0f,

        // Batch 1, Channel 1
        27.0f, 28.0f,

        // Batch 1, Channel 2
        29.0f, 30.0f
    }));

    auto input2 = MakeTensor<T, 3>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        13.0f, 14.0f,

        // Batch 0, Channel 1
        15.0f, 16.0f,

        // Batch 0, Channel 2
        17.0f, 18.0f,

        // Batch 1, Channel 0
        31.0f, 32.0f,

        // Batch 1, Channel 1
        33.0f, 34.0f,

        // Batch 1, Channel 2
        35.0f, 36.0f
    }));

    LayerTestResult<T, 3> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { inputTensorInfo, inputTensorInfo, inputTensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   dimension,
                   useSubtensor);

    result.output = MakeTensor<T, 3>(outputTensorInfo, output);
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Concatenation3dDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 6, 3, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 3> result = Concatenation3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 0, true, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f,

        // Batch 2, Channel 0
        7.0f, 8.0f,

        // Batch 2, Channel 1
        9.0f, 10.0f,

        // Batch 2, Channel 2
        11.0f, 12.0f,

        // Batch 3, Channel 0
        25.0f, 26.0f,

        // Batch 3, Channel 1
        27.0f, 28.0f,

        // Batch 3, Channel 2
        29.0f, 30.0f,

        // Batch 4, Channel 0
        13.0f, 14.0f,

        // Batch 4, Channel 1
        15.0f, 16.0f,

        // Batch 4, Channel 2
        17.0f, 18.0f,

        // Batch 5, Channel 0
        31.0f, 32.0f,

        // Batch 5, Channel 1
        33.0f, 34.0f,

        // Batch 5, Channel 2
        35.0f, 36.0f
    }));

    return result;
}

LayerTestResult<float, 3> Concatenation3dDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim0TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Concatenation3dDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 2, 9, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 3> result = Concatenation3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 1, true, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f,

        // Batch 0, Channel 3
        7.0f, 8.0f,

        // Batch 0, Channel 4
        9.0f, 10.0f,

        // Batch 0, Channel 5
        11.0f, 12.0f,

        // Batch 0, Channel 6
        13.0f, 14.0f,

        // Batch 0, Channel 7
        15.0f, 16.0f,

        // Batch 0, Channel 8
        17.0f, 18.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f,

        // Batch 1, Channel 3
        25.0f, 26.0f,

        // Batch 1, Channel 4
        27.0f, 28.0f,

        // Batch 1, Channel 5
        29.0f, 30.0f,

        // Batch 1, Channel 6
        31.0f, 32.0f,

        // Batch 1, Channel 7
        33.0f, 34.0f,

        // Batch 1, Channel 8
        35.0f, 36.0f
    }));

    return result;
}

LayerTestResult<float, 3> Concatenation3dDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim1TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Concatenation3dDim2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 2, 3, 6 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 3> result = Concatenation3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 2, useSubtensor, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f, 7.0f, 8.0f, 13.0f, 14.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f, 9.0f, 10.0f, 15.0f, 16.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f, 11.0f, 12.0f, 17.0f, 18.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f, 25.0f, 26.0f, 31.0f, 32.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f, 27.0f, 28.0f, 33.0f, 34.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f, 29.0f, 30.0f, 35.0f, 36.0f,
    }));

    return result;
}

LayerTestResult<float, 3> Concatenation3dDim2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation3dDim2TestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, useSubtensor, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Concatenation3dDim0DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3, 2 }, ArmnnType);
    auto input0 = MakeTensor<T, 3>(input0TensorInfo, QuantizedVector<T>(qScale, qOffset, {
            // Batch 0, Channel 0
            1.0f, 2.0f,

            // Batch 0, Channel 1
            3.0f, 4.0f,

            // Batch 0, Channel 2
            5.0f, 6.0f,

            // Batch 1, Channel 0
            19.0f, 20.0f,

            // Batch 1, Channel 1
            21.0f, 22.0f,

            // Batch 1, Channel 2
            23.0f, 24.0f
    }));

    armnn::TensorInfo input1TensorInfo({ 1, 3, 2 }, ArmnnType);
    auto input1 = MakeTensor<T, 3>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
            // Batch 0, Channel 0
            7.0f, 8.0f,

            // Batch 0, Channel 1
            9.0f, 10.0f,

            // Batch 0, Channel 2
            11.0f, 12.0f,
    }));

    armnn::TensorInfo input2TensorInfo({ 3, 3, 2 }, ArmnnType);
    auto input2 = MakeTensor<T, 3>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
            // Batch 0, Channel 0
            25.0f, 26.0f,

            // Batch 0, Channel 1
            27.0f, 28.0f,

            // Batch 0, Channel 2
            29.0f, 30.0f,

            // Batch 1, Channel 0
            13.0f, 14.0f,

            // Batch 1, Channel 1
            15.0f, 16.0f,

            // Batch 1, Channel 2
            17.0f, 18.0f,

            // Batch 2, Channel 0
            31.0f, 32.0f,

            // Batch 2, Channel 1
            33.0f, 34.0f,

            // Batch 2, Channel 2
            35.0f, 36.0f
    }));

    armnn::TensorInfo outputTensorInfo({ 6, 3, 2 }, ArmnnType);
    LayerTestResult<T, 3> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { input0TensorInfo, input1TensorInfo, input2TensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   0,
                   true);

    result.output = MakeTensor<T, 3>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f,

        // Batch 2, Channel 0
        7.0f, 8.0f,

        // Batch 2, Channel 1
        9.0f, 10.0f,

        // Batch 2, Channel 2
        11.0f, 12.0f,

        // Batch 3, Channel 0
        25.0f, 26.0f,

        // Batch 3, Channel 1
        27.0f, 28.0f,

        // Batch 3, Channel 2
        29.0f, 30.0f,

        // Batch 4, Channel 0
        13.0f, 14.0f,

        // Batch 4, Channel 1
        15.0f, 16.0f,

        // Batch 4, Channel 2
        17.0f, 18.0f,

        // Batch 5, Channel 0
        31.0f, 32.0f,

        // Batch 5, Channel 1
        33.0f, 34.0f,

        // Batch 5, Channel 2
        35.0f, 36.0f
    }));

    return result;
}

LayerTestResult<float, 3> Concatenation3dDim0DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim0DiffInputDimsTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Concatenation3dDim1DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3, 2 }, ArmnnType, qScale, qOffset);
    auto input0 = MakeTensor<T, 3>(input0TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f
    }));

    armnn::TensorInfo input1TensorInfo({ 2, 4, 2 }, ArmnnType, qScale, qOffset);
    auto input1 = MakeTensor<T, 3>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        7.0f, 8.0f,

        // Batch 0, Channel 1
        9.0f, 10.0f,

        // Batch 0, Channel 2
        11.0f, 12.0f,

        // Batch 0, Channel 3
        25.0f, 26.0f,

        // Batch 1, Channel 0
        27.0f, 28.0f,

        // Batch 1, Channel 1
        29.0f, 30.0f,

        // Batch 1, Channel 2
        13.0f, 14.0f,

        // Batch 1, Channel 3
        15.0f, 16.0f,
    }));

    armnn::TensorInfo input2TensorInfo({ 2, 1, 2 }, ArmnnType, qScale, qOffset);
    auto input2 = MakeTensor<T, 3>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        17.0f, 18.0f,

        // Batch 1, Channel 0
        31.0f, 32.0f,
    }));

    armnn::TensorInfo outputTensorInfo({ 2, 8, 2 }, ArmnnType, qScale, qOffset);
    LayerTestResult<T, 3> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { input0TensorInfo, input1TensorInfo, input2TensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   1,
                   true);

    result.output = MakeTensor<T, 3>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f,

        // Batch 0, Channel 3
        7.0f, 8.0f,

        // Batch 0, Channel 4
        9.0f, 10.0f,

        // Batch 0, Channel 5
        11.0f, 12.0f,

        // Batch 0, Channel 6
        25.0f, 26.0f,

        // Batch 0, Channel 7
        17.0f, 18.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f,

        // Batch 1, Channel 3
        27.0f, 28.0f,

        // Batch 1, Channel 4
        29.0f, 30.0f,

        // Batch 1, Channel 5
        13.0f, 14.0f,

        // Batch 1, Channel 6
        15.0f, 16.0f,

        // Batch 1, Channel 7
        31.0f, 32.0f,
    }));

    return result;
}

LayerTestResult<float, 3> Concatenation3dDim1DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim1DiffInputDimsTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Concatenation3dDim2DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3, 2 }, ArmnnType, qScale, qOffset);
    auto input0 = MakeTensor<T, 3>(input0TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f
    }));

    armnn::TensorInfo input1TensorInfo({ 2, 3, 1 }, ArmnnType, qScale, qOffset);
    auto input1 = MakeTensor<T, 3>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        7.0f,

        // Batch 0, Channel 1
        9.0f,

        // Batch 0, Channel 2
        11.0f,

        // Batch 1, Channel 0
        25.0f,

        // Batch 1, Channel 1
        27.0f,

        // Batch 1, Channel 2
        29.0f
    }));

    armnn::TensorInfo input2TensorInfo({ 2, 3, 3 }, ArmnnType, qScale, qOffset);
    auto input2 = MakeTensor<T, 3>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        13.0f, 14.0f, 50.0f,

        // Batch 0, Channel 1
        15.0f, 16.0f, 51.0f,

        // Batch 0, Channel 2
        17.0f, 18.0f, 52.0f,

        // Batch 1, Channel 0
        31.0f, 32.0f, 53.0f,

        // Batch 1, Channel 1
        33.0f, 34.0f, 54.0f,

        // Batch 1, Channel 2
        35.0f, 36.0f, 55.0f,
    }));

    armnn::TensorInfo outputTensorInfo({ 2, 3, 6 }, ArmnnType, qScale, qOffset);
    LayerTestResult<T, 3> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory, memoryManager,
                   { input0TensorInfo, input1TensorInfo, input2TensorInfo },
                   { input0.data(), input1.data(), input2.data() },
                   outputTensorInfo,
                   output.data(),
                   2,
                   useSubtensor);

    result.output = MakeTensor<T, 3>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        1.0f, 2.0f, 7.0f, 13.0f, 14.0f, 50.0f,

        // Batch 0, Channel 1
        3.0f, 4.0f, 9.0f, 15.0f, 16.0f, 51.0f,

        // Batch 0, Channel 2
        5.0f, 6.0f, 11.0f, 17.0f, 18.0f, 52.0f,

        // Batch 1, Channel 0
        19.0f, 20.0f, 25.0f, 31.0f, 32.0f, 53.0f,

        // Batch 1, Channel 1
        21.0f, 22.0f, 27.0f, 33.0f, 34.0f, 54.0f,

        // Batch 1, Channel 2
        23.0f, 24.0f, 29.0f, 35.0f, 36.0f, 55.0f,
    }));

    return result;
}

LayerTestResult<float, 3> Concatenation3dDim2DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation3dDim2DiffInputDimsTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, useSubtensor, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& outputTensorInfo,
    unsigned int dimension,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 1, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    auto input1 = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,
        19.0f, 20.0f,
        21.0f, 22.0f
    }));

    auto input2 = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        21.0f, 22.0f,
        23.0f, 24.0f,
        25.0f, 26.0f,
        27.0f, 28.0f,
        29.0f, 30.0f,
        31.0f, 32.0f
    }));

    LayerTestResult<T, 4> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());

    Concatenate<T>(workloadFactory,
                   memoryManager,
                   {inputTensorInfo, inputTensorInfo, inputTensorInfo},
                   {input0.data(), input1.data(), input2.data()},
                   outputTensorInfo,
                   output.data(),
                   dimension,
                   useSubtensor);

    result.output = MakeTensor<T, 4>(outputTensorInfo, output);
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 3, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 0, true, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f,

        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,
        19.0f, 20.0f,
        21.0f, 22.0f,

        21.0f, 22.0f,
        23.0f, 24.0f,
        25.0f, 26.0f,
        27.0f, 28.0f,
        29.0f, 30.0f,
        31.0f, 32.0f
    }));
    return result;
}

LayerTestResult<float, 4> Concatenation4dDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim0TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 1, 9, 2, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 1, true, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f,

        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,
        19.0f, 20.0f,
        21.0f, 22.0f,

        21.0f, 22.0f,
        23.0f, 24.0f,
        25.0f, 26.0f,
        27.0f, 28.0f,
        29.0f, 30.0f,
        31.0f, 32.0f
    }));

    return result;
}

LayerTestResult<float, 4> Concatenation4dDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim1TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDim2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 1, 3, 6, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 2, true, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        11.0f, 12.0f,
        13.0f, 14.0f,
        21.0f, 22.0f,
        23.0f, 24.0f,

        5.0f, 6.0f,
        7.0f, 8.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,
        25.0f, 26.0f,
        27.0f, 28.0f,

        9.0f, 10.0f,
        11.0f, 12.0f,
        19.0f, 20.0f,
        21.0f, 22.0f,
        29.0f, 30.0f,
        31.0f, 32.0f
    }));

    return result;
}

LayerTestResult<float, 4> Concatenation4dDim2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim2TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDim3TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool useSubtensor)
{
    armnn::TensorInfo outputTensorInfo({ 1, 3, 2, 6 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, outputTensorInfo, 3, useSubtensor, qScale, qOffset);

    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        11.0f, 12.0f,
        21.0f, 22.0f,
        3.0f, 4.0f,
        13.0f, 14.0f,
        23.0f, 24.0f,

        5.0f, 6.0f,
        15.0f, 16.0f,
        25.0f, 26.0f,
        7.0f, 8.0f,
        17.0f, 18.0f,
        27.0f, 28.0f,

        9.0f, 10.0f,
        19.0f, 20.0f,
        29.0f, 30.0f,
        11.0f, 12.0f,
        21.0f, 22.0f,
        31.0f, 32.0f
    }));

    return result;
}

LayerTestResult<float, 4> Concatenation4dDim3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation4dDim3TestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, useSubtensor);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    unsigned int dimension = 0;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 2, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset, {
        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,
        19.0f, 20.0f,
        21.0f, 22.0f,

        21.0f, 22.0f,
        23.0f, 24.0f,
        25.0f, 26.0f,
        27.0f, 28.0f,
        29.0f, 30.0f,
        31.0f, 32.0f

    }));

    armnn::TensorInfo outputTensorInfo({ 3, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory,
                   memoryManager,
                   {inputTensorInfo0, inputTensorInfo1},
                   {input0.data(), input1.data()},
                   outputTensorInfo,
                   output.data(),
                   dimension,
                   true);

    result.output = MakeTensor<T, 4>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f,

        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,
        19.0f, 20.0f,
        21.0f, 22.0f,

        21.0f, 22.0f,
        23.0f, 24.0f,
        25.0f, 26.0f,
        27.0f, 28.0f,
        29.0f, 30.0f,
        31.0f, 32.0f
    }));

    return result;
}

LayerTestResult<float, 4> Concatenation4dDiffShapeDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim0TestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    unsigned int dimension = 1;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 1, 2, 2, 2 }, ArmnnType, qScale, qOffset);

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset, {
        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,

    }));

    armnn::TensorInfo outputTensorInfo({ 1, 5, 2, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory,
                   memoryManager,
                   {inputTensorInfo0, inputTensorInfo1},
                   {input0.data(), input1.data()},
                   outputTensorInfo,
                   output.data(),
                   dimension,
                   true);

    result.output = MakeTensor<T, 4>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f,
        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f
    }));

    return result;
}

LayerTestResult<float, 4> Concatenation4dDiffShapeDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim1TestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    unsigned int dimension = 2;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 1, 3, 3, 2 }, ArmnnType, qScale, qOffset);

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset, {
        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,
        19.0f, 20.0f,
        21.0f, 22.0f,
        23.0f, 24.0f,
        25.0f, 26.0f,
        27.0f, 28.0f
    }));

    armnn::TensorInfo outputTensorInfo({ 1, 3, 5, 2 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory,
                   memoryManager,
                   {inputTensorInfo0, inputTensorInfo1},
                   {input0.data(), input1.data()},
                   outputTensorInfo,
                   output.data(),
                   dimension,
                   true);

    result.output = MakeTensor<T, 4>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,

        5.0f, 6.0f,
        7.0f, 8.0f,
        17.0f, 18.0f,
        19.0f, 20.0f,
        21.0f, 22.0f,

        9.0f, 10.0f,
        11.0f, 12.0f,
        23.0f, 24.0f,
        25.0f, 26.0f,
        27.0f, 28.0f
    }));

    return result;
}

LayerTestResult<float, 4> Concatenation4dDiffShapeDim2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim2TestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim3TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool useSubtensor)
{
    unsigned int dimension = 3;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, ArmnnType, qScale, qOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 1, 3, 2, 3 }, ArmnnType, qScale, qOffset);

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset, {
        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,

        17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f,

        23.0f, 24.0f, 25.0f,
        26.0f, 27.0f, 28.0f
    }));

    armnn::TensorInfo outputTensorInfo({ 1, 3, 2, 5 }, ArmnnType, qScale, qOffset);

    LayerTestResult<T, 4> result(outputTensorInfo);

    std::vector<T> output;
    output.resize(outputTensorInfo.GetNumElements());
    Concatenate<T>(workloadFactory,
                   memoryManager,
                   {inputTensorInfo0, inputTensorInfo1},
                   {input0.data(), input1.data()},
                   outputTensorInfo,
                   output.data(),
                   dimension,
                   useSubtensor);

    result.output = MakeTensor<T, 4>(outputTensorInfo, output);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f, 11.0f, 12.0f, 13.0f,
        3.0f, 4.0f, 14.0f, 15.0f, 16.0f,
        5.0f, 6.0f, 17.0f, 18.0f, 19.0f,
        7.0f, 8.0f, 20.0f, 21.0f, 22.0f,
        9.0f, 10.0f, 23.0f, 24.0f, 25.0f,
        11.0f, 12.0f, 26.0f, 27.0f, 28.0f
    }));

    return result;
}

LayerTestResult<float, 4> Concatenation4dDiffShapeDim3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation4dDiffShapeDim3TestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, useSubtensor);
}

LayerTestResult<float, 2> FakeQuantizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    constexpr unsigned int width = 2;
    constexpr unsigned int height = 3;

    const armnn::TensorInfo tensorInfo({height, width },
        armnn::DataType::Float32);
    auto input = MakeTensor<float, 2>(tensorInfo, std::vector<float>({
       -10.0f,  -5.0f,
         0.0f,   5.0f,
        10.0f,  10.0f
    }));

    LayerTestResult<float, 2> ret(tensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(tensorInfo);

    std::unique_ptr<armnn::ITensorHandle> outputHandle  = workloadFactory.CreateTensorHandle(tensorInfo);

    armnn::FakeQuantizationQueueDescriptor data;
    armnn::WorkloadInfo info;

    AddInputToWorkload(data, info, tensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, tensorInfo, outputHandle.get());
    float min = -10.f;
    float max = 10.f;

    data.m_Parameters.m_Min = min;
    data.m_Parameters.m_Max = max;

    armnn::PassthroughCpuTensorHandle refHandle(tensorInfo, &ret.outputExpected[0][0]);
    armnn::FakeQuantizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadOutput(refData, refInfo, 0, tensorInfo, &refHandle);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateFakeQuantization(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0], outputHandle.get());

    ret.outputExpected = MakeTensor<float, 2>(tensorInfo, std::vector<float>({
        0.0f,     63.0f,
        128.0f,   191.0f,
        255.0f,   255.0f
    }));
    return ret;
}

namespace
{
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2NormalizationTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorShape& inputOutputTensorShape,
    float scale,
    int32_t offset,
    const std::vector<float>& inputValues,
    float outScale,
    int32_t outOffset,
    const std::vector<float>& expectedOutputValues,
    const armnn::DataLayout layout,
    float epsilon = 1e-12f)
{
    const armnn::TensorInfo inputTensorInfo(inputOutputTensorShape, ArmnnType, scale, offset);
    const armnn::TensorInfo outputTensorInfo(inputOutputTensorShape, ArmnnType, outScale, outOffset);

    // at this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    std::vector<float> inputData = inputValues;
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;
    }

    auto inputTensor = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(
                                                         inputTensorInfo.GetQuantizationScale(),
                                                         inputTensorInfo.GetQuantizationOffset(),
                                                         inputData));

    std::vector<float> expectedOutputData = expectedOutputValues;
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(expectedOutputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, expectedOutputData.data(), tmp.data(),
                            sizeof(float));
        expectedOutputData = tmp;
    }

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(
                                                               outputTensorInfo.GetQuantizationScale(),
                                                               outputTensorInfo.GetQuantizationOffset(),
                                                               expectedOutputData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::L2NormalizationQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Eps = epsilon;
    descriptor.m_Parameters.m_DataLayout = layout;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateL2Normalization(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0][0]);

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

float CalcInvL2Norm(std::initializer_list<float> elements)
{
    const float reduction = std::accumulate(elements.begin(), elements.end(), 0.0f,
        [](float acc, float element) { return acc + element * element; });
    return 1.0f / sqrtf(reduction);
}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Pad2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    const float customPaddingValue)
{
    const armnn::TensorShape inputShape{ 3, 3 };
    const armnn::TensorShape outputShape{ 7, 7 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues(
    QuantizedVector<T>(qScale, qOffset,
    {
      // Height (3) x Width (3)
      4, 8, 6,
      7, 4, 4,
      3, 2, 4
    }));

    auto p = customPaddingValue;
    std::vector<T> expectedOutputValues;
    expectedOutputValues = (
    QuantizedVector<T>(qScale, qOffset,
    {
      p, p, p, p, p, p, p,
      p, p, p, p, p, p, p,
      p, p, 4, 8, 6, p, p,
      p, p, 7, 4, 4, p, p,
      p, p, 3, 2, 4, p, p,
      p, p, p, p, p, p, p,
      p, p, p, p, p, p, p
    }));

    auto inputTensor = MakeTensor<T, 2>(inputTensorInfo, std::vector<T>(inputValues));

    LayerTestResult<T, 2> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo, std::vector<T>(expectedOutputValues));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PadQueueDescriptor descriptor;

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    descriptor.m_Parameters.m_PadList = padList;
    descriptor.m_Parameters.m_PadValue = customPaddingValue;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePad(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Pad3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 2 };
    const armnn::TensorShape outputShape{ 3, 5, 6 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues(
      QuantizedVector<T>(qScale,qOffset,
    {
        // Channel 0, Height (2) x Width (2)
        0, 4,
        2, 5,

        // Channel 1, Height (2) x Width (2)
        6, 1,
        5, 2
    }));

    std::vector<T> expectedOutputValues(
      QuantizedVector<T>(qScale,qOffset,
    {

        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 4, 0, 0,
        0, 0, 2, 5, 0, 0,
        0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 6, 1, 0, 0,
        0, 0, 5, 2, 0, 0,
        0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0

    }));

    auto inputTensor = MakeTensor<T, 3>(inputTensorInfo, std::vector<T>(inputValues));

    LayerTestResult<T, 3> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, std::vector<T>(expectedOutputValues));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PadQueueDescriptor descriptor;

    std::vector<std::pair<unsigned int, unsigned int>> PadList;
    PadList.push_back(std::pair<unsigned int, unsigned int>(0,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(2,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    descriptor.m_Parameters.m_PadList = PadList;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePad(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> Pad4dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 3, 2 };
    const armnn::TensorShape outputShape{ 4, 5, 7, 4 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues(
      QuantizedVector<T>(qScale,qOffset,
    {
        // Batch 0, Channel 0, Height (3) x Width (2)
        0, 1,
        2, 3,
        4, 5,

        // Batch 0, Channel 1, Height (3) x Width (2)
        6, 7,
        8, 9,
        10, 11,

        // Batch 1, Channel 0, Height (3) x Width (2)
        12, 13,
        14, 15,
        16, 17,

        // Batch 1, Channel 1, Height (3) x Width (2)
        18, 19,
        20, 21,
        22, 23
    }));

    std::vector<T> expectedOutputValues(
      QuantizedVector<T>(qScale,qOffset,
    {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 2, 3, 0,
        0, 4, 5, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 6, 7, 0,
        0, 8, 9, 0,
        0, 10, 11, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 12, 13, 0,
        0, 14, 15, 0,
        0, 16, 17, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 18, 19, 0,
        0, 20, 21, 0,
        0, 22, 23, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    }));

    auto inputTensor = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(inputValues));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, std::vector<T>(expectedOutputValues));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PadQueueDescriptor descriptor;

    std::vector<std::pair<unsigned int, unsigned int>> PadList;
    PadList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(2,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(3,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(1,1));

    descriptor.m_Parameters.m_PadList = PadList;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePad(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

LayerTestResult<uint8_t, 2> PadUint82dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad2dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<uint8_t, 2> PadUint82dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad2dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.0f, 0, 1.0f);
}

LayerTestResult<uint8_t, 3> PadUint83dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad3dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<uint8_t, 4> PadUint84dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad4dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.0f, 0);
}


template LayerTestResult<typename armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 2>
Pad2dTestCommon<armnn::DataType::QuantisedSymm16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    const float customPaddingValue);

template LayerTestResult<typename armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 3>
Pad3dTestCommon<armnn::DataType::QuantisedSymm16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset);

template LayerTestResult<typename armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 4>
Pad4dTestCommon<armnn::DataType::QuantisedSymm16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset);

LayerTestResult<float, 2> PadFloat322dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<float, 2> PadFloat322dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0, 1.0f);
}

LayerTestResult<float, 3> PadFloat323dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad3dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<float, 4> PadFloat324dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Pad4dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2NormalizationEpsilonTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float scale,
        int32_t offset,
        float outScale,
        int32_t outOffset,
        const armnn::DataLayout layout,
        float epsilon)
{
    // Width: 1
    // Height: 1
    // Channels: 3
    // BatchSize: 1
    unsigned int numberOfBatches = 1;
    unsigned int numberOfChannels = 3;
    unsigned int height = 1;
    unsigned int width = 1;

    const armnn::TensorShape inputOutputShape = armnnUtils::GetTensorShape(
            numberOfBatches, numberOfChannels, height, width, layout);

    // 0.0000001^2 + 0.00000002^2 + 0.00000003^2 < 1e-12
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (1) x Width (1)
        0.00000001f,

        // Batch 0, Channel 1, Height (1) x Width (1)
        0.00000002f,

        // Batch 0, Channel 2, Height (1) x Width (1)
        0.00000003f,
    };

    const float approxInvL2Norm = 1.f / sqrtf(epsilon);
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (1) x Width (1)
        0.00000001f * approxInvL2Norm,
        0.00000002f * approxInvL2Norm,
        0.00000003f * approxInvL2Norm,
    };

    return L2NormalizationTestImpl<ArmnnType>(workloadFactory, memoryManager, inputOutputShape, scale, offset,
                                              inputValues, outScale, outOffset, expectedOutputValues, layout,
                                              epsilon);
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization1dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float scale,
        int32_t offset,
        float outScale,
        int32_t outOffset,
        const armnn::DataLayout layout)
{
    // Width: 1
    // Height: 1
    // Channels: 10
    // BatchSize: 1
    unsigned int numberOfBatches = 1;
    unsigned int numberOfChannels = 10;
    unsigned int height = 1;
    unsigned int width = 1;


    const armnn::TensorShape inputOutputShape = armnnUtils::GetTensorShape(
            numberOfBatches, numberOfChannels, height, width, layout);
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (1) x Width (1)
        1.0f,

        // Batch 0, Channel 1, Height (1) x Width (1)
        2.0f,

        // Batch 0, Channel 2, Height (1) x Width (1)
        3.0f,

        // Batch 0, Channel 3, Height (1) x Width (1)
        4.0f,

        // Batch 0, Channel 4, Height (1) x Width (1)
        5.0f,

        // Batch 0, Channel 5, Height (1) x Width (1)
        6.0f,

        // Batch 0, Channel 6, Height (1) x Width (1)
        7.0f,

        // Batch 0, Channel 7, Height (1) x Width (1)
        8.0f,

        // Batch 0, Channel 8, Height (1) x Width (1)
        9.0f,

        // Batch 0, Channel 9, Height (1) x Width (1)
        10.0f
    };
    const float approxInvL2Norm = 0.050964719f;
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (1) x Width (1)
        1.0f * approxInvL2Norm,
        2.0f * approxInvL2Norm,
        3.0f * approxInvL2Norm,
        4.0f * approxInvL2Norm,
        5.0f * approxInvL2Norm,
        6.0f * approxInvL2Norm,
        7.0f * approxInvL2Norm,
        8.0f * approxInvL2Norm,
        9.0f * approxInvL2Norm,
        10.0f * approxInvL2Norm
    };


    return L2NormalizationTestImpl<ArmnnType>(workloadFactory, memoryManager, inputOutputShape, scale, offset,
                                              inputValues, outScale, outOffset, expectedOutputValues, layout);
}

LayerTestResult<float, 4> L2NormalizationDefaultEpsilonTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout)
{
    // Dummy descriptor to get the default value of epsilon.
    armnn::L2NormalizationDescriptor descriptor;

    return L2NormalizationEpsilonTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.f, 0, 0.f, 0,
                                                                      layout, descriptor.m_Eps);
}

LayerTestResult<float, 4> L2NormalizationNonDefaultEpsilonTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout)
{
    return L2NormalizationEpsilonTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.f, 0, 0.f, 0,
                                                                      layout, 1e-9f);
}

LayerTestResult<float, 4> L2Normalization1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.f, 0, 0.f, 0,layout);
}

LayerTestResult<int16_t, 4> L2Normalization1dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 1.f, 0, 1.f, 0,
                                                                         layout);
}

LayerTestResult<uint8_t, 4> L2Normalization1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.f, 0,
                                                                         1.f/128, 128, layout);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float scale,
    int32_t offset,
    float outScale,
    int32_t outOffset,
    const armnn::DataLayout layout)
{
    // Width: 5
    // Height: 1
    // Channels: 2
    // BatchSize: 1
    unsigned int numberOfBatches = 1;
    unsigned int numberOfChannels = 2;
    unsigned int height = 1;
    unsigned int width = 5;

    const armnn::TensorShape inputOutputShape = armnnUtils::GetTensorShape(
            numberOfBatches, numberOfChannels, height, width, layout);
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (1) x Width (5)
        1.0f, 3.0f, 5.0f, 7.0f,  9.0f,

        // Batch 0, Channel 1, Height (1) x Width (5)
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (1) x Width (5)
        1.0f * CalcInvL2Norm({ 1.0f,  2.0f }),
        3.0f * CalcInvL2Norm({ 3.0f,  4.0f }),
        5.0f * CalcInvL2Norm({ 5.0f,  6.0f }),
        7.0f * CalcInvL2Norm({ 7.0f,  8.0f }),
        9.0f * CalcInvL2Norm({ 9.0f, 10.0f }),

        // Batch 0, Channel 1, Height (1) x Width (5)
        2.0f * CalcInvL2Norm({ 1.0f,  2.0f }),
        4.0f * CalcInvL2Norm({ 3.0f,  4.0f }),
        6.0f * CalcInvL2Norm({ 5.0f,  6.0f }),
        8.0f * CalcInvL2Norm({ 7.0f,  8.0f }),
        10.0f * CalcInvL2Norm({ 9.0f, 10.0f })
    };

    return L2NormalizationTestImpl<ArmnnType>(workloadFactory, memoryManager, inputOutputShape, scale, offset,
                                              inputValues, outScale, outOffset, expectedOutputValues, layout);
}

LayerTestResult<float, 4> L2Normalization2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.f, 0, 0.f, 0,
                                                                 layout);
}

LayerTestResult<int16_t, 4> L2Normalization2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 1.f, 0, 1.f, 0,
                                                                         layout);
}

LayerTestResult<uint8_t, 4> L2Normalization2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.f, 0,
                                                                         1.f/128, 128, layout);
}

LayerTestResult<float, 2> L2Normalization2dShapeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::DataLayout layout = armnn::DataLayout::NHWC;
    const armnn::TensorShape inputOutputTensorShape = armnn::TensorShape({ 5, 2 });

    std::vector<float> inputData
    {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f
    };
    std::vector<float> expectedOutputData
    {
        1.0f * CalcInvL2Norm({ 1.0f,  2.0f }),
        2.0f * CalcInvL2Norm({ 1.0f,  2.0f }),
        3.0f * CalcInvL2Norm({ 3.0f,  4.0f }),
        4.0f * CalcInvL2Norm({ 3.0f,  4.0f }),
        5.0f * CalcInvL2Norm({ 5.0f,  6.0f }),
        6.0f * CalcInvL2Norm({ 5.0f,  6.0f }),
        7.0f * CalcInvL2Norm({ 7.0f,  8.0f }),
        8.0f * CalcInvL2Norm({ 7.0f,  8.0f }),
        9.0f  * CalcInvL2Norm({ 9.0f, 10.0f }),
        10.0f * CalcInvL2Norm({ 9.0f, 10.0f })
    };

    const armnn::TensorInfo inputTensorInfo(inputOutputTensorShape, armnn::DataType::Float32, 0.f, 0);
    const armnn::TensorInfo outputTensorInfo(inputOutputTensorShape, armnn::DataType::Float32, 0.f, 0);

    auto inputTensor = MakeTensor<float, 2>(inputTensorInfo, QuantizedVector<float>(
                                                             inputTensorInfo.GetQuantizationScale(),
                                                             inputTensorInfo.GetQuantizationOffset(),
                                                             inputData));

    LayerTestResult<float, 2> result(outputTensorInfo);
    result.outputExpected = MakeTensor<float, 2>(outputTensorInfo, QuantizedVector<float>(
                                                                   outputTensorInfo.GetQuantizationScale(),
                                                                   outputTensorInfo.GetQuantizationOffset(),
                                                                   expectedOutputData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::L2NormalizationQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Eps = 1e-12f;
    descriptor.m_Parameters.m_DataLayout = layout;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateL2Normalization(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0]);

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float scale,
    int32_t offset,
    float outScale,
    int32_t outOffset,
    const armnn::DataLayout layout)
{
    // Width: 3
    // Height: 4
    // Channels: 2
    // BatchSize: 1
    unsigned int numberOfBatches = 1;
    unsigned int numberOfChannels = 2;
    unsigned int height = 4;
    unsigned int width = 3;

    const armnn::TensorShape inputOutputShape = armnnUtils::GetTensorShape(
            numberOfBatches, numberOfChannels, height, width, layout);
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (4) x Width (3)
        119.0f,  21.0f, 150.0f,
        149.0f,  32.0f, 179.0f,
        15.0f, 227.0f, 141.0f,
        147.0f, 199.0f, 220.0f,

        // Batch 0, Channel 1, Height (4) x Width (3)
        110.0f, 140.0f,  73.0f,
        211.0f, 212.0f,  89.0f,
        24.0f, 138.0f, 188.0f,
        162.0f,  12.0f, 161.0f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (4) x Width (3)
        119.0f * CalcInvL2Norm({ 119.0f, 110.0f }),
        21.0f * CalcInvL2Norm({  21.0f, 140.0f }),
        150.0f * CalcInvL2Norm({ 150.0f,  73.0f }),
        149.0f * CalcInvL2Norm({ 149.0f, 211.0f }),
        32.0f * CalcInvL2Norm({  32.0f, 212.0f }),
        179.0f * CalcInvL2Norm({ 179.0f,  89.0f }),
        15.0f * CalcInvL2Norm({  15.0f,  24.0f }),
        227.0f * CalcInvL2Norm({ 227.0f, 138.0f }),
        141.0f * CalcInvL2Norm({ 141.0f, 188.0f }),
        147.0f * CalcInvL2Norm({ 147.0f, 162.0f }),
        199.0f * CalcInvL2Norm({ 199.0f,  12.0f }),
        220.0f * CalcInvL2Norm({ 220.0f, 161.0f }),

        // Batch 0, Channel 1, Height (4) x Width (3)
        110.0f * CalcInvL2Norm({ 119.0f, 110.0f }),
        140.0f * CalcInvL2Norm({  21.0f, 140.0f }),
        73.0f * CalcInvL2Norm({ 150.0f,  73.0f }),
        211.0f * CalcInvL2Norm({ 149.0f, 211.0f }),
        212.0f * CalcInvL2Norm({  32.0f, 212.0f }),
        89.0f * CalcInvL2Norm({ 179.0f,  89.0f }),
        24.0f * CalcInvL2Norm({  15.0f,  24.0f }),
        138.0f * CalcInvL2Norm({ 227.0f, 138.0f }),
        188.0f * CalcInvL2Norm({ 141.0f, 188.0f }),
        162.0f * CalcInvL2Norm({ 147.0f, 162.0f }),
        12.0f * CalcInvL2Norm({ 199.0f,  12.0f }),
        161.0f * CalcInvL2Norm({ 220.0f, 161.0f })
    };

    return L2NormalizationTestImpl<ArmnnType>(workloadFactory, memoryManager, inputOutputShape, scale, offset,
                                              inputValues, outScale, outOffset, expectedOutputValues, layout);
}

LayerTestResult<float, 4> L2Normalization3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization3dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.f, 0, 0.f, 0,
                                                                 layout);
}

LayerTestResult<int16_t, 4> L2Normalization3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 1.f, 0, 1.f, 0,
                                                                         layout);
}

LayerTestResult<uint8_t, 4> L2Normalization3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.f, 0,
                                                                         1.f/128, 128, layout);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization4dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float scale,
    int32_t offset,
    float outScale,
    int32_t outOffset,
    const armnn::DataLayout layout)
{
    // Width: 3
    // Height: 4
    // Channels: 3
    // BatchSize: 2
    unsigned int numberOfBatches = 2;
    unsigned int numberOfChannels = 3;
    unsigned int height = 4;
    unsigned int width = 3;

    const armnn::TensorShape inputOutputShape = armnnUtils::GetTensorShape(
            numberOfBatches, numberOfChannels, height, width, layout);
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (4) x Width (3)
        235.0f,  46.0f, 178.0f,
        100.0f, 123.0f,  19.0f,
        172.0f,  74.0f, 250.0f,
        6.0f, 195.0f,  80.0f,

        // Batch 0, Channel 1, Height (4) x Width (3)
        113.0f,  95.0f, 202.0f,
        77.0f, 114.0f,  71.0f,
        122.0f, 246.0f, 166.0f,
        82.0f,  28.0f,  37.0f,

        // Batch 0, Channel 2, Height (4) x Width (3)
        56.0f, 170.0f, 162.0f,
        194.0f,  89.0f, 254.0f,
        12.0f, 209.0f, 200.0f,
        1.0f,  64.0f,  54.0f,

        // Batch 1, Channel 0, Height (4) x Width (3)
        67.0f,  90.0f,  49.0f,
        7.0f, 163.0f,  18.0f,
        25.0f, 117.0f, 103.0f,
        247.0f,  59.0f, 189.0f,

        // Batch 1, Channel 1, Height (4) x Width (3)
        239.0f, 104.0f, 199.0f,
        17.0f, 124.0f, 153.0f,
        222.0f, 217.0f, 75.0f,
        32.0f, 126.0f, 21.0f,

        // Batch 1, Channel 2, Height (4) x Width (3)
        97.0f, 145.0f, 215.0f,
        115.0f, 116.0f, 238.0f,
        226.0f,  16.0f, 132.0f,
        92.0f, 125.0f,  88.0f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (4) x Width (3)
        235.0f * CalcInvL2Norm({ 235.0f, 113.0f,  56.0f }),
        46.0f * CalcInvL2Norm({  46.0f,  95.0f, 170.0f }),
        178.0f * CalcInvL2Norm({ 178.0f, 202.0F, 162.0f }),
        100.0f * CalcInvL2Norm({ 100.0f,  77.0f, 194.0f }),
        123.0f * CalcInvL2Norm({ 123.0f, 114.0f,  89.0f }),
        19.0f * CalcInvL2Norm({  19.0f,  71.0f, 254.0f }),
        172.0f * CalcInvL2Norm({ 172.0f, 122.0f,  12.0f }),
        74.0f * CalcInvL2Norm({  74.0f, 246.0f, 209.0f }),
        250.0f * CalcInvL2Norm({ 250.0f, 166.0f, 200.0f }),
        6.0f * CalcInvL2Norm({   6.0f,  82.0f,   1.0f }),
        195.0f * CalcInvL2Norm({ 195.0f,  28.0f,  64.0f }),
        80.0f * CalcInvL2Norm({  80.0f,  37.0f,  54.0f }),

        // Batch 0, Channel 1, Height (4) x Width (3)
        113.0f * CalcInvL2Norm({ 235.0f, 113.0f,  56.0f }),
        95.0f * CalcInvL2Norm({  46.0f,  95.0f, 170.0f }),
        202.0f * CalcInvL2Norm({ 178.0f, 202.0F, 162.0f }),
        77.0f * CalcInvL2Norm({ 100.0f,  77.0f, 194.0f }),
        114.0f * CalcInvL2Norm({ 123.0f, 114.0f,  89.0f }),
        71.0f * CalcInvL2Norm({  19.0f,  71.0f, 254.0f }),
        122.0f * CalcInvL2Norm({ 172.0f, 122.0f,  12.0f }),
        246.0f * CalcInvL2Norm({  74.0f, 246.0f, 209.0f }),
        166.0f * CalcInvL2Norm({ 250.0f, 166.0f, 200.0f }),
        82.0f * CalcInvL2Norm({   6.0f,  82.0f,   1.0f }),
        28.0f * CalcInvL2Norm({ 195.0f,  28.0f,  64.0f }),
        37.0f * CalcInvL2Norm({  80.0f,  37.0f,  54.0f }),

        // Batch 0, Channel 2, Height (4) x Width (3)
        56.0f * CalcInvL2Norm({ 235.0f, 113.0f,  56.0f }),
        170.0f * CalcInvL2Norm({  46.0f,  95.0f, 170.0f }),
        162.0f * CalcInvL2Norm({ 178.0f, 202.0F, 162.0f }),
        194.0f * CalcInvL2Norm({ 100.0f,  77.0f, 194.0f }),
        89.0f * CalcInvL2Norm({ 123.0f, 114.0f,  89.0f }),
        254.0f * CalcInvL2Norm({  19.0f,  71.0f, 254.0f }),
        12.0f * CalcInvL2Norm({ 172.0f, 122.0f,  12.0f }),
        209.0f * CalcInvL2Norm({  74.0f, 246.0f, 209.0f }),
        200.0f * CalcInvL2Norm({ 250.0f, 166.0f, 200.0f }),
        1.0f * CalcInvL2Norm({   6.0f,  82.0f,   1.0f }),
        64.0f * CalcInvL2Norm({ 195.0f,  28.0f,  64.0f }),
        54.0f * CalcInvL2Norm({  80.0f,  37.0f,  54.0f }),

        // Batch 1, Channel 0, Height (4) x Width (3)
        67.0f * CalcInvL2Norm({  67.0f, 239.0f,  97.0f }),
        90.0f * CalcInvL2Norm({  90.0f, 104.0f, 145.0f }),
        49.0f * CalcInvL2Norm({  49.0f, 199.0f, 215.0f }),
        7.0f * CalcInvL2Norm({   7.0f,  17.0f, 115.0f }),
        163.0f * CalcInvL2Norm({ 163.0f, 124.0f, 116.0f }),
        18.0f * CalcInvL2Norm({  18.0f, 153.0f, 238.0f }),
        25.0f * CalcInvL2Norm({  25.0f, 222.0f, 226.0f }),
        117.0f * CalcInvL2Norm({ 117.0f, 217.0f,  16.0f }),
        103.0f * CalcInvL2Norm({ 103.0f,  75.0f, 132.0f }),
        247.0f * CalcInvL2Norm({ 247.0f,  32.0f,  92.0f }),
        59.0f * CalcInvL2Norm({  59.0f, 126.0f, 125.0f }),
        189.0f * CalcInvL2Norm({ 189.0f,  21.0f,  88.0f }),

        // Batch 1, Channel 1, Height (4) x Width (3)
        239.0f * CalcInvL2Norm({  67.0f, 239.0f,  97.0f }),
        104.0f * CalcInvL2Norm({  90.0f, 104.0f, 145.0f }),
        199.0f * CalcInvL2Norm({  49.0f, 199.0f, 215.0f }),
        17.0f * CalcInvL2Norm({   7.0f,  17.0f, 115.0f }),
        124.0f * CalcInvL2Norm({ 163.0f, 124.0f, 116.0f }),
        153.0f * CalcInvL2Norm({  18.0f, 153.0f, 238.0f }),
        222.0f * CalcInvL2Norm({  25.0f, 222.0f, 226.0f }),
        217.0f * CalcInvL2Norm({ 117.0f, 217.0f,  16.0f }),
        75.0f * CalcInvL2Norm({ 103.0f,  75.0f, 132.0f }),
        32.0f * CalcInvL2Norm({ 247.0f,  32.0f,  92.0f }),
        126.0f * CalcInvL2Norm({  59.0f, 126.0f, 125.0f }),
        21.0f * CalcInvL2Norm({ 189.0f,  21.0f,  88.0f }),

        // Batch 1, Channel 2, Height (4) x Width (3)
        97.0f * CalcInvL2Norm({  67.0f, 239.0f,  97.0f }),
        145.0f * CalcInvL2Norm({  90.0f, 104.0f, 145.0f }),
        215.0f * CalcInvL2Norm({  49.0f, 199.0f, 215.0f }),
        115.0f * CalcInvL2Norm({   7.0f,  17.0f, 115.0f }),
        116.0f * CalcInvL2Norm({ 163.0f, 124.0f, 116.0f }),
        238.0f * CalcInvL2Norm({  18.0f, 153.0f, 238.0f }),
        226.0f * CalcInvL2Norm({  25.0f, 222.0f, 226.0f }),
        16.0f * CalcInvL2Norm({ 117.0f, 217.0f,  16.0f }),
        132.0f * CalcInvL2Norm({ 103.0f,  75.0f, 132.0f }),
        92.0f * CalcInvL2Norm({ 247.0f,  32.0f,  92.0f }),
        125.0f * CalcInvL2Norm({  59.0f, 126.0f, 125.0f }),
        88.0f * CalcInvL2Norm({ 189.0f,  21.0f,  88.0f })
    };

    return L2NormalizationTestImpl<ArmnnType>(workloadFactory, memoryManager, inputOutputShape, scale, offset,
                                              inputValues, outScale, outOffset, expectedOutputValues, layout);
}

LayerTestResult<float, 4> L2Normalization4dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization4dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.f, 0, 0.f, 0,
                                                                 layout);
}

LayerTestResult<int16_t, 4> L2Normalization4dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 1.f, 0, 1.f, 0,
                                                                         layout);
}

LayerTestResult<uint8_t, 4> L2Normalization4dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.f, 0,
                                                                         1.f/128, 128, layout);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ConstantTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    constexpr unsigned int inputWidth = 3;
    constexpr unsigned int inputHeight = 4;
    constexpr unsigned int inputChannels = 3;
    constexpr unsigned int inputBatchSize = 2;

    constexpr unsigned int outputWidth = inputWidth;
    constexpr unsigned int outputHeight = inputHeight;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
                                        ArmnnType, qScale, qOffset);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
                                         ArmnnType, qScale, qOffset);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        235.0f,  46.0f, 178.0f,
        100.0f, 123.0f,  19.0f,
        172.0f,  74.0f, 250.0f,
          6.0f, 195.0f,  80.0f,

        // Batch 0, Channel 1
        113.0f,  95.0f, 202.0f,
         77.0f, 114.0f,  71.0f,
        122.0f, 246.0f, 166.0f,
         82.0f,  28.0f,  37.0f,

        // Batch 0, Channel 2
         56.0f, 170.0f, 162.0f,
        194.0f,  89.0f, 254.0f,
         12.0f, 209.0f, 200.0f,
          1.0f,  64.0f,  54.0f,

        // Batch 1, Channel 0
         67.0f,  90.0f,  49.0f,
          7.0f, 163.0f,  18.0f,
         25.0f, 117.0f, 103.0f,
        247.0f,  59.0f, 189.0f,

        // Batch 1, Channel 1
        239.0f, 104.0f, 199.0f,
         17.0f, 124.0f, 153.0f,
        222.0f, 217.0f, 75.0f,
         32.0f, 126.0f, 21.0f,

        // Batch 1, Channel 2
         97.0f, 145.0f, 215.0f,
        115.0f, 116.0f, 238.0f,
        226.0f,  16.0f, 132.0f,
         92.0f, 125.0f,  88.0f,
    })));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = input;

    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ScopedCpuTensorHandle constantTensor(inputTensorInfo);
    AllocateAndCopyDataToITensorHandle(&constantTensor, &input[0][0][0][0]);

    armnn::ConstantQueueDescriptor descriptor;
    descriptor.m_LayerOutput = &constantTensor;

    armnn::WorkloadInfo info;
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConstant(descriptor, info);

    outputHandle->Allocate();

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<float, 4> ConstantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<int16_t, 4> ConstantInt16SimpleQuantizationScaleNoOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<uint8_t, 4> ConstantUint8SimpleQuantizationScaleNoOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<uint8_t, 3> ConcatUint8DifferentQParamsTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int outputWidth = 3;
    unsigned int outputHeight = 6;
    unsigned int outputChannels = 3;

    unsigned int inputWidth1 = 3;
    unsigned int inputHeight1 = 6;
    unsigned int inputChannels1 = 2;

    unsigned int inputWidth2 = 3;
    unsigned int inputHeight2 = 6;
    unsigned int inputChannels2 = 1;

    // Defines the tensor descriptors.
    armnn::TensorInfo outputTensorInfo({ outputChannels, outputHeight, outputWidth }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo inputTensorInfo1({ inputChannels1, inputHeight1, inputWidth1 }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo inputTensorInfo2({ inputChannels2, inputHeight2, inputWidth2 }, armnn::DataType::QuantisedAsymm8);

    // Quantized input1 tensor. Range [-3, 1]
    const float inputScale1 = 0.015686f;
    const int32_t inputOffset1 = 192;

    auto input1 = MakeTensor<uint8_t, 3>(inputTensorInfo1, std::vector<uint8_t>(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,
    })
    );

    // Quatized input2 tensor. Range [-1, 4]
    const float inputScale2 = 0.019608f;
    const int32_t inputOffset2 = 50;

    auto input2 = MakeTensor<uint8_t, 3>(inputTensorInfo2, std::vector<uint8_t>(
    {
        37, 38, 39,
        40, 41, 42,
        43, 44, 45,
        46, 47, 48,
        49, 50, 51,
        52, 53, 54,
    })
    );

    // Output has the same quantization parameters than input1,
    // so that only the requantization of input2 is required
    const float outputScale = 0.015686f;
    const int32_t outputOffset = 192;

    LayerTestResult<uint8_t, 3> ret(outputTensorInfo);

    ret.outputExpected = MakeTensor<uint8_t, 3>(outputTensorInfo, std::vector<uint8_t>(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,

        176, 177, 178,
        179, 181, 182,
        183, 184, 186,
        187, 188, 189,
        191, 192, 193,
        195, 196, 197,
    })
    );

    outputTensorInfo.SetQuantizationScale(outputScale);
    outputTensorInfo.SetQuantizationOffset(outputOffset);
    inputTensorInfo1.SetQuantizationScale(inputScale1);
    inputTensorInfo1.SetQuantizationOffset(inputOffset1);
    inputTensorInfo2.SetQuantizationScale(inputScale2);
    inputTensorInfo2.SetQuantizationOffset(inputOffset2);

    std::vector<unsigned int> wOrigin1 = { 0, 0, 0 }; //Extent of the window is defined by size of input[0].
    armnn::ConcatQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = { 2, 0, 0 }; //Extent of the window is defined by size of input[1].
    armnn::ConcatQueueDescriptor::ViewOrigin window2(wOrigin2);

    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    bool subTensorsSupported = workloadFactory.SupportsSubTensors();

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 =
            subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo1.GetShape(), wOrigin1.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo1);

    std::unique_ptr<armnn::ITensorHandle> inputHandle2 =
            subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo2.GetShape(), wOrigin2.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo2);

    armnn::ConcatQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConcat(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<uint8_t, 3> ConcatUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int outputWidth = 3;
    unsigned int outputHeight = 6;
    unsigned int outputChannels = 3;

    unsigned int inputWidth1 = 3;
    unsigned int inputHeight1 = 6;
    unsigned int inputChannels1 = 2;

    unsigned int inputWidth2 = 3;
    unsigned int inputHeight2 = 6;
    unsigned int inputChannels2 = 1;

    // Defines the tensor descriptors.
    armnn::TensorInfo outputTensorInfo({ outputChannels, outputHeight, outputWidth }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo inputTensorInfo1({ inputChannels1, inputHeight1, inputWidth1 }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo inputTensorInfo2({ inputChannels2, inputHeight2, inputWidth2 }, armnn::DataType::QuantisedAsymm8);

    // Arbitrary scale and offsets. They don't really matter as the Concat operator doesn't dequantize/quantize them.
    const float scale = 0.13497836f;
    const int32_t offset = -7;

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);
    inputTensorInfo1.SetQuantizationScale(scale);
    inputTensorInfo1.SetQuantizationOffset(offset);
    inputTensorInfo2.SetQuantizationScale(scale);
    inputTensorInfo2.SetQuantizationOffset(offset);

    LayerTestResult<uint8_t, 3> ret(outputTensorInfo);

    ret.outputExpected = MakeTensor<uint8_t, 3>(outputTensorInfo, std::vector<uint8_t>(
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,

            19, 20, 21,
            22, 23, 24,
            25, 26, 27,
            28, 29, 30,
            31, 32, 33,
            34, 35, 36,

            37, 38, 39,
            40, 41, 42,
            43, 44, 45,
            46, 47, 48,
            49, 50, 51,
            52, 53, 54,
        })
    );

    auto input1 = MakeTensor<uint8_t, 3>(inputTensorInfo1, std::vector<uint8_t>(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,
    })
    );

    auto input2 = MakeTensor<uint8_t, 3>(inputTensorInfo2, std::vector<uint8_t>(
    {
        37, 38, 39,
        40, 41, 42,
        43, 44, 45,
        46, 47, 48,
        49, 50, 51,
        52, 53, 54,
    })
    );

    std::vector<unsigned int> wOrigin1 = { 0, 0, 0 }; //Extent of the window is defined by size of input[0].
    armnn::ConcatQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = { 2, 0, 0 }; //Extent of the window is defined by size of input[1].
    armnn::ConcatQueueDescriptor::ViewOrigin window2(wOrigin2);


    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    bool subTensorsSupported = workloadFactory.SupportsSubTensors();

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo1.GetShape(), wOrigin1.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo1);

    std::unique_ptr<armnn::ITensorHandle> inputHandle2 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo2.GetShape(), wOrigin2.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo2);


    armnn::ConcatQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConcat(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<uint16_t, 3> ConcatUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int outputWidth = 3;
    unsigned int outputHeight = 6;
    unsigned int outputChannels = 3;

    unsigned int inputWidth1 = 3;
    unsigned int inputHeight1 = 6;
    unsigned int inputChannels1 = 2;

    unsigned int inputWidth2 = 3;
    unsigned int inputHeight2 = 6;
    unsigned int inputChannels2 = 1;

    // Defines the tensor descriptors.
    armnn::TensorInfo outputTensorInfo({ outputChannels, outputHeight, outputWidth }, armnn::DataType::QuantisedSymm16);
    armnn::TensorInfo inputTensorInfo1({ inputChannels1, inputHeight1, inputWidth1 }, armnn::DataType::QuantisedSymm16);
    armnn::TensorInfo inputTensorInfo2({ inputChannels2, inputHeight2, inputWidth2 }, armnn::DataType::QuantisedSymm16);

    // Arbitrary scale and offsets. They don't really matter as the Concat operator doesn't dequantize/quantize them.
    const float scale = 0.13497836f;
    const int32_t offset = -7;

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);
    inputTensorInfo1.SetQuantizationScale(scale);
    inputTensorInfo1.SetQuantizationOffset(offset);
    inputTensorInfo2.SetQuantizationScale(scale);
    inputTensorInfo2.SetQuantizationOffset(offset);

    LayerTestResult<uint16_t, 3> ret(outputTensorInfo);

    ret.outputExpected = MakeTensor<uint16_t, 3>(outputTensorInfo, std::vector<uint16_t>(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,

        37, 38, 39,
        40, 41, 42,
        43, 44, 45,
        46, 47, 48,
        49, 50, 51,
        52, 53, 54,
    }));

    auto input1 = MakeTensor<uint16_t, 3>(inputTensorInfo1, std::vector<uint16_t>(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,
    }));

    auto input2 = MakeTensor<uint16_t, 3>(inputTensorInfo2, std::vector<uint16_t>(
    {
        37, 38, 39,
        40, 41, 42,
        43, 44, 45,
        46, 47, 48,
        49, 50, 51,
        52, 53, 54,
    }));

    std::vector<unsigned int> wOrigin1 = { 0, 0, 0 }; //Extent of the window is defined by size of input[0].
    armnn::ConcatQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = { 2, 0, 0 }; //Extent of the window is defined by size of input[1].
    armnn::ConcatQueueDescriptor::ViewOrigin window2(wOrigin2);


    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    bool subTensorsSupported = workloadFactory.SupportsSubTensors();

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 =
            subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo1.GetShape(), wOrigin1.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo1);

    std::unique_ptr<armnn::ITensorHandle> inputHandle2 =
            subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo2.GetShape(), wOrigin2.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo2);


    armnn::ConcatQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConcat(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0], outputHandle.get());

    return ret;
}

namespace
{
template <typename T>
LayerTestResult<T, 4> AdditionQuantizeTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int shape0[4],
    const std::vector<T>& values0,
    float scale0,
    int32_t offset0,
    const unsigned int shape1[4],
    const std::vector<T> & values1,
    float scale1,
    int32_t offset1,
    const unsigned int outShape[4],
    const std::vector<T> & outValues,
    float outScale,
    int32_t outOffset)
{
    auto dataType = (std::is_same<T, uint8_t>::value ?
                     armnn::DataType::QuantisedAsymm8 :
                     armnn::DataType::QuantisedSymm16);

    armnn::TensorInfo inputTensorInfo0(4, shape0, dataType);
    armnn::TensorInfo inputTensorInfo1(4, shape1, dataType);
    armnn::TensorInfo outputTensorInfo(4, outShape, dataType);

    inputTensorInfo0.SetQuantizationScale(scale0);
    inputTensorInfo0.SetQuantizationOffset(offset0);

    inputTensorInfo1.SetQuantizationScale(scale1);
    inputTensorInfo1.SetQuantizationOffset(offset1);

    outputTensorInfo.SetQuantizationScale(outScale);
    outputTensorInfo.SetQuantizationOffset(outOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, values1);

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outValues);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(data,  info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateAddition(data, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}
} // anonymous namespace

LayerTestResult<uint8_t, 4> AdditionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 2, 2, 3 };

    std::vector<uint8_t> input0(
    {
        63,  35,  77,  70,  56, 112, //  420, 224,  518,  469,  371, 763
        203,  28, 252, 168, 245,  91  // 1400, 175, 1743, 1155, 1694, 616
    });

    std::vector<uint8_t> input1(
    {
        21,   7, 175, 231, 175, 210, // 126,   28, 1204, 1596, 1204, 1449
        126, 161,  63,  21, 105, 126  // 861, 1106,  420,  126,  714,  861
    });

    std::vector<uint8_t> output(
    {
        81,  39, 249, 255, 228, 255, //  546,  252, 1722, 2065(clamped), 1575, 2212(clamped)
        255, 186, 255, 186, 255, 214, // 2261(clamped), 1281, 2163(clamped), 1281, 2408(clamped), 1477
    });

    return AdditionQuantizeTestHelper(workloadFactory,
                                      memoryManager,
                                      shape0, input0, 7.0f, 3,
                                      shape1, input1, 7.0f, 3,
                                      shape0, output, 7.0f, 3);
}

LayerTestResult<int16_t, 4> AdditionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 2, 2, 3 };

    std::vector<int16_t> input0(
        {
            63,  35,  77,  70,  56, 112, //  441, 245,  539,  490,  392, 184
            203,  28, 252, 168, 245,  91  // 1421, 196, 1764, 1176, 1715, 637
        });

    std::vector<int16_t> input1(
        {
            21,   7, 175, 231, 175, 210, // 126,   28, 1204, 1596, 1204, 1449
            126, 161,  63,  21, 105, 126  // 861, 1106,  420,  126,  714,  861
        });

    std::vector<int16_t> output(
        {
            84,  42, 252, 301, 231, 322, //  588,  294, 1764, 2107(clamped), 1617, 2254(clamped)
            329, 189, 315, 189, 350, 217, // 2303(clamped), 1323, 2205(clamped), 1323, 2450(clamped), 1519
        });

    return AdditionQuantizeTestHelper(workloadFactory,
                                      memoryManager,
                                      shape0, input0, 7.0f, 0,
                                      shape1, input1, 7.0f, 0,
                                      shape0, output, 7.0f, 0);
}

namespace
{
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> MultiplicationQuantizeTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int shape0[4],
    const std::vector<T> & values0,
    float scale0,
    int32_t offset0,
    const unsigned int shape1[4],
    const std::vector<T> & values1,
    float scale1,
    int32_t offset1,
    const unsigned int outShape[4],
    const std::vector<T> & outValues,
    float outScale,
    int32_t outOffset)
{
    armnn::TensorInfo inputTensorInfo0(4, shape0, ArmnnType);
    armnn::TensorInfo inputTensorInfo1(4, shape1, ArmnnType);
    armnn::TensorInfo outputTensorInfo(4, outShape, ArmnnType);

    inputTensorInfo0.SetQuantizationScale(scale0);
    inputTensorInfo0.SetQuantizationOffset(offset0);

    inputTensorInfo1.SetQuantizationScale(scale1);
    inputTensorInfo1.SetQuantizationOffset(offset1);

    outputTensorInfo.SetQuantizationScale(outScale);
    outputTensorInfo.SetQuantizationOffset(outOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, values1);

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outValues);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::MultiplicationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(data,  info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMultiplication(data, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}
} // anonymous namespace

LayerTestResult<uint8_t, 4> MultiplicationUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int batchSize = 1;
    unsigned int channels = 2;
    unsigned int height = 2;
    unsigned int width = 3;
    const unsigned int shape[] = { batchSize, channels, height, width };

    // See dequantized values to the right.
    std::vector<uint8_t> input0({
         62,  37,   3, 172,  13, 111, // 244, 144,   8, 684,  48, 440,
        188,  20,  73,  31,  23,  31  // 748,  76, 288, 120,  88, 120
    });

    // See dequantized values to the right.
    std::vector<uint8_t> input1({
        126, 240, 252, 183, 121, 247, // 384, 726, 762, 555, 369, 747,
         48, 115, 151,  79,  78,  97  // 150, 351, 459, 243, 240, 297
    });

    // See dequantized values to the right.
    std::vector<uint8_t> output(
    {
         64,  72,   0, 255,   8, 236, //  93696, 104544, 6096(clamped), 379620(clamped), 17712, 328680,
         77,  15,  92,  16,  10,  21, // 112200,  26676,        132192,           29160, 21120,  35640
    });

    // Scale/offset chosen to have output values out of range.
    return MultiplicationQuantizeTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                              memoryManager,
                                                                              shape,
                                                                              input0,
                                                                              4.0f,
                                                                              1,
                                                                              shape,
                                                                              input1,
                                                                              3.0f,
                                                                              -2,
                                                                              shape,
                                                                              output,
                                                                              1366.255f,
                                                                              -5);
}

LayerTestResult<uint8_t, 4> MultiplicationBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0({
        1, 2, 3,    4,  5,  6,
        7, 8, 9,   10, 11, 12
    });

    std::vector<uint8_t> input1({2});

    std::vector<uint8_t> output({
        2,  4,   6,     8, 10, 12,
        14, 16, 18,    20, 22, 24
    });

    return MultiplicationQuantizeTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                              memoryManager,
                                                                              shape0,
                                                                              input0,
                                                                              1.0f,
                                                                              0,
                                                                              shape1,
                                                                              input1,
                                                                              1.0f,
                                                                              0,
                                                                              shape0,
                                                                              output,
                                                                              1.0f,
                                                                              0);
}

LayerTestResult<uint8_t, 4> MultiplicationBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0({
        1, 2, 3,    4,  5,  6,
        7, 8, 9,   10, 11, 12
    });

    std::vector<uint8_t> input1({1, 2, 3});

    std::vector<uint8_t> output({
        1,  4,   9,     4, 10, 18,
        7, 16,  27,    10, 22, 36
    });

    return MultiplicationQuantizeTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                              memoryManager,
                                                                              shape0,
                                                                              input0,
                                                                              1.0f,
                                                                              0,
                                                                              shape1,
                                                                              input1,
                                                                              1.0f,
                                                                              0,
                                                                              shape0,
                                                                              output,
                                                                              1.0f,
                                                                              0);
}

LayerTestResult<int16_t, 4> MultiplicationInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape[] = { 1, 2, 2, 3 };

    std::vector<int16_t> input0(
    {
        6,   7,  8,  9, 10, 11,
        12, 13, 14, 15, 16, 17
    });

    std::vector<int16_t> input1(
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    });

    std::vector<int16_t> output(
    {
        6,   14,  24,  36,  50,  66,
        84, 104, 126, 150, 176, 204
    });

    return MultiplicationQuantizeTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                              memoryManager,
                                                                              shape,
                                                                              input0,
                                                                              1.0f,
                                                                              0,
                                                                              shape,
                                                                              input1,
                                                                              1.0f,
                                                                              0,
                                                                              shape,
                                                                              output,
                                                                              1.0f,
                                                                              0);
}

LayerTestResult<int16_t, 4> MultiplicationBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0(
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    });

    std::vector<int16_t> input1({2});

    std::vector<int16_t> output(
    {
        2,   4,  6,  8, 10, 12,
        14, 16, 18, 20, 22, 24
    });

    return MultiplicationQuantizeTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                              memoryManager,
                                                                              shape0,
                                                                              input0,
                                                                              1.0f,
                                                                              0,
                                                                              shape1,
                                                                              input1,
                                                                              1.0f,
                                                                              0,
                                                                              shape0,
                                                                              output,
                                                                              1.0f,
                                                                              0);
}

LayerTestResult<int16_t, 4> MultiplicationBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int16_t> input0(
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    });

    std::vector<int16_t> input1({1, 2, 3});

    std::vector<int16_t> output(
    {
        1,  4,  9,  4, 10, 18,
        7, 16, 27, 10, 22, 36
    });

    return MultiplicationQuantizeTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                              memoryManager,
                                                                              shape0,
                                                                              input0,
                                                                              1.0f,
                                                                              0,
                                                                              shape1,
                                                                              input1,
                                                                              1.0f,
                                                                              0,
                                                                              shape0,
                                                                              output,
                                                                              1.0f,
                                                                              0);
}

namespace
{
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SubtractionTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int shape0[4],
    const std::vector<T>& values0,
    float scale0,
    int32_t offset0,
    const unsigned int shape1[4],
    const std::vector<T> & values1,
    float scale1,
    int32_t offset1,
    const unsigned int outShape[4],
    const std::vector<T> & outValues,
    float outScale,
    int32_t outOffset)
{
    armnn::TensorInfo inputTensorInfo0(4, shape0, ArmnnType);
    armnn::TensorInfo inputTensorInfo1(4, shape1, ArmnnType);
    armnn::TensorInfo outputTensorInfo(4, outShape, ArmnnType);

    inputTensorInfo0.SetQuantizationScale(scale0);
    inputTensorInfo0.SetQuantizationOffset(offset0);

    inputTensorInfo1.SetQuantizationScale(scale1);
    inputTensorInfo1.SetQuantizationOffset(offset1);

    outputTensorInfo.SetQuantizationScale(outScale);
    outputTensorInfo.SetQuantizationOffset(outOffset);

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, values1);

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outValues);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::SubtractionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(data,  info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSubtraction(data, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}
} // anonymous namespace

LayerTestResult<uint8_t, 4> SubtractionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 2 };

    std::vector<uint8_t> input0({ 10, 12, 14, 16 });
    std::vector<uint8_t> input1({ 1, 2, 1, 2 });
    std::vector<uint8_t> output({ 3, 3, 5, 5 });

    return SubtractionTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                   memoryManager,
                                                                   shape0, input0, 0.5f, 2,
                                                                   shape1, input1, 1.0f, 0,
                                                                   shape0, output, 1.0f, 0);
}

LayerTestResult<uint8_t, 4> SubtractionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0({ 10, 12, 14, 16 });
    std::vector<uint8_t> input1({ 2 });
    std::vector<uint8_t> output({ 5, 6, 7, 8 });

    return SubtractionTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                   memoryManager,
                                                                   shape0, input0, 0.5f, 2,
                                                                   shape1, input1, 1.0f, 0,
                                                                   shape0, output, 1.0f, 3);
}

LayerTestResult<uint8_t, 4> SubtractionBroadcastUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 1 };

    std::vector<uint8_t> input0({ 10, 12, 14, 16 });
    std::vector<uint8_t> input1({ 2, 1 });
    std::vector<uint8_t> output({ 8, 11, 12, 15 });

    return SubtractionTestHelper<armnn::DataType::QuantisedAsymm8>(workloadFactory,
                                                                   memoryManager,
                                                                   shape0, input0, 1.0f, 0,
                                                                   shape1, input1, 1.0f, 0,
                                                                   shape0, output, 1.0f, 0);
}

LayerTestResult<float, 4> SubtractionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 2 };

    std::vector<float> input0({ 1,  2, 3, 4 });
    std::vector<float> input1({ 1, -1, 0, 2 });
    std::vector<float> output({ 0,  3, 3, 2 });

    return SubtractionTestHelper<armnn::DataType::Float32>(workloadFactory,
                                                           memoryManager,
                                                           shape0, input0, 1.0f, 0,
                                                           shape1, input1, 1.0f, 0,
                                                           shape0, output, 1.0f, 0);
}

LayerTestResult<float, 4> SubtractionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<float> input0({ 1,  2, 3, 4 });
    std::vector<float> input1({ 10 });
    std::vector<float> output({ -9,  -8, -7, -6 });

    return SubtractionTestHelper<armnn::DataType::Float32>(workloadFactory,
                                                           memoryManager,
                                                           shape0, input0, 1.0f, 0,
                                                           shape1, input1, 1.0f, 0,
                                                           shape0, output, 1.0f, 0);
}

LayerTestResult<float, 4> SubtractionBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<float> input0({ 1,  2, 3, 4 });
    std::vector<float> input1({ 10, -5 });
    std::vector<float> output({ -9,  7, -7, 9 });

    return SubtractionTestHelper<armnn::DataType::Float32>(workloadFactory,
                                                           memoryManager,
                                                           shape0, input0, 1.0f, 0,
                                                           shape1, input1, 1.0f, 0,
                                                           shape0, output, 1.0f, 0);
}

LayerTestResult<int16_t, 4> SubtractionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 2 };

    std::vector<int16_t> input0({ 10, 12, 14, 16 });
    std::vector<int16_t> input1({ 1, 2, 1, 2 });
    std::vector<int16_t> output({ 3, 3, 5, 5 });

    return SubtractionTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                   memoryManager,
                                                                   shape0, input0, 0.5f, 0,
                                                                   shape1, input1, 1.0f, 0,
                                                                   shape0, output, 1.0f, 0);
}

LayerTestResult<int16_t, 4> SubtractionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0({ 10, 12, 14, 16 });
    std::vector<int16_t> input1({ 2 });
    std::vector<int16_t> output({ 3, 4, 5, 6 });

    return SubtractionTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                   memoryManager,
                                                                   shape0, input0, 0.5f, 0,
                                                                   shape1, input1, 1.0f, 0,
                                                                   shape0, output, 1.0f, 0);
}

LayerTestResult<int16_t, 4> SubtractionBroadcastInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 1 };

    std::vector<int16_t> input0({ 10, 12, 14, 16 });
    std::vector<int16_t> input1({ 2, 1 });
    std::vector<int16_t> output({ 8, 11, 12, 15 });

    return SubtractionTestHelper<armnn::DataType::QuantisedSymm16>(workloadFactory,
                                                                   memoryManager,
                                                                   shape0, input0, 1.0f, 0,
                                                                   shape1, input1, 1.0f, 0,
                                                                   shape0, output, 1.0f, 0);
}

LayerTestResult<float, 4> BatchNormTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // BatchSize: 1
    // Channels: 2
    // Height: 3
    // Width: 2

    const armnn::TensorShape inputOutputShape{ 1, 2, 3, 2 };
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (3) x Width (2)
         1.f, 4.f,
         4.f, 2.f,
         1.f, 6.f,

        // Batch 0, Channel 1, Height (3) x Width (2)
         1.f, 1.f,
         4.f, 1.f,
        -2.f, 4.f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (3) x Width (2)
        1.f, 4.f,
        4.f, 2.f,
        1.f, 6.f,

        // Batch 0, Channel 1, Height (3) x Width (2)
        3.f, 3.f,
        4.f, 3.f,
        2.f, 4.f
    };

    return BatchNormTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager,
        inputOutputShape, inputValues, expectedOutputValues,
        0.f, 0, armnn::DataLayout::NCHW);
}

LayerTestResult<float, 4> BatchNormNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // BatchSize: 1
    // Height: 3
    // Width: 2
    // Channels: 2

    const armnn::TensorShape inputOutputShape{ 1, 3, 2, 2 };
    std::vector<float> inputValues
    {
        // Batch 0, Height 0, Width (2) x Channel (2)
        1.f,  1.f,
        4.f,  1.f,

        // Batch 0, Height 1, Width (2) x Channel (2)
        4.f,  4.f,
        2.f,  1.f,

        // Batch 0, Height 2, Width (2) x Channel (2)
        1.f, -2.f,
        6.f,  4.f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Height 0, Width (2) x Channel (2)
        1.f, 3.f,
        4.f, 3.f,

        // Batch 0, Height 1, Width (2) x Channel (2)
        4.f, 4.f,
        2.f, 3.f,

        // Batch 0, Height 2, Width (2) x Channel (2)
        1.f, 2.f,
        6.f, 4.f
    };

    return BatchNormTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager,
        inputOutputShape, inputValues, expectedOutputValues,
        0.f, 0, armnn::DataLayout::NHWC);
}

LayerTestResult<uint8_t, 4> BatchNormUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // BatchSize: 1
    // Channels: 2
    // Height: 3
    // Width: 2

    const armnn::TensorShape inputOutputShape{ 1, 2, 3, 2 };
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (3) x Width (2)
         1.f, 4.f,
         4.f, 2.f,
         1.f, 6.f,

        // Batch 0, Channel 1, Height (3) x Width (2)
         1.f, 1.f,
         4.f, 1.f,
        -2.f, 4.f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (3) x Width (2)
        1.f, 4.f,
        4.f, 2.f,
        1.f, 6.f,

        // Batch 0, Channel 1, Height (3) x Width (2)
        3.f, 3.f,
        4.f, 3.f,
        2.f, 4.f
    };

    return BatchNormTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager,
        inputOutputShape, inputValues, expectedOutputValues,
        1.f/20.f, 50, armnn::DataLayout::NCHW);
}

LayerTestResult<uint8_t, 4> BatchNormUint8NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // BatchSize: 1
    // Height: 3
    // Width: 2
    // Channels: 2

    const armnn::TensorShape inputOutputShape{ 1, 3, 2, 2 };
    std::vector<float> inputValues
    {
        // Batch 0, Height 0, Width (2) x Channel (2)
        1.f,  1.f,
        4.f,  1.f,

        // Batch 0, Height 1, Width (2) x Channel (2)
        4.f,  4.f,
        2.f,  1.f,

        // Batch 0, Height 2, Width (2) x Channel (2)
        1.f, -2.f,
        6.f,  4.f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Height 0, Width (2) x Channel (2)
        1.f, 3.f,
        4.f, 3.f,

        // Batch 0, Height 1, Width (2) x Channel (2)
        4.f, 4.f,
        2.f, 3.f,

        // Batch 0, Height 2, Width (2) x Channel (2)
        1.f, 2.f,
        6.f, 4.f
    };

    return BatchNormTestImpl<armnn::DataType::QuantisedAsymm8>
        (workloadFactory, memoryManager,
         inputOutputShape, inputValues, expectedOutputValues,
         1.f/20.f, 50, armnn::DataLayout::NHWC);
}

LayerTestResult<int16_t, 4> BatchNormInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // BatchSize: 1
    // Channels: 2
    // Height: 3
    // Width: 2

    const armnn::TensorShape inputOutputShape{ 1, 2, 3, 2 };
    std::vector<float> inputValues
    {
        // Batch 0, Channel 0, Height (3) x Width (2)
         1.f, 4.f,
         4.f, 2.f,
         1.f, 6.f,

        // Batch 0, Channel 1, Height (3) x Width (2)
         1.f, 1.f,
         4.f, 1.f,
        -2.f, 4.f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Channel 0, Height (3) x Width (2)
        1.f, 4.f,
        4.f, 2.f,
        1.f, 6.f,

        // Batch 0, Channel 1, Height (3) x Width (2)
        3.f, 3.f,
        4.f, 3.f,
        2.f, 4.f
    };

    return BatchNormTestImpl<armnn::DataType::QuantisedSymm16>(
        workloadFactory, memoryManager,
        inputOutputShape, inputValues, expectedOutputValues,
        1.f/20.f, 50, armnn::DataLayout::NCHW);
}

LayerTestResult<int16_t, 4> BatchNormInt16NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // BatchSize: 1
    // Height: 3
    // Width: 2
    // Channels: 2

    const armnn::TensorShape inputOutputShape{ 1, 3, 2, 2 };
    std::vector<float> inputValues
    {
        // Batch 0, Height 0, Width (2) x Channel (2)
        1.f,  1.f,
        4.f,  1.f,

        // Batch 0, Height 1, Width (2) x Channel (2)
        4.f,  4.f,
        2.f,  1.f,

        // Batch 0, Height 2, Width (2) x Channel (2)
        1.f, -2.f,
        6.f,  4.f
    };
    std::vector<float> expectedOutputValues
    {
        // Batch 0, Height 0, Width (2) x Channel (2)
        1.f, 3.f,
        4.f, 3.f,

        // Batch 0, Height 1, Width (2) x Channel (2)
        4.f, 4.f,
        2.f, 3.f,

        // Batch 0, Height 2, Width (2) x Channel (2)
        1.f, 2.f,
        6.f, 4.f
    };

    return BatchNormTestImpl<armnn::DataType::QuantisedSymm16>
        (workloadFactory, memoryManager,
         inputOutputShape, inputValues, expectedOutputValues,
         1.f/20.f, 50, armnn::DataLayout::NHWC);
}

LayerTestResult<uint8_t, 4> ConstantUint8CustomQuantizationScaleAndOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 2e-6f, 1);
}

LayerTestResult<int16_t, 4> ConstantInt16CustomQuantizationScaleAndOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, 2e-6f, 1);
}

LayerTestResult<uint8_t, 1> Concatenation1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation1dTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim0TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim1TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim0DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim0DiffInputDimsTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim1DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim1DiffInputDimsTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim0TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim1TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation3dDim2TestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, useSubtensor, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim0DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim0TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim1DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim1DiffInputDimsTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim2DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation3dDim2DiffInputDimsTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, useSubtensor, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim0TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim1TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim2TestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager, bool useSubtensor)
{
    return Concatenation4dDim3TestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1, useSubtensor);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim0TestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim1TestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim2TestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation4dDiffShapeDim3TestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5f, -1, useSubtensor);
}

LayerTestResult<float, 4> SimpleMaxPooling2dSize2x2Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize2x2Stride2x2TestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize2x2Stride2x2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize2x2Stride2x2TestCommon<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, forceNoPadding, 3.0f, -5);
}

LayerTestResult<int16_t, 4> SimpleMaxPooling2dSize2x2Stride2x2Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize2x2Stride2x2TestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<float, 4> SimpleMaxPooling2dSize3x3Stride2x4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize3x3Stride2x4TestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize3x3Stride2x4Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize3x3Stride2x4TestCommon<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, forceNoPadding, 0.1f, 128);
}

LayerTestResult<int16_t, 4> SimpleMaxPooling2dSize3x3Stride2x4Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize3x3Stride2x4TestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<float, 4> SimpleMaxPooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<int16_t, 4> SimpleMaxPooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling2dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, dataLayout);
}
LayerTestResult<float, 4> IgnorePaddingSimpleMaxPooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleMaxPooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleMaxPooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleMaxPooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(
            workloadFactory, memoryManager, 1.0f, -5);
}

LayerTestResult<int16_t, 4> IgnorePaddingSimpleMaxPooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleMaxPooling2dTestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingMaxPooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingMaxPooling2dSize3TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingMaxPooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingMaxPooling2dSize3TestCommon<armnn::DataType::QuantisedAsymm8>(
            workloadFactory, memoryManager, 1.0f, -5);
}

LayerTestResult<int16_t, 4> IgnorePaddingMaxPooling2dSize3Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingMaxPooling2dSize3TestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SimpleAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<uint8_t, 4> SimpleAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, dataLayout, 0.5, -1);
}

LayerTestResult<int16_t, 4> SimpleAveragePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling2dTestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<float, 4> IgnorePaddingAveragePooling2dSize3x2Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return IgnorePaddingAveragePooling2dSize3x2Stride2x2TestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<float, 4> LargeTensorsAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return LargeTensorsAveragePooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> LargeTensorsAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return LargeTensorsAveragePooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, 0.5, -1);
}

LayerTestResult<int16_t, 4> LargeTensorsAveragePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return LargeTensorsAveragePooling2dTestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager);
}
LayerTestResult<float, 4> IgnorePaddingSimpleAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(
            workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> IgnorePaddingSimpleAveragePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dTestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dNoPaddingTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dNoPaddingTestCommon<armnn::DataType::QuantisedAsymm8>(
            workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dNoPaddingTestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingAveragePooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingAveragePooling2dSize3TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingAveragePooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingAveragePooling2dSize3TestCommon<armnn::DataType::QuantisedAsymm8>(
            workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> IgnorePaddingAveragePooling2dSize3Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingAveragePooling2dSize3TestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SimpleL2Pooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<uint8_t, 4> SimpleL2Pooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<int16_t, 4> SimpleL2Pooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling2dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<float, 4> L2Pooling2dSize3Stride1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride1TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride1TestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> L2Pooling2dSize3Stride1Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride1TestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> L2Pooling2dSize3Stride3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride3TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride3TestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> L2Pooling2dSize3Stride3Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride3TestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}
LayerTestResult<float, 4> L2Pooling2dSize3Stride4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride4TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride4Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride4TestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> L2Pooling2dSize3Stride4Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride4TestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> L2Pooling2dSize7Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize7TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize7Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize7TestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> L2Pooling2dSize7Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize7TestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> L2Pooling2dSize9Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize9TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize9Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize9TestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> L2Pooling2dSize9Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize9TestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}
LayerTestResult<float, 4> IgnorePaddingSimpleL2Pooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleL2Pooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleL2Pooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleL2Pooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> IgnorePaddingSimpleL2Pooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleL2Pooling2dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingL2Pooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingL2Pooling2dSize3TestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingL2Pooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingL2Pooling2dSize3TestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> IgnorePaddingL2Pooling2dSize3Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingL2Pooling2dSize3TestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> AsymmetricNonSquarePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AsymmetricNonSquarePooling2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> AsymmetricNonSquarePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AsymmetricNonSquarePooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> AsymmetricNonSquarePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AsymmetricNonSquarePooling2dTestCommon<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> ComparePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType)
{
    return ComparePooling2dTestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager, refWorkloadFactory, poolingType);
}

LayerTestResult<uint8_t, 4> ComparePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType)
{
    return ComparePooling2dTestCommon<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager, refWorkloadFactory, poolingType, 0.1f, 128);
}

LayerTestResult<int16_t, 4> ComparePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType)
{
    return ComparePooling2dTestCommon<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager, refWorkloadFactory, poolingType);
}

LayerTestResult<float, 2> FullyConnectedLargeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool transposeWeights)
{
    return FullyConnectedLargeTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, transposeWeights);
}

LayerTestResult<float, 4> AdditionAfterMaxPoolTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // Create Initial Tensor
    // 1, 2, 3
    // 4, 5, 6
    // 7, 8, 9

    armnn::TensorInfo poolingInputTensorInfo({ 1, 1, 3, 3}, armnn::DataType::Float32);
    armnn::TensorInfo poolingOutputTensorInfo({ 1, 1, 2, 2}, armnn::DataType::Float32);

    boost::multi_array<float, 4> poolingInput = MakeTensor<float,4>(poolingInputTensorInfo,
                                                            {1, 2, 3,
                                                             4, 5, 6,
                                                             7, 8, 9
                                                            });

    std::unique_ptr<armnn::ITensorHandle> poolingInputHandle =
            workloadFactory.CreateTensorHandle(poolingInputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> poolingOutputHandle =
            workloadFactory.CreateTensorHandle(poolingOutputTensorInfo);

    // Apply MaxPool poolSize = 1x1, stride=2x2
    // Result =
    // 1, 3
    // 7, 9
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolHeight = 1;
    descriptor.m_PoolWidth = 1;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 2;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;

    armnn::Pooling2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, poolingInputTensorInfo, poolingInputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, poolingOutputTensorInfo, poolingOutputHandle.get());

    // Create the MaxPool
    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePooling2d(queueDescriptor, workloadInfo);

    //LayerTestResult<float, 4> result(poolingOutputTensorInfo);
    auto shape( GetTensorShapeAsArray<4>(poolingOutputTensorInfo));
    boost::multi_array<float, 4> resultMaxPool;
    resultMaxPool.resize(shape);


    // Create addition with another tensor the same size
    // This would be the result to apply a Conv2d with kernel ones(2) and stride 1x1
    // with the initial tensor.
    // 12, 16
    // 24, 28

    armnn::TensorInfo addInputTensorInfo({ 1,1,2,2}, armnn::DataType::Float32);
    armnn::TensorInfo addOutputTensorInfo({ 1,1,2,2}, armnn::DataType::Float32);

    boost::multi_array<float, 4> addInput = MakeTensor<float,4>(addInputTensorInfo,
                                                                    {12, 16,
                                                                     24, 28,
                                                                    });

    // Expected output tensor after MaxPool and Addition.
    LayerTestResult<float,4> addRet(addOutputTensorInfo);
    addRet.outputExpected = MakeTensor<float, 4>(addOutputTensorInfo, std::vector<float>(
            {
                    13, 19,
                    31, 37
            }));

    std::unique_ptr<armnn::ITensorHandle> addInputHandle = workloadFactory.CreateTensorHandle(addInputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> addOutputHandle = workloadFactory.CreateTensorHandle(addOutputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;

    // Add the output of the MaxPool and the new tensor
    AddInputToWorkload(data, info, poolingOutputTensorInfo, poolingOutputHandle.get());
    AddInputToWorkload(data, info, addInputTensorInfo, addInputHandle.get());
    AddOutputToWorkload(data, info, addOutputTensorInfo, addOutputHandle.get());

    std::unique_ptr<armnn::IWorkload> addWorkload = workloadFactory.CreateAddition(data, info);

    poolingInputHandle->Allocate();
    poolingOutputHandle->Allocate();
    addInputHandle->Allocate();
    addOutputHandle->Allocate();

    CopyDataToITensorHandle(poolingInputHandle.get(), &poolingInput[0][0][0][0]);
    CopyDataFromITensorHandle(&resultMaxPool[0][0][0][0], poolingOutputHandle.get());

    CopyDataToITensorHandle(poolingOutputHandle.get(), &resultMaxPool[0][0][0][0]);
    CopyDataToITensorHandle(addInputHandle.get(), &addInput[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();
    addWorkload->PostAllocationConfigure();
    addWorkload->Execute();

    CopyDataFromITensorHandle(&addRet.output[0][0][0][0], addOutputHandle.get());

    return addRet;
}

LayerTestResult<float, 4> SpaceToBatchNdSimpleFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiChannelsFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiBlockFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdPaddingFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiChannelsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiBlockUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdPaddingUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdSimpleNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleNHWCTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiChannelsNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsNHWCTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiBlockNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockNHWCTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdPaddingNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingNHWCTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdSimpleNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleNHWCTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiChannelsNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsNHWCTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiBlockNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockNHWCTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdPaddingNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingNHWCTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdSimpleUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiChannelsUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiBlockUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdPaddingUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdSimpleNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleNHWCTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiChannelsNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsNHWCTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiBlockNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockNHWCTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToBatchNdPaddingNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingNHWCTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToDepthNHWCAsymmQ8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToDepthNCHWAsymmQ8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        armnn::DataLayout::NCHW);
}

LayerTestResult<float, 4> SpaceToDepthNHWCFloat32Test1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager);
}

LayerTestResult<float, 4> SpaceToDepthNCHWFloat32Test1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        armnn::DataLayout::NCHW);
}

LayerTestResult<float, 4> SpaceToDepthNHWCFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager);
}

LayerTestResult<float, 4> SpaceToDepthNCHWFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        armnn::DataLayout::NCHW);
}

LayerTestResult<int16_t, 4> SpaceToDepthNHWCQSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager);
}

LayerTestResult<int16_t, 4> SpaceToDepthNCHWQSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::QuantisedSymm16>(
        workloadFactory,
        memoryManager,
        armnn::DataLayout::NCHW);
}

namespace {

} // anonymous namespace

LayerTestResult<float, 4> StridedSlice4DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSlice4DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DReverseTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSliceSimpleStrideFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSliceSimpleRangeMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> StridedSlice3DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> StridedSlice3DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DReverseTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSlice2DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSlice2DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DReverseTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSlice4DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSlice4DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DReverseTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleStrideUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleRangeMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> StridedSlice3DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> StridedSlice3DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DReverseTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSlice2DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSlice2DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DReverseTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSlice4DInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSlice4DReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DReverseTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSliceSimpleStrideInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSliceSimpleRangeMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> StridedSliceShrinkAxisMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 3> StridedSlice3DInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 3> StridedSlice3DReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DReverseTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> StridedSlice2DInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> StridedSlice2DReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DReverseTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> Debug4DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug4DTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> Debug3DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug3DTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> Debug2DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug2DTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 1> Debug1DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug1DTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> Debug4DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug4DTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> Debug3DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug3DTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> Debug2DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug2DTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 1> Debug1DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug1DTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 1> Gather1DParamsFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Gather1DParamsTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 1> Gather1DParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Gather1DParamsTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 1> Gather1DParamsInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Gather1DParamsTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> GatherMultiDimParamsFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> GatherMultiDimParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> GatherMultiDimParamsInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> GatherMultiDimParamsMultiDimIndicesFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsMultiDimIndicesTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> GatherMultiDimParamsMultiDimIndicesUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsMultiDimIndicesTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> GatherMultiDimParamsMultiDimIndicesInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsMultiDimIndicesTestImpl<armnn::DataType::QuantisedSymm16>(
            workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeOffsetUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeOffsetTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeSimpleInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> QuantizeSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return QuantizeSimpleTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> QuantizeClampUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return QuantizeClampTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> QuantizeClampInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return QuantizeClampTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}
