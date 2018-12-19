//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "LayerTests.hpp"
#include "WorkloadTestUtils.hpp"
#include "TensorUtils.hpp"

#include "test/TensorHelpers.hpp"
#include "TensorCopyUtils.hpp"
#include "Permute.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/assert.hpp>

#include <armnn/LayerSupport.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <reference/workloads/RefWorkloads.hpp>

#include <algorithm>
#include <boost/cast.hpp>

#include "WorkloadTestUtils.hpp"
#include "Conv2dTestImpl.hpp"
#include "BatchNormTestImpl.hpp"
#include "ActivationTestImpl.hpp"
#include "Pooling2dTestImpl.hpp"
#include "ReshapeTestImpl.hpp"
#include "FullyConnectedTestImpl.hpp"
#include "SpaceToBatchNdTestImpl.hpp"
#include "SplitterTestImpl.hpp"
#include "SoftmaxTestImpl.hpp"
#include "StridedSliceTestImpl.hpp"
#include "NormTestImpl.hpp"
#include "PermuteTestImpl.hpp"
#include "LstmTestImpl.hpp"
#include "ConvertFp16ToFp32TestImpl.hpp"
#include "ConvertFp32ToFp16TestImpl.hpp"
#include "DebugTestImpl.hpp"

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

// Helper function that returns either Bias2 or an empty vector depending on whether bias is enabled.
template<typename T>
boost::multi_array<T, 1> GetBias2(bool biasEnabled, float qScale, int32_t qOffset)
{
    if(biasEnabled)
    {
        armnn::TensorInfo biasDesc({static_cast<unsigned int>(Bias2.size())}, armnn::GetDataType<T>());
        boost::multi_array<T, 1> bias = MakeTensor<T, 1>(biasDesc, QuantizedVector<T>(qScale, qOffset, Bias2));
        return bias;
    }
    else
    {
        return boost::multi_array<T, 1>();
    }
}

template<typename T>
LayerTestResult<T, 4> SimpleConvolution2d3x5TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use common single-batch 3-channel 16x8 image.
    armnn::TensorInfo inputDesc({1, 3, 8, 16}, armnn::GetDataType<T>());
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, QuantizedVector<T>(qScale, qOffset, ConvInput3x8x16));

    // Use a 2-element batch with 3-channel 3x5 kernels.
    armnn::TensorInfo kernelDesc({2, 3, 5, 3}, armnn::GetDataType<T>());
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
    armnn::TensorInfo outputDesc({1, 2, 4, 14}, armnn::GetDataType<T>());
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

    return SimpleConvolution2dTestImpl<T>(workloadFactory,
      memoryManager,
      input,
      kernel,
      GetBias2<typename FullyConnectedBiasTypeForInputType<T>::Type>(biasEnabled, qScale, qOffset),
      expectedOutput,
      qScale,
      qOffset,
      layout);
}

template<typename T>
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
    armnn::TensorInfo inputDesc({1, 3, 8, 16}, armnn::GetDataType<T>());
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, QuantizedVector<T>(qScale, qOffset, ConvInput3x8x16));

    // Use a 2-element batch of 3-channel 3x3 kernels.
    armnn::TensorInfo kernelDesc({2, 3, 3, 3}, armnn::GetDataType<T>());
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
    armnn::TensorInfo outputDesc({1, 2, 6, 14}, armnn::GetDataType<T>());
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

    return SimpleConvolution2dTestImpl<T>(workloadFactory,
      memoryManager,
      input,
      kernel,
      GetBias2<typename FullyConnectedBiasTypeForInputType<T>::Type>(biasEnabled, qScale, qOffset),
      expectedOutput,
      qScale,
      qOffset,
      layout);
}

template<typename T>
LayerTestResult<T, 4> SimpleConvolution2d3x3NhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    armnn::DataLayout dataLayout)
{
    // Use common single-batch 5x5 image.

    armnn::TensorInfo inputDesc({1, 3, 4, 1}, armnn::GetDataType<T>());
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc,
                                                      {
                                                       1, 5, 2, 3,
                                                       8, 7, 3, 6,
                                                       3, 3, 9, 1
                                                       });


    // Use a 2-element batch of 3-channel 3x3 kernels.
    armnn::TensorInfo kernelDesc({1, 3, 3, 1}, armnn::GetDataType<T>());
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc, {
                                                                    4, 5, 6,
                                                                    0, 0, 0,
                                                                    3, 2, 1
                                                                    });

    // Expected output is 1 batch of a 5x5 image.
    armnn::TensorInfo outputDesc({1, 3, 4, 1}, armnn::GetDataType<T>());

    const std::vector<float> outputData =
            {
                    23, 41, 33, 21,
                    44, 65, 76, 52,
                    82, 85, 79, 42
            };

    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, outputData);

    return SimpleConvolution2dNhwcTestImpl<T>(workloadFactory,
                                              memoryManager,
                                              input,
                                              kernel,
                                              boost::multi_array<T, 1>(),
                                              expectedOutput,
                                              dataLayout,
                                              qScale,
                                              qOffset);
}

LayerTestResult<float, 4> SimpleConvolution2d3x5Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x5TestCommon<float>(workloadFactory, memoryManager, 0.f, 0, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x5Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x5TestCommon<uint8_t>(workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<float>(workloadFactory, memoryManager, 0.f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return SimpleConvolution2d3x3NhwcTestCommon<float>(workloadFactory,
                                                       memoryManager,
                                                       0.f,
                                                       0,
                                                       biasEnabled,
                                                       armnn::DataLayout::NHWC);
}

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<uint8_t>(workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

template<typename T>
LayerTestResult<T, 4> Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout,
    float qScale,
    int32_t qOffset)
{
    // Use a single-batch 1-channel 3x3 image as input.
    armnn::TensorInfo inputDesc({1, 1, 3, 3}, armnn::GetDataType<T>());
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            11,21,31,
            12,22,32,
            13,23,33
        })));

    // Use 1 batch of a 1-channel 2x2 kernel.
    armnn::TensorInfo kernelDesc({1, 1, 2, 2}, armnn::GetDataType<T>());
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
    armnn::TensorInfo outputDesc({1, 1, 8, 6}, armnn::GetDataType<T>());
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

    return SimpleConvolution2dTestImpl<T>(workloadFactory,
      memoryManager,
      input,
      kernel,
      GetBias2<typename FullyConnectedBiasTypeForInputType<T>::Type>(false, qScale, qOffset),
      expectedOutput,
      qScale,
      qOffset,
      layout,
      1,  // Padding left.
      2,  // Padding top.
      3,  // Padding right.
      4); // Padding bottom.
}

template<typename T>
LayerTestResult<T, 4> SimpleConvolution2dAsymmetricPaddingTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout,
    float qScale,
    int32_t qOffset)
{
    // Use a single-batch 1-channel 5x5 image as input.
    armnn::TensorInfo inputDesc({ 1, 1, 5, 5 }, armnn::GetDataType<T>());
    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            11,21,31,41,51,
            12,22,32,42,52,
            13,23,33,43,53,
            14,24,34,44,54,
            15,25,35,45,55,
        })));

    // Use 1 batch of a 1-channel 4x4 kernel.
    armnn::TensorInfo kernelDesc({ 1, 1, 4, 4 }, armnn::GetDataType<T>());
    boost::multi_array<T, 4> kernel = MakeTensor<T, 4>(kernelDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            -11,-21,-31,-41,
            -12,-22,-32,-42,
            -13,-23,-33,-43,
            -14,-24,-34,-44,
        })));

    // Expected output is 1 batch of a 1-channel 5x5 image.
    armnn::TensorInfo outputDesc({ 1, 1, 5, 5 }, armnn::GetDataType<T>());
    std::vector<T> myVec(outputDesc.GetNumElements(), 0);
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputDesc, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            -7140, -10580, -13940,  -9300, -5230,
            -9590, -14120, -18520, -12290, -6860,
            -9980, -14560, -18960, -12560, -7000,
            -7518, -10904, -14144,  -9318, -5152,
            -5032,  -7256,  -9376,  -6142, -3368,
        })));

    return SimpleConvolution2dTestImpl<T>(workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<typename FullyConnectedBiasTypeForInputType<T>::Type>(false, qScale, qOffset),
        expectedOutput,
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2); // Padding bottom.
}

template<typename T>
LayerTestResult<T, 4> DepthwiseConvolution2dAsymmetricTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use a single-batch 2-channel 5x5 image as input.
    armnn::TensorInfo inputTensorInfo({ 1, 2, 5, 5 }, armnn::GetDataType<T>());
    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset(), {
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
    armnn::TensorInfo kernelTensorInfo({ 1, 2, 4, 4 }, armnn::GetDataType<T>());
    auto kernel = MakeTensor<T, 4>(kernelTensorInfo, std::vector<T>(
        QuantizedVector<T>(kernelTensorInfo.GetQuantizationScale(), kernelTensorInfo.GetQuantizationOffset(), {
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
    armnn::TensorInfo outputTensorInfo({ 1, 2, 5, 5 }, armnn::GetDataType<T>());
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputTensorInfo, std::vector<T>(
        QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(), {
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

    return DepthwiseConvolution2dAsymmetricTestImpl<T>(workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<typename FullyConnectedBiasTypeForInputType<T>::Type>(biasEnabled, qScale, qOffset),
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

template<typename T>
LayerTestResult<T, 4> DepthwiseConvolution2dNhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool biasEnabled)
{
    armnn::TensorInfo inputTensorInfo({ 1, 5, 5, 2}, armnn::GetDataType<T>());
    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset(), {
            0, 25,
            1, 26,
            2, 27,
            3, 28,
            4, 29,

            5, 30,
            6, 31,
            7, 32,
            8, 33,
            9, 34,

            10, 35,
            11, 36,
            12, 37,
            13, 38,
            14, 39,

            15, 40,
            16, 41,
            17, 42,
            18, 43,
            19, 44,

            20, 45,
            21, 46,
            22, 47,
            23, 48,
            24, 49
        })));

    armnn::TensorInfo kernelTensorInfo({ 1, 4, 4, 2}, armnn::GetDataType<T>());
    auto kernel = MakeTensor<T, 4>(kernelTensorInfo, std::vector<T>(
        QuantizedVector<T>(kernelTensorInfo.GetQuantizationScale(), kernelTensorInfo.GetQuantizationOffset(), {
             32, 16,
             31, 15,
             30, 14,
             29, 13,

             28, 12,
             27, 11,
             26, 10,
             25,  9,

             24,  8,
             23,  7,
             22,  6,
             21,  5,

             20,  4,
             19,  3,
             18,  2,
             17,  1
        })));

    armnn::TensorInfo outputTensorInfo({ 1, 5, 5, 2}, armnn::GetDataType<T>());
    boost::multi_array<T, 4> expectedOutput = MakeTensor<T, 4>(outputTensorInfo, std::vector<T>(
        QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(), {
        1062, 1550,
        1580, 2284,
        1850, 2362,
        1530, 1955,
        1117, 1428,

        2140, 2910,
        3108, 4206,
        3500, 4342,
        2842, 3528,
        2042, 2536,

        3580, 3390,
        5068, 4886,
        5460, 5022,
        4342, 4068,
        3062, 2916,

        3618, 3566,
        5072, 5056,
        5390, 5182,
        4248, 4133,
        2971, 2922,

        3074, 3100,
        4282, 4352,
        4510, 4452,
        3533, 3517,
        2457, 2465
        })));

    return DepthwiseConvolution2dNhwcTestImpl<T>(workloadFactory,
        memoryManager,
        input,
        kernel,
        GetBias2<typename FullyConnectedBiasTypeForInputType<T>::Type>(biasEnabled, qScale, qOffset),
        expectedOutput,
        qScale,
        qOffset,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2,  // Padding bottom.
        1,  // strideX
        1);  // strideY
}

LayerTestResult<float, 4>
Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTestCommon<float>(
         workloadFactory, memoryManager, layout, 0.0f, 0);
}

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2dAsymmetricPaddingTestCommon<float>(
        workloadFactory, memoryManager, layout, 0.0f, 0);
}

LayerTestResult<float, 4> DepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<float, float>(
        workloadFactory, memoryManager, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return DepthwiseConvolution2dNhwcTestCommon<float>(workloadFactory, memoryManager, 0.0f, 0, biasEnabled);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<float, float>(
        workloadFactory, memoryManager, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> DepthwiseConvolution2dAsymmetricTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dAsymmetricTestCommon<float>(
        workloadFactory, memoryManager, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<uint8_t, int32_t>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dDepthMul1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<uint8_t, int32_t>(
        workloadFactory, memoryManager, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> Convolution1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return Convolution1dTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0, biasEnabled);
}

LayerTestResult<uint8_t, 4> Convolution1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled)
{
    return Convolution1dTestImpl<uint8_t>(workloadFactory, memoryManager, 0.1f, 128, biasEnabled);
}

LayerTestResult<float,4> CompareConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory)
{
    return CompareConvolution2dTestImpl<float>(workloadFactory, memoryManager, refWorkloadFactory);
}

template<typename T>
LayerTestResult<T,4> CompareDepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::DataLayout layout)
{
    return CompareDepthwiseConvolution2dTestImpl<T>(workloadFactory, memoryManager, refWorkloadFactory, layout);
}

template LayerTestResult<float, 4> CompareDepthwiseConvolution2dTest<float>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    armnn::IWorkloadFactory&,
    const armnn::DataLayout);

template LayerTestResult<uint8_t, 4> CompareDepthwiseConvolution2dTest<uint8_t>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    armnn::IWorkloadFactory&,
    const armnn::DataLayout);

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
    return SimpleSoftmaxTestImpl<float>(workloadFactory, memoryManager, beta);
}

LayerTestResult<uint8_t,2> SimpleSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta)
{
    return SimpleSoftmaxTestImpl<uint8_t>(workloadFactory, memoryManager, beta);
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
    return CompareSoftmaxTestImpl<float>(workloadFactory, memoryManager, refWorkloadFactory, beta);
}

LayerTestResult<uint8_t,2> CompareSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float beta)
{
    return CompareSoftmaxTestImpl<uint8_t>(workloadFactory, memoryManager, refWorkloadFactory, beta);
}

std::vector<LayerTestResult<float,3>> SplitterTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SplitterTestCommon<float>(workloadFactory, memoryManager);
}

std::vector<LayerTestResult<uint8_t,3>> SplitterUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SplitterTestCommon<uint8_t>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<float, 3> CopyViaSplitterTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return CopyViaSplitterTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<uint8_t, 3> CopyViaSplitterUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return CopyViaSplitterTestImpl<uint8_t>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<float, 2> LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({ 2, 2 }, armnn::GetDataType<float>());
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            { 2., 3., 3., 4. }));

    armnn::TensorInfo outputDesc({ 2, 4 }, armnn::GetDataType<float>());
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(outputDesc, std::vector<float>(
            {-0.36444446f, -0.00352185f, 0.12886585f, -0.05163646f,
             -0.42734814f, -0.00478661f,  0.13455015f, -0.03560682f}));
    return LstmLayerWithCifgWithPeepholeNoProjectionTestImpl(
        workloadFactory, memoryManager, input, expectedOutput);
}

LayerTestResult<float, 2> LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({ 2, 5 }, armnn::GetDataType<float>());
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            {0.787926f, 0.151646f, 0.071352f, 0.118426f, 0.458058f,
             0.295743f, 0.544053f, 0.690064f, 0.858138f, 0.497181f}));

    armnn::TensorInfo outputDesc({ 2, 16 }, armnn::GetDataType<float>());
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(outputDesc, std::vector<float>(
            {-0.00396806f, 0.029352f,     -0.00279226f, 0.0159977f,   -0.00835576f,
             -0.0211779f,  0.0283512f,    -0.0114597f,  0.00907307f,  -0.0244004f,
             -0.0152191f,  -0.0259063f,   0.00914318f,  0.00415118f,  0.017147f,
             0.0134203f, -0.013869f,    0.0287268f,   -0.00334693f, 0.00733398f,  -0.0287926f,
             -0.0186926f,   0.0193662f,   -0.0115437f,  0.00422612f,  -0.0345232f,
             0.00223253f,   -0.00957321f, 0.0210624f,   0.013331f,    0.0150954f,
             0.02168f}));
    return LstmLayerNoCifgWithPeepholeWithProjectionTestImpl(workloadFactory, memoryManager, input, expectedOutput);
}

LayerTestResult<float, 2> LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputDesc({2, 2}, armnn::GetDataType<float>());
    boost::multi_array<float, 2> input = MakeTensor<float, 2>(inputDesc, std::vector<float>(
            {2., 3., 3., 4.}));


    armnn::TensorInfo outputDesc({2, 4}, armnn::GetDataType<float>());
    boost::multi_array<float, 2> expectedOutput = MakeTensor<float, 2>(outputDesc, std::vector<float>(
            {{-0.02973187f, 0.1229473f,   0.20885126f, -0.15358765f,
              -0.0185422f,   0.11281417f,  0.24466537f, -0.1826292f}}));

    return LstmNoCifgNoPeepholeNoProjectionTestImpl(
        workloadFactory, memoryManager, input, expectedOutput);
}

LayerTestResult<float,3> MergerTest(
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
    armnn::MergerQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = {2, 0, 0}; //Extent of the window is defined by size of input[1].
    armnn::MergerQueueDescriptor::ViewOrigin window2(wOrigin2);

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

    armnn::MergerQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMerger(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0]);

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

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template <typename T>
LayerTestResult<T, 4> AdditionBroadcastTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo1 = armnn::TensorInfo({1, 3, 2, 1}, armnn::GetDataType<T>());
    armnn::TensorInfo inputTensorInfo2 = armnn::TensorInfo({1, 1, 2, 3}, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 2, 3}, armnn::GetDataType<T>());

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

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template <typename T>
LayerTestResult<T, 4> AdditionBroadcast1ElementTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo1 = armnn::TensorInfo({1, 3, 2, 3}, armnn::GetDataType<T>());
    armnn::TensorInfo inputTensorInfo2 = armnn::TensorInfo({1, 1, 1, 1}, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 2, 3}, armnn::GetDataType<T>());

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

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<float, 4> AdditionBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcastTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<uint8_t, 4> AdditionBroadcastUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcastTestImpl<uint8_t>(workloadFactory, memoryManager, 2.f, 0);
}

LayerTestResult<float, 4> AdditionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcast1ElementTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<uint8_t, 4> AdditionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AdditionBroadcast1ElementTestImpl<uint8_t>(workloadFactory, memoryManager, 0.1333333f, 128);
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

    workload->Execute();
    workloadRef->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());
    CopyDataFromITensorHandle(&ret.outputExpected[0][0][0][0], outputHandleRef.get());

    return ret;
}

namespace {
template <typename T>
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
    auto dataType = (std::is_same<T, uint8_t>::value ?
                     armnn::DataType::QuantisedAsymm8 :
                     armnn::DataType::Float32);

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

    return DivisionTestHelper<float>(workloadFactory,
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


    return DivisionTestHelper<float>(workloadFactory,
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


    return DivisionTestHelper<float>(workloadFactory,
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

    return DivisionTestHelper<float>(workloadFactory,
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


    return DivisionTestHelper<uint8_t>(workloadFactory,
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

    return DivisionTestHelper<uint8_t>(workloadFactory,
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

    return DivisionTestHelper<uint8_t>(workloadFactory,
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
    template <typename Descriptor, typename dataType>
    LayerTestResult<dataType, 4> ElementwiseTestHelper
        (armnn::IWorkloadFactory & workloadFactory,
         const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
         const unsigned int shape0[4], std::vector<dataType> values0,
         const unsigned int shape1[4], std::vector<dataType> values1,
         const unsigned int outShape[4], std::vector<dataType> outValues,
         float qScale = 0.0f, int qOffset = 0)
    {
        const size_t dimensionCount = 4;
        armnn::TensorInfo inputTensorInfo0{dimensionCount, shape0, armnn::GetDataType<dataType>()};
        armnn::TensorInfo inputTensorInfo1{dimensionCount, shape1, armnn::GetDataType<dataType>()};
        armnn::TensorInfo outputTensorInfo{dimensionCount, outShape, armnn::GetDataType<dataType>()};

        auto input0 = MakeTensor<dataType, 4>(inputTensorInfo0, values0);
        auto input1 = MakeTensor<dataType, 4>(inputTensorInfo1, values1);

        if (armnn::IsQuantizedType<dataType>())
        {
            inputTensorInfo0.SetQuantizationScale(qScale);
            inputTensorInfo0.SetQuantizationOffset(qOffset);

            inputTensorInfo1.SetQuantizationScale(qScale);
            inputTensorInfo1.SetQuantizationOffset(qOffset);

            outputTensorInfo.SetQuantizationScale(qScale);
            outputTensorInfo.SetQuantizationOffset(qOffset);
        }

        LayerTestResult<dataType,4> ret(outputTensorInfo);

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

        ExecuteWorkload(*workload, memoryManager);

        CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

        ret.outputExpected = MakeTensor<dataType, 4>(outputTensorInfo, outValues);
        return ret;
    }
}

LayerTestResult<float, 4> EqualSimpleTest(armnn::IWorkloadFactory& workloadFactory,
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

    std::vector<float> output({ 1, 1, 1, 1,  0, 0, 0, 0,
                                0, 0, 0, 0,  1, 1, 1, 1 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, float>
            (workloadFactory,
             memoryManager,
             shape,
             input0,
             shape,
             input1,
             shape,
             output);
}

LayerTestResult<float, 4> EqualBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 2, 3, 4, 5, 6, 7, 8});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 1 });

    std::vector<float> output({ 1, 0, 0, 0, 0, 0, 0, 0});

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, float>
            (workloadFactory,
             memoryManager,
             shape0,
             input0,
             shape1,
             input1,
             shape0,
             output);
}

LayerTestResult<float, 4> EqualBroadcast1DVectorTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0({ 1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12 });

    std::vector<float> input1({ 1, 2, 3});

    std::vector<float> output({ 1, 1, 1, 0, 0, 0,
                                0, 0, 0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, float>
            (workloadFactory,
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
                                  3, 3, 3, 3, 5, 5, 5, 5 });

    std::vector<uint8_t> input1({ 2, 2, 2, 2, 6, 6, 6, 6,
                                  3, 3, 3, 3, 5, 5, 5, 5 });

    std::vector<uint8_t> output({ 0, 0, 0, 0, 1, 1, 1, 1,
                                  1, 1, 1, 1, 0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, uint8_t >
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, uint8_t >
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::EqualQueueDescriptor, uint8_t>
            (workloadFactory,
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

LayerTestResult<float, 4> GreaterSimpleTest(armnn::IWorkloadFactory& workloadFactory,
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

    std::vector<float> output({ 0, 0, 0, 0,  1, 1, 1, 1,
                                0, 0, 0, 0,  0, 0, 0, 0 });

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, float>
            (workloadFactory,
             memoryManager,
             shape,
             input0,
             shape,
             input1,
             shape,
             output);
}

LayerTestResult<float, 4> GreaterBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    std::vector<float> input0({ 1, 2, 3, 4, 5, 6, 7, 8});

    unsigned int shape1[] = { 1, 1, 1, 1 };
    std::vector<float> input1({ 1 });

    std::vector<float> output({ 0, 1, 1, 1, 1, 1, 1, 1});

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, float>
            (workloadFactory,
             memoryManager,
             shape0,
             input0,
             shape1,
             input1,
             shape0,
             output);
}

LayerTestResult<float, 4> GreaterBroadcast1DVectorTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0({ 1, 2.9f, 2.1f, 4, 5, 6,
                                7, 8, 9, 10, 11, 12 });

    std::vector<float> input1({ 1, 3, 2});

    std::vector<float> output({ 0, 0, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1 });

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, float>
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, uint8_t >
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, uint8_t >
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::GreaterQueueDescriptor, uint8_t>
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, float>
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, float>
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, float>
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, uint8_t >
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, uint8_t >
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::MaximumQueueDescriptor, uint8_t>
            (workloadFactory,
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

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, float>(workloadFactory,
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

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, float>(workloadFactory,
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

    return ElementwiseTestHelper<armnn::MinimumQueueDescriptor, uint8_t>(workloadFactory,
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
LayerTestResult<float,4> MultiplicationTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int shape0[4],
    const std::vector<float> & values0,
    const unsigned int shape1[4],
    const std::vector<float> & values1,
    const unsigned int outShape[4],
    const std::vector<float> & outValues)
{
    const size_t dimensionCount = 4;
    armnn::TensorInfo inputTensorInfo0{dimensionCount, shape0, armnn::DataType::Float32};
    armnn::TensorInfo inputTensorInfo1{dimensionCount, shape1, armnn::DataType::Float32};
    armnn::TensorInfo outputTensorInfo{dimensionCount, outShape, armnn::DataType::Float32};

    auto input0 = MakeTensor<float, 4>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<float, 4>(inputTensorInfo1, values1);

    LayerTestResult<float,4> ret(outputTensorInfo);

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

    CopyDataToITensorHandle(inputHandle0.get(), &input0[0][0][0][0]);
    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    ret.outputExpected = MakeTensor<float, 4>(outputTensorInfo, outValues);
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

    return MultiplicationTestHelper(workloadFactory,
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

    return MultiplicationTestHelper(workloadFactory,
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

    return MultiplicationTestHelper(workloadFactory,
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

    workload->Execute();
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

    workload->Execute();
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

    workload->Execute();

    outputData.resize(outputTensorInfo.GetNumElements());
    CopyDataFromITensorHandle(&outputData[0], outputHandle.get());
    inputTensorInfo = outputTensorInfo;
}

armnn::OriginsDescriptor CreateMergerDescriptorForConcatenation(
        const std::vector<armnn::TensorInfo> & inputTensorInfos,
        unsigned int concatDim)
{
    std::vector<armnn::TensorShape> shapes;
    shapes.reserve(inputTensorInfos.size());
    for (const armnn::TensorInfo& it: inputTensorInfos)
    {
        shapes.push_back(it.GetShape());
    }

    return armnn::CreateMergerDescriptorForConcatenation(shapes.begin(),
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

    armnn::MergerQueueDescriptor queueDescriptor;
    armnn::OriginsDescriptor viewsDescriptor = CreateMergerDescriptorForConcatenation(inputTensorInfos, concatDim);
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

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMerger(queueDescriptor, workloadInfo);

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

template <typename T>
LayerTestResult<T, 1> Concatenation1dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 3 }, armnn::GetDataType<T>());

    auto input0 = MakeTensor<T, 1>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, { 1.0f, 2.0f, 3.0f }));
    auto input1 = MakeTensor<T, 1>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, { 4.0f, 5.0f, 6.0f }));
    auto input2 = MakeTensor<T, 1>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, { 7.0f, 8.0f, 9.0f }));

    armnn::TensorInfo outputTensorInfo({ 9 }, armnn::GetDataType<T>());

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
    return Concatenation1dTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 2> Concatenation2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& outputTensorInfo,
    unsigned int dimension,
    const float qScale,
    const int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 2, 3 }, armnn::GetDataType<T>());

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

template <typename T>
LayerTestResult<T, 2> Concatenation2dDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 6, 3 }, armnn::GetDataType<T>());

    LayerTestResult<T, 2> result =
        Concatenation2dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 0, qScale, qOffset);
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
    return Concatenation2dDim0TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 2> Concatenation2dDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 2, 9 }, armnn::GetDataType<T>());

    LayerTestResult<T, 2> result =
        Concatenation2dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 1, qScale, qOffset);
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
    return Concatenation2dDim1TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 2> Concatenation2dDim0DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3 }, armnn::GetDataType<T>());
    auto input0 = MakeTensor<T, 2>(input0TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f,
    }));

    armnn::TensorInfo input1TensorInfo({ 3, 3 }, armnn::GetDataType<T>());
    auto input1 = MakeTensor<T, 2>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        4.0f, 5.0f, 6.0f,

        // Batch 1
        13.0f, 14.0f, 15.0f,

        // Batch 0
        7.0f, 8.0f, 9.0f,
    }));

    armnn::TensorInfo input2TensorInfo({ 1, 3 }, armnn::GetDataType<T>());
    auto input2 = MakeTensor<T, 2>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 1
        16.0f, 17.0f, 18.0f,
    }));

    armnn::TensorInfo outputTensorInfo({ 6, 3 }, armnn::GetDataType<T>());
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
    return Concatenation2dDim0DiffInputDimsTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 2> Concatenation2dDim1DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3 }, armnn::GetDataType<T>());
    auto input0 = MakeTensor<T, 2>(input0TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        1.0f, 2.0f, 3.0f,

        // Batch 1
        10.0f, 11.0f, 12.0f,
    }));

    armnn::TensorInfo input1TensorInfo({ 2, 5 }, armnn::GetDataType<T>());
    auto input1 = MakeTensor<T, 2>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,

        // Batch 1
        13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
    }));

    armnn::TensorInfo input2TensorInfo({ 2, 1 }, armnn::GetDataType<T>());
    auto input2 = MakeTensor<T, 2>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0
        9.0f,

        // Batch 1
        18.0f
    }));

    armnn::TensorInfo outputTensorInfo({ 2, 9 }, armnn::GetDataType<T>());
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
    return Concatenation2dDim1DiffInputDimsTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 3> Concatenation3dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& outputTensorInfo,
    unsigned int dimension,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 2, 3, 2 }, armnn::GetDataType<T>());

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

template <typename T>
LayerTestResult<T, 3> Concatenation3dDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 6, 3, 2 }, armnn::GetDataType<T>());

    LayerTestResult<T, 3> result =
        Concatenation3dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 0, true, qScale, qOffset);
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
    return Concatenation3dDim0TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 3> Concatenation3dDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 2, 9, 2 }, armnn::GetDataType<T>());

    LayerTestResult<T, 3> result =
        Concatenation3dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 1, true, qScale, qOffset);

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
    return Concatenation3dDim1TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 3> Concatenation3dDim2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 2, 3, 6 }, armnn::GetDataType<T>());

    LayerTestResult<T, 3> result =
        Concatenation3dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 2, useSubtensor, qScale, qOffset);

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
    return Concatenation3dDim2TestImpl<float>(workloadFactory, memoryManager, useSubtensor, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 3> Concatenation3dDim0DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3, 2 }, armnn::GetDataType<T>());
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

    armnn::TensorInfo input1TensorInfo({ 1, 3, 2 }, armnn::GetDataType<T>());
    auto input1 = MakeTensor<T, 3>(input1TensorInfo, QuantizedVector<T>(qScale, qOffset, {
            // Batch 0, Channel 0
            7.0f, 8.0f,

            // Batch 0, Channel 1
            9.0f, 10.0f,

            // Batch 0, Channel 2
            11.0f, 12.0f,
    }));

    armnn::TensorInfo input2TensorInfo({ 3, 3, 2 }, armnn::GetDataType<T>());
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

    armnn::TensorInfo outputTensorInfo({ 6, 3, 2 }, armnn::GetDataType<T>());
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
    return Concatenation3dDim0DiffInputDimsTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 3> Concatenation3dDim1DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3, 2 }, armnn::GetDataType<T>());
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

    armnn::TensorInfo input1TensorInfo({ 2, 4, 2 }, armnn::GetDataType<T>());
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

    armnn::TensorInfo input2TensorInfo({ 2, 1, 2 }, armnn::GetDataType<T>());
    auto input2 = MakeTensor<T, 3>(input2TensorInfo, QuantizedVector<T>(qScale, qOffset, {
        // Batch 0, Channel 0
        17.0f, 18.0f,

        // Batch 1, Channel 0
        31.0f, 32.0f,
    }));

    armnn::TensorInfo outputTensorInfo({ 2, 8, 2 }, armnn::GetDataType<T>());
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
    return Concatenation3dDim1DiffInputDimsTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 3> Concatenation3dDim2DiffInputDimsTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo input0TensorInfo({ 2, 3, 2 }, armnn::GetDataType<T>());
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

    armnn::TensorInfo input1TensorInfo({ 2, 3, 1 }, armnn::GetDataType<T>());
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

    armnn::TensorInfo input2TensorInfo({ 2, 3, 3 }, armnn::GetDataType<T>());
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

    armnn::TensorInfo outputTensorInfo({ 2, 3, 6 }, armnn::GetDataType<T>());
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
    return Concatenation3dDim2DiffInputDimsTestImpl<float>(workloadFactory, memoryManager, useSubtensor, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& outputTensorInfo,
    unsigned int dimension,
    bool useSubtensor,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo inputTensorInfo({ 1, 3, 2, 2 }, armnn::GetDataType<T>());

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

template <typename T>
LayerTestResult<T, 4> Concatenation4dDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 3, 3, 2, 2 }, armnn::GetDataType<T>());

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 0,
                                                              true, qScale, qOffset);
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
    return Concatenation4dDim0TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 1, 9, 2, 2 }, armnn::GetDataType<T>());

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 1,
                                                              true, qScale, qOffset);
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
    return Concatenation4dDim1TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dDim2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    armnn::TensorInfo outputTensorInfo({ 1, 3, 6, 2 }, armnn::GetDataType<T>());

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 2,
                                                              true, qScale, qOffset);
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
    return Concatenation4dDim2TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dDim3TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool useSubtensor)
{
    armnn::TensorInfo outputTensorInfo({ 1, 3, 2, 6 }, armnn::GetDataType<T>());

    LayerTestResult<T, 4> result = Concatenation4dTestImpl<T>(workloadFactory, memoryManager, outputTensorInfo, 3,
                                                              useSubtensor, qScale, qOffset);
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
    return Concatenation4dDim3TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0, useSubtensor);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim0TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    unsigned int dimension = 0;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, armnn::GetDataType<T>());

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 2, 3, 2, 2 }, armnn::GetDataType<T>());

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

    armnn::TensorInfo outputTensorInfo({ 3, 3, 2, 2 }, armnn::GetDataType<T>());

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
    return Concatenation4dDiffShapeDim0TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    unsigned int dimension = 1;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, armnn::GetDataType<T>());

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 1, 2, 2, 2 }, armnn::GetDataType<T>());

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset, {
        11.0f, 12.0f,
        13.0f, 14.0f,
        15.0f, 16.0f,
        17.0f, 18.0f,

    }));

    armnn::TensorInfo outputTensorInfo({ 1, 5, 2, 2 }, armnn::GetDataType<T>());

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
    return Concatenation4dDiffShapeDim1TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    unsigned int dimension = 2;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, armnn::GetDataType<T>());

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 1, 3, 3, 2 }, armnn::GetDataType<T>());

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

    armnn::TensorInfo outputTensorInfo({ 1, 3, 5, 2 }, armnn::GetDataType<T>());

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
    return Concatenation4dDiffShapeDim2TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

template <typename T>
LayerTestResult<T, 4> Concatenation4dDiffShapeDim3TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    bool useSubtensor)
{
    unsigned int dimension = 3;
    armnn::TensorInfo inputTensorInfo0({ 1, 3, 2, 2 }, armnn::GetDataType<T>());

    auto input0 = MakeTensor<T, 4>(inputTensorInfo0, QuantizedVector<T>(qScale, qOffset, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }));

    armnn::TensorInfo inputTensorInfo1({ 1, 3, 2, 3 }, armnn::GetDataType<T>());

    auto input1 = MakeTensor<T, 4>(inputTensorInfo1, QuantizedVector<T>(qScale, qOffset, {
        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,

        17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f,

        23.0f, 24.0f, 25.0f,
        26.0f, 27.0f, 28.0f
    }));

    armnn::TensorInfo outputTensorInfo({ 1, 3, 2, 5 }, armnn::GetDataType<T>());

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
    return Concatenation4dDiffShapeDim3TestImpl<float>(workloadFactory, memoryManager, 0.0f, 0, useSubtensor);
}

LayerTestResult<float, 4> ResizeBilinearNopTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    const armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 4, 4, dataLayout);
    const armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 4, 4, dataLayout);

    std::vector<float> inputData({
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,

        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    });

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;
    }

    auto input = MakeTensor<float, 4>(inputTensorInfo, inputData);

    LayerTestResult<float, 4> result(outputTensorInfo);
    result.outputExpected = input;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<float, 4> SimpleResizeBilinearTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    const armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 2, 2, dataLayout);
    const armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 1, 1, dataLayout);

    std::vector<float> inputData({
          1.0f, 255.0f,
        200.0f, 250.0f,

        250.0f, 200.0f,
        250.0f,   1.0f
    });

    // The 'resize bilinear' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel. Thus, for a input matrix of 2x2, we'll expect the output 1x1 matrix to contain, as
    // its single element, the value that was at position (0,0) of the input matrix (rather than an average,
    // which we would expect if projecting the centre).

    std::vector<float> outputData({
          1.0f,

        250.0f
    });

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data());
        outputData = tmp1;
    }

    auto input = MakeTensor<float, 4>(inputTensorInfo, inputData);

    LayerTestResult<float, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<float, 4>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<float, 4> ResizeBilinearSqMinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    const armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 4, 4, dataLayout);
    const armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 2, 2, dataLayout);

    std::vector<float> inputData({
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,

        7.0f, 6.0f, 5.0f, 4.0f,
        6.0f, 5.0f, 4.0f, 3.0f,
        5.0f, 4.0f, 3.0f, 2.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    });

    std::vector<float> outputData({
        1.0f, 3.0f,
        3.0f, 5.0f,

        7.0f, 5.0f,
        5.0f, 3.0f
    });

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data());
        outputData = tmp1;
    }

    auto input = MakeTensor<float, 4>(inputTensorInfo, inputData);

    LayerTestResult<float, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<float, 4>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<float, 4> ResizeBilinearMinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    const armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 3, 5, dataLayout);
    const armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 2, 3, dataLayout);

    std::vector<float> inputData({
          1.0f,   2.0f,   3.0f,   5.0f,   8.0f,
         13.0f,  21.0f,  34.0f,  55.0f,  89.0f,
        144.0f, 233.0f, 377.0f, 610.0f, 987.0f,

        987.0f, 610.0f, 377.0f, 233.0f, 144.0f,
         89.0f,  55.0f,  34.0f,  21.0f,  13.0f,
          8.0f,   5.0f,   3.0f,   2.0f,   1.0f
    });

    std::vector<float> outputData({
          1.0f,   2.6666f,   6.00f,
         78.5f, 179.3333f, 401.00f,

        987.0f, 454.6670f, 203.33f,
         48.5f,  22.3333f,  10.00f
    });

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data());
        outputData = tmp1;
    }

    auto input = MakeTensor<float, 4>(inputTensorInfo, inputData);

    LayerTestResult<float, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<float, 4>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<float, 4> ResizeBilinearMagTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    const armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 3, 2, dataLayout);
    const armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo<float>(1, 2, 3, 5, dataLayout);

    std::vector<float> inputData({
          1.0f,   2.0f,
         13.0f,  21.0f,
        144.0f, 233.0f,

        233.0f, 144.0f,
         21.0f,  13.0f,
          2.0f,   1.0f
    });

    std::vector<float> outputData({
          1.0f,   1.4f,   1.8f,   2.0f,   2.0f,
         13.0f,  16.2f,  19.4f,  21.0f,  21.0f,
        144.0f, 179.6f, 215.2f, 233.0f, 233.0f,

        233.0f, 197.4f, 161.8f, 144.0f, 144.0f,
         21.0f,  17.8f,  14.6f,  13.0f,  13.0f,
          2.0f,   1.6f,   1.2f,   1.0f,   1.0f
    });

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data());
        outputData = tmp1;
    }

    auto input = MakeTensor<float, 4>(inputTensorInfo, inputData);

    LayerTestResult<float, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<float, 4>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
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

LayerTestResult<float, 4> L2NormalizationTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorShape& inputOutputTensorShape,
    const std::vector<float>& inputValues,
    const std::vector<float>& expectedOutputValues,
    const armnn::DataLayout layout)
{
    const armnn::TensorInfo inputTensorInfo(inputOutputTensorShape, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo(inputOutputTensorShape, armnn::DataType::Float32);

    // at this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    std::vector<float> inputData = inputValues;
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;
    }

    auto inputTensor = MakeTensor<float, 4>(inputTensorInfo, std::vector<float>(inputData));

    LayerTestResult<float, 4> result(outputTensorInfo);
    std::vector<float> expectedOutputData = expectedOutputValues;
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(expectedOutputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, expectedOutputData.data(), tmp.data());
        expectedOutputData = tmp;
    }
    result.outputExpected = MakeTensor<float, 4>(inputTensorInfo, std::vector<float>(expectedOutputData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::L2NormalizationQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = layout;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateL2Normalization(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0][0]);

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

template<typename T>
LayerTestResult<T, 2> Pad2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
  const armnn::TensorShape inputShape{ 3, 3 };
  const armnn::TensorShape outputShape{ 7, 7 };

  const armnn::TensorInfo inputTensorInfo(inputShape, armnn::GetDataType<T>());
  const armnn::TensorInfo outputTensorInfo(outputShape, armnn::GetDataType<T>());

  std::vector<T> inputValues(
    QuantizedVector<T>(qScale, qOffset,
    {
      // Height (3) x Width (3)
      4, 8, 6,
      7, 4, 4,
      3, 2, 4
    }));

 std::vector<T> expectedOutputValues(
  QuantizedVector<T>(qScale, qOffset,
    {
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 4, 8, 6, 0, 0,
      0, 0, 7, 4, 4, 0, 0,
      0, 0, 3, 2, 4, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0
    }));

  auto inputTensor = MakeTensor<T, 2>(inputTensorInfo, std::vector<T>(inputValues));

  LayerTestResult<T, 2> result(outputTensorInfo);
  result.outputExpected = MakeTensor<T, 2>(outputTensorInfo, std::vector<T>(expectedOutputValues));

  std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
  std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

  armnn::PadQueueDescriptor descriptor;

  std::vector<std::pair<unsigned int, unsigned int>> PadList;
  PadList.push_back(std::pair<unsigned int, unsigned int>(2,2));
  PadList.push_back(std::pair<unsigned int, unsigned int>(2,2));

  descriptor.m_Parameters.m_PadList = PadList;
  armnn::WorkloadInfo info;

  AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
  AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

  std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePad(descriptor, info);

  inputHandle->Allocate();
  outputHandle->Allocate();

  CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0]);

  workload->Execute();

  CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

  return result;
}

template <typename T>
LayerTestResult<T, 3> Pad3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 2 };
    const armnn::TensorShape outputShape{ 3, 5, 6 };

    const armnn::TensorInfo inputTensorInfo(inputShape, armnn::GetDataType<T>());
    const armnn::TensorInfo outputTensorInfo(outputShape, armnn::GetDataType<T>());

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

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0], outputHandle.get());

    return result;
}

template <typename T>
LayerTestResult<T, 4> Pad4dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 3, 2 };
    const armnn::TensorShape outputShape{ 4, 5, 7, 4 };

    const armnn::TensorInfo inputTensorInfo(inputShape, armnn::GetDataType<T>());
    const armnn::TensorInfo outputTensorInfo(outputShape, armnn::GetDataType<T>());

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

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

LayerTestResult<uint8_t, 2> PadUint82dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
  return Pad2dTestCommon<uint8_t>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<uint8_t, 3> PadUint83dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
  return Pad3dTestCommon<uint8_t>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<uint8_t, 4> PadUint84dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
  return Pad4dTestCommon<uint8_t>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<float, 2> PadFloat322dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
  return Pad2dTestCommon<float>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<float, 3> PadFloat323dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
  return Pad3dTestCommon<float>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<float, 4> PadFloat324dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
  return Pad4dTestCommon<float>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<float, 4> L2Normalization1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
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


    return L2NormalizationTestImpl(workloadFactory, memoryManager, inputOutputShape,
                                   inputValues, expectedOutputValues, layout);
}

LayerTestResult<float, 4> L2Normalization2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
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

    return L2NormalizationTestImpl(workloadFactory, memoryManager, inputOutputShape,
                                   inputValues, expectedOutputValues, layout);
}

LayerTestResult<float, 4> L2Normalization3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
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

    return L2NormalizationTestImpl(workloadFactory, memoryManager, inputOutputShape,
                                   inputValues, expectedOutputValues, layout);
}

LayerTestResult<float, 4> L2Normalization4dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
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

    return L2NormalizationTestImpl(workloadFactory, memoryManager, inputOutputShape,
                                   inputValues, expectedOutputValues, layout);
}

template <typename T>
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
        armnn::GetDataType<T>());

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
        armnn::GetDataType<T>());

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

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<float, 4> ConstantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<float>(workloadFactory, memoryManager, 0.0f, 0);
}

LayerTestResult<uint8_t, 4> ConstantTestUint8(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<uint8_t>(workloadFactory, memoryManager, 1.0f, 0);
}

LayerTestResult<uint8_t, 3> MergerUint8Test(
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

    // Arbitrary scale and offsets. They don't really matter as the merger operator doesn't dequantize/quantize them.
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
    armnn::MergerQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = { 2, 0, 0 }; //Extent of the window is defined by size of input[1].
    armnn::MergerQueueDescriptor::ViewOrigin window2(wOrigin2);


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


    armnn::MergerQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMerger(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<uint8_t, 4> AdditionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int batchSize = 1;
    unsigned int channels = 2;
    unsigned int height = 2;
    unsigned int width = 3;

    const float scale = 7.0f;
    const int32_t offset = 3;

    armnn::TensorInfo inputTensorInfo1, inputTensorInfo2;
    armnn::TensorInfo outputTensorInfo;

    const unsigned int shape[] = { batchSize, channels, height, width };
    inputTensorInfo1 = armnn::TensorInfo(4, shape, armnn::DataType::QuantisedAsymm8);
    inputTensorInfo1.SetQuantizationScale(scale);
    inputTensorInfo1.SetQuantizationOffset(offset);

    inputTensorInfo2 = armnn::TensorInfo(4, shape, armnn::DataType::QuantisedAsymm8);
    inputTensorInfo2.SetQuantizationScale(scale);
    inputTensorInfo2.SetQuantizationOffset(offset);

    outputTensorInfo = armnn::TensorInfo(4, shape, armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);

    // See dequantized values to the right.
    auto input1 = MakeTensor<uint8_t, 4>(inputTensorInfo1, std::vector<uint8_t>(
    {
         63,  35,  77,  70,  56, 112, //  420, 224,  518,  469,  371, 763
        203,  28, 252, 168, 245,  91  // 1400, 175, 1743, 1155, 1694, 616
    }));

    // See dequantized values to the right.
    auto input2 = MakeTensor<uint8_t, 4>(inputTensorInfo1, std::vector<uint8_t>(
    {
         21,   7, 175, 231, 175, 210, // 126,   28, 1204, 1596, 1204, 1449
        126, 161,  63,  21, 105, 126  // 861, 1106,  420,  126,  714,  861
    }));

    // See dequantized values to the right.
    LayerTestResult<uint8_t, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<uint8_t, 4>(outputTensorInfo, std::vector<uint8_t>(
    {
         81,  39, 249, 255, 228, 255, //  546,  252, 1722, 2065(clamped), 1575, 2212(clamped)
        255, 186, 255, 186, 255, 214, // 2261(clamped), 1281, 2163(clamped), 1281, 2408(clamped), 1477
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

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

namespace
{
LayerTestResult<uint8_t, 4> MultiplicationUint8TestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int shape0[4],
    const std::vector<uint8_t> & values0,
    float scale0,
    int32_t offset0,
    const unsigned int shape1[4],
    const std::vector<uint8_t> & values1,
    float scale1,
    int32_t offset1,
    const unsigned int outShape[4],
    const std::vector<uint8_t> & outValues,
    float outScale,
    int32_t outOffset)
{
    armnn::TensorInfo inputTensorInfo0(4, shape0, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo inputTensorInfo1(4, shape1, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo outputTensorInfo(4, outShape, armnn::DataType::QuantisedAsymm8);

    inputTensorInfo0.SetQuantizationScale(scale0);
    inputTensorInfo0.SetQuantizationOffset(offset0);

    inputTensorInfo1.SetQuantizationScale(scale1);
    inputTensorInfo1.SetQuantizationOffset(offset1);

    outputTensorInfo.SetQuantizationScale(outScale);
    outputTensorInfo.SetQuantizationOffset(outOffset);

    auto input0 = MakeTensor<uint8_t, 4>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<uint8_t, 4>(inputTensorInfo1, values1);

    LayerTestResult<uint8_t, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<uint8_t, 4>(outputTensorInfo, outValues);

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

    return MultiplicationUint8TestHelper(workloadFactory,
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
                                         1366.255f, // Scale/offset chosen to have output values out of range.
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

    return MultiplicationUint8TestHelper(workloadFactory,
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

    return MultiplicationUint8TestHelper(workloadFactory,
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
template <typename T>
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
    auto dataType = (std::is_same<T, uint8_t>::value ?
                     armnn::DataType::QuantisedAsymm8 :
                     armnn::DataType::Float32);

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

    return SubtractionTestHelper(workloadFactory,
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

    return SubtractionTestHelper(workloadFactory,
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

    return SubtractionTestHelper(workloadFactory,
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

    return SubtractionTestHelper(workloadFactory,
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

    return SubtractionTestHelper(workloadFactory,
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

    return SubtractionTestHelper(workloadFactory,
                                 memoryManager,
                                 shape0, input0, 1.0f, 0,
                                 shape1, input1, 1.0f, 0,
                                 shape0, output, 1.0f, 0);
}

LayerTestResult<uint8_t, 4> ResizeBilinearNopUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    constexpr unsigned int inputWidth = 4;
    constexpr unsigned int inputHeight = 4;
    constexpr unsigned int inputChannels = 1;
    constexpr unsigned int inputBatchSize = 1;

    constexpr unsigned int outputWidth = inputWidth;
    constexpr unsigned int outputHeight = inputHeight;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
        armnn::DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationScale(1.5f);
    inputTensorInfo.SetQuantizationOffset(-3);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
        armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(1.5f);
    outputTensorInfo.SetQuantizationOffset(-3);

    auto input = MakeTensor<uint8_t, 4>(inputTensorInfo, std::vector<uint8_t>({
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7
    }));

    LayerTestResult<uint8_t, 4> result(outputTensorInfo);
    result.outputExpected = input;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<uint8_t, 4> SimpleResizeBilinearUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    constexpr unsigned int inputWidth = 2;
    constexpr unsigned int inputHeight = 2;
    constexpr unsigned int inputChannels = 1;
    constexpr unsigned int inputBatchSize = 1;

    constexpr unsigned int outputWidth = inputWidth / 2;
    constexpr unsigned int outputHeight = inputHeight / 2;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
        armnn::DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationScale(0.1567f);
    inputTensorInfo.SetQuantizationOffset(1);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
        armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(0.1567f);
    outputTensorInfo.SetQuantizationOffset(1);

    auto input = MakeTensor<uint8_t, 4>(inputTensorInfo, std::vector<uint8_t>({
        1, 255,
        200, 250
    }));

    // The 'resize bilinear' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel - and thus we'll expect the output 1x1 matrix to contain, as its single element, the value
    // that was at position (0,0) of the input matrix (rather than an average, which we would expect if projecting
    // the centre).
    LayerTestResult<uint8_t, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<uint8_t, 4>(outputTensorInfo, std::vector<uint8_t>({
        1
    }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<uint8_t, 4> ResizeBilinearSqMinUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    constexpr unsigned int inputWidth = 4;
    constexpr unsigned int inputHeight = 4;
    constexpr unsigned int inputChannels = 1;
    constexpr unsigned int inputBatchSize = 1;

    constexpr unsigned int outputWidth = inputWidth / 2;
    constexpr unsigned int outputHeight = inputHeight / 2;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
        armnn::DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationScale(3.141592f);
    inputTensorInfo.SetQuantizationOffset(3);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
        armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(3.141592f);
    outputTensorInfo.SetQuantizationOffset(3);

    auto input = MakeTensor<uint8_t, 4>(inputTensorInfo, std::vector<uint8_t>({
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7
    }));

    LayerTestResult<uint8_t, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<uint8_t, 4>(outputTensorInfo, std::vector<uint8_t>({
        1, 3,
        3, 5
    }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<uint8_t, 4> ResizeBilinearMinUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    constexpr unsigned int inputWidth = 3;
    constexpr unsigned int inputHeight = 2;
    constexpr unsigned int inputChannels = 1;
    constexpr unsigned int inputBatchSize = 1;

    constexpr unsigned int outputWidth = 2;
    constexpr unsigned int outputHeight = 1;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
        armnn::DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationScale(1.5f);
    inputTensorInfo.SetQuantizationOffset(-1);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
        armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(1.5f);
    outputTensorInfo.SetQuantizationOffset(-1);

    auto input = MakeTensor<uint8_t, 4>(inputTensorInfo, std::vector<uint8_t>({
        1,  2,  3, // 3.0, 4.5, 6.0
        5,  8, 13  // 9.0, 13.5, 21.0
    }));

    LayerTestResult<uint8_t, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<uint8_t, 4>(outputTensorInfo, std::vector<uint8_t>({
        1, 3 // 3.0, 5.25
    }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

LayerTestResult<uint8_t, 4> ResizeBilinearMagUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    constexpr unsigned int inputWidth = 2;
    constexpr unsigned int inputHeight = 3;
    constexpr unsigned int inputChannels = 1;
    constexpr unsigned int inputBatchSize = 1;

    constexpr unsigned int outputWidth = 5;
    constexpr unsigned int outputHeight = 3;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
        armnn::DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationScale(0.010765f);
    inputTensorInfo.SetQuantizationOffset(7);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
        armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(0.010132f);
    outputTensorInfo.SetQuantizationOffset(-18);

    auto input = MakeTensor<uint8_t, 4>(inputTensorInfo, std::vector<uint8_t>({
         24, 228, // 0.183005, 2.379065,
        105, 128, // 1.05497, 1.302565
        230,  71  // 2.400595, 0.68896
    }));

    LayerTestResult<uint8_t, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<uint8_t, 4>(outputTensorInfo, std::vector<uint8_t>({
          0,  87, 173, 217, 217, // 0.18300501, 1.06142902, 1.93985295, 2.37906504, 2.37906504
         86,  96, 106, 111, 111, // 1.05497003, 1.15400803, 1.25304604, 1.30256498, 1.30256498
        219, 151,  84,  50,  50  // 2.40059495, 1.71594095, 1.03128707, 0.68896002, 0.68896002
    }));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeBilinearQueueDescriptor descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResizeBilinear(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
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

    return BatchNormTestImpl<float>(workloadFactory, memoryManager,
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

    return BatchNormTestImpl<float>(workloadFactory, memoryManager,
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

    return BatchNormTestImpl<uint8_t>(workloadFactory, memoryManager,
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

    return BatchNormTestImpl<uint8_t>(workloadFactory, memoryManager,
                                      inputOutputShape, inputValues, expectedOutputValues,
                                      1.f/20.f, 50, armnn::DataLayout::NHWC);
}

LayerTestResult<uint8_t, 4> ConstantUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return ConstantTestImpl<uint8_t>(workloadFactory, memoryManager, 2e-6f, 1);
}

LayerTestResult<uint8_t, 1> Concatenation1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation1dTestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim0TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim1TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim0DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim0DiffInputDimsTestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 2> Concatenation2dDim1DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation2dDim1DiffInputDimsTestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim0TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim1TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation3dDim2TestImpl<uint8_t>(workloadFactory, memoryManager, useSubtensor, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim0DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim0TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim1DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation3dDim1DiffInputDimsTestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 3> Concatenation3dDim2DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation3dDim2DiffInputDimsTestImpl<uint8_t>(workloadFactory, memoryManager, useSubtensor, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim0TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim1TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDim2TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDim3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager, bool useSubtensor)
{
    return Concatenation4dDim3TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1, useSubtensor);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim0TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim1TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Concatenation4dDiffShapeDim2TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1);
}

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor)
{
    return Concatenation4dDiffShapeDim3TestImpl<uint8_t>(workloadFactory, memoryManager, 0.5f, -1, useSubtensor);
}

LayerTestResult<float, 4> SimpleMaxPooling2dSize2x2Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize2x2Stride2x2TestCommon<float>(workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize2x2Stride2x2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize2x2Stride2x2TestCommon<uint8_t>(
        workloadFactory, memoryManager, forceNoPadding, 3.0f, -5);
}

LayerTestResult<float, 4> SimpleMaxPooling2dSize3x3Stride2x4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize3x3Stride2x4TestCommon<float>(workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize3x3Stride2x4Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return SimpleMaxPooling2dSize3x3Stride2x4TestCommon<uint8_t>(
        workloadFactory, memoryManager, forceNoPadding, 0.1f, 128);
}

LayerTestResult<float, 4> SimpleMaxPooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling2dTestCommon<float>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling2dTestCommon<uint8_t>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<float, 4> SimpleAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling2dTestCommon<float>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<uint8_t, 4> SimpleAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling2dTestCommon<uint8_t>(
        workloadFactory, memoryManager, dataLayout, 0.5, -1);
}

LayerTestResult<float, 4> IgnorePaddingAveragePooling2dSize3x2Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding)
{
    return IgnorePaddingAveragePooling2dSize3x2Stride2x2TestCommon<float>(
        workloadFactory, memoryManager, forceNoPadding);
}

LayerTestResult<float, 4> LargeTensorsAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return LargeTensorsAveragePooling2dTestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> LargeTensorsAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return LargeTensorsAveragePooling2dTestCommon<uint8_t>(workloadFactory, memoryManager, 0.5, -1);
}

LayerTestResult<float, 4> SimpleL2Pooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling2dTestCommon<float>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<uint8_t, 4> SimpleL2Pooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling2dTestCommon<uint8_t>(workloadFactory, memoryManager, dataLayout);
}

LayerTestResult<float, 4> L2Pooling2dSize3Stride1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride1TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride1TestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> L2Pooling2dSize3Stride3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride3TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride3TestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> L2Pooling2dSize3Stride4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride4TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride4Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize3Stride4TestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> L2Pooling2dSize7Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize7TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize7Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize7TestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> L2Pooling2dSize9Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize9TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> L2Pooling2dSize9Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return L2Pooling2dSize9TestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> AsymmetricNonSquarePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AsymmetricNonSquarePooling2dTestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> AsymmetricNonSquarePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return AsymmetricNonSquarePooling2dTestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> ComparePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType)
{
    return ComparePooling2dTestCommon<float>(
        workloadFactory, memoryManager, refWorkloadFactory, poolingType);
}

LayerTestResult<uint8_t, 4> ComparePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType)
{
    return ComparePooling2dTestCommon<uint8_t>(
        workloadFactory, memoryManager, refWorkloadFactory, poolingType, 0.1f, 128);
}

LayerTestResult<float, 2> FullyConnectedLargeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool transposeWeights)
{
    return FullyConnectedLargeTestCommon<float>(workloadFactory, memoryManager, transposeWeights);
}

LayerTestResult<float, 4> IgnorePaddingSimpleMaxPooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleMaxPooling2dTestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleMaxPooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleMaxPooling2dTestCommon<uint8_t>(workloadFactory, memoryManager, 1.0f, -5);
}

LayerTestResult<float, 4> IgnorePaddingMaxPooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingMaxPooling2dSize3TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingMaxPooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingMaxPooling2dSize3TestCommon<uint8_t>(workloadFactory, memoryManager, 1.0f, -5);
}

LayerTestResult<float, 4> IgnorePaddingSimpleAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dTestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dTestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dNoPaddingTestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleAveragePooling2dNoPaddingTestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingAveragePooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingAveragePooling2dSize3TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingAveragePooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingAveragePooling2dSize3TestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingSimpleL2Pooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleL2Pooling2dTestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleL2Pooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingSimpleL2Pooling2dTestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> IgnorePaddingL2Pooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingL2Pooling2dSize3TestCommon<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> IgnorePaddingL2Pooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return IgnorePaddingL2Pooling2dSize3TestCommon<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SimplePermuteFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SimplePermuteFloat32TestCommon(workloadFactory, memoryManager);
};

LayerTestResult<uint8_t, 4> SimplePermuteUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SimplePermuteUint8TestCommon(workloadFactory, memoryManager);
};

LayerTestResult<float, 4> PermuteFloat32ValueSet1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PermuteFloat32ValueSet1TestCommon(workloadFactory, memoryManager);
};

LayerTestResult<float, 4> PermuteFloat32ValueSet2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PermuteFloat32ValueSet2TestCommon(workloadFactory, memoryManager);
};

LayerTestResult<float, 4> PermuteFloat32ValueSet3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PermuteFloat32ValueSet3TestCommon(workloadFactory, memoryManager);
};

namespace
{

template <typename T, std::size_t InputDim, std::size_t OutputDim>
LayerTestResult<T, OutputDim> MeanTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const unsigned int* inputShape,
    const std::vector<T>& inputData,
    const std::vector<unsigned int>& axis,
    bool keepDims,
    const unsigned int* outputShape,
    const std::vector<T>& outputData,
    float scale = 1.0f,
    int32_t offset = 0)
{
    auto dataType = (std::is_same<T, uint8_t>::value ? armnn::DataType::QuantisedAsymm8 : armnn::DataType::Float32);

    armnn::TensorInfo inputTensorInfo(InputDim, inputShape, dataType);
    armnn::TensorInfo outputTensorInfo(OutputDim, outputShape, dataType);

    inputTensorInfo.SetQuantizationScale(scale);
    inputTensorInfo.SetQuantizationOffset(offset);

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);

    auto input = MakeTensor<T, InputDim>(inputTensorInfo, inputData);

    LayerTestResult<T, OutputDim> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, OutputDim>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::MeanQueueDescriptor data;
    data.m_Parameters.m_Axis = axis;
    data.m_Parameters.m_KeepDims = keepDims;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMean(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    workload->Execute();

    CopyDataFromITensorHandle(result.output.origin(), outputHandle.get());

    return result;
}

} // anonymous namespace

LayerTestResult<uint8_t, 1> MeanUint8SimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 3, 2 };
    const unsigned int outputShape[] = { 1 };

    std::vector<uint8_t> input({ 1, 1, 2, 2, 3, 3 });
    std::vector<uint8_t> output({ 2 });

    return MeanTestHelper<uint8_t, 2, 1>(
        workloadFactory, memoryManager, inputShape, input, {}, false, outputShape, output);
}

LayerTestResult<uint8_t, 3> MeanUint8SimpleAxisTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 1, 1, 3, 2 };
    const unsigned int outputShape[] = { 1, 1, 2 };

    std::vector<uint8_t> input({ 1, 1, 2, 2, 3, 3 });
    std::vector<uint8_t> output({ 2, 2 });

    return MeanTestHelper<uint8_t, 4, 3>(
        workloadFactory, memoryManager, inputShape, input, { 2 }, false, outputShape, output);
}

LayerTestResult<uint8_t, 4> MeanUint8KeepDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 1, 1, 3, 2 };
    const unsigned int outputShape[] = { 1, 1, 1, 2 };

    std::vector<uint8_t> input({ 1, 1, 2, 2, 3, 3 });
    std::vector<uint8_t> output({ 2, 2 });

    return MeanTestHelper<uint8_t, 4, 4>(
        workloadFactory, memoryManager, inputShape, input, { 2 }, true, outputShape, output);
}

LayerTestResult<uint8_t, 4> MeanUint8MultipleDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 2, 3, 1, 2 };
    const unsigned int outputShape[] = { 1, 3, 1, 1 };

    std::vector<uint8_t> input({ 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 });
    std::vector<uint8_t> output({ 1, 3, 5 });

    return MeanTestHelper<uint8_t, 4, 4>(
        workloadFactory, memoryManager, inputShape, input, { 0, 3 }, true, outputShape, output);
}

LayerTestResult<uint8_t, 1> MeanVtsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 4, 3, 2 };
    const unsigned int outputShape[] = { 2 };

    std::vector<uint8_t> input({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                 24 });
    std::vector<uint8_t> output({ 12, 13 });

    return MeanTestHelper<uint8_t, 3, 1>(workloadFactory, memoryManager,
                                         inputShape, input, { 0, 1 }, false, outputShape,
                                         output, 0.8f, 5);
}

LayerTestResult<float, 1> MeanFloatSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 3, 2 };
    const unsigned int outputShape[] = { 1 };

    std::vector<float> input({ 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f });
    std::vector<float> output({ 2.0f });

    return MeanTestHelper<float, 2, 1>(
        workloadFactory, memoryManager, inputShape, input, {}, false, outputShape, output);
}

LayerTestResult<float, 3> MeanFloatSimpleAxisTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 2, 3, 1, 2 };
    const unsigned int outputShape[] = { 3, 1, 2 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    std::vector<float> output({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });

    return MeanTestHelper<float, 4, 3>(
        workloadFactory, memoryManager, inputShape, input, { 0 }, false, outputShape, output);
}

LayerTestResult<float, 4> MeanFloatKeepDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 1, 1, 3, 2 };
    const unsigned int outputShape[] = { 1, 1, 1, 2 };

    std::vector<float> input({ 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f });
    std::vector<float> output({ 2.0f, 2.0f });

    return MeanTestHelper<float, 4, 4>(
        workloadFactory, memoryManager, inputShape, input, { 2 }, true, outputShape, output);
}

LayerTestResult<float, 4> MeanFloatMultipleDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 2, 3, 1, 2 };
    const unsigned int outputShape[] = { 1, 3, 1, 1 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    std::vector<float> output({ 1.5f, 3.5f, 5.5f });

    return MeanTestHelper<float, 4, 4>(
        workloadFactory, memoryManager, inputShape, input, { 0, 3 }, true, outputShape, output);
}

LayerTestResult<float, 1> MeanVtsFloat1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 4, 3, 2 };
    const unsigned int outputShape[] = { 2 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                               15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f });
    std::vector<float> output({ 12.0f, 13.0f });

    return MeanTestHelper<float, 3, 1>(
        workloadFactory, memoryManager, inputShape, input, { 0, 1 }, false, outputShape, output);
}

LayerTestResult<float, 3> MeanVtsFloat2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 4, 3, 2 };
    const unsigned int outputShape[] = { 1, 3, 1 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                               15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f });
    std::vector<float> output({ 10.5f, 12.5f, 14.5f });

    return MeanTestHelper<float, 3, 3>(
        workloadFactory, memoryManager, inputShape, input, { 0, 2 }, true, outputShape, output);
}

LayerTestResult<float, 3> MeanVtsFloat3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 1, 2, 2, 1 };
    const unsigned int outputShape[] = { 1, 2, 1 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f });
    std::vector<float> output({ 1.5f, 3.5f });

    return MeanTestHelper<float, 4, 3>(
        workloadFactory, memoryManager, inputShape, input, { 2 }, false, outputShape, output);
}

LayerTestResult<float, 4> AdditionAfterMaxPoolTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // Create Initial Tensor
    // 1, 2, 3
    // 4, 5, 6
    // 7, 8, 9

    armnn::TensorInfo poolingInputTensorInfo({ 1, 1, 3, 3}, armnn::GetDataType<float>());
    armnn::TensorInfo poolingOutputTensorInfo({ 1, 1, 2, 2}, armnn::GetDataType<float>());

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

    armnn::TensorInfo addInputTensorInfo({ 1,1,2,2}, armnn::GetDataType<float>());
    armnn::TensorInfo addOutputTensorInfo({ 1,1,2,2}, armnn::GetDataType<float>());

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

    workload->Execute();
    addWorkload->Execute();

    CopyDataFromITensorHandle(&addRet.output[0][0][0][0], addOutputHandle.get());

    return addRet;
}

LayerTestResult<float, 4> SpaceToBatchNdSimpleFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiChannelsFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiBlockFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdPaddingFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiChannelsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiBlockUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdPaddingUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdSimpleNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleNHWCTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiChannelsNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsNHWCTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdMultiBlockNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockNHWCTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> SpaceToBatchNdPaddingNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingNHWCTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdSimpleNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdSimpleNHWCTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiChannelsNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiChannelsNHWCTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiBlockNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdMultiBlockNHWCTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> SpaceToBatchNdPaddingNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return SpaceToBatchNdPaddingNHWCTest<uint8_t>(workloadFactory, memoryManager);
}

namespace {

template<typename T, std::size_t InputDim, std::size_t OutputDim>
LayerTestResult<T, OutputDim> BatchToSpaceNdHelper(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout& dataLayout,
    const unsigned int *inputShape,
    const std::vector<T> &inputData,
    const std::vector<unsigned int> &blockShape,
    const std::vector<std::pair<unsigned int, unsigned int>> &crops,
    const unsigned int *outputShape,
    const std::vector<T> &outputData,
    float scale = 1.0f,
    int32_t offset = 0)
  {
    auto dataType = (std::is_same<T, uint8_t>::value ? armnn::DataType::QuantisedAsymm8 : armnn::DataType::Float32);

    armnn::TensorInfo inputTensorInfo(InputDim, inputShape, dataType);
    armnn::TensorInfo outputTensorInfo(OutputDim, outputShape, dataType);

    inputTensorInfo.SetQuantizationScale(scale);
    inputTensorInfo.SetQuantizationOffset(offset);

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);

    auto input = MakeTensor<T, InputDim>(inputTensorInfo, inputData);

    LayerTestResult<T, OutputDim> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, OutputDim>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::BatchToSpaceNdQueueDescriptor data;
    data.m_Parameters.m_DataLayout = dataLayout;
    data.m_Parameters.m_BlockShape = blockShape;
    data.m_Parameters.m_Crops = crops;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateBatchToSpaceNd(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

} // anonymous namespace

LayerTestResult<float, 4> BatchToSpaceNdNhwcFloat32Test1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 2, 2, 1};
    const unsigned int outputShape[] = {1, 4, 4, 1 };

    std::vector<float> input
    ({
        // Batch 0, Height 0, Width (2) x Channel (1)
        1.0f, 3.0f,
        // Batch 0, Height 1, Width (2) x Channel (1)
        9.0f, 11.0f,


        // Batch 1, Height 0, Width (2) x Channel (1)
        2.0f, 4.0f,
        // Batch 1, Height 1, Width (2) x Channel (1)
        10.0f, 12.0f,


        // Batch 2, Height 0, Width (2) x Channel (1)
        5.0f, 7.0f,
        // Batch 2, Height 1, Width (2) x Channel (1)
        13.0f, 15.0f,

        // Batch 3, Height 0, Width (2) x Channel (3)
        6.0f, 8.0f,
        // Batch 3, Height 1, Width (2) x Channel (1)
        14.0f, 16.0f
    });

    std::vector<float> expectedOutput
    ({
        1.0f,   2.0f,  3.0f,  4.0f,
        5.0f,   6.0f,  7.0f,  8.0f,
        9.0f,  10.0f, 11.0f,  12.0f,
        13.0f, 14.0f, 15.0f,  16.0f
    });

    std::vector<unsigned int> blockShape {2, 2};
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<float, 4, 4>(workloadFactory, memoryManager,
            armnn::DataLayout::NHWC, inputShape, input, blockShape,
            crops, outputShape, expectedOutput);
}

LayerTestResult<float, 4> BatchToSpaceNdNhwcFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 2, 2, 1};

    std::vector<float> input
    ({
         // Batch 0, Height 0, Width (2) x Channel (1)
         1.0f, 2.0f, 3.0f, 4.0f
    });

    std::vector<float> expectedOutput({1.0f,   2.0f,  3.0f,  4.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<float, 4, 4>(workloadFactory, memoryManager,
        armnn::DataLayout::NHWC, inputShape, input, blockShape,
        crops, outputShape, expectedOutput);
}

LayerTestResult<float, 4> BatchToSpaceNdNhwcFloat32Test3(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });

    std::vector<float> expectedOutput({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<float, 4, 4>(workloadFactory, memoryManager,
        armnn::DataLayout::NHWC, inputShape, input, blockShape,
        crops, outputShape, expectedOutput);
}

LayerTestResult<float, 4> BatchToSpaceNdNchwFloat32Test1(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });

    std::vector<float> expectedOutput
    ({
         // Batch 0, Channel 0, Height (2) x Width (2)
         1.0f,  4.0f,
         7.0f, 10.0f,

         // Batch 0, Channel 1, Height (2) x Width (2)
         2.0f,  5.0f,
         8.0f, 11.0f,

         // Batch 0, Channel 2, Height (2) x Width (2)
         3.0f,  6.0f,
         9.0f, 12.0f,
    });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<float, 4, 4>(workloadFactory, memoryManager,
        armnn::DataLayout::NCHW, inputShape, input, blockShape,
        crops, outputShape, expectedOutput);
}

LayerTestResult<float, 4> BatchToSpaceNdNchwFloat32Test2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 1, 2, 2};

    std::vector<float> input
            ({
                     // Batch 0, Height 0, Width (2) x Channel (1)
                     1.0f, 2.0f, 3.0f, 4.0f
             });

    std::vector<float> expectedOutput({1.0f,   2.0f,  3.0f,  4.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<float, 4, 4>(workloadFactory, memoryManager,
                                             armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                             crops, outputShape, expectedOutput);
}

LayerTestResult<float, 4> BatchToSpaceNdNchwFloat32Test3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({ 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f });

    std::vector<float> expectedOutput
            ({
                     // Batch 0, Channel 0, Height (2) x Width (2)
                     1.0f,  7.0f,
                     2.0f,  8.0f,

                     // Batch 0, Channel 1, Height (2) x Width (2)
                     3.0f,  9.0f,
                     4.0f, 10.0f,

                     // Batch 0, Channel 2, Height (2) x Width (2)
                     5.0f, 11.0f,
                     6.0f, 12.0f,
             });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<float, 4, 4>(workloadFactory, memoryManager,
                                             armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                             crops, outputShape, expectedOutput);
}

LayerTestResult<uint8_t, 4> BatchToSpaceNdNhwcUintTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 2, 2, 1};
    const unsigned int outputShape[] = {1, 4, 4, 1};

    std::vector<uint8_t> input({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    std::vector<uint8_t> expectedOutput({ 1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<uint8_t, 4, 4>(workloadFactory, memoryManager, armnn::DataLayout::NHWC, inputShape,
                                               input, blockShape, crops, outputShape, expectedOutput);
}

LayerTestResult<float, 4> StridedSlice4DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSlice4DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DReverseTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSliceSimpleStrideFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSliceSimpleRangeMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> StridedSlice3DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> StridedSlice3DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DReverseTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSlice2DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSlice2DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DReverseTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSlice4DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSlice4DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4DReverseTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleStrideUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleRangeMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> StridedSlice3DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> StridedSlice3DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3DReverseTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSlice2DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSlice2DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2DReverseTest<uint8_t>(workloadFactory, memoryManager);
}
LayerTestResult<uint8_t, 4> BatchToSpaceNdNhwcUintTest2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 2, 2, 1};

    std::vector<uint8_t> input
            ({
                     // Batch 0, Height 0, Width (2) x Channel (1)
                     1, 2, 3, 4
             });

    std::vector<uint8_t> expectedOutput({1, 2, 3, 4});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<uint8_t, 4, 4>(workloadFactory, memoryManager,
                                               armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                               crops, outputShape, expectedOutput);
}

LayerTestResult<uint8_t, 4> BatchToSpaceNdNhwcUintTest3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    std::vector<uint8_t> input({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> expectedOutput({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<uint8_t, 4, 4>(workloadFactory, memoryManager,
                                               armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                               crops, outputShape, expectedOutput);
}


LayerTestResult<uint8_t, 4> BatchToSpaceNdNchwUintTest1(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<uint8_t> input({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    std::vector<uint8_t> expectedOutput
            ({
                     // Batch 0, Channel 0, Height (2) x Width (2)
                     1,  4,
                     7, 10,

                     // Batch 0, Channel 1, Height (2) x Width (2)
                     2,  5,
                     8, 11,

                     // Batch 0, Channel 2, Height (2) x Width (2)
                     3,  6,
                     9, 12,
             });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<uint8_t, 4, 4>(workloadFactory, memoryManager,
                                               armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                               crops, outputShape, expectedOutput);
}

LayerTestResult<uint8_t, 4> BatchToSpaceNdNchwUintTest2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 1, 2, 2};

    std::vector<uint8_t> input
            ({
                     // Batch 0, Height 0, Width (2) x Channel (1)
                     1, 2, 3, 4
             });

    std::vector<uint8_t> expectedOutput({1, 2, 3, 4});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<uint8_t, 4, 4>(workloadFactory, memoryManager,
                                             armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                             crops, outputShape, expectedOutput);
}

LayerTestResult<uint8_t, 4> BatchToSpaceNdNchwUintTest3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<uint8_t> input({ 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12 });

    std::vector<uint8_t> expectedOutput
            ({
                     // Batch 0, Channel 0, Height (2) x Width (2)
                     1,  7,
                     2,  8,

                     // Batch 0, Channel 1, Height (2) x Width (2)
                     3,  9,
                     4, 10,

                     // Batch 0, Channel 2, Height (2) x Width (2)
                     5, 11,
                     6, 12,
             });
    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<uint8_t, 4, 4>(workloadFactory, memoryManager,
                                               armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                               crops, outputShape, expectedOutput);
}

LayerTestResult<float, 4> Debug4DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug4DTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> Debug3DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug3DTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> Debug2DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug2DTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<float, 1> Debug1DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug1DTest<float>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> Debug4DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug4DTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> Debug3DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug3DTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> Debug2DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug2DTest<uint8_t>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 1> Debug1DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Debug1DTest<uint8_t>(workloadFactory, memoryManager);
}
