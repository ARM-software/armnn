//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "L2NormalizationTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <armnnUtils/TensorUtils.hpp>
#include <armnnUtils/Permute.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <numeric>

namespace
{

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2NormalizationTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::TensorShape& inputOutputTensorShape,
    float scale,
    int32_t offset,
    const std::vector<float>& inputValues,
    float outScale,
    int32_t outOffset,
    std::vector<float>& expectedOutputValues,
    const armnn::DataLayout layout,
    float epsilon = 1e-12f)
{
    IgnoreUnused(memoryManager);
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

    auto inputTensor = armnnUtils::QuantizedVector<T>(inputData,
                                                      inputTensorInfo.GetQuantizationScale(),
                                                      inputTensorInfo.GetQuantizationOffset());

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(expectedOutputValues.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, expectedOutputValues.data(), tmp.data(),
                            sizeof(float));
        expectedOutputValues = tmp;
    }

    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputValues,
                                                                       outputTensorInfo.GetQuantizationScale(),
                                                                       outputTensorInfo.GetQuantizationOffset());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::L2NormalizationQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Eps = epsilon;
    descriptor.m_Parameters.m_DataLayout = layout;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::L2Normalization,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputTensor.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutputData,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

float CalcInvL2Norm(std::initializer_list<float> elements)
{
    const float reduction = std::accumulate(elements.begin(), elements.end(), 0.0f,
        [](float acc, float element) { return acc + element * element; });
    return 1.0f / sqrtf(reduction);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2NormalizationEpsilonTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
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

    return L2NormalizationTestImpl<ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputOutputShape,
        scale,
        offset,
        inputValues,
        outScale,
        outOffset,
        expectedOutputValues,
        layout,
        epsilon);
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization1dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
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


    return L2NormalizationTestImpl<ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputOutputShape,
        scale,
        offset,
        inputValues,
        outScale,
        outOffset,
        expectedOutputValues,
        layout);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
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

    return L2NormalizationTestImpl<ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputOutputShape,
        scale,
        offset,
        inputValues,
        outScale,
        outOffset,
        expectedOutputValues,
        layout);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
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

    return L2NormalizationTestImpl<ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputOutputShape,
        scale,
        offset,
        inputValues,
        outScale,
        outOffset,
        expectedOutputValues,
        layout);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Normalization4dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
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

    return L2NormalizationTestImpl<ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputOutputShape,
        scale,
        offset,
        inputValues,
        outScale,
        outOffset,
        expectedOutputValues,
        layout);
}

} // anonymous namespace

LayerTestResult<float, 4> L2NormalizationDefaultEpsilonTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout layout)
{
    // Dummy descriptor to get the default value of epsilon.
    armnn::L2NormalizationDescriptor descriptor;

    return L2NormalizationEpsilonTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        0.f,
        0,
        layout,
        descriptor.m_Eps);
}

LayerTestResult<float, 4> L2NormalizationNonDefaultEpsilonTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout layout)
{
    return L2NormalizationEpsilonTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        0.f,
        0,
        layout,
        1e-9f);
}

LayerTestResult<float, 4> L2Normalization1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        0.f,
        0,
        layout);
}

LayerTestResult<int16_t, 4> L2Normalization1dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f,
        0,
        layout);
}

LayerTestResult<uint8_t, 4> L2Normalization1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f / 128,
        128,
        layout);
}

LayerTestResult<float, 4> L2Normalization2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization2dTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        0.f,
        0,
        layout);
}

LayerTestResult<int16_t, 4> L2Normalization2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f,
        0,
        layout);
}

LayerTestResult<uint8_t, 4> L2Normalization2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f / 128,
        128,
        layout);
}

LayerTestResult<float, 2> L2Normalization2dShapeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
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

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::L2NormalizationQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Eps = 1e-12f;
    descriptor.m_Parameters.m_DataLayout = layout;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::L2Normalization,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<float, 2>(actualOutput,
                                     expectedOutputData,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
}

LayerTestResult<float, 4> L2Normalization3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization3dTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        0.f,
        0,
        layout);
}

LayerTestResult<int16_t, 4> L2Normalization3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f,
        0,
        layout);
}

LayerTestResult<uint8_t, 4> L2Normalization3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f / 128,
        128,
        layout);
}

LayerTestResult<float, 4> L2Normalization4dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization4dTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        0.f,
        0,
        layout);
}

LayerTestResult<int16_t, 4> L2Normalization4dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f,
        0,
        layout);
}

LayerTestResult<uint8_t, 4> L2Normalization4dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    return L2Normalization1dTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        1.f,
        0,
        1.f / 128,
        128,
        layout);
}
