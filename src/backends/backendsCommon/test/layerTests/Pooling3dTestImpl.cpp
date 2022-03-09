//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include "Pooling3dTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <armnnUtils/TensorUtils.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/Permute.hpp>

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnn/BackendHelper.hpp>
#include <backendsCommon/WorkloadInfo.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

using namespace armnnUtils;

template<typename T>
void PermuteNCDHWToNDHWC(std::vector<T> &src, armnn::TensorInfo &srcInfo)
{
    const armnn::PermutationVector NCDHWToNDHWC = { 0, 4, 1, 2, 3 };
    std::vector<T> tmp(src.size());
    armnnUtils::Permute(srcInfo.GetShape(), NCDHWToNDHWC, src.data(), tmp.data(), sizeof(T));
    src = tmp;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> SimplePooling3dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::Pooling3dDescriptor descriptor,
    float qScale,
    int32_t qOffset,
    const std::vector<T>& input,
    const std::vector<T>& outputExpected,
    const armnn::TensorShape& inputShape,
    const armnn::TensorShape& outputShape)
{
    IgnoreUnused(memoryManager);
    const armnn::DataLayout dataLayout = descriptor.m_DataLayout;
    const armnnUtils::DataLayoutIndexed dimensionIndices = dataLayout;
    auto heightIndex = dimensionIndices.GetHeightIndex();
    auto widthIndex = dimensionIndices.GetWidthIndex();
    auto depthIndex = dimensionIndices.GetDepthIndex();
    auto channelsIndex = dimensionIndices.GetChannelsIndex();

    unsigned int inputDepth      = armnn::numeric_cast<unsigned int>(inputShape[depthIndex]);
    unsigned int inputHeight     = armnn::numeric_cast<unsigned int>(inputShape[heightIndex]);
    unsigned int inputWidth      = armnn::numeric_cast<unsigned int>(inputShape[widthIndex]);
    unsigned int inputChannels   = armnn::numeric_cast<unsigned int>(inputShape[channelsIndex]);
    unsigned int inputBatchSize  = armnn::numeric_cast<unsigned int>(inputShape[0]);

    unsigned int outputDepth     = armnn::numeric_cast<unsigned int>(outputShape[depthIndex]);
    unsigned int outputHeight    = armnn::numeric_cast<unsigned int>(outputShape[heightIndex]);
    unsigned int outputWidth     = armnn::numeric_cast<unsigned int>(outputShape[widthIndex]);
    unsigned int outputChannels  = armnn::numeric_cast<unsigned int>(outputShape[channelsIndex]);
    unsigned int outputBatchSize = armnn::numeric_cast<unsigned int>(outputShape[0]);

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(
        inputBatchSize, inputChannels, inputDepth, inputHeight, inputWidth, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(
        outputBatchSize, outputChannels, outputDepth, outputHeight, outputWidth, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    LayerTestResult<T, 5> result(outputTensorInfo);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Pooling3dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;
    queueDescriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    // Don't execute if Pooling is not supported, as an exception will be raised.
    armnn::BackendId backend = workloadFactory.GetBackendId();
    std::string reasonIfUnsupported;
    armnn::LayerSupportHandle handle = armnn::GetILayerSupportByBackendId(backend);
    result.m_Supported = handle.IsPooling3dSupported(inputTensorInfo,
                                                     outputTensorInfo,
                                                     queueDescriptor.m_Parameters,
                                                     reasonIfUnsupported);
    if (!result.m_Supported)
    {
        return result;
    }

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Pooling3d,
                                                                                queueDescriptor,
                                                                                workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    result.m_ActualData = actualOutput;
    result.m_ExpectedData = outputExpected;

    return result;
}

//
// Tests max pooling with the following parameters:
//
//   Pooling size: 2x2x2
//   Stride:       (1,1,1)
//   input size:   3x3x3
//   channels:     2
//   batch size:   2
//
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> SimpleMaxPooling3dSize2x2x2Stride1x1x1TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = 2;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolDepth = 2;
    descriptor.m_StrideX = 1;
    descriptor.m_StrideY = 1;
    descriptor.m_StrideZ = 1;
    descriptor.m_PadLeft = descriptor.m_PadRight = 0;
    descriptor.m_PadTop = descriptor.m_PadBottom = 0;
    descriptor.m_PadFront = descriptor.m_PadBack = 0;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    unsigned int inputWidth = 3;
    unsigned int inputHeight = 3;
    unsigned int inputDepth = 3;
    unsigned int outputWidth =
        (inputWidth + descriptor.m_PadLeft + descriptor.m_PadRight + descriptor.m_StrideX - descriptor.m_PoolWidth) /
        descriptor.m_StrideX;
    unsigned int outputHeight =
        (inputHeight + descriptor.m_PadTop + descriptor.m_PadBottom + descriptor.m_StrideY - descriptor.m_PoolHeight) /
        descriptor.m_StrideY;
    unsigned int outputDepth =
        (inputDepth + descriptor.m_PadFront + descriptor.m_PadBack + descriptor.m_StrideZ - descriptor.m_PoolDepth) /
        descriptor.m_StrideZ;
    unsigned int channels = 2;
    unsigned int batchSize = 2;

    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( batchSize, channels, inputDepth, inputHeight,
                                                                   inputWidth, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( batchSize, channels, outputDepth, outputHeight,
                                                                    outputWidth, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<float> singleChannelData({
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
    });

    // Constructs input data.
    std::vector<float> inputData;
    auto negator = [](float f) { return -f; };

    // First image (two channels where the second channel is the negative of the first one).
    inputData.insert(inputData.end(), singleChannelData.begin(), singleChannelData.end());
    std::transform(singleChannelData.begin(), singleChannelData.end(), std::back_inserter(inputData), negator);

    // Second image (same as first image).
    inputData.insert(inputData.end(), singleChannelData.begin(), singleChannelData.end());
    std::transform(singleChannelData.begin(), singleChannelData.end(), std::back_inserter(inputData), negator);

    auto input = QuantizedVector<T>(inputData, qScale, qOffset);

    // These were calculated manually.
    std::vector<T> outputExpected = QuantizedVector<T>(
            {
                1.0f, 1.0f,
                1.0f, 1.0f,

                1.0f, 1.0f,
                1.0f, 1.0f,

                -1.0f, -1.0f,
                -1.0f, -1.0f,

                -1.0f, -1.0f,
                -1.0f, -1.0f,


                1.0f, 1.0f,
                1.0f, 1.0f,

                1.0f, 1.0f,
                1.0f, 1.0f,

                -1.0f, -1.0f,
                -1.0f, -1.0f,

                -1.0f, -1.0f,
                -1.0f, -1.0f,
            },
            qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC(input, inputTensorInfo);
        PermuteNCDHWToNDHWC(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> SimpleMaxPooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = descriptor.m_PoolDepth = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = descriptor.m_StrideZ = 2;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(1, 1, 4, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(1, 1, 2, 2, 2, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> inputData(
        QuantizedVector<T>({
             1.0f,  2.0f,  5.0f,  6.0f,
             3.0f,  4.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 13.0f, 14.0f,
            11.0f, 12.0f, 15.0f, 16.0f,

            17.0f, 18.0f, 21.0f, 22.0f,
            19.0f, 20.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 29.0f, 30.0f,
            27.0f, 28.0f, 31.0f, 32.0f,

            33.0f, 34.0f, 37.0f, 38.0f,
            35.0f, 36.0f, 39.0f, 40.0f,
            41.0f, 42.0f, 45.0f, 46.0f,
            43.0f, 44.0f, 47.0f, 48.0f,

            49.0f, 50.0f, 53.0f, 54.0f,
            51.0f, 52.0f, 55.0f, 56.0f,
            57.0f, 58.0f, 61.0f, 62.0f,
            59.0f, 60.0f, 63.0f, 64.0f,
        },
        qScale, qOffset));

    std::vector<T> outputData(
        QuantizedVector<T>({
            20.0f, 24.0f,
            28.0f, 32.0f,

            52.0f, 56.0f,
            60.0f, 64.0f,
        },
        qScale, qOffset));

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC(inputData, inputTensorInfo);
        PermuteNCDHWToNDHWC(outputData, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        inputData, outputData, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> IgnorePaddingSimpleMaxPooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = descriptor.m_PoolDepth = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = descriptor.m_StrideZ = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 4, 4, 4 , dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 3, 3, 3 , dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = QuantizedVector<T>(
        {
            -1.0f, -2.0f,  3.0f,  4.0f,
            -1.0f, -2.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,

            -1.0f, -2.0f,  3.0f,  4.0f,
            -1.0f, -2.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,

            -1.0f, -2.0f,  3.0f,  4.0f,
            -1.0f, -2.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,

            -1.0f, -2.0f,  3.0f,  4.0f,
            -1.0f, -2.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
        },
        qScale, qOffset);

    auto outputExpected = QuantizedVector<T>(
        {
            -1.0f,  3.0f,  4.0f,
             1.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -4.0f,

            -1.0f,  3.0f,  4.0f,
             1.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -4.0f,

            -1.0f,  3.0f,  4.0f,
             1.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -4.0f,
        },
        qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC(input, inputTensorInfo);
        PermuteNCDHWToNDHWC(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> SimpleAveragePooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout = armnn::DataLayout::NCDHW,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = descriptor.m_PoolDepth = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = descriptor.m_StrideZ = 2;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(1, 1, 4, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(1, 1, 2, 2, 2, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> inputData(
        QuantizedVector<T>({
             1.0f,  2.0f,  5.0f,  6.0f,
             3.0f,  4.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 13.0f, 14.0f,
            11.0f, 12.0f, 15.0f, 16.0f,

            17.0f, 18.0f, 21.0f, 22.0f,
            19.0f, 20.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 29.0f, 30.0f,
            27.0f, 28.0f, 31.0f, 32.0f,

            33.0f, 34.0f, 37.0f, 38.0f,
            35.0f, 36.0f, 39.0f, 40.0f,
            41.0f, 42.0f, 45.0f, 46.0f,
            43.0f, 44.0f, 47.0f, 48.0f,

            49.0f, 50.0f, 53.0f, 54.0f,
            51.0f, 52.0f, 55.0f, 56.0f,
            57.0f, 58.0f, 61.0f, 62.0f,
            59.0f, 60.0f, 63.0f, 64.0f,
        },
        qScale, qOffset));

    std::vector<T> outputData(
        QuantizedVector<T>({
            10.5f, 14.5f,
            18.5f, 22.5f,

            42.5f, 46.5f,
            50.5f, 54.5f,
        },
        qScale, qOffset));

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC(inputData, inputTensorInfo);
        PermuteNCDHWToNDHWC(outputData, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        inputData, outputData, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> LargeTensorsAveragePooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = descriptor.m_PoolDepth = 100;
    descriptor.m_StrideX = descriptor.m_StrideY = descriptor.m_StrideZ = 5;
    descriptor.m_PadLeft = 50;
    descriptor.m_PadRight = 50;
    descriptor.m_PadTop = 50;
    descriptor.m_PadBottom = 50;
    descriptor.m_PadFront = 50;
    descriptor.m_PadBack = 50;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 5, 3, 52, 60, 68, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 5, 3, 11, 13, 14, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.armnnUtils::GetTensorInfo(
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input;

    for (unsigned int i = 0 ; i < inputTensorInfo.GetShape().GetNumElements(); ++i)
    {
        input.push_back(1);
    }

    std::vector<T> outputExpected;

    for (unsigned int i = 0 ; i < outputTensorInfo.GetShape().GetNumElements(); ++i)
    {
        outputExpected.push_back(1);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> IgnorePaddingSimpleAveragePooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = descriptor.m_PoolDepth = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = descriptor.m_StrideZ = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    descriptor.m_DataLayout = dataLayout;


    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo ( 1, 1, 4, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 3, 3, 3, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = QuantizedVector<T>(
        {
            12.0f, 20.0f, 32.0f, 40.0f,
            12.0f, 20.0f, 32.0f, 40.0f,
            12.0f, 20.0f, 32.0f, 40.0f,
            12.0f, 20.0f, 32.0f, 40.0f,

            24.0f, 40.0f, 64.0f, 80.0f,
            24.0f, 40.0f, 64.0f, 80.0f,
            24.0f, 40.0f, 64.0f, 80.0f,
            24.0f, 40.0f, 64.0f, 80.0f,

            36.0f, 60.0f, 96.0f, 120.0f,
            36.0f, 60.0f, 96.0f, 120.0f,
            36.0f, 60.0f, 96.0f, 120.0f,
            36.0f, 60.0f, 96.0f, 120.0f,

            48.0f, 80.0f, 128.0f, 160.0f,
            48.0f, 80.0f, 128.0f, 160.0f,
            48.0f, 80.0f, 128.0f, 160.0f,
            48.0f, 80.0f, 128.0f, 160.0f,
        },
        qScale, qOffset);

    auto outputExpected = QuantizedVector<T>(
        {
            1.5f,  6.5f,  5.0f,
            3.0f,  13.0f,  10.0f,
            1.5f,  6.5f,  5.0f,

            7.5f,  32.5f,  25.0f,
            15.0f,  65.0f,  50.0f,
            7.5f,  32.5f,  25.0f,

            6.0f,  26.0f,  20.0f,
            12.0f,  52.0f,  40.0f,
            6.0f,  26.0f,  20.0f,
        },
        qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC(input, inputTensorInfo);
        PermuteNCDHWToNDHWC(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> SimpleL2Pooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = descriptor.m_PoolDepth = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = descriptor.m_StrideZ = 2;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(1, 1, 4, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(1, 1, 2, 2, 2, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> inputData(
        QuantizedVector<T>({
             1.0f,  2.0f,  5.0f,  6.0f,
             3.0f,  4.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 13.0f, 14.0f,
            11.0f, 12.0f, 15.0f, 16.0f,

            17.0f, 18.0f, 21.0f, 22.0f,
            19.0f, 20.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 29.0f, 30.0f,
            27.0f, 28.0f, 31.0f, 32.0f,

            33.0f, 34.0f, 37.0f, 38.0f,
            35.0f, 36.0f, 39.0f, 40.0f,
            41.0f, 42.0f, 45.0f, 46.0f,
            43.0f, 44.0f, 47.0f, 48.0f,

            49.0f, 50.0f, 53.0f, 54.0f,
            51.0f, 52.0f, 55.0f, 56.0f,
            57.0f, 58.0f, 61.0f, 62.0f,
            59.0f, 60.0f, 63.0f, 64.0f,
        },
        qScale, qOffset));

    std::vector<T> outputData(
        QuantizedVector<T>({
            13.2476412995f, 16.5981926727f,
            20.1866292382f, 23.9060661758f,

            43.2608367926f, 47.1963981677f,
            51.1419592898f, 55.0953718564f,
        },
        qScale, qOffset));

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC(inputData, inputTensorInfo);
        PermuteNCDHWToNDHWC(outputData, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        inputData, outputData, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> IgnorePaddingSimpleL2Pooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = descriptor.m_PoolDepth = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = descriptor.m_StrideZ = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 4, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 3, 3, 3, dataLayout,ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = QuantizedVector<T>(
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,

            2.0f, 3.0f, 4.0f, 5.0f,
            2.0f, 3.0f, 4.0f, 5.0f,
            2.0f, 3.0f, 4.0f, 5.0f,
            2.0f, 3.0f, 4.0f, 5.0f,

            3.0f, 4.0f, 5.0f, 6.0f,
            3.0f, 4.0f, 5.0f, 6.0f,
            3.0f, 4.0f, 5.0f, 6.0f,
            3.0f, 4.0f, 5.0f, 6.0f,

            4.0f, 5.0f, 6.0f, 7.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
        },
        qScale, qOffset);

    float v111 = float(sqrt(pow(1,2)/8.0f));
    float v112 = float(sqrt((pow(2,2)+pow(3,2))/8.0f));
    float v113 = float(sqrt(pow(4,2)/8));

    float v121 = float(sqrt((2*pow(1,2))/8.0f));
    float v122 = float(sqrt((2*pow(2,2)+2*pow(3,2))/8.0f));
    float v123 = float(sqrt((2*pow(4,2))/8.0f));

    float v131 = v111;
    float v132 = v112;
    float v133 = v113;

    float v211 = float(sqrt((pow(2,2)+pow(3,2))/8.0f));
    float v212 = float(sqrt((pow(3,2)+2*pow(4,2)+pow(5,2))/8.0f));
    float v213 = float(sqrt((pow(5,2)+pow(6,2))/8.0f));

    float v221 = float(sqrt((2*pow(2,2)+2*pow(3,2))/8.0f));
    float v222 = float(sqrt((2*pow(3,2)+4*pow(4,2)+2*pow(5,2))/8.0f));
    float v223 = float(sqrt((2*pow(5,2)+2*pow(6,2))/8.0f));

    float v231 = v211;
    float v232 = v212;
    float v233 = v213;

    float v311 = float(sqrt(pow(4,2)/8.0f));
    float v312 = float(sqrt((pow(5,2)+pow(6,2))/8.0f));
    float v313 = float(sqrt(pow(7,2)/8));

    float v321 = float(sqrt((2*pow(4,2))/8.0f));
    float v322 = float(sqrt((2*pow(5,2)+2*pow(6,2))/8.0f));
    float v323 = float(sqrt((2*pow(7,2))/8.0f));

    float v331 = v311;
    float v332 = v312;
    float v333 = v313;

    auto outputExpected = QuantizedVector<T>(
        {
            v111,  v112,  v113,
            v121,  v122,  v123,
            v131,  v132,  v133,

            v211,  v212,  v213,
            v221,  v222,  v223,
            v231,  v232,  v233,

            v311,  v312,  v313,
            v321,  v322,  v323,
            v331,  v332,  v333,
        },
        qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC(input, inputTensorInfo);
        PermuteNCDHWToNDHWC(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout,
        float qScale = 1.0f,
        int32_t qOffset = 0)
{
    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 1, 3, 1, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 2, 2, 1, dataLayout, ArmnnType);

    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = 1;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolDepth = 3;
    descriptor.m_StrideX = 1;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 2;
    descriptor.m_PadBottom = 0;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 2;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    // Construct input data.
    auto input = QuantizedVector<T>( { 1.0f, 3.0f, 4.0f, }, qScale, qOffset);

    // These were calculated manually.
    auto outputExpected = QuantizedVector<T>( { 0.0f, 3.0f, 0.0f, 3.0f, }, qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC<T>(input, inputTensorInfo);
        PermuteNCDHWToNDHWC<T>(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
            workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
            input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> AsymmetricNonSquareMaxPooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 1, 3, 1, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 2, 2, 1, dataLayout, ArmnnType);

    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = 1;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolDepth = 3;
    descriptor.m_StrideX = 1;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 0;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 2;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    // Construct input data.
    auto input = QuantizedVector<T>( { 1.0f, 3.0f, 4.0f, }, qScale, qOffset);

    // These were calculated manually.
    auto outputExpected = QuantizedVector<T>( { 1.0f, 4.0f, 1.0f, 4.0f, }, qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC<T>(input, inputTensorInfo);
        PermuteNCDHWToNDHWC<T>(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 1, 3, 1, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 2, 2, 1, dataLayout, ArmnnType);

    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = 1;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolDepth = 3;
    descriptor.m_StrideX = 1;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 2;
    descriptor.m_PadBottom = 0;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 2;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    // Construct input data.
    auto input = QuantizedVector<T>({ 1.0f, 3.0f, 4.0f, }, qScale, qOffset);

    // These were calculated manually.
    auto outputExpected = QuantizedVector<T>( { 0.0f, 2.0f, 0.0f, 2.0f, }, qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC<T>(input, inputTensorInfo);
        PermuteNCDHWToNDHWC<T>(outputExpected, outputTensorInfo);
    }
    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> AsymmetricNonSquareAveragePooling3dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout,
        float qScale = 1.0f,
        int32_t qOffset = 0)
{
    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 1, 3, 1, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 2, 2, 1, dataLayout, ArmnnType);

    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = 1;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolDepth = 3;
    descriptor.m_StrideX = 1;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 0;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 2;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    // Construct input data.
    auto input = QuantizedVector<T>( { 1.0f, 3.0f, 4.0f, }, qScale, qOffset);

    // These were calculated manually.
    auto outputExpected = QuantizedVector<T>( { 1.0f, 3.5f, 1.0f, 3.5f, }, qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC<T>(input, inputTensorInfo);
        PermuteNCDHWToNDHWC<T>(outputExpected, outputTensorInfo);
    }
    return SimplePooling3dTestImpl<ArmnnType>(
            workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
            input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 1, 3, 1, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 2, 2, 1, dataLayout, ArmnnType);

    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = 1;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolDepth = 3;
    descriptor.m_StrideX = 0;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 2;
    descriptor.m_PadBottom = 0;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 2;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    // Construct input data.
    auto input = QuantizedVector<T>( { 1.0f, 3.0f, 4.0f, }, qScale, qOffset);

    // These were calculated manually.
    auto outputExpected = QuantizedVector<T>( { 0.0f, 2.2360679775f, 0.0f, 2.2360679775f, }, qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC<T>(input, inputTensorInfo);
        PermuteNCDHWToNDHWC<T>(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
        input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> AsymmetricNonSquareL2Pooling3dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout,
        float qScale = 1.0f,
        int32_t qOffset = 0)
{
    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 1, 3, 1, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo( 1, 1, 2, 2, 1, dataLayout, ArmnnType);

    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = 1;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolDepth = 3;
    descriptor.m_StrideX = 1;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 0;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 2;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    // Construct input data.
    auto input = QuantizedVector<T>( { 1.0f, 3.0f, 4.0f, }, qScale, qOffset);

    // These were calculated manually.
    auto outputExpected = QuantizedVector<T>( { 1.0f, 3.53553390593f, 1.0f, 3.53553390593f, }, qScale, qOffset);

    if (dataLayout == armnn::DataLayout::NDHWC)
    {
        PermuteNCDHWToNDHWC<T>(input, inputTensorInfo);
        PermuteNCDHWToNDHWC<T>(outputExpected, outputTensorInfo);
    }

    return SimplePooling3dTestImpl<ArmnnType>(
            workloadFactory, memoryManager, tensorHandleFactory, descriptor, qScale, qOffset,
            input, outputExpected, inputTensorInfo.GetShape(), outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> ComparePooling3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::PoolingAlgorithm poolingType,
    const armnn::DataLayout dataLayout,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);
    const unsigned int inputWidth = 16;
    const unsigned int inputHeight = 32;
    const unsigned int inputDepth = 48;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 5;

    const unsigned int poolSize = 3;
    const unsigned int strideX = 2;
    const unsigned int strideY = 4;
    const unsigned int strideZ = 6;
    const unsigned int padX = 0;
    const unsigned int padY = 0;
    const unsigned int padZ = 0;

    const unsigned int outputWidth = (inputWidth + 2 * padX + strideX - poolSize) / strideX;
    const unsigned int outputHeight = (inputHeight + 2 * padY + strideY - poolSize) / strideY;
    const unsigned int outputDepth = (inputDepth + 2 * padZ + strideZ - poolSize) / strideZ;

    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo(batchSize, channelCount, inputDepth, inputHeight,
                                                                  inputWidth, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(batchSize, channelCount, outputDepth, outputHeight,
                                                                   outputWidth, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = MakeRandomTensor<T>(inputTensorInfo, 81715);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput(outputTensorInfo.GetNumElements());
    LayerTestResult<T, 5> comparisonResult(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Pooling3dQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_PoolType = poolingType;
    data.m_Parameters.m_PoolWidth = poolSize;
    data.m_Parameters.m_PoolHeight = poolSize;
    data.m_Parameters.m_PoolDepth = poolSize;
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_StrideZ = strideZ;
    data.m_Parameters.m_PadLeft = padX;
    data.m_Parameters.m_PadRight = padX;
    data.m_Parameters.m_PadTop = padY;
    data.m_Parameters.m_PadBottom = padY;
    data.m_Parameters.m_PadFront = padZ;
    data.m_Parameters.m_PadBack = padZ;
    data.m_Parameters.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    data.m_Parameters.m_DataLayout = dataLayout;

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef =
                                          refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo);

    // Don't execute if Pooling is not supported, as an exception will be raised.
    armnn::BackendId backend = workloadFactory.GetBackendId();
    std::string reasonIfUnsupported;
    armnn::LayerSupportHandle handle = armnn::GetILayerSupportByBackendId(backend);
    comparisonResult.m_Supported = handle.IsPooling3dSupported(inputTensorInfo,
                                                               outputTensorInfo,
                                                               data.m_Parameters,
                                                               reasonIfUnsupported);
    if (!comparisonResult.m_Supported)
    {
        return comparisonResult;
    }

    armnn::Pooling3dQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::Pooling3d, data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef
            = refWorkloadFactory.CreateWorkload(armnn::LayerType::Pooling3d, refData, refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());
    CopyDataToITensorHandle(inputHandleRef.get(), input.data());

    workload->Execute();
    workloadRef->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    CopyDataFromITensorHandle(expectedOutput.data(), outputHandleRef.get());

    comparisonResult.m_ActualData = actualOutput;
    comparisonResult.m_ExpectedData = expectedOutput;

    return comparisonResult;
}

} // anonymous namespace

LayerTestResult<float, 5> SimpleMaxPooling3dSize2x2x2Stride1x1x1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling3dSize2x2x2Stride1x1x1TestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> SimpleMaxPooling3dSize2x2x2Stride1x1x1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling3dSize2x2x2Stride1x1x1TestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, tensorHandleFactory, dataLayout, 0.1f, 128);
}

LayerTestResult<int16_t, 5> SimpleMaxPooling3dSize2x2x2Stride1x1x1Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling3dSize2x2x2Stride1x1x1TestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> SimpleMaxPooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> SimpleMaxPooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> SimpleMaxPooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleMaxPooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> IgnorePaddingSimpleMaxPooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleMaxPooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> IgnorePaddingSimpleMaxPooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleMaxPooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory,dataLayout, 1.0f, -5);
}

LayerTestResult<int16_t, 5> IgnorePaddingSimpleMaxPooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleMaxPooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> SimpleAveragePooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> SimpleAveragePooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> SimpleAveragePooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleAveragePooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> SimpleL2Pooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> SimpleL2Pooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> SimpleL2Pooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return SimpleL2Pooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> LargeTensorsAveragePooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return LargeTensorsAveragePooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> LargeTensorsAveragePooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return LargeTensorsAveragePooling3dTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, tensorHandleFactory, dataLayout, 0.5, -1);
}

LayerTestResult<int16_t, 5> LargeTensorsAveragePooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return LargeTensorsAveragePooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> IgnorePaddingSimpleAveragePooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleAveragePooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> IgnorePaddingSimpleAveragePooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleAveragePooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout, 1.0f, -5);
}

LayerTestResult<int16_t, 5> IgnorePaddingSimpleAveragePooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleAveragePooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> IgnorePaddingSimpleL2Pooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleL2Pooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> IgnorePaddingSimpleL2Pooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleL2Pooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout, 1.0f, -5);
}

LayerTestResult<int16_t, 5> IgnorePaddingSimpleL2Pooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return IgnorePaddingSimpleL2Pooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> AsymmetricNonSquareMaxPooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareMaxPooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> AsymmetricNonSquareMaxPooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareMaxPooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> AsymmetricNonSquareMaxPooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareMaxPooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> AsymmetricNonSquareAveragePooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareAveragePooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> AsymmetricNonSquareAveragePooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareAveragePooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> AsymmetricNonSquareAveragePooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareAveragePooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> AsymmetricNonSquareL2Pooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareL2Pooling3dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> AsymmetricNonSquareL2Pooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareL2Pooling3dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> AsymmetricNonSquareL2Pooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareL2Pooling3dTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<uint8_t, 5> AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<int16_t, 5> AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout)
{
    return AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 5> ComparePooling3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::PoolingAlgorithm  poolingType,
    const armnn::DataLayout dataLayout)
{
    return ComparePooling3dTestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager,  refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory,
        poolingType, dataLayout);
}

LayerTestResult<uint8_t, 5> ComparePooling3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::PoolingAlgorithm  poolingType,
    const armnn::DataLayout dataLayout)
{
    return ComparePooling3dTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager,  refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory,
        poolingType, dataLayout, 0.1f, 128);
}

LayerTestResult<int16_t, 5> ComparePooling3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::PoolingAlgorithm  poolingType,
    const armnn::DataLayout dataLayout)
{
    return ComparePooling3dTestCommon<armnn::DataType::QSymmS16>(
        workloadFactory, memoryManager,  refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory,
        poolingType, dataLayout);
}