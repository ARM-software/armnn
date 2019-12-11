//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "WorkloadTestUtils.hpp"
#include "TensorUtils.hpp"

#include "QuantizeHelper.hpp"

#include <armnn/ArmNN.hpp>

#include <Permute.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <backendsCommon/WorkloadInfo.hpp>

#include <test/TensorHelpers.hpp>

#include <DataLayoutIndexed.hpp>

#include <boost/numeric/conversion/cast.hpp>

#include <algorithm>
#include <string>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimplePooling2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::Pooling2dDescriptor descriptor,
    float qScale,
    int32_t qOffset,
    const boost::multi_array<T, 4>& input,
    const boost::multi_array<T, 4>& outputExpected)
{
    const armnn::DataLayout dataLayout = descriptor.m_DataLayout;
    const armnnUtils::DataLayoutIndexed dimensionIndices = dataLayout;
    auto heightIndex = dimensionIndices.GetHeightIndex();
    auto widthIndex = dimensionIndices.GetWidthIndex();
    auto channelsIndex = dimensionIndices.GetChannelsIndex();

    unsigned int inputHeight     = boost::numeric_cast<unsigned int>(input.shape()[heightIndex]);
    unsigned int inputWidth      = boost::numeric_cast<unsigned int>(input.shape()[widthIndex]);
    unsigned int inputChannels   = boost::numeric_cast<unsigned int>(input.shape()[channelsIndex]);
    unsigned int inputBatchSize  = boost::numeric_cast<unsigned int>(input.shape()[0]);

    unsigned int outputHeight    = boost::numeric_cast<unsigned int>(outputExpected.shape()[heightIndex]);
    unsigned int outputWidth     = boost::numeric_cast<unsigned int>(outputExpected.shape()[widthIndex]);
    unsigned int outputChannels  = boost::numeric_cast<unsigned int>(outputExpected.shape()[channelsIndex]);
    unsigned int outputBatchSize = boost::numeric_cast<unsigned int>(outputExpected.shape()[0]);

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(
        inputBatchSize, inputChannels, inputHeight, inputWidth, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(
        outputBatchSize, outputChannels, outputHeight, outputWidth, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    LayerTestResult<T, 4> result(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Pooling2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;
    queueDescriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    // Don't execute if Pooling is not supported, as an exception will be raised.
    armnn::BackendId backend = workloadFactory.GetBackendId();
    const size_t reasonIfUnsupportedMaxLen = 255;
    char reasonIfUnsupported[reasonIfUnsupportedMaxLen+1];
    result.supported = armnn::IsPooling2dSupported(backend, inputTensorInfo, outputTensorInfo,
                                                   queueDescriptor.m_Parameters,
                                                   reasonIfUnsupported, reasonIfUnsupportedMaxLen);
    if (!result.supported)
    {
        return result;
    }

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePooling2d(queueDescriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    result.outputExpected = outputExpected;

    return result;
}

//
// Tests max pooling with the following parameters:
//
//   Pooling size: 3x3
//   Stride:       (2,4)
//   input size:   8x13
//   channels:     2
//   batch size:   2
//
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleMaxPooling2dSize3x3Stride2x4TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 4;
    // forceNoPadding is mainly used for compatibility with ARM Compute.
    // As of 16/05/2017, it errors if padX or padY are equal to or greater than the pool size.
    descriptor.m_PadLeft = descriptor.m_PadRight = forceNoPadding ? 0 : 3;
    descriptor.m_PadTop = descriptor.m_PadBottom = 0;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    unsigned int inputWidth = 8;
    unsigned int inputHeight = 13;
    unsigned int outputWidth =
        (inputWidth + descriptor.m_PadLeft + descriptor.m_PadRight + descriptor.m_StrideX - descriptor.m_PoolWidth) /
        descriptor.m_StrideX;
    unsigned int outputHeight =
        (inputHeight + descriptor.m_PadTop + descriptor.m_PadBottom + descriptor.m_StrideY - descriptor.m_PoolHeight) /
        descriptor.m_StrideY;
    unsigned int channels = 2;
    unsigned int batchSize = 2;

    armnn::TensorInfo inputTensorInfo({ batchSize, channels, inputHeight, inputWidth }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ batchSize, channels, outputHeight, outputWidth }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<float> singleChannelData({
        0.0f, 4.0f, 8.0f, 1.0f, 6.0f, 4.0f, 5.0f, 8.0f,
        1.0f, 1.0f, 6.0f, 0.0f, 3.0f, 7.0f, 4.0f, 7.0f,
        8.0f, 5.0f, 0.0f, 0.0f, 8.0f, 3.0f, 4.0f, 3.0f,
        8.0f, 2.0f, 5.0f, 4.0f, 1.0f, 9.0f, 2.0f, 0.0f,
        5.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 7.0f, 2.0f,
        1.0f, 2.0f, 6.0f, 2.0f, 7.0f, 9.0f, 5.0f, 2.0f,
        9.0f, 7.0f, 3.0f, 1.0f, 3.0f, 4.0f, 8.0f, 3.0f,
        1.0f, 0.0f, 0.0f, 5.0f, 5.0f, 4.0f, 2.0f, 0.0f,
        6.0f, 4.0f, 3.0f, 6.0f, 9.0f, 5.0f, 5.0f, 6.0f,
        8.0f, 7.0f, 9.0f, 6.0f, 1.0f, 4.0f, 1.0f, 9.0f,
        7.0f, 1.0f, 9.0f, 2.0f, 9.0f, 9.0f, 8.0f, 1.0f,
        4.0f, 4.0f, 5.0f, 9.0f, 2.0f, 6.0f, 6.0f, 4.0f,
        3.0f, 5.0f, 4.0f, 0.0f, 1.0f, 5.0f, 9.0f, 7.0f,
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

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, inputData));

    // These were calculated manually.
    auto shape(GetTensorShapeAsArray<4>(outputTensorInfo));
    boost::multi_array<T, 4> outputExpected(shape);
    if (forceNoPadding)
    {
        outputExpected = MakeTensor<T, 4>(outputTensorInfo,
            QuantizedVector<T>(qScale, qOffset, {
                 8.0f,  8.0f,  8.0f,
                 9.0f,  7.0f,  9.0f,
                 9.0f,  9.0f,  9.0f,

                 0.0f,  0.0f, -3.0f,
                -1.0f,  0.0f,  0.0f,
                -1.0f, -1.0f, -1.0f,

                 8.0f,  8.0f,  8.0f,
                 9.0f,  7.0f,  9.0f,
                 9.0f,  9.0f,  9.0f,

                 0.0f,  0.0f, -3.0f,
                -1.0f,  0.0f,  0.0f,
                -1.0f, -1.0f, -1.0f
        }));
    }
    else
    {
        outputExpected = MakeTensor<T, 4>(outputTensorInfo,
            QuantizedVector<T>(qScale, qOffset, {
                0.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f,
                0.0f, 9.0f, 7.0f, 9.0f, 9.0f, 3.0f,
                0.0f, 8.0f, 9.0f, 9.0f, 9.0f, 9.0f,

                0.0f, 0.0f, 0.0f, 0.0f,-3.0f,-3.0f,
                0.0f,-1.0f, 0.0f, 0.0f, 0.0f,-2.0f,
                0.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,

                0.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f,
                0.0f, 9.0f, 7.0f, 9.0f, 9.0f, 3.0f,
                0.0f, 8.0f, 9.0f, 9.0f, 9.0f, 9.0f,

                0.0f, 0.0f, 0.0f, 0.0f,-3.0f,-3.0f,
                0.0f,-1.0f, 0.0f, 0.0f, 0.0f,-2.0f,
                0.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f
        }));
    }

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleMaxPooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout = armnn::DataLayout::NCHW,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> inputData(
        QuantizedVector<T>(qScale, qOffset, {
             1.0f,  2.0f,  5.0f,  6.0f,
             3.0f,  4.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 13.0f, 14.0f,
            11.0f, 12.0f, 15.0f, 16.0f,

            17.0f, 18.0f, 21.0f, 22.0f,
            19.0f, 20.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 29.0f, 30.0f,
            27.0f, 28.0f, 31.0f, 32.0f,
        }));

    std::vector<T> outputData(
        QuantizedVector<T>(qScale, qOffset, {
             4.0f,  8.0f,
            12.0f, 16.0f,

            20.0f, 24.0f,
            28.0f, 32.0f,
        }));

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(T));
        inputData = tmp;

        std::vector<T> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(T));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, inputData);

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputData);

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleAveragePooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout = armnn::DataLayout::NCHW,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> inputData(
        QuantizedVector<T>(qScale, qOffset, {
             2.0f,  2.0f,  6.0f,  6.0f,
             4.0f,  4.0f,  8.0f,  8.0f,
            10.0f, 12.0f, 14.0f, 16.0f,
            10.0f, 12.0f, 16.0f, 14.0f,

            18.0f, 20.0f, 24.0f, 22.0f,
            20.0f, 18.0f, 22.0f, 24.0f,
            26.0f, 28.0f,  0.0f,  0.0f,
            26.0f, 28.0f,  0.0f,  0.0f,
        }));

    std::vector<T> outputData(
        QuantizedVector<T>(qScale, qOffset, {
             3.0f,  7.0f,
            11.0f, 15.0f,

            19.0f, 23.0f,
            27.0f,  0.0f,
        }));

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(T));
        inputData = tmp;

        std::vector<T> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(T));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, inputData);

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputData);

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> LargeTensorsAveragePooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 100;
    descriptor.m_StrideX = descriptor.m_StrideY = 5;
    descriptor.m_PadLeft = 50;
    descriptor.m_PadRight = 50;
    descriptor.m_PadTop = 50;
    descriptor.m_PadBottom = 50;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 5, 3, 52, 60 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 5, 3, 11, 13 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> inputVec;

    for (unsigned int i = 0 ; i < inputTensorInfo.GetShape().GetNumElements(); ++i)
    {
        inputVec.push_back(1);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, inputVec);

    std::vector<T> outputVec;

    for (unsigned int i = 0 ; i < outputTensorInfo.GetShape().GetNumElements(); ++i)
    {
        outputVec.push_back(1);
    }

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputVec);

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleL2Pooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout = armnn::DataLayout::NCHW,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout = dataLayout;

    armnn::TensorInfo inputTensorInfo  = armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    std::vector<T> inputData(
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 7.0f, 5.0f, 5.0f,
            1.0f, 7.0f, 5.0f, 5.0f,
            3.0f, 3.0f, 1.0f, 1.0f,
            3.0f, 3.0f, 1.0f, 1.0f,

            1.0f, 7.0f, 0.0f, 0.0f,
            1.0f, 7.0f, 2.0f, 0.0f,
            0.0f, 2.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f,
        }));

    std::vector<T> outputData(
        QuantizedVector<T>(qScale, qOffset, {
            5.0f, 5.0f,
            3.0f, 1.0f,

            5.0f, 1.0f,
            1.0f, 1.0f,
        }));

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(T));
        inputData = tmp;

        std::vector<T> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(T));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, inputData);

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputData);

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Pooling2dSize3Stride1TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            2.0f, 1.0f, 5.0f, 2.0f,
            1.0f, 2.0f, 2.0f, 1.0f,
            5.0f, 4.0f, 1.0f, 5.0f,
            2.0f, 1.0f, 5.0f, 2.0f,
        }));

    armnn::TensorInfo outputTensorInfo({ 1, 1, 2, 2 }, ArmnnType);
    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            3.0f, 3.0f,
            3.0f, 3.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Pooling2dSize3Stride3TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 3;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 9, 9 }, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f,
            2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f,
            2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f,
        }));

    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, ArmnnType);
    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Pooling2dSize3Stride4TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 4;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 7, 7 }, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            2.0f, 1.0f, 5.0f, 0.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 0.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 0.0f, 5.0f, 4.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            2.0f, 1.0f, 5.0f, 0.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 0.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 0.0f, 5.0f, 4.0f, 1.0f,
        }));

    armnn::TensorInfo outputTensorInfo({ 1, 1, 2, 2 }, ArmnnType);
    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            3.0f, 3.0f,
            3.0f, 3.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Pooling2dSize7TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 7;
    descriptor.m_StrideX = descriptor.m_StrideY = 7;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 7, 7 }, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 0.0f, 2.0f, 0.0f,  3.0f, 0.0f, 4.0f,
            0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f,
            0.0f, 5.0f, 0.0f, 6.0f,  0.0f, 7.0f, 0.0f,
            8.0f, 0.0f, 9.0f, 0.0f, 10.0f, 0.0f, 5.0f,
            0.0f, 5.0f, 0.0f, 2.0f,  0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f,
        }));

    armnn::TensorInfo outputTensorInfo({ 1, 1, 1, 1 }, ArmnnType);
    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            3.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> L2Pooling2dSize9TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 9;
    descriptor.m_StrideX = descriptor.m_StrideY = 9;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 9, 9 }, ArmnnType);
    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f,
            2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f,
            2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f, 2.0f, 1.0f, 5.0f,
            1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 2.0f,
            5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f, 5.0f, 4.0f, 1.0f,
        }));

    armnn::TensorInfo outputTensorInfo({ 1, 1, 1, 1 }, ArmnnType);
    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            3.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> AsymmetricNonSquarePooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::TensorInfo inputTensorInfo({ 1, 1, 1, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 2, 2 }, ArmnnType);

    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = 2;
    descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 2;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    // Construct input data.
    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 3.0f, 4.0f,
        }));

    // These were calculated manually.
    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            0.0f, 3.0f, 0.0f, 3.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ComparePooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm poolingType,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    const unsigned int inputWidth = 16;
    const unsigned int inputHeight = 32;
    const unsigned int channelCount = 2;
    const unsigned int batchSize = 5;

    const unsigned int poolSize = 3;
    const unsigned int strideX = 2;
    const unsigned int strideY = 4;
    const unsigned int padX = 0;
    const unsigned int padY = 0;

    const unsigned int outputWidth = (inputWidth + 2 * padX + strideX - poolSize) / strideX;
    const unsigned int outputHeight = (inputHeight + 2 * padY + strideY - poolSize) / strideY;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { batchSize, channelCount, inputHeight, inputWidth };
    unsigned int outputShape[] = { batchSize, channelCount, outputHeight, outputWidth };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    boost::multi_array<T, 4> input = MakeRandomTensor<T, 4>(inputTensorInfo, 81715);

    LayerTestResult<T, 4> comparisonResult(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Pooling2dQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_PoolType = poolingType;
    data.m_Parameters.m_PoolWidth = poolSize;
    data.m_Parameters.m_PoolHeight = poolSize;
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padX;
    data.m_Parameters.m_PadRight = padX;
    data.m_Parameters.m_PadTop = padY;
    data.m_Parameters.m_PadBottom = padY;
    data.m_Parameters.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refWorkloadFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refWorkloadFactory.CreateTensorHandle(inputTensorInfo);

    // Don't execute if Pooling is not supported, as an exception will be raised.
    armnn::BackendId backend = workloadFactory.GetBackendId();
    const size_t reasonIfUnsupportedMaxLen = 255;
    char reasonIfUnsupported[reasonIfUnsupportedMaxLen+1];
    comparisonResult.supported = armnn::IsPooling2dSupported(backend, inputTensorInfo, outputTensorInfo,
                                                             data.m_Parameters,
                                                             reasonIfUnsupported, reasonIfUnsupportedMaxLen);
    if (!comparisonResult.supported)
    {
        return comparisonResult;
    }

    armnn::Pooling2dQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePooling2d(data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreatePooling2d(refData, refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);
    CopyDataToITensorHandle(inputHandleRef.get(), &input[0][0][0][0]);

    workload->Execute();
    workloadRef->Execute();

    CopyDataFromITensorHandle(&comparisonResult.output[0][0][0][0], outputHandle.get());
    CopyDataFromITensorHandle(&comparisonResult.outputExpected[0][0][0][0], outputHandleRef.get());

    return comparisonResult;
}

//
// Tests max pooling with the following parameters:
//
//   Pooling size: 2x2
//   Stride:       (2,2)
//   input size:   4x4
//   channels:     1
//   batch size:   1
//
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleMaxPooling2dSize2x2Stride2x2TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = descriptor.m_PadRight = forceNoPadding ? 0 : 3;
    descriptor.m_PadTop = descriptor.m_PadBottom = 0;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;


    unsigned int inputWidth = 4;

    unsigned int inputHeight = 4;

    unsigned int outputWidth =
        (inputWidth + descriptor.m_PadLeft + descriptor.m_PadRight + descriptor.m_StrideX - descriptor.m_PoolWidth) /
        descriptor.m_StrideX;
    unsigned int outputHeight =
        (inputHeight + descriptor.m_PadTop + descriptor.m_PadBottom + descriptor.m_StrideY - descriptor.m_PoolHeight) /
        descriptor.m_StrideY;
    unsigned int channels = 1;
    unsigned int batchSize = 1;

    std::vector<float> inputData = {
        510.0f, 222.0f, 780.0f, 654.0f,
        141.0f, 276.0f,  15.0f, 546.0f,
        303.0f, 618.0f, 582.0f, 339.0f,
        438.0f, 564.0f, 573.0f, 402.0f
    };

    // Note that left and right edges will be 0.f, due to the 2x2 max pooling only accessing zeros here.
    std::vector<float> expectedOutputDataWithPadding = {
        0.0f, 510.0f, 780.0f, 654.0f, 0.0f,
        0.0f, 438.0f, 618.0f, 402.0f, 0.0f
    };

    std::vector<float> expectedOutputDataNoPadding = {
        510.0f, 780.0f,
        618.0f, 582.0f
    };

    armnn::TensorInfo inputTensorInfo({ batchSize, channels, inputHeight, inputWidth }, ArmnnType);

    // Scale and offset should match input - we're just calculating maximum values.
    armnn::TensorInfo outputTensorInfo({ batchSize, channels, outputHeight, outputWidth }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, inputData));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        forceNoPadding ? QuantizedVector<T>(qScale, qOffset, expectedOutputDataNoPadding) :
                         QuantizedVector<T>(qScale, qOffset, expectedOutputDataWithPadding));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

//
// Tests max pooling with the following parameters:
//
//   Pooling size: 3x2
//   Stride:       (2,2)
//   input size:   3x2
//   channels:     1
//   batch size:   1
//
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingAveragePooling2dSize3x2Stride2x2TestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool forceNoPadding,
        float qScale = 1.0f,
        int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = 3;
    descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = (forceNoPadding) ? 0 : 1;
    descriptor.m_PadRight = descriptor.m_PadLeft;
    descriptor.m_PadTop = 0;
    descriptor.m_PadBottom = 0;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    unsigned int inputWidth = 3;
    unsigned int inputHeight = 2;
    unsigned int outputWidth =
        (inputWidth + descriptor.m_PadLeft + descriptor.m_PadRight + descriptor.m_StrideX - descriptor.m_PoolWidth) /
        descriptor.m_StrideX;
    unsigned int outputHeight =
        (inputHeight + descriptor.m_PadTop + descriptor.m_PadBottom + descriptor.m_StrideY - descriptor.m_PoolHeight) /
        descriptor.m_StrideY;
    unsigned int channels = 1;
    unsigned int batchSize = 1;

    std::vector<float> inputData = {
        3.0f, 6.0f, 9.0f,
        12.0f, 15.0f, 18.0f,
    };

    std::vector<float> expectedOutputDataWithPadding = {
        6.0f, 8.0f,
    };

    std::vector<float> expectedOutputDataNoPadding = {
        10.5f,
    };

    armnn::TensorInfo inputTensorInfo({ batchSize, channels, inputHeight, inputWidth }, ArmnnType);

    // Scale and offset should match input - we're just calculating average values.
    armnn::TensorInfo outputTensorInfo({ batchSize, channels, outputHeight, outputWidth }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, inputData));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        forceNoPadding ? QuantizedVector<T>(qScale, qOffset, expectedOutputDataNoPadding) :
                         QuantizedVector<T>(qScale, qOffset, expectedOutputDataWithPadding));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingSimpleMaxPooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            -1.0f, -2.0f,  3.0f,  4.0f,
            -1.0f, -2.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
        }));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            -1.0f,  3.0f,  4.0f,
             1.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -4.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingMaxPooling2dSize3TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            -1.0f, -2.0f,  3.0f,  4.0f,
            -1.0f, -2.0f,  3.0f,  4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
             1.0f,  2.0f, -3.0f, -4.0f,
        }));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            -1.0f,  3.0f,  4.0f,  4.0f,
             2.0f,  3.0f,  4.0f,  4.0f,
             2.0f,  3.0f,  4.0f,  4.0f,
             2.0f,  2.0f,  2.0f, -3.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingSimpleAveragePooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            12.0f, 20.0f, 32.0f, 40.0f,
            12.0f, 20.0f, 32.0f, 40.0f,
            12.0f, 20.0f, 32.0f, 40.0f,
            12.0f, 20.0f, 32.0f, 40.0f,
        }));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            3.0f,  13.0f,  10.0f,
            6.0f,  26.0f,  20.0f,
            3.0f,  13.0f,  10.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 0;
    descriptor.m_PadBottom = 0;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Ceiling;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 2, 2 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
        }));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            2.0f, 3.5f,
            2.0f, 3.5f
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingAveragePooling2dSize3TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            9.0f,   27.0f,  18.0f,  36.0f,
            18.0f,   9.0f,  18.0f,   9.0f,
            27.0f,  18.0f,   9.0f,  27.0f,
            9.0f,   27.0f,   9.0f,  18.0f,
        }));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
             7.0f,  11.0f,  13.0f, 9.0f,
            12.0f,  17.0f,  19.0f, 13.0f,
            12.0f,  16.0f,  16.0f, 10.0f,
             9.0f,  11.0f,  12.0f, 7.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingSimpleL2Pooling2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            2.0f,  4.0f, 8.0f, 16.0f,
            4.0f,  2.0f, 2.0f, 4.0f,
            8.0f,  2.0f, 4.0f, 2.0f,
            16.0f, 2.0f, 2.0f, 8.0f,
        }));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
               1.0f,     4.4721f,   8.0f,
            4.4721f,     2.6457f,   2.236f,
               8.0f,     1.4142f,   4.0f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> IgnorePaddingL2Pooling2dSize3TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::L2;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
        }));

    auto outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            1.0540f, 1.7638f, 2.5385f, 2.3570f,
            1.2909f, 2.1602f, 3.1091f, 2.8867f,
            1.2909f, 2.1602f, 3.1091f, 2.8867f,
            1.0540f, 1.7638f, 2.5385f, 2.3570f,
        }));

    return SimplePooling2dTestImpl<ArmnnType>(
        workloadFactory, memoryManager, descriptor, qScale, qOffset, input, outputExpected);
}
