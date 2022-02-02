//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AdditionTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::AdditionQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::AdditionQueueDescriptor& descriptor)
{
    return workloadFactory.CreateWorkload(armnn::LayerType::Addition, descriptor, info);
}

LayerTestResult<float,4> AdditionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int batchSize = 2u;
    unsigned int channels  = 2u;
    unsigned int height    = 2u;
    unsigned int width     = 3u;

    unsigned int shape[] = { batchSize, channels, height, width };

    std::vector<float> input1 =
    {
        0.0f, 2.0f, 1.0f,
        0.2f, 1.0f, 2.0f,

        1.0f, 2.0f, 1.0f,
        0.2f, 1.0f, 2.0f,

        0.0f, 2.0f, 1.0f,
        4.2f, 1.0f, 2.0f,

        0.0f, 0.0f, 1.0f,
        0.2f, 1.0f, 2.0f,
    };

    std::vector<float> input2 =
    {
        1.0f, 2.0f,  1.0f,
        0.0f, 1.0f,  2.0f,

        1.0f, 2.0f, -2.0f,
        0.2f, 1.0f,  2.0f,

        0.0f, 2.0f,  1.0f,
        4.2f, 0.0f, -3.0f,

        0.0f, 0.0f,  1.0f,
        0.7f, 1.0f,  5.0f,
    };


    std::vector<float> output
    {
        1.0f, 4.0f,  2.0f,
        0.2f, 2.0f,  4.0f,

        2.0f, 4.0f, -1.0f,
        0.4f, 2.0f,  4.0f,

        0.0f, 4.0f,  2.0f,
        8.4f, 1.0f, -1.0f,

        0.0f, 0.0f,  2.0f,
        0.9f, 2.0f,  7.0f,
    };

    return ElementwiseTestHelper<4, armnn::AdditionQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape,
        input1,
        shape,
        input2,
        shape,
        output,
        tensorHandleFactory);
}

LayerTestResult<float, 5> Addition5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int depth     = 2u;
    unsigned int batchSize = 2u;
    unsigned int channels  = 2u;
    unsigned int height    = 2u;
    unsigned int width     = 3u;

    unsigned int shape[] = { depth, batchSize, channels, height, width };

    std::vector<float> input1 =
    {
        2.6f, 4.0f, 4.4f,  2.7f, 4.6f, 2.8f,
        2.3f, 1.9f, 3.4f,  2.9f, 2.2f, 4.5f,

        2.8f, 1.9f, 2.3f,  2.6f, 4.7f, 3.5f,
        0.4f, 1.5f, 2.1f,  0.7f, 5.0f, 1.1f,


        1.0f, 2.7f, 0.0f,  0.6f, 0.8f, 0.9f,
        1.0f, 2.6f, 0.4f,  3.8f, 0.4f, 0.8f,

        0.5f, 4.3f, 3.1f,  4.4f, 0.7f, 1.4f,
        0.4f, 4.4f, 0.7f,  0.6f, 4.7f, 1.2f,

    };

    std::vector<float> input2 =
    {
        4.4f, 3.0f, 1.0f,  0.0f, 3.9f, 3.1f,
        1.7f, 2.9f, 1.3f,  0.4f, 0.4f, 4.3f,

        4.5f, 0.2f, 2.2f,  4.1f, 3.9f, 3.0f,
        0.1f, 2.5f, 4.1f,  4.6f, 1.5f, 0.0f,


        0.5f, 4.9f, 2.5f,  1.5f, 3.4f, 4.5f,
        2.0f, 3.0f, 4.9f,  1.6f, 2.4f, 3.4f,

        3.6f, 1.8f, 1.3f,  2.6f, 2.1f, 4.8f,
        2.0f, 4.3f, 4.0f,  0.2f, 0.6f, 4.4f,
    };

    std::vector<float> output =
    {
        7.0f, 7.0f, 5.4f,  2.7f, 8.5f, 5.9f,
        4.0f, 4.8f, 4.7f,  3.3f, 2.6f, 8.8f,

        7.3f, 2.1f, 4.5f,  6.7f, 8.6f, 6.5f,
        0.5f, 4.0f, 6.2f,  5.3f, 6.5f, 1.1f,


        1.5f, 7.6f, 2.5f,  2.1f, 4.2f, 5.4f,
        3.0f, 5.6f, 5.3f,  5.4f, 2.8f, 4.2f,

        4.1f, 6.1f, 4.4f,  7.0f, 2.8f, 6.2f,
        2.4f, 8.7f, 4.7f,  0.8f, 5.3f, 5.6f,
    };

    return ElementwiseTestHelper<5, armnn::AdditionQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape,
        input1,
        shape,
        input2,
        shape,
        output,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> AdditionBroadcastTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
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

    auto input1 = armnnUtils::QuantizedVector<T>(
    {
        0.0f,
        1.0f,

        2.0f,
        3.0f,

        4.0f,
        5.0f,
    },
    qScale, qOffset);

    auto input2 = armnnUtils::QuantizedVector<T>(
    {
        0.5f, 1.5f, 2.5f,
        3.5f, 4.5f, 5.5f,
    },
    qScale, qOffset);

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    auto expectedOutput = armnnUtils::QuantizedVector<T>(
    {
        0.5f, 1.5f, 2.5f,
        4.5f, 5.5f, 6.5f,

        2.5f, 3.5f, 4.5f,
        6.5f, 7.5f, 8.5f,

        4.5f, 5.5f, 6.5f,
        8.5f, 9.5f, 10.5f,
    },
    qScale, qOffset);

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Addition,
                                                                                data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), input1.data());
    CopyDataToITensorHandle(inputHandle2.get(), input2.data());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> AdditionBroadcast1ElementTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
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

    auto input1 = armnnUtils::QuantizedVector<T>(
    {
         0.0f,  1.0f,  2.0f,
         3.0f,  4.0f,  5.0f,
         6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f,
    },
    qScale, qOffset);

    auto input2 = armnnUtils::QuantizedVector<T>(
    {
        0.5f,
    },
    qScale, qOffset);

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    auto expectedOutput = armnnUtils::QuantizedVector<T>(
    {
         0.5f,  1.5f,  2.5f,
         3.5f,  4.5f,  5.5f,
         6.5f,  7.5f,  8.5f,
         9.5f, 10.5f, 11.5f,
        12.5f, 13.5f, 14.5f,
        15.5f, 16.5f, 17.5f,
    },
    qScale, qOffset);

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Addition,
                                                                                data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), input1.data());
    CopyDataToITensorHandle(inputHandle2.get(), input2.data());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

LayerTestResult<float, 4> AdditionBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcastTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> AdditionBroadcastUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcastTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, 2.f, 0, tensorHandleFactory);
}

LayerTestResult<int16_t, 4> AdditionBroadcastInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcastTestImpl<armnn::DataType::QSymmS16>(
        workloadFactory, memoryManager, 2.f, 0, tensorHandleFactory);
}

LayerTestResult<int32_t, 4> AdditionBroadcastInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcastTestImpl<armnn::DataType::Signed32>(
            workloadFactory, memoryManager, 1.f, 0, tensorHandleFactory);
}

LayerTestResult<float, 4> AdditionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcast1ElementTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, 0.0f, 0, tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> AdditionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcast1ElementTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, 0.1333333f, 128, tensorHandleFactory);
}

LayerTestResult<int16_t, 4> AdditionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcast1ElementTestImpl<armnn::DataType::QSymmS16>(
        workloadFactory, memoryManager, 0.1333333f, 0, tensorHandleFactory);
}

LayerTestResult<int32_t, 4> AdditionBroadcast1ElementInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AdditionBroadcast1ElementTestImpl<armnn::DataType::Signed32>(
            workloadFactory, memoryManager, 1.f, 0, tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> AdditionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
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

    return ElementwiseTestHelper<4, armnn::AdditionQueueDescriptor, armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        7.0f,
        3,
        shape1,
        input1,
        7.0f,
        3,
        shape0,
        output,
        tensorHandleFactory,
        7.0f,
        3);
}

LayerTestResult<int16_t, 4> AdditionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 2, 2, 3 };

    std::vector<int16_t> input0 =
    {
        63,  35,  77,  70,  56, 112, //  441, 245,  539,  490,  392, 184
        203,  28, 252, 168, 245,  91  // 1421, 196, 1764, 1176, 1715, 637
    };

    std::vector<int16_t> input1 =
    {
        21,   7, 175, 231, 175, 210, // 126,   28, 1204, 1596, 1204, 1449
        126, 161,  63,  21, 105, 126  // 861, 1106,  420,  126,  714,  861
    };

    std::vector<int16_t> output =
    {
        84,  42, 252, 301, 231, 322, //  588,  294, 1764, 2107(clamped), 1617, 2254(clamped)
        329, 189, 315, 189, 350, 217, // 2303(clamped), 1323, 2205(clamped), 1323, 2450(clamped), 1519
    };

    return ElementwiseTestHelper<4, armnn::AdditionQueueDescriptor, armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        7.0f,
        0,
        shape1,
        input1,
        7.0f,
        0,
        shape0,
        output,
        tensorHandleFactory,
        7.0f,
        0);
}

LayerTestResult<int32_t, 4> AdditionInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 2, 2, 3 };

    std::vector<int32_t> input0 =
    {
        63,  35,  77,  70,  56, 112, //  441, 245,  539,  490,  392, 184
        203,  28, 252, 168, 245,  91  // 1421, 196, 1764, 1176, 1715, 637
    };

    std::vector<int32_t> input1 =
    {
        21,   7, 175, 231, 175, 210, // 126,   28, 1204, 1596, 1204, 1449
        126, 161,  63,  21, 105, 126  // 861, 1106,  420,  126,  714,  861
    };

    std::vector<int32_t> output =
    {
        84,  42, 252, 301, 231, 322, //  588,  294, 1764, 2107(clamped), 1617, 2254(clamped)
        329, 189, 315, 189, 350, 217, // 2303(clamped), 1323, 2205(clamped), 1323, 2450(clamped), 1519
    };

    return ElementwiseTestHelper<4, armnn::AdditionQueueDescriptor, armnn::DataType::Signed32>(
        workloadFactory,
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
        tensorHandleFactory,
        1.0f,
        0);
}

LayerTestResult<float, 4> AdditionAfterMaxPoolTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);

    // Create Initial Tensor
    // 1, 2, 3
    // 4, 5, 6
    // 7, 8, 9

    armnn::TensorInfo poolingInputTensorInfo({ 1, 1, 3, 3}, armnn::DataType::Float32);
    armnn::TensorInfo poolingOutputTensorInfo({ 1, 1, 2, 2}, armnn::DataType::Float32);

    std::vector<float> poolingInput = {1, 2, 3,
                                       4, 5, 6,
                                       7, 8, 9
                                      };
    std::unique_ptr<armnn::ITensorHandle> poolingInputHandle =
            tensorHandleFactory.CreateTensorHandle(poolingInputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> poolingOutputHandle =
            tensorHandleFactory.CreateTensorHandle(poolingOutputTensorInfo);

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
    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Pooling2d,
                                                                                queueDescriptor,
                                                                                workloadInfo);

    std::vector<float> resultMaxPool(poolingOutputTensorInfo.GetNumElements());

    // Create addition with another tensor the same size
    // This would be the result to apply a Conv2d with kernel ones(2) and stride 1x1
    // with the initial tensor.
    // 12, 16
    // 24, 28
    armnn::TensorInfo addInputTensorInfo({ 1,1,2,2 }, armnn::DataType::Float32);
    armnn::TensorInfo addOutputTensorInfo({ 1,1,2,2 }, armnn::DataType::Float32);

    std::vector<float> addInput = { 12, 16,
                                    24, 28 };

    // Expected output tensor after MaxPool and Addition.
    std::vector<float> actualOutput(addOutputTensorInfo.GetNumElements());
    std::vector<float> expectedOutput = { 13, 19,
                                          31, 37 };

    std::unique_ptr<armnn::ITensorHandle> addInputHandle = tensorHandleFactory.CreateTensorHandle(addInputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> addOutputHandle = tensorHandleFactory.CreateTensorHandle(addOutputTensorInfo);

    armnn::AdditionQueueDescriptor data;
    armnn::WorkloadInfo info;

    // Add the output of the MaxPool and the new tensor
    AddInputToWorkload(data, info, poolingOutputTensorInfo, poolingOutputHandle.get());
    AddInputToWorkload(data, info, addInputTensorInfo, addInputHandle.get());
    AddOutputToWorkload(data, info, addOutputTensorInfo, addOutputHandle.get());

    std::unique_ptr<armnn::IWorkload> addWorkload = workloadFactory.CreateWorkload(armnn::LayerType::Addition,
                                                                                   data, info);

    poolingInputHandle->Allocate();
    poolingOutputHandle->Allocate();
    addInputHandle->Allocate();
    addOutputHandle->Allocate();

    CopyDataToITensorHandle(poolingInputHandle.get(), poolingInput.data());
    CopyDataFromITensorHandle(resultMaxPool.data(), poolingOutputHandle.get());

    CopyDataToITensorHandle(poolingOutputHandle.get(), resultMaxPool.data());
    CopyDataToITensorHandle(addInputHandle.get(), addInput.data());

    workload->PostAllocationConfigure();
    workload->Execute();
    addWorkload->PostAllocationConfigure();
    addWorkload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), addOutputHandle.get());

    return LayerTestResult<float, 4>(actualOutput,
                                     expectedOutput,
                                     addOutputHandle->GetShape(),
                                     addOutputTensorInfo.GetShape());
}

LayerTestResult<float,4> CompareAdditionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory)
{
    IgnoreUnused(memoryManager);
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

    auto input1 = MakeRandomTensor<float>(inputTensorInfo1, 1232);
    auto input2 = MakeRandomTensor<float>(inputTensorInfo2, 456);

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<float> expectedOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle1Ref = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> inputHandle2Ref = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo2);
    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);

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

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Addition,
                                                                                data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateWorkload(armnn::LayerType::Addition,
                                                                                      refData, refInfo);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();
    inputHandle1Ref->Allocate();
    inputHandle2Ref->Allocate();
    outputHandleRef->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), input1.data());
    CopyDataToITensorHandle(inputHandle2.get(), input2.data());
    CopyDataToITensorHandle(inputHandle1Ref.get(), input1.data());
    CopyDataToITensorHandle(inputHandle2Ref.get(), input2.data());

    workload->PostAllocationConfigure();
    workload->Execute();
    workloadRef->PostAllocationConfigure();
    workloadRef->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    CopyDataFromITensorHandle(expectedOutput.data(), outputHandleRef.get());

    return LayerTestResult<float, 4>(actualOutput,
                                     expectedOutput,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
}