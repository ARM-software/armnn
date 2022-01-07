//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MultiplicationTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"
#include <reference/test/RefWorkloadFactoryHelper.hpp>

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::MultiplicationQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::MultiplicationQueueDescriptor& descriptor)
{
    return workloadFactory.CreateWorkload(armnn::LayerType::Multiplication, descriptor, info);
}

LayerTestResult<float, 4> MultiplicationTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int width        = 2u;
    const unsigned int height       = 2u;
    const unsigned int channelCount = 2u;
    const unsigned int batchSize    = 2u;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0 =
    {
        1, 1, 1, 1,  2, 2, 2, 2,
        3, 3, 3, 3,  4, 4, 4, 4
    };

    std::vector<float> input1 =
    {
        2, 2, 2, 2,  3, 3, 3, 3,
        4, 4, 4, 4,  5, 5, 5, 5
    };

    std::vector<float> output =
    {
         2,  2,  2,  2,   6,  6,  6,  6,
        12, 12, 12, 12,  20, 20, 20, 20
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        tensorHandleFactory);
}

LayerTestResult<float, 5> Multiplication5dTest(armnn::IWorkloadFactory& workloadFactory,
                                               const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                               const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int width        = 3u;
    const unsigned int height       = 2u;
    const unsigned int channelCount = 2u;
    const unsigned int batchSize    = 2u;;
    const unsigned int depth        = 2u;

    unsigned int shape[] = { depth, batchSize, channelCount, height, width };

    std::vector<float> input0 =
    {
        1.80f, 0.20f, 2.30f,  1.30f, 2.10f, 1.00f,
        2.60f, 0.60f, 2.10f,  2.30f, 2.30f, 2.00f,

        2.50f, 1.00f, 2.90f,  3.10f, 1.50f, 2.40f,
        2.80f, 1.10f, 1.00f,  3.20f, 1.00f, 2.30f,


        0.30f, 2.20f, 1.00f,  0.20f, 1.60f, 1.40f,
        0.80f, 3.20f, 0.10f,  0.10f, 3.10f, 2.10f,

        1.50f, 2.40f, 1.40f,  0.70f, 2.40f, 1.40f,
        1.60f, 1.20f, 1.90f,  0.80f, 0.00f, 0.10f,
    };

    std::vector<float> input1 =
    {
        0.70f, 1.00f, 2.90f,  2.20f, 3.10f, 2.80f,
        1.80f, 2.00f, 0.50f,  2.30f, 1.20f, 2.70f,

        2.40f, 0.20f, 3.20f,  1.60f, 0.20f, 2.50f,
        2.30f, 0.70f, 2.70f,  1.80f, 2.90f, 2.70f,


        3.20f, 3.20f, 0.70f,  1.90f, 2.70f, 2.50f,
        2.40f, 0.90f, 2.30f,  1.80f, 2.50f, 2.00f,

        1.60f, 2.20f, 1.60f,  2.00f, 0.30f, 3.20f,
        0.40f, 3.00f, 2.60f,  0.30f, 0.00f, 2.50f,
    };

    std::vector<float> output =
    {
        1.26f, 0.20f, 6.67f,  2.86f, 6.51f, 2.80f,
        4.68f, 1.20f, 1.05f,  5.29f, 2.76f, 5.40f,

        6.00f, 0.20f, 9.28f,  4.96f, 0.30f, 6.00f,
        6.44f, 0.77f, 2.70f,  5.76f, 2.90f, 6.21f,


        0.96f, 7.04f, 0.70f,  0.38f, 4.32f, 3.50f,
        1.92f, 2.88f, 0.23f,  0.18f, 7.75f, 4.20f,

        2.40f, 5.28f, 2.24f,  1.40f, 0.72f, 4.48f,
        0.64f, 3.60f, 4.94f,  0.24f, 0.00f, 0.25f,
    };

    return ElementwiseTestHelper<5, armnn::MultiplicationQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        tensorHandleFactory);
}

LayerTestResult<float, 4> MultiplicationBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<float> input0 = { 1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<float> input1 = { 2 };

    std::vector<float> output = { 2, 4, 6, 8, 10, 12, 14, 16};

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<float, 4> MultiplicationBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<float> input0 =
    {
        1,   2,    3,  4,    5,  6,
        7,   8,    9, 10,   11, 12,
        13, 14,   15, 16,   17, 18
    };

    std::vector<float> input1 = { 1, 2 };

    std::vector<float> output =
    {
         1,  4,    3,  8,    5, 12,
         7, 16,    9, 20,   11, 24,
        13, 28,   15, 32,   17, 36
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> MultiplicationUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    constexpr unsigned int batchSize = 1u;
    constexpr unsigned int channels  = 2u;
    constexpr unsigned int height    = 2u;
    constexpr unsigned int width     = 3u;

    const unsigned int shape[] = { batchSize, channels, height, width };

    // See dequantized values to the right
    std::vector<uint8_t> input0 =
    {
         62,  37,   3, 172,  13, 111, // 244, 144,   8, 684,  48, 440,
        188,  20,  73,  31,  23,  31  // 748,  76, 288, 120,  88, 120
    };

    // See dequantized values to the right
    std::vector<uint8_t> input1 =
    {
        126, 240, 252, 183, 121, 247, // 384, 726, 762, 555, 369, 747,
         48, 115, 151,  79,  78,  97  // 150, 351, 459, 243, 240, 297
    };

    // See dequantized values to the right
    std::vector<uint8_t> output =
    {
         64,  72,   0, 255,   8, 236, //  93696, 104544, 6096(clamped), 379620(clamped), 17712, 328680,
         77,  15,  92,  16,  10,  21, // 112200,  26676,        132192,           29160, 21120,  35640
    };

    // Scale/offset chosen to have output values out of range
    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::QAsymmU8>(
        workloadFactory,
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
        tensorHandleFactory,
        1366.255f,
        -5);
}

LayerTestResult<uint8_t, 4> MultiplicationBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0 =
    {
        1, 2, 3,    4,  5,  6,
        7, 8, 9,   10, 11, 12
    };

    std::vector<uint8_t> input1 = { 2 };

    std::vector<uint8_t> output =
    {
        2,  4,   6,    8, 10, 12,
        14, 16, 18,   20, 22, 24
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> MultiplicationBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0 =
    {
        1, 2, 3,    4,  5,  6,
        7, 8, 9,   10, 11, 12
    };

    std::vector<uint8_t> input1 = { 1, 2, 3 };

    std::vector<uint8_t> output =
    {
        1,  4,   9,     4, 10, 18,
        7, 16,  27,    10, 22, 36
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<int16_t, 4> MultiplicationInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape[] = { 1, 2, 2, 3 };

    std::vector<int16_t> input0 =
    {
        6,   7,  8,  9, 10, 11,
        12, 13, 14, 15, 16, 17
    };

    std::vector<int16_t> input1 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<int16_t> output =
    {
        6,   14,  24,  36,  50,  66,
        84, 104, 126, 150, 176, 204
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        tensorHandleFactory);
}

LayerTestResult<int16_t, 4> MultiplicationBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<int16_t> input1 = { 2 };

    std::vector<int16_t> output =
    {
        2,   4,  6,  8, 10, 12,
        14, 16, 18, 20, 22, 24
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<int16_t, 4> MultiplicationBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int16_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<int16_t> input1 = { 1, 2, 3 };

    std::vector<int16_t> output =
    {
        1,  4,  9,  4, 10, 18,
        7, 16, 27, 10, 22, 36
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<int32_t, 4> MultiplicationInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape[] = { 1, 2, 2, 3 };

    std::vector<int32_t> input0 =
    {
        6,   7,  8,  9, 10, 11,
        12, 13, 14, 15, 16, 17
    };

    std::vector<int32_t> input1 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<int32_t> output =
    {
        6,   14,  24,  36,  50,  66,
        84, 104, 126, 150, 176, 204
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::Signed32>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        tensorHandleFactory);
}

LayerTestResult<int32_t, 4> MultiplicationBroadcast1ElementInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int32_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<int32_t> input1 = { 2 };

    std::vector<int32_t> output =
    {
        2,   4,  6,  8, 10, 12,
        14, 16, 18, 20, 22, 24
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::Signed32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<int32_t, 4> MultiplicationBroadcast1DVectorInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int32_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<int32_t> input1 = { 1, 2, 3 };

    std::vector<int32_t> output =
    {
        1,  4,  9,  4, 10, 18,
        7, 16, 27, 10, 22, 36
    };

    return ElementwiseTestHelper<4, armnn::MultiplicationQueueDescriptor, armnn::DataType::Signed32>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output,
        tensorHandleFactory);
}

LayerTestResult<float,4> CompareMultiplicationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory)
{
    IgnoreUnused(memoryManager);
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

    auto input0 = MakeRandomTensor<float>(inputTensorInfo0, 803506992);
    auto input1 = MakeRandomTensor<float>(inputTensorInfo1, 54902257);

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<float> expectedOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0Ref = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1Ref = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);

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

    std::unique_ptr<armnn::IWorkload> workload
                = workloadFactory.CreateWorkload(armnn::LayerType::Multiplication, data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef
                = refWorkloadFactory.CreateWorkload(armnn::LayerType::Multiplication, refData, refInfo);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();
    inputHandle0Ref->Allocate();
    inputHandle1Ref->Allocate();
    outputHandleRef->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), input0.data());
    CopyDataToITensorHandle(inputHandle1.get(), input1.data());
    CopyDataToITensorHandle(inputHandle0Ref.get(), input0.data());
    CopyDataToITensorHandle(inputHandle1Ref.get(), input1.data());

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
