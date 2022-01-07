//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DivisionTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::DivisionQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::DivisionQueueDescriptor& descriptor)
{
    return workloadFactory.CreateWorkload(armnn::LayerType::Division, descriptor, info);
}

LayerTestResult<float, 4> DivisionByZeroTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int width        = 2u;
    const unsigned int height       = 2u;
    const unsigned int channelCount = 2u;
    const unsigned int batchSize    = 2u;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<float> input0 =
    {
         1.f,  1.f,  1.f,  1.f,  0.f, 0.f, 0.f, 0.f,
        -1.f, -1.f, -1.f, -1.f,  5.f, 5.f, 5.f, 5.f
    };

    std::vector<float> input1 =
    {
        0.f, 0.f, -0.f, -0.f,  0.f, 0.f, -0.f, -0.f,
        0.f, 0.f, -0.f, -0.f,  5.f, 5.f,  5.f,  5.f
    };

    std::vector<float> output =
    {
         INFINITY,  INFINITY, -INFINITY, -INFINITY,  NAN,  NAN, -NAN, -NAN,
        -INFINITY, -INFINITY,  INFINITY,  INFINITY,    1,    1,    1,    1
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<float, 4> DivisionTest(
    armnn::IWorkloadFactory& workloadFactory,
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
        2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
        4.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f, 5.f
    };

    std::vector<float> input1 =
    {
        1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f,
        4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f
    };

    std::vector<float> output =
    {
        2.f, 2.f, 2.f, 2.f, 1.50f, 1.50f, 1.50f, 1.50f,
        1.f, 1.f, 1.f, 1.f, 1.25f, 1.25f, 1.25f, 1.25f
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<float, 4> DivisionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<float> input0({ 2, 4, 6, 8, 10, 12, 14, 16});

    std::vector<float> input1({ 2 });

    std::vector<float> output({ 1, 2, 3, 4, 5, 6, 7, 8});

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<float, 4> DivisionBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<float> input0 =
    {
         1.f,  4.f,  3.f,  8.f,  5.f, 12.f,
         7.f, 16.f,  9.f, 20.f, 11.f, 24.f,
        13.f, 28.f, 15.f, 32.f, 17.f, 36.f
    };

    std::vector<float> input1 = { 1.f, 2.f };

    std::vector<float> output =
    {
         1.f,  2.f,  3.f,  4.f,  5.f,  6.f,
         7.f,  8.f,  9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f, 17.f, 18.f
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<armnn::Half, 4> DivisionFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    const unsigned int width        = 2u;
    const unsigned int height       = 2u;
    const unsigned int channelCount = 2u;
    const unsigned int batchSize    = 2u;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<armnn::Half> input0 =
    {
        2._h, 2._h, 2._h, 2._h, 3._h, 3._h, 3._h, 3._h,
        4._h, 4._h, 4._h, 4._h, 5._h, 5._h, 5._h, 5._h
    };

    std::vector<armnn::Half> input1 =
    {
        1._h, 1._h, 1._h, 1._h, 2._h, 2._h, 2._h, 2._h,
        4._h, 4._h, 4._h, 4._h, 4._h, 4._h, 4._h, 4._h
    };

    std::vector<armnn::Half> output =
    {
        2._h, 2._h, 2._h, 2._h, 1.50_h, 1.50_h, 1.50_h, 1.50_h,
        1._h, 1._h, 1._h, 1._h, 1.25_h, 1.25_h, 1.25_h, 1.25_h
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<armnn::Half, 4> DivisionBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<armnn::Half> input0({ 2._h, 4._h, 6._h, 8._h, 10._h, 12._h, 14._h, 16._h});

    std::vector<armnn::Half> input1({ 2._h });

    std::vector<armnn::Half> output({ 1._h, 2._h, 3._h, 4._h, 5._h, 6._h, 7._h, 8._h});

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<armnn::Half, 4> DivisionBroadcast1DVectorFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    unsigned int shape0[] = { 1, 3, 3, 2 };
    unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<armnn::Half> input0 =
    {
         1._h,  4._h,  3._h,  8._h,  5._h, 12._h,
         7._h, 16._h,  9._h, 20._h, 11._h, 24._h,
        13._h, 28._h, 15._h, 32._h, 17._h, 36._h
    };

    std::vector<armnn::Half> input1 = { 1._h, 2._h };

    std::vector<armnn::Half> output =
    {
         1._h,  2._h,  3._h,  4._h,  5._h,  6._h,
         7._h,  8._h,  9._h, 10._h, 11._h, 12._h,
        13._h, 14._h, 15._h, 16._h, 17._h, 18._h
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<uint8_t, 4> DivisionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int width        = 2u;
    const unsigned int height       = 2u;
    const unsigned int channelCount = 2u;
    const unsigned int batchSize    = 2u;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<uint8_t> input0 =
    {
        2, 2, 2, 2,  3, 3, 3, 3,
        4, 4, 4, 4,  5, 5, 5, 5
    };

    std::vector<uint8_t> input1 =
    {
        1, 1, 1, 1,  2, 2, 2, 2,
        4, 4, 4, 4,  4, 4, 4, 4
    };

    std::vector<uint8_t> output =
    {
        8, 8, 8, 8,  6, 6, 6, 6,
        4, 4, 4, 4,  5, 5, 5, 5
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        tensorHandleFactory,
        0.25f,
        0);
}

LayerTestResult<uint8_t, 4> DivisionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0 = { 2, 4, 6, 8, 10, 12, 14, 16};

    std::vector<uint8_t> input1 = { 2 };

    std::vector<uint8_t> output = { 1, 2, 3, 4, 5, 6, 7, 8};

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::QAsymmU8>(
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

LayerTestResult<uint8_t, 4> DivisionBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<uint8_t> input0 =
    {
        1,  4,    3,  8,    5, 12,
        7, 16,    9, 20,   11, 24,
       13, 28,   15, 32,   17, 36
    };

    std::vector<uint8_t> input1 = { 1, 2 };

    std::vector<uint8_t> output =
    {
        1,  2,    3,  4,    5,  6,
        7,  8,    9, 10,   11, 12,
       13, 14,   15, 16,   17, 18
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::QAsymmU8>(
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

LayerTestResult<int16_t,4> DivisionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int16_t> input0 =
    {
        2, 2, 2, 2,  3, 3, 3, 3,
        4, 4, 4, 4,  5, 5, 5, 5
    };

    std::vector<int16_t> input1 =
    {
        1, 1, 1, 1,  2, 2, 2, 2,
        4, 4, 4, 4,  4, 4, 4, 4
    };

    std::vector<int16_t> output =
    {
        8, 8, 8, 8,  6, 6, 6, 6,
        4, 4, 4, 4,  5, 5, 5, 5
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        tensorHandleFactory,
        0.25f,
        0);
}

LayerTestResult<int16_t, 4> DivisionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0 = { 2, 4, 6, 8, 10, 12, 14, 16};

    std::vector<int16_t> input1 = { 2 };

    std::vector<int16_t> output = { 1, 2, 3, 4, 5, 6, 7, 8};

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::QSymmS16>(
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

LayerTestResult<int16_t, 4> DivisionBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<int16_t> input0 =
    {
         1,  4,    3,  8,    5, 12,
         7, 16,    9, 20,   11, 24,
        13, 28,   15, 32,   17, 36
    };

    std::vector<int16_t> input1 = { 1, 2 };

    std::vector<int16_t> output =
    {
         1,  2,    3,  4,    5,  6,
         7,  8,    9, 10,   11, 12,
        13, 14,   15, 16,   17, 18
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::QSymmS16>(
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

LayerTestResult<int32_t, 4> DivisionInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int width        = 2u;
    const unsigned int height       = 2u;
    const unsigned int channelCount = 2u;
    const unsigned int batchSize    = 2u;

    unsigned int shape[] = { batchSize, channelCount, height, width };

    std::vector<int32_t> input0 =
    {
        8, 8, 8, 8,  6, 6, 6, 6,
        8, 8, 8, 8,  5, 5, 5, 5
    };

    std::vector<int32_t> input1 =
    {
        4, 4, 4, 4,  2, 2, 2, 2,
        2, 2, 2, 2,  1, 1, 1, 1
    };

    std::vector<int32_t> output =
    {
        2, 2, 2, 2,  3, 3, 3, 3,
        4, 4, 4, 4,  5, 5, 5, 5
    };


    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Signed32>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output,
        tensorHandleFactory,
        1.f,
        0);
}

LayerTestResult<int32_t, 4> DivisionBroadcast1ElementInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int32_t> input0 = { 2, 4, 6, 8, 10, 12, 14, 16};

    std::vector<int32_t> input1 = { 2 };

    std::vector<int32_t> output = { 1, 2, 3, 4, 5, 6, 7, 8};

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Signed32>(
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

LayerTestResult<int32_t, 4> DivisionBroadcast1DVectorInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 3, 3, 2 };
    unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<int32_t> input0 =
    {
        1,  4,    3,  8,    5, 12,
        7, 16,    9, 20,   11, 24,
        13, 28,   15, 32,   17, 36
    };

    std::vector<int32_t> input1 = { 1, 2 };

    std::vector<int32_t> output =
    {
        1,  2,    3,  4,    5,  6,
        7,  8,    9, 10,   11, 12,
        13, 14,   15, 16,   17, 18
    };

    return ElementwiseTestHelper<4, armnn::DivisionQueueDescriptor, armnn::DataType::Signed32>(
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