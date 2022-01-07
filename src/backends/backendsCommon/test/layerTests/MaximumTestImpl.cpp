//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MaximumTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::MaximumQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::MaximumQueueDescriptor& descriptor)
{
    return workloadFactory.CreateWorkload(armnn::LayerType::Maximum, descriptor, info);
}

LayerTestResult<float, 4> MaximumSimpleTest(
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
        1.f, 1.f, 1.f, 1.f,  5.f, 5.f, 5.f, 5.f,
        3.f, 3.f, 3.f, 3.f,  4.f, 4.f, 4.f, 4.f
    };

    std::vector<float> input1 =
    {
        2.f, 2.f, 2.f, 2.f,  3.f, 3.f, 3.f, 3.f,
        4.f, 4.f, 4.f, 4.f,  5.f, 5.f, 5.f, 5.f
    };

    std::vector<float> output =
    {
        2.f, 2.f, 2.f, 2.f,  5.f, 5.f, 5.f, 5.f,
        4.f, 4.f, 4.f, 4.f,  5.f, 5.f, 5.f, 5.f
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<float, 4> MaximumBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<float> input0 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    std::vector<float> input1 = { 2.f };

    std::vector<float> output = { 2.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<float, 4> MaximumBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0 =
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };

    std::vector<float> input1 = { 1.f, 2.f, 3.f };

    std::vector<float> output =
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<armnn::Half, 4> MaximumFloat16Test(
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
        1._h, 1._h, 1._h, 1._h,  5._h, 5._h, 5._h, 5._h,
        3._h, 3._h, 3._h, 3._h,  4._h, 4._h, 4._h, 4._h
    };

    std::vector<armnn::Half> input1 =
    {
        2._h, 2._h, 2._h, 2._h,  3._h, 3._h, 3._h, 3._h,
        4._h, 4._h, 4._h, 4._h,  5._h, 5._h, 5._h, 5._h
    };

    std::vector<armnn::Half> output =
    {
        2._h, 2._h, 2._h, 2._h,  5._h, 5._h, 5._h, 5._h,
        4._h, 4._h, 4._h, 4._h,  5._h, 5._h, 5._h, 5._h
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<armnn::Half, 4> MaximumBroadcast1ElementFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<armnn::Half> input0 = { 1._h, 2._h, 3._h, 4._h, 5._h, 6._h, 7._h, 8._h };

    std::vector<armnn::Half> input1 = { 2._h };

    std::vector<armnn::Half> output = { 2._h, 2._h, 3._h, 4._h, 5._h, 6._h, 7._h, 8._h };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<armnn::Half, 4> MaximumBroadcast1DVectorFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<armnn::Half> input0 =
    {
        1._h, 2._h, 3._h,  4._h,  5._h,  6._h,
        7._h, 8._h, 9._h, 10._h, 11._h, 12._h
    };

    std::vector<armnn::Half> input1 = { 1._h, 2._h, 3._h };

    std::vector<armnn::Half> output =
    {
        1._h, 2._h, 3._h,  4._h,  5._h,  6._h,
        7._h, 8._h, 9._h, 10._h, 11._h, 12._h
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<uint8_t, 4> MaximumUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    // See dequantized values to the right.
    std::vector<uint8_t> input0 =
    {
        1, 1, 1, 1, 6, 6, 6, 6,
        3, 3, 3, 3, 4, 4, 4, 4
    };

    std::vector<uint8_t> input1 =
    {
        2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5
    };

    std::vector<uint8_t> output =
    {
        2, 2, 2, 2, 6, 6, 6, 6,
        4, 4, 4, 4, 5, 5, 5, 5
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::QAsymmU8>(
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

LayerTestResult<uint8_t, 4> MaximumBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<uint8_t> input1 = { 2 };

    std::vector<uint8_t> output =
    {
        2, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::QAsymmU8>(
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

LayerTestResult<uint8_t, 4> MaximumBroadcast1DVectorUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<uint8_t> input1 = { 1, 10, 3 };

    std::vector<uint8_t> output = {
        1, 10, 3,  4, 10,  6,
        7, 10, 9, 10, 11, 12
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::QAsymmU8>(
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

LayerTestResult<int16_t, 4> MaximumInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int16_t> input0({ 1, 1, 1, 1, 6, 6, 6, 6,
                                  3, 3, 3, 3, 4, 4, 4, 4 });

    std::vector<int16_t> input1({ 2, 2, 2, 2, 3, 3, 3, 3,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    std::vector<int16_t> output({ 2, 2, 2, 2, 6, 6, 6, 6,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::QSymmS16>(
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

LayerTestResult<int16_t, 4> MaximumBroadcast1ElementInt16Test(
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
        2, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::QSymmS16>(
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

LayerTestResult<int16_t, 4> MaximumBroadcast1DVectorInt16Test(
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

    std::vector<int16_t> input1 = { 1, 10, 3 };

    std::vector<int16_t> output =
    {
        1, 10, 3,  4, 10,  6,
        7, 10, 9, 10, 11, 12
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::QSymmS16>(
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

LayerTestResult<int32_t, 4> MaximumInt32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int32_t> input0({ 1, 1, 1, 1, 6, 6, 6, 6,
                                  3, 3, 3, 3, 4, 4, 4, 4 });

    std::vector<int32_t> input1({ 2, 2, 2, 2, 3, 3, 3, 3,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    std::vector<int32_t> output({ 2, 2, 2, 2, 6, 6, 6, 6,
                                  4, 4, 4, 4, 5, 5, 5, 5 });

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Signed32>(
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

LayerTestResult<int32_t, 4> MaximumBroadcast1ElementInt32Test(
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
        2, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Signed32>(
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

LayerTestResult<int32_t, 4> MaximumBroadcast1DVectorInt32Test(
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

    std::vector<int32_t> input1 = { 1, 10, 3 };

    std::vector<int32_t> output =
    {
        1, 10, 3,  4, 10,  6,
        7, 10, 9, 10, 11, 12
    };

    return ElementwiseTestHelper<4, armnn::MaximumQueueDescriptor, armnn::DataType::Signed32>(
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