//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubtractionTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::SubtractionQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::SubtractionQueueDescriptor& descriptor)
{
    return workloadFactory.CreateWorkload(armnn::LayerType::Subtraction, descriptor, info);
}

LayerTestResult<uint8_t, 4> SubtractionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 2 };

    std::vector<uint8_t> input0 = { 10, 12, 14, 16 };
    std::vector<uint8_t> input1 = {  1,  2,  1,  2 };
    std::vector<uint8_t> output = {  3,  3,  5,  5 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        0.5f,
        2,
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

LayerTestResult<uint8_t, 4> SubtractionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0 = { 10, 12, 14, 16 };

    std::vector<uint8_t> input1 = { 2 };

    std::vector<uint8_t> output = { 5, 6, 7, 8 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        0.5f,
        2,
        shape1,
        input1,
        1.0f,
        0,
        shape0,
        output,
        tensorHandleFactory,
        1.0f,
        3);
}

LayerTestResult<uint8_t, 4> SubtractionBroadcastUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 1 };

    std::vector<uint8_t> input0 = { 10, 12, 14, 16 };

    std::vector<uint8_t> input1 = { 2, 1 };

    std::vector<uint8_t> output = { 8, 11, 12, 15 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::QAsymmU8>(
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

LayerTestResult<float, 4> SubtractionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 2 };

    std::vector<float> input0 = { 1,  2, 3, 4 };
    std::vector<float> input1 = { 1, -1, 0, 2 };
    std::vector<float> output = { 0,  3, 3, 2 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<float, 4> SubtractionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<float> input0 = { 1,  2, 3, 4 };

    std::vector<float> input1 = { 10 };

    std::vector<float> output = { -9,  -8, -7, -6 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<float, 4> SubtractionBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<float> input0 = { 1,  2, 3, 4 };

    std::vector<float> input1 = { 10, -5 };

    std::vector<float> output = { -9,  7, -7, 9 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Float32>(
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

LayerTestResult<armnn::Half, 4> SubtractionFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 2 };

    std::vector<armnn::Half> input0 = { 1._h,  2._h, 3._h, 4._h };
    std::vector<armnn::Half> input1 = { 1._h, -1._h, 0._h, 2._h };
    std::vector<armnn::Half> output = { 0._h,  3._h, 3._h, 2._h };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<armnn::Half, 4> SubtractionBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<armnn::Half> input0 = { 1._h,  2._h, 3._h, 4._h };

    std::vector<armnn::Half> input1 = { 10._h };

    std::vector<armnn::Half> output = { -9._h,  -8._h, -7._h, -6._h };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<armnn::Half, 4> SubtractionBroadcastFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<armnn::Half> input0 = { 1._h,  2._h, 3._h, 4._h };

    std::vector<armnn::Half> input1 = { 10._h, -5._h };

    std::vector<armnn::Half> output = { -9._h,  7._h, -7._h, 9._h };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Float16>(
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

LayerTestResult<int16_t, 4> SubtractionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape[] = { 1, 1, 2, 2 };

    std::vector<int16_t> input0 = { 10, 12, 14, 16 };
    std::vector<int16_t> input1 = {  1,  2,  1,  2 };
    std::vector<int16_t> output = {  3,  3,  5,  5 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        0.5f,
        0,
        shape,
        input1,
        1.0f,
        0,
        shape,
        output,
        tensorHandleFactory,
        1.0f,
        0);
}

LayerTestResult<int16_t, 4> SubtractionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0 = { 10, 12, 14, 16 };

    std::vector<int16_t> input1 = { 2 };

    std::vector<int16_t> output = { 3, 4, 5, 6 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        0.5f,
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

LayerTestResult<int16_t, 4> SubtractionBroadcastInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 1 };

    std::vector<int16_t> input0 = { 10, 12, 14, 16 };

    std::vector<int16_t> input1 = { 2, 1 };

    std::vector<int16_t> output = { 8, 11, 12, 15 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::QSymmS16>(
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

LayerTestResult<int32_t, 4> SubtractionInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape[] = { 1, 1, 2, 2 };

    std::vector<int32_t> input0 = { 5, 6, 7, 8 };
    std::vector<int32_t> input1 = { 1, 2, 1, 2 };
    std::vector<int32_t> output = { 4, 4, 6, 6 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Signed32>(
        workloadFactory,
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
        tensorHandleFactory,
        1.0f,
        0);
}

LayerTestResult<int32_t, 4> SubtractionBroadcast1ElementInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int32_t> input0 = { 5, 6, 7, 8 };

    std::vector<int32_t> input1 = { 2 };

    std::vector<int32_t> output = { 3, 4, 5, 6 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Signed32>(
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

LayerTestResult<int32_t, 4> SubtractionBroadcastInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 1 };

    std::vector<int32_t> input0 = { 10, 12, 14, 16 };

    std::vector<int32_t> input1 = { 2, 1 };

    std::vector<int32_t> output = { 8, 11, 12, 15 };

    return ElementwiseTestHelper<4, armnn::SubtractionQueueDescriptor, armnn::DataType::Signed32>(
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