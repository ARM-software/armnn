//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "EqualTestImpl.hpp"
#include "ElementwiseTestImpl.hpp"

#include <Half.hpp>

template<>
std::unique_ptr<armnn::IWorkload> CreateWorkload<armnn::EqualQueueDescriptor>(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::EqualQueueDescriptor& descriptor)
{
    return workloadFactory.CreateEqual(descriptor, info);
}

LayerTestResult<uint8_t, 4> EqualSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
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

    std::vector<float> input1({ 1, 1, 1, 1,  3, 3, 3, 3,
                                5, 5, 5, 5,  4, 4, 4, 4 });

    std::vector<uint8_t> output({ 1, 1, 1, 1,  0, 0, 0, 0,
                                  0, 0, 0, 0,  1, 1, 1, 1 });

    return ElementwiseTestHelper<4, armnn::EqualQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
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

    return ElementwiseTestHelper<4, armnn::EqualQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
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

    return ElementwiseTestHelper<4, armnn::EqualQueueDescriptor, armnn::DataType::Float32, armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> EqualFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using namespace half_float::literal;

    unsigned int shape[] = { 2, 2, 2, 2 };

    // See dequantized values to the right.
    std::vector<armnn::Half> input0({ 1._h, 1._h, 1._h, 1._h, 6._h, 6._h, 6._h, 6._h,
                                      3._h, 3._h, 3._h, 3._h, 7._h, 7._h, 7._h, 7._h });

    std::vector<armnn::Half> input1({ 2._h, 2._h, 2._h, 2._h, 6._h, 6._h, 6._h, 6._h,
                                      3._h, 3._h, 3._h, 3._h, 5._h, 5._h, 5._h, 5._h });

    std::vector<uint8_t> output({ 0, 0, 0, 0, 1, 1, 1, 1,
                                  1, 1, 1, 1, 0, 0, 0, 0 });

    return ElementwiseTestHelper<4,
                                 armnn::EqualQueueDescriptor,
                                 armnn::DataType::Float16,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<armnn::Half> input0({ 1._h, 2._h, 3._h, 4._h, 5._h, 6._h,
                                      7._h, 8._h, 9._h, 10._h, 11._h, 12._h });

    std::vector<armnn::Half> input1({ 1._h });

    std::vector<uint8_t> output({ 1, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0 });

    return ElementwiseTestHelper<4,
                                 armnn::EqualQueueDescriptor,
                                 armnn::DataType::Float16,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1DVectorFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<armnn::Half> input0({ 1._h, 2._h, 3._h, 4._h, 5._h, 6._h,
                                      7._h, 8._h, 9._h, 10._h, 11._h, 12._h });

    std::vector<armnn::Half> input1({ 1._h, 1._h, 3._h });

    std::vector<uint8_t> output({ 1, 0, 1, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0 });

    return ElementwiseTestHelper<4,
                                 armnn::EqualQueueDescriptor,
                                 armnn::DataType::Float16,
                                 armnn::DataType::Boolean>(
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

    return ElementwiseTestHelper<4,
                                 armnn::EqualQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
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

    return ElementwiseTestHelper<4,
                                 armnn::EqualQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
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

    return ElementwiseTestHelper<4,
                                 armnn::EqualQueueDescriptor,
                                 armnn::DataType::QuantisedAsymm8,
                                 armnn::DataType::Boolean>(
        workloadFactory,
        memoryManager,
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}
