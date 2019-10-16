//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GreaterTestImpl.hpp"

#include "ComparisonTestImpl.hpp"

#include <Half.hpp>

LayerTestResult<uint8_t, 4> GreaterSimpleTest(armnn::IWorkloadFactory& workloadFactory,
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

    std::vector<float> input1 =
    {
        1.f, 1.f, 1.f, 1.f,  3.f, 3.f, 3.f, 3.f,
        5.f, 5.f, 5.f, 5.f,  4.f, 4.f, 4.f, 4.f
    };

    std::vector<uint8_t> output =
    {
        0, 0, 0, 0,  1, 1, 1, 1,
        0, 0, 0, 0,  0, 0, 0, 0
    };

    return ComparisonTestImpl<4, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<float> input0 = { 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> input1 = { 1 };

    std::vector<uint8_t> output = { 0, 1, 1, 1, 1, 1, 1, 1};

    return ComparisonTestImpl<4, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1DVectorTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0 =
    {
        1.0f, 2.9f, 2.1f,  4.0f,  5.0f,  6.0f,
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f
    };

    std::vector<float> input1 = { 1.f, 3.f, 2.f };

    std::vector<uint8_t> output =
    {
        0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };

    return ComparisonTestImpl<4, armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> GreaterFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        1._h, 1._h, 1._h, 1._h,  3._h, 3._h, 3._h, 3._h,
        5._h, 5._h, 5._h, 5._h,  4._h, 4._h, 4._h, 4._h
    };

    std::vector<uint8_t> output =
    {
        0, 0, 0, 0,  1, 1, 1, 1,
        0, 0, 0, 0,  0, 0, 0, 0
    };

    return ComparisonTestImpl<4,armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using namespace half_float::literal;

    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<armnn::Half> input0 = { 1._h, 2._h, 3._h, 4._h, 5._h, 6._h, 7._h, 8._h };
    std::vector<armnn::Half> input1 = { 1._h };

    std::vector<uint8_t> output = { 0, 1, 1, 1, 1, 1, 1, 1};

    return ComparisonTestImpl<4, armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1DVectorFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<armnn::Half> input0 =
    {
        1.0_h, 2.9_h, 2.1_h,  4.0_h,  5.0_h,  6.0_h,
        7.0_h, 8.0_h, 9.0_h, 10.0_h, 11.0_h, 12.0_h
    };

    std::vector<armnn::Half> input1 = { 1._h, 3._h, 2._h };

    std::vector<uint8_t> output =
    {
        0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };

    return ComparisonTestImpl<4, armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
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
    std::vector<uint8_t> input0 =
    {
        1, 1, 1, 1, 6, 6, 6, 6,
        3, 3, 3, 3, 5, 5, 5, 5
    };

    std::vector<uint8_t> input1 =
    {
        2, 2, 2, 2, 6, 6, 6, 6,
        2, 2, 2, 2, 5, 5, 5, 5
    };

    std::vector<uint8_t> output =
    {
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0
    };

    return ComparisonTestImpl<4, armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape,
        input0,
        shape,
        input1,
        shape,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<uint8_t> input1 = { 1 };

    std::vector<uint8_t> output =
    {
        0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };

    return ComparisonTestImpl<4, armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1DVectorUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<uint8_t> input0 =
    {
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    };

    std::vector<uint8_t> input1 = { 1, 1, 3 };

    std::vector<uint8_t> output =
    {
        0, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };

    return ComparisonTestImpl<4, armnn::DataType::QuantisedAsymm8>(
        workloadFactory,
        memoryManager,
        armnn::ComparisonDescriptor(armnn::ComparisonOperation::Greater),
        shape0,
        input0,
        shape1,
        input1,
        shape0,
        output);
}
