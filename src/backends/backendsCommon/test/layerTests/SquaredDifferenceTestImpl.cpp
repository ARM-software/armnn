//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SquaredDifferenceTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"

LayerTestResult<float, 4> SquaredDifferenceTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<float> input0 =
            {
                 7.f, 3.f, 4.f, 2.f, 6.f, 4.f, 2.f, 1.f,
                 3.f, 1.f, 0.f, 1.f, 4.f, 3.f, 4.f, 3.f
            };

    std::vector<float> input1 =
            {
                 5.f, 3.f, 2.f, 5.f, 3.f, 3.f, 4.f, 3.f,
                 4.f, 4.f, 3.f, 2.f, 5.f, 5.f, 5.f, 5.f
            };

    std::vector<float> output
            {
                 4.f, 0.f, 4.f, 9.f, 9.f, 1.f, 4.f, 4.f,
                 1.f, 9.f, 9.f, 1.f, 1.f, 4.f, 1.f, 4.f
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<float, 4> SquaredDiffBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    unsigned int shape0[] = { 1, 2, 2, 2 };
    unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<float> input0 =
            {
                   1.f, 2.f, 3.f, 4.f, 5.f, 0.f, 2.f, 1.f
            };

    std::vector<float> input1 = { 2.f };

    std::vector<float> output =
            {
                1.f, 0.f, 1.f, 4.f, 9.f, 4.f, 0.f, 1.f
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<float, 4> SquaredDiffBroadcastTest(
       armnn::IWorkloadFactory & workloadFactory,
       const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
       const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0 =
            {
                1.f, 2.f, 3.f, 3.f, 6.f, 4.f,
                4.f, 0.f, 2.f, 3.f, 4.f, 4.f
            };

    std::vector<float> input1 = { 1.f, 3.f, 1.f };

    std::vector<float> output =
            {
                0.f, 1.f, 4.f, 4.f, 9.f, 9.f,
                9.f, 9.f, 1.f, 4.f, 1.f, 9.f
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> SquaredDifferenceFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    using namespace half_float::literal;

    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<armnn::Half> input0 =
            {
                1._h, 5._h, 1._h, 4._h, 6._h, 1._h, 3._h, 5._h,
                3._h, 7._h, 6._h, 3._h, 8._h, 4._h, 4._h, 2._h
            };

    std::vector<armnn::Half> input1 =
            {
                2._h, 2._h, 2._h, 2._h, 3._h, 3._h, 3._h, 3._h,
                4._h, 4._h, 4._h, 4._h, 5._h, 6._h, 5._h, 5._h
            };

    std::vector<armnn::Half> output
            {
                1._h, 9._h, 1._h, 4._h, 9._h, 4._h, 0._h, 4._h,
                1._h, 9._h, 4._h, 1._h, 9._h, 4._h, 1._h, 9._h
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> SquaredDiffBroadcast1ElementFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    using namespace half_float::literal;

    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<armnn::Half> input0 =
            {
                1._h, 2._h, 3._h, 4._h, 5._h, 4._h,
                1._h, 5._h, 4._h, 2._h, 0._h, 1._h
            };

    std::vector<armnn::Half> input1 = { 2._h };

    std::vector<armnn::Half> output =
            {
                1._h, 0._h, 1._h, 4._h, 9._h, 4._h,
                1._h, 9._h, 4._h, 0._h, 4._h, 1._h
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> SquaredDiffBroadcastFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    using namespace half_float::literal;
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<armnn::Half> input0 =
            {
                4._h, 2._h, 3._h, 4._h, 5._h,  5._h,
                2._h, 8._h, 1._h, 1._h, 2._h, 4._h
            };

    std::vector<armnn::Half> input1 = { 1._h, 5._h, 3._h };

    std::vector<armnn::Half> output =
            {
                9._h, 9._h, 0._h, 9._h, 0._h, 4._h,
                1._h, 9._h, 4._h, 0._h, 9._h, 1._h
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> SquaredDifferenceUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 2, 2 };

    std::vector<uint8_t> input0 = { 4, 2, 4, 3 };

    std::vector<uint8_t> input1 = { 1, 2, 2, 2 };

    std::vector<uint8_t> output = { 9, 0, 4, 1 };

    return ElementwiseTestHelper<4, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> SquaredDiffBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0 = { 4, 5, 1, 0 };

    std::vector<uint8_t> input1 = { 2 };

    std::vector<uint8_t> output = { 4, 9, 1, 4 };

    return ElementwiseTestHelper<4, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> SquaredDiffBroadcastUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<uint8_t> input0 = { 4, 12, 3, 6 };

    std::vector<uint8_t> input1 = { 2, 9 };

    std::vector<uint8_t> output = { 4, 9, 1, 9 };

    return ElementwiseTestHelper<4, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int16_t, 4> SquaredDifferenceInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int16_t> input0 =
            {
                1, 5, 1, 4, 6, 9, 6, 5,
                3, 2, 3, 6, 4, 4, 1, 4
            };

    std::vector<int16_t> input1 =
            {
                2, 2, 0, 4, 3, 7, 3, 3,
                4, 4, 4, 9, 7, 5, 4, 5
            };

    std::vector<int16_t> output
            {
                1, 9, 1, 0, 9, 4, 9, 4,
                1, 4, 1, 9, 9, 1, 9, 1
            };

    return ElementwiseTestHelper<4, armnn::DataType::QSymmS16, armnn::DataType::QSymmS16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<int16_t, 4> SquaredDiffBroadcast1ElementInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int16_t> input0 =
            {
                1, 2, 3, 4, 5, 0,
                5, 4, 1, 4, 5, 2
            };

    std::vector<int16_t> input1 = { 2 };

    std::vector<int16_t> output =
            {
                1, 0, 1, 4, 9, 4,
                9, 4, 1, 4, 9, 0
            };

    return ElementwiseTestHelper<4, armnn::DataType::QSymmS16, armnn::DataType::QSymmS16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int16_t, 4> SquaredDiffBroadcastInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int16_t> input0 =
            {
                4, 2, 1, 4, 5, 6,
                7, 3, 5, 8, 1, 5
            };

    std::vector<int16_t> input1 = { 7, 2, 3 };

    std::vector<int16_t> output =
            {
                9, 0, 4, 9, 9, 9,
                0, 1, 4, 1, 1, 4
            };

    return ElementwiseTestHelper<4, armnn::DataType::QSymmS16, armnn::DataType::QSymmS16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int32_t, 4> SquaredDifferenceInt32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int32_t> input0 =
            {
                1, 3, 4, 3, 6, 4, 2, 6,
                3, 1, 3, 1, 4, 3, 4, 3
            };

    std::vector<int32_t> input1 =
            {
                2, 2, 2, 2, 3, 3, 4, 3,
                4, 4, 4, 4, 5, 5, 5, 5
            };

    std::vector<int32_t> output
            {
                1, 1, 4, 1, 9, 1, 4, 9,
                1, 9, 1, 9, 1, 4, 1, 4
            };

    return ElementwiseTestHelper<4, armnn::DataType::Signed32, armnn::DataType::Signed32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<int32_t, 4> SquaredDiffBroadcastInt32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int32_t> input0 =
            {
                4, 4, 3, 4, 5, 6,
                5, 8, 6, 3, 9, 5
            };

    std::vector<int32_t> input1 = { 2, 7, 3 };

    std::vector<int32_t> output =
            {
                4, 9, 0, 4, 4, 9,
                9, 1, 9, 1, 4, 4
            };

    return ElementwiseTestHelper<4, armnn::DataType::Signed32, armnn::DataType::Signed32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int32_t, 4> SquaredDiffBroadcast1ElementInt32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<int32_t> input0 =
            {
                1, 2, 3, 4, 5, 3,
                3, 1, 0, 2, 1, 5
            };

    std::vector<int32_t> input1 = { 2 };

    std::vector<int32_t> output =
            {
                1, 0, 1, 4, 9, 1,
                1, 1, 4, 0, 1, 9
            };

    return ElementwiseTestHelper<4, armnn::DataType::Signed32, armnn::DataType::Signed32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::SqDiff,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}
