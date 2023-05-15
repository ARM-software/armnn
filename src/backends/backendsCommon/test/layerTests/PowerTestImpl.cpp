//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PowerTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"

LayerTestResult<float, 4> PowerTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<float> input0 =
            {
                    7.f, 3.f, 4.f, 2.f, 6.f, 4.f, 2.f, 1.f,
                    1.f, 1.f, 0.f, 2.f, 9.f, 3.f, 5.f, 3.f
            };

    std::vector<float> input1 =
            {
                    2.f, 3.f, 2.f, 1.f, 2.f, 3.f, 4.f, 3.f,
                    4.f, 5.f, 3.f, 5.f, 2.f, 3.f, 2.f, 0.f
            };

    std::vector<float> output
            {
                    49.f, 27.f, 16.f, 2.f, 36.f, 64.f, 16.f, 1.f,
                    1.f, 1.f, 0.f, 32.f, 81.f, 27.f, 25.f, 1.f
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<float, 4> PowerBroadcast1ElementTest(
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
                    1.f, 4.f, 9.f, 16.f, 25.f, 0.f, 4.f, 1.f
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<float, 4> PowerBroadcastTest(
        armnn::IWorkloadFactory & workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<float> input0 =
            {
                    1.f, 2.f, 3.f, 3.f, 4.f, 4.f,
                    4.f, 0.f, 2.f, 3.f, 4.f, 4.f
            };

    std::vector<float> input1 = { 1.f, 3.f, 1.f };

    std::vector<float> output =
            {
                    1.f, 8.f, 3.f, 3.f, 64.f, 4.f,
                    4.f, 0.f, 2.f, 3.f, 64.f, 4.f
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> PowerFloat16Test(
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
                    2._h, 2._h, 2._h, 2._h, 2._h, 3._h, 3._h, 2._h,
                    1._h, 2._h, 2._h, 4._h, 2._h, 1._h, 3._h, 5._h
            };

    std::vector<armnn::Half> output
            {
                    1._h, 25._h, 1._h, 16._h, 36._h, 1._h, 27._h, 25._h,
                    3._h, 49._h, 36._h, 81._h, 64._h, 4._h, 64._h, 32._h
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> PowerBroadcast1ElementFloat16Test(
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
                    1._h, 4._h, 9._h, 16._h, 25._h, 16._h,
                    1._h, 25._h, 16._h, 4._h, 0._h, 1._h
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> PowerBroadcastFloat16Test(
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
                    4._h, 2._h, 3._h, 4._h, 1._h,  0._h,
                    8._h, 1._h, 1._h, 1._h, 2._h, 4._h
            };

    std::vector<armnn::Half> input1 = { 1._h, 5._h, 3._h };

    std::vector<armnn::Half> output =
            {
                    4._h, 32._h, 27._h, 4._h, 1._h, 0._h,
                    8._h, 1._h, 1._h, 1._h, 32._h, 64._h
            };

    return ElementwiseTestHelper<4, armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> PowerUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape[] = { 1, 1, 2, 2 };

    std::vector<uint8_t> input0 = { 4, 2, 4, 3 };

    std::vector<uint8_t> input1 = { 1, 2, 2, 2 };

    std::vector<uint8_t> output = { 4, 4, 16, 9 };

    return ElementwiseTestHelper<4, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> PowerBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 1 };

    std::vector<uint8_t> input0 = { 4, 5, 1, 0 };

    std::vector<uint8_t> input1 = { 2 };

    std::vector<uint8_t> output = { 16, 25, 1, 0 };

    return ElementwiseTestHelper<4, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> PowerBroadcastUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 1, 2, 2 };
    const unsigned int shape1[] = { 1, 1, 1, 2 };

    std::vector<uint8_t> input0 = { 4, 1, 6, 2 };

    std::vector<uint8_t> input1 = { 2, 6 };

    std::vector<uint8_t> output = { 16, 1, 36, 64 };

    return ElementwiseTestHelper<4, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int16_t, 4> PowerInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int16_t> input0 =
            {
                    1, 5, 1, 4, 4, 9, 3, 7,
                    3, 2, 9, 6, 1, 2, 1, 4
            };

    std::vector<int16_t> input1 =
            {
                    2, 2, 0, 3, 2, 1, 3, 2,
                    4, 4, 2, 1, 7, 5, 4, 2
            };

    std::vector<int16_t> output
            {
                    1, 25, 0, 64, 16, 9, 27, 49,
                    81, 16, 81, 6, 1, 32, 1, 16
            };

    return ElementwiseTestHelper<4, armnn::DataType::QSymmS16, armnn::DataType::QSymmS16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<int16_t, 4> PowerBroadcast1ElementInt16Test(
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
                    1, 4, 9, 16, 25, 0,
                    25, 16, 1, 16, 25, 4
            };

    return ElementwiseTestHelper<4, armnn::DataType::QSymmS16, armnn::DataType::QSymmS16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int16_t, 4> PowerBroadcastInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int16_t> input0 =
            {
                    4, 2, 1, 4, 5, 3,
                    7, 3, 4, 8, 1, 2
            };

    std::vector<int16_t> input1 = { 1, 2, 3 };

    std::vector<int16_t> output =
            {
                    4, 4, 1, 4, 25, 27,
                    7, 9, 64, 8, 1, 8
            };

    return ElementwiseTestHelper<4, armnn::DataType::QSymmS16, armnn::DataType::QSymmS16>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int32_t, 4> PowerInt32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    unsigned int shape[] = { 2, 2, 2, 2 };

    std::vector<int32_t> input0 =
            {
                    1, 3, 4, 3, 1, 4, 2, 1,
                    2, 1, 2, 1, 4, 3, 4, 3
            };

    std::vector<int32_t> input1 =
            {
                    2, 2, 2, 2, 3, 3, 4, 3,
                    4, 4, 4, 4, 1, 3, 1, 3
            };

    std::vector<int32_t> output
            {
                    1, 9, 16, 9, 1, 64, 16, 1,
                    16, 1, 16, 1, 4, 27, 4, 27
            };

    return ElementwiseTestHelper<4, armnn::DataType::Signed32, armnn::DataType::Signed32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape,
            input0,
            shape,
            input1,
            shape,
            output,
            tensorHandleFactory);
}

LayerTestResult<int32_t, 4> PowerBroadcastInt32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const unsigned int shape0[] = { 1, 2, 2, 3 };
    const unsigned int shape1[] = { 1, 1, 1, 3 };

    std::vector<int32_t> input0 =
            {
                    4, 4, 3, 4, 5, 0,
                    5, 8, 1, 3, 9, 2
            };

    std::vector<int32_t> input1 = { 2, 1, 3 };

    std::vector<int32_t> output =
            {
                    16, 4, 27, 16, 5, 0,
                    25, 8, 1, 9, 9, 8
            };

    return ElementwiseTestHelper<4, armnn::DataType::Signed32, armnn::DataType::Signed32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}

LayerTestResult<int32_t, 4> PowerBroadcast1ElementInt32Test(
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
                    1, 4, 9, 16, 25, 9,
                    9, 1, 0, 4, 1, 25
            };

    return ElementwiseTestHelper<4, armnn::DataType::Signed32, armnn::DataType::Signed32>(
            workloadFactory,
            memoryManager,
            armnn::BinaryOperation::Power,
            shape0,
            input0,
            shape1,
            input1,
            shape0,
            output,
            tensorHandleFactory);
}
