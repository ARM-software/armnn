//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/WorkloadUtils.hpp>
#include <armnnTestUtils/MockMemoryManager.hpp>
#include <armnnTestUtils/MockTensorHandle.hpp>
#include <doctest/doctest.h>

TEST_SUITE("WorkloadUtilsTest")
{
using namespace armnn;

TEST_CASE("CopyTensorContents")
{
    // Two tensors of 1 float. Set source to a value and destination to 0.
    // Copy source to destination and make sure destination is copied.
    std::shared_ptr<MockMemoryManager> memoryManager = std::make_shared<MockMemoryManager>();
    TensorInfo info({ 1, 1, 1, 1 }, DataType::Float32);
    MockTensorHandle srcTensorHandle(info, memoryManager);
    MockTensorHandle destTensorHandle(info, memoryManager);
    srcTensorHandle.Allocate();
    float* buffer = reinterpret_cast<float*>(srcTensorHandle.Map());
    buffer[0] = 2.5f;

    destTensorHandle.Allocate();
    buffer = reinterpret_cast<float*>(destTensorHandle.Map());
    buffer[0] = 0.0f;

    auto copyFunc = [](void* dst, const void* src, size_t size)
    {
        memcpy(dst, src, size);
    };
    CopyTensorContentsGeneric(&srcTensorHandle, &destTensorHandle, copyFunc);
    // After copy the destination should be 2.5.
    buffer = reinterpret_cast<float*>(destTensorHandle.Map());
    CHECK(buffer[0] == 2.5f);
    // Also make sure the first buffer hasn't changed.
    buffer = reinterpret_cast<float*>(srcTensorHandle.Map());
    CHECK(buffer[0] == 2.5f);
}

TEST_CASE("CopyTensorContents_UnallocatedTensors")
{
    // Standard copy lambda
    auto copyFunc = [](void* dst, const void* src, size_t size)
    {
        memcpy(dst, src, size);
    };

    // Two tensors of 1 float. The source will be managed but unallocated. This should throw an exception.
    std::shared_ptr<MockMemoryManager> memoryManager = std::make_shared<MockMemoryManager>();
    TensorInfo info({ 1, 1, 1, 1 }, DataType::Float32);
    MockTensorHandle unallocatedSource(info, memoryManager);
    unallocatedSource.Manage();
    MockTensorHandle destTensorHandle(info, memoryManager);
    destTensorHandle.Allocate();
    CHECK_THROWS_AS(CopyTensorContentsGeneric(&unallocatedSource, &destTensorHandle, copyFunc),
                    const armnn::MemoryValidationException&);

    // Same test for destination tensor.
    MockTensorHandle unallocatedDest(info, memoryManager);
    unallocatedDest.Manage();
    MockTensorHandle srcTensorHandle(info, memoryManager);
    srcTensorHandle.Allocate();
    CHECK_THROWS_AS(CopyTensorContentsGeneric(&srcTensorHandle, &unallocatedDest, copyFunc),
                    const armnn::MemoryValidationException&);

}

TEST_CASE("CopyTensorContents_DifferentTensorSizes")
{
    std::shared_ptr<MockMemoryManager> memoryManager = std::make_shared<MockMemoryManager>();
    // Standard copy lambda
    auto copyFunc = [](void* dst, const void* src, size_t size)
    {
        memcpy(dst, src, size);
    };

    // This is an odd test case. We'll make the destination tensor 1 element smaller than the source.
    // In this case the tensor will be truncated.
    TensorInfo largerInfo({ 1, 1, 1, 6 }, DataType::Float32);
    TensorInfo smallerInfo({ 1, 1, 1, 5 }, DataType::Float32);
    MockTensorHandle srcLargerTensorHandle(largerInfo, memoryManager);
    srcLargerTensorHandle.Allocate();
    float* buffer = reinterpret_cast<float*>(srcLargerTensorHandle.Map());
    // We'll set a value in the 5th elements, this should be copied over.
    buffer[4] = 5.1f;
    MockTensorHandle destSmallerTensorHandle(smallerInfo, memoryManager);
    destSmallerTensorHandle.Allocate();
    buffer = reinterpret_cast<float*>(destSmallerTensorHandle.Map());
    buffer[4] = 5.2f; // This should be overwritten.
    CopyTensorContentsGeneric(&srcLargerTensorHandle, &destSmallerTensorHandle, copyFunc);
    CHECK(buffer[4] == 5.1f);

    // Same test case but with destination being larger than source.
    MockTensorHandle srcSmallerTensorHandle(smallerInfo, memoryManager);
    srcSmallerTensorHandle.Allocate();
    buffer = reinterpret_cast<float*>(srcSmallerTensorHandle.Map());
    // We'll set a value in the 5th elements, this should be copied over.
    buffer[4] = 5.1f;
    MockTensorHandle destLargerTensorHandle(largerInfo, memoryManager);
    destLargerTensorHandle.Allocate();
    buffer = reinterpret_cast<float*>(destLargerTensorHandle.Map());
    buffer[4] = 5.2f; // This should be overwritten.
    buffer[5] = 6.2f; // This should NOT be overwritten.
    CopyTensorContentsGeneric(&srcSmallerTensorHandle, &destLargerTensorHandle, copyFunc);
    CHECK(buffer[4] == 5.1f); // Has been copied.
    CHECK(buffer[5] == 6.2f); // Should be untouched.
}

TEST_CASE("CopyTensorContents_MixedDataTypes")
{
    // This is almost a pointless test, but it may detect problems with future changes.
    // We'll copy a float tensor into a uint8 tensor of the same size. It should
    // work without error.
    std::shared_ptr<MockMemoryManager> memoryManager = std::make_shared<MockMemoryManager>();
    // Standard copy lambda
    auto copyFunc = [](void* dst, const void* src, size_t size)
    {
        memcpy(dst, src, size);
    };

    TensorInfo floatInfo({ 1, 1, 1, 2 }, DataType::Float32);
    TensorInfo intInfo({ 1, 1, 1, 8 }, DataType::QAsymmU8);
    MockTensorHandle floatTensorHandle(floatInfo, memoryManager);
    floatTensorHandle.Allocate();
    float* floatBuffer = reinterpret_cast<float*>(floatTensorHandle.Map());
    floatBuffer[0] = 1.1f; // This should be 0x3f8ccccd or something very close.
    MockTensorHandle uintTensorHandle(intInfo, memoryManager);
    uintTensorHandle.Allocate();
    uint8_t* intBuffer = reinterpret_cast<uint8_t*>(uintTensorHandle.Map());
    intBuffer[0] = 0;

    CopyTensorContentsGeneric(&floatTensorHandle, &uintTensorHandle, copyFunc);
    CHECK(intBuffer[0] == 0xcd); // Make sure the data has been copied over.
}

}
