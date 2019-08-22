//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <reference/RefTensorHandle.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(RefTensorHandleTests)
using namespace armnn;

BOOST_AUTO_TEST_CASE(AcquireAndRelease)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();

    TensorInfo info({1,1,1,1}, DataType::Float32);
    RefTensorHandle handle(info, memoryManager);

    handle.Manage();
    handle.Allocate();

    memoryManager->Acquire();
    {
        float *buffer = reinterpret_cast<float *>(handle.Map());

        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 2.5f;

        BOOST_CHECK(buffer[0] == 2.5f); // Memory is writable and readable

    }
    memoryManager->Release();

    memoryManager->Acquire();
    {
        float *buffer = reinterpret_cast<float *>(handle.Map());

        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 3.5f;

        BOOST_CHECK(buffer[0] == 3.5f); // Memory is writable and readable
    }
    memoryManager->Release();
}

#if !defined(__ANDROID__)
// Only run these tests on non Android platforms
BOOST_AUTO_TEST_CASE(CheckSourceType)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();

    TensorInfo info({1}, DataType::Float32);
    RefTensorHandle handle(info, memoryManager, static_cast<unsigned int>(MemorySource::Malloc));

    int* testPtr = new int(4);

    // Not supported
    BOOST_CHECK(!handle.Import(static_cast<void *>(testPtr), MemorySource::DmaBuf));

    // Not supported
    BOOST_CHECK(!handle.Import(static_cast<void *>(testPtr), MemorySource::DmaBufProtected));

    // Supported
    BOOST_CHECK(handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc));

    delete testPtr;
}

BOOST_AUTO_TEST_CASE(ReusePointer)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();

    TensorInfo info({1}, DataType::Float32);
    RefTensorHandle handle(info, memoryManager, static_cast<unsigned int>(MemorySource::Malloc));

    int* testPtr = new int(4);

    handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc);

    // Reusing previously Imported pointer
    BOOST_CHECK(handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc));

    delete testPtr;
}

BOOST_AUTO_TEST_CASE(MisalignedPointer)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();

    TensorInfo info({2}, DataType::Float32);
    RefTensorHandle handle(info, memoryManager, static_cast<unsigned int>(MemorySource::Malloc));

    // Allocates a 2 int array
    int* testPtr = new int[2];
    int* misalignedPtr = testPtr + 1;

    BOOST_CHECK(!handle.Import(static_cast<void *>(misalignedPtr), MemorySource::Malloc));

    delete[] testPtr;
}

#endif

BOOST_AUTO_TEST_SUITE_END()