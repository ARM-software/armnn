//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <reference/RefTensorHandle.hpp>
#include <reference/RefTensorHandleFactory.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(RefTensorHandleTests)
using namespace armnn;

BOOST_AUTO_TEST_CASE(AcquireAndRelease)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();

    TensorInfo info({ 1, 1, 1, 1 }, DataType::Float32);
    RefTensorHandle handle(info, memoryManager);

    handle.Manage();
    handle.Allocate();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle.Map());

        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 2.5f;

        BOOST_CHECK(buffer[0] == 2.5f); // Memory is writable and readable

    }
    memoryManager->Release();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle.Map());

        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 3.5f;

        BOOST_CHECK(buffer[0] == 3.5f); // Memory is writable and readable
    }
    memoryManager->Release();
}

BOOST_AUTO_TEST_CASE(RefTensorHandleFactoryMemoryManaged)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();
    RefTensorHandleFactory handleFactory(memoryManager);
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);

    // create TensorHandle with memory managed
    auto handle = handleFactory.CreateTensorHandle(info, true);
    handle->Manage();
    handle->Allocate();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle->Map());
        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer
        buffer[0] = 1.5f;
        buffer[1] = 2.5f;
        BOOST_CHECK(buffer[0] == 1.5f); // Memory is writable and readable
        BOOST_CHECK(buffer[1] == 2.5f); // Memory is writable and readable
    }
    memoryManager->Release();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle->Map());
        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer
        buffer[0] = 3.5f;
        buffer[1] = 4.5f;
        BOOST_CHECK(buffer[0] == 3.5f); // Memory is writable and readable
        BOOST_CHECK(buffer[1] == 4.5f); // Memory is writable and readable
    }
    memoryManager->Release();

    float testPtr[2] = { 2.5f, 5.5f };
    // Cannot import as import is disabled
    BOOST_CHECK(!handle->Import(static_cast<void*>(testPtr), MemorySource::Malloc));
}

BOOST_AUTO_TEST_CASE(RefTensorHandleFactoryImport)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();
    RefTensorHandleFactory handleFactory(memoryManager);
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);

    // create TensorHandle without memory managed
    auto handle = handleFactory.CreateTensorHandle(info, false);
    handle->Manage();
    handle->Allocate();
    memoryManager->Acquire();

    // No buffer allocated when import is enabled
    BOOST_CHECK_THROW(handle->Map(), armnn::NullPointerException);

    float testPtr[2] = { 2.5f, 5.5f };
    // Correctly import
    BOOST_CHECK(handle->Import(static_cast<void*>(testPtr), MemorySource::Malloc));
    float* buffer = reinterpret_cast<float*>(handle->Map());
    BOOST_CHECK(buffer != nullptr); // Yields a valid pointer after import
    BOOST_CHECK(buffer == testPtr); // buffer is pointing to testPtr
    // Memory is writable and readable with correct value
    BOOST_CHECK(buffer[0] == 2.5f);
    BOOST_CHECK(buffer[1] == 5.5f);
    buffer[0] = 3.5f;
    buffer[1] = 10.0f;
    BOOST_CHECK(buffer[0] == 3.5f);
    BOOST_CHECK(buffer[1] == 10.0f);
    memoryManager->Release();
}

BOOST_AUTO_TEST_CASE(RefTensorHandleImport)
{
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    handle.Manage();
    handle.Allocate();

    // No buffer allocated when import is enabled
    BOOST_CHECK_THROW(handle.Map(), armnn::NullPointerException);

    float testPtr[2] = { 2.5f, 5.5f };
    // Correctly import
    BOOST_CHECK(handle.Import(static_cast<void*>(testPtr), MemorySource::Malloc));
    float* buffer = reinterpret_cast<float*>(handle.Map());
    BOOST_CHECK(buffer != nullptr); // Yields a valid pointer after import
    BOOST_CHECK(buffer == testPtr); // buffer is pointing to testPtr
    // Memory is writable and readable with correct value
    BOOST_CHECK(buffer[0] == 2.5f);
    BOOST_CHECK(buffer[1] == 5.5f);
    buffer[0] = 3.5f;
    buffer[1] = 10.0f;
    BOOST_CHECK(buffer[0] == 3.5f);
    BOOST_CHECK(buffer[1] == 10.0f);
}

BOOST_AUTO_TEST_CASE(RefTensorHandleGetCapabilities)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();
    RefTensorHandleFactory handleFactory(memoryManager);

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* input = net->AddInputLayer(0);
    IConnectableLayer* output = net->AddOutputLayer(0);
    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    std::vector<Capability> capabilities = handleFactory.GetCapabilities(input,
                                                                         output,
                                                                         CapabilityClass::PaddingRequired);
    BOOST_CHECK(capabilities.empty());
}

BOOST_AUTO_TEST_CASE(RefTensorHandleSupportsInPlaceComputation)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();
    RefTensorHandleFactory handleFactory(memoryManager);

    // RefTensorHandleFactory does not support InPlaceComputation
    ARMNN_ASSERT(!(handleFactory.SupportsInPlaceComputation()));
}

#if !defined(__ANDROID__)
// Only run these tests on non Android platforms
BOOST_AUTO_TEST_CASE(CheckSourceType)
{
    TensorInfo info({1}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

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
    TensorInfo info({1}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    int* testPtr = new int(4);

    handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc);

    // Reusing previously Imported pointer
    BOOST_CHECK(handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc));

    delete testPtr;
}

BOOST_AUTO_TEST_CASE(MisalignedPointer)
{
    TensorInfo info({2}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    // Allocate a 2 int array
    int* testPtr = new int[2];

    // Increment pointer by 1 byte
    void* misalignedPtr = static_cast<void*>(reinterpret_cast<char*>(testPtr) + 1);

    BOOST_CHECK(!handle.Import(misalignedPtr, MemorySource::Malloc));

    delete[] testPtr;
}

#endif

BOOST_AUTO_TEST_SUITE_END()
