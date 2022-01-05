//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <reference/RefTensorHandle.hpp>
#include <reference/RefTensorHandleFactory.hpp>

#include <doctest/doctest.h>

TEST_SUITE("RefTensorHandleTests")
{
using namespace armnn;

TEST_CASE("AcquireAndRelease")
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();

    TensorInfo info({ 1, 1, 1, 1 }, DataType::Float32);
    RefTensorHandle handle(info, memoryManager);

    handle.Manage();
    handle.Allocate();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle.Map());

        CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 2.5f;

        CHECK(buffer[0] == 2.5f); // Memory is writable and readable

    }
    memoryManager->Release();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle.Map());

        CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 3.5f;

        CHECK(buffer[0] == 3.5f); // Memory is writable and readable
    }
    memoryManager->Release();
}

TEST_CASE("RefTensorHandleFactoryMemoryManaged")
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
        CHECK(buffer != nullptr); // Yields a valid pointer
        buffer[0] = 1.5f;
        buffer[1] = 2.5f;
        CHECK(buffer[0] == 1.5f); // Memory is writable and readable
        CHECK(buffer[1] == 2.5f); // Memory is writable and readable
    }
    memoryManager->Release();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle->Map());
        CHECK(buffer != nullptr); // Yields a valid pointer
        buffer[0] = 3.5f;
        buffer[1] = 4.5f;
        CHECK(buffer[0] == 3.5f); // Memory is writable and readable
        CHECK(buffer[1] == 4.5f); // Memory is writable and readable
    }
    memoryManager->Release();

    float testPtr[2] = { 2.5f, 5.5f };
    // Cannot import as import is disabled
    CHECK(!handle->Import(static_cast<void*>(testPtr), MemorySource::Malloc));
}

TEST_CASE("RefTensorHandleFactoryImport")
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
    CHECK_THROWS_AS(handle->Map(), armnn::NullPointerException);

    float testPtr[2] = { 2.5f, 5.5f };
    // Correctly import
    CHECK(handle->Import(static_cast<void*>(testPtr), MemorySource::Malloc));
    float* buffer = reinterpret_cast<float*>(handle->Map());
    CHECK(buffer != nullptr); // Yields a valid pointer after import
    CHECK(buffer == testPtr); // buffer is pointing to testPtr
    // Memory is writable and readable with correct value
    CHECK(buffer[0] == 2.5f);
    CHECK(buffer[1] == 5.5f);
    buffer[0] = 3.5f;
    buffer[1] = 10.0f;
    CHECK(buffer[0] == 3.5f);
    CHECK(buffer[1] == 10.0f);
    memoryManager->Release();
}

TEST_CASE("RefTensorHandleImport")
{
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    handle.Manage();
    handle.Allocate();

    // No buffer allocated when import is enabled
    CHECK_THROWS_AS(handle.Map(), armnn::NullPointerException);

    float testPtr[2] = { 2.5f, 5.5f };
    // Correctly import
    CHECK(handle.Import(static_cast<void*>(testPtr), MemorySource::Malloc));
    float* buffer = reinterpret_cast<float*>(handle.Map());
    CHECK(buffer != nullptr); // Yields a valid pointer after import
    CHECK(buffer == testPtr); // buffer is pointing to testPtr
    // Memory is writable and readable with correct value
    CHECK(buffer[0] == 2.5f);
    CHECK(buffer[1] == 5.5f);
    buffer[0] = 3.5f;
    buffer[1] = 10.0f;
    CHECK(buffer[0] == 3.5f);
    CHECK(buffer[1] == 10.0f);
}

TEST_CASE("RefTensorHandleGetCapabilities")
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
    CHECK(capabilities.empty());
}

TEST_CASE("RefTensorHandleSupportsInPlaceComputation")
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();
    RefTensorHandleFactory handleFactory(memoryManager);

    // RefTensorHandleFactory does not support InPlaceComputation
    ARMNN_ASSERT(!(handleFactory.SupportsInPlaceComputation()));
}

TEST_CASE("TestManagedConstTensorHandle")
{
    // Initialize arguments
    void* mem = nullptr;
    TensorInfo info;

    // Use PassthroughTensor as others are abstract
    auto passThroughHandle = std::make_shared<PassthroughTensorHandle>(info, mem);

    // Test managed handle is initialized with m_Mapped unset and once Map() called its set
    ManagedConstTensorHandle managedHandle(passThroughHandle);
    CHECK(!managedHandle.IsMapped());
    managedHandle.Map();
    CHECK(managedHandle.IsMapped());

    // Test it can then be unmapped
    managedHandle.Unmap();
    CHECK(!managedHandle.IsMapped());

    // Test member function
    CHECK(managedHandle.GetTensorInfo() == info);

    // Test that nullptr tensor handle doesn't get mapped
    ManagedConstTensorHandle managedHandleNull(nullptr);
    CHECK(!managedHandleNull.IsMapped());
    CHECK_THROWS_AS(managedHandleNull.Map(), armnn::Exception);
    CHECK(!managedHandleNull.IsMapped());

    // Check Unmap() when m_Mapped already false
    managedHandleNull.Unmap();
    CHECK(!managedHandleNull.IsMapped());
}

#if !defined(__ANDROID__)
// Only run these tests on non Android platforms
TEST_CASE("CheckSourceType")
{
    TensorInfo info({1}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    int* testPtr = new int(4);

    // Not supported
    CHECK(!handle.Import(static_cast<void *>(testPtr), MemorySource::DmaBuf));

    // Not supported
    CHECK(!handle.Import(static_cast<void *>(testPtr), MemorySource::DmaBufProtected));

    // Supported
    CHECK(handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc));

    delete testPtr;
}

TEST_CASE("ReusePointer")
{
    TensorInfo info({1}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    int* testPtr = new int(4);

    handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc);

    // Reusing previously Imported pointer
    CHECK(handle.Import(static_cast<void *>(testPtr), MemorySource::Malloc));

    delete testPtr;
}

TEST_CASE("MisalignedPointer")
{
    TensorInfo info({2}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    // Allocate a 2 int array
    int* testPtr = new int[2];

    // Increment pointer by 1 byte
    void* misalignedPtr = static_cast<void*>(reinterpret_cast<char*>(testPtr) + 1);

    CHECK(!handle.Import(misalignedPtr, MemorySource::Malloc));

    delete[] testPtr;
}

TEST_CASE("CheckCanBeImported")
{
    TensorInfo info({1}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    int* testPtr = new int(4);

    // Not supported
    CHECK(!handle.CanBeImported(static_cast<void *>(testPtr), MemorySource::DmaBuf));

    // Supported
    CHECK(handle.CanBeImported(static_cast<void *>(testPtr), MemorySource::Malloc));

    delete testPtr;

}

TEST_CASE("MisalignedCanBeImported")
{
    TensorInfo info({2}, DataType::Float32);
    RefTensorHandle handle(info, static_cast<unsigned int>(MemorySource::Malloc));

    // Allocate a 2 int array
    int* testPtr = new int[2];

    // Increment pointer by 1 byte
    void* misalignedPtr = static_cast<void*>(reinterpret_cast<char*>(testPtr) + 1);

    CHECK(!handle.Import(misalignedPtr, MemorySource::Malloc));

    delete[] testPtr;
}

#endif

}
