//
// Copyright © 2020-2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <Graph.hpp>
#include <Network.hpp>

#include <aclCommon/BaseMemoryManager.hpp>

#include <neon/NeonTensorHandle.hpp>
#include <neon/NeonTensorHandleFactory.hpp>

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <GraphUtils.hpp>
#include <arm_compute/runtime/Allocator.h>
#include <CommonTestUtils.hpp>

#include <doctest/doctest.h>

TEST_SUITE("NeonTensorHandleTests")
{
using namespace armnn;

TEST_CASE("NeonTensorHandleGetCapabilitiesNoPadding")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    INetworkPtr network(INetwork::Create());

    // Add the layers
    IConnectableLayer* input = network->AddInputLayer(0);
    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;
    IConnectableLayer* softmax = network->AddSoftmaxLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // No padding required for input
    std::vector<Capability> capabilities = handleFactory.GetCapabilities(input,
                                                                         softmax,
                                                                         CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // No padding required for Softmax
    capabilities = handleFactory.GetCapabilities(softmax, output, CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // No padding required for output
    capabilities = handleFactory.GetCapabilities(output, nullptr, CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());
}

TEST_CASE("NeonTensorHandleGetCapabilitiesPadding")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    INetworkPtr network(INetwork::Create());

    // Add the layers
    IConnectableLayer* input = network->AddInputLayer(0);
    Pooling2dDescriptor descriptor;
    IConnectableLayer* pooling = network->AddPooling2dLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // No padding required for input
    std::vector<Capability> capabilities = handleFactory.GetCapabilities(input,
                                                                         pooling,
                                                                         CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // No padding required for output
    capabilities = handleFactory.GetCapabilities(output, nullptr, CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // Padding required for Pooling2d
    capabilities = handleFactory.GetCapabilities(pooling, output, CapabilityClass::PaddingRequired);
    CHECK(capabilities.size() == 1);
    CHECK((capabilities[0].m_CapabilityClass == CapabilityClass::PaddingRequired));
    CHECK(capabilities[0].m_Value);
}

TEST_CASE("NeonTensorHandleFactoryMemoryManaged")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>(
        std::make_unique<arm_compute::Allocator>(),
        BaseMemoryManager::MemoryAffinity::Offset);
    NeonTensorHandleFactory handleFactory(memoryManager);
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
    CHECK_THROWS_AS(handle->Import(static_cast<void*>(testPtr), MemorySource::Malloc), MemoryImportException);
}

TEST_CASE("NeonTensorHandleFactoryImport")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>(
        std::make_unique<arm_compute::Allocator>(),
        BaseMemoryManager::MemoryAffinity::Offset);
    NeonTensorHandleFactory handleFactory(memoryManager);
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);

    // create TensorHandle without memory managed
    auto handle = handleFactory.CreateTensorHandle(info, false);
    handle->Manage();
    handle->Allocate();
    memoryManager->Acquire();

    // No buffer allocated when import is enabled
    CHECK((PolymorphicDowncast<NeonTensorHandle*>(handle.get()))->GetTensor().buffer() == nullptr);

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

TEST_CASE("NeonTensorHandleCanBeImported")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>(
        std::make_unique<arm_compute::Allocator>(),
        BaseMemoryManager::MemoryAffinity::Offset);
    NeonTensorHandleFactory handleFactory(memoryManager);
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);

    // create TensorHandle (Memory Managed status is irrelevant)
    auto handle = handleFactory.CreateTensorHandle(info, false);

    // Create an aligned buffer
    float alignedBuffer[2] = { 2.5f, 5.5f };
    // Check aligned buffers return true
    CHECK(handle->CanBeImported(&alignedBuffer, MemorySource::Malloc) == true);

    // Create a misaligned buffer from the aligned one
    float* misalignedBuffer = reinterpret_cast<float*>(reinterpret_cast<char*>(alignedBuffer) + 1);
    // Check misaligned buffers return false
    CHECK(handle->CanBeImported(static_cast<void*>(misalignedBuffer), MemorySource::Malloc) == false);
}

TEST_CASE("NeonTensorHandleSupportsInPlaceComputation")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    // NeonTensorHandleFactory supports InPlaceComputation
    CHECK(handleFactory.SupportsInPlaceComputation() == true);
}

}
