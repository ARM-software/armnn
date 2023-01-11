//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/ICustomAllocator.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/backends/TensorHandle.hpp>
// Requires the OpenCl backend to be included (GpuAcc)
#include <cl/ClBackend.hpp>
#include <doctest/doctest.h>
#include <backendsCommon/DefaultAllocator.hpp>
#include <armnnTestUtils/MockBackend.hpp>
#include <cl/ClBackendDefaultAllocator.hpp>

using namespace armnn;


namespace
{

TEST_SUITE("DefaultAllocatorTests")
{

TEST_CASE("DefaultAllocatorTest")
{
    float number = 3;

    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<DefaultAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", std::move(customAllocator)}};
    IRuntimePtr run = IRuntime::Create(options);

    // Creates structures for input & output
    unsigned int numElements = inputTensorInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    void* alignedInputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, 0);

    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    std::fill_n(inputPtr, numElements, number);
    CHECK(inputPtr[0] == 3);

    auto& backendRegistry = armnn::BackendRegistryInstance();
    backendRegistry.DeregisterAllocator(ClBackend::GetIdStatic());
}

TEST_CASE("DefaultAllocatorTestMulti")
{
    float number = 3;

    TensorInfo inputTensorInfo(TensorShape({2, 1}), DataType::Float32);

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<DefaultAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", std::move(customAllocator)}};
    IRuntimePtr run = IRuntime::Create(options);

    // Creates structures for input & output
    unsigned int numElements = inputTensorInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    void* alignedInputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, 0);
    void* alignedInputPtr2 = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, 0);

    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    std::fill_n(inputPtr, numElements, number);
    CHECK(inputPtr[0] == 3);
    CHECK(inputPtr[1] == 3);

    auto* inputPtr2 = reinterpret_cast<float*>(alignedInputPtr2);
    std::fill_n(inputPtr2, numElements, number);
    CHECK(inputPtr2[0] == 3);
    CHECK(inputPtr2[1] == 3);

    // No overlap
    CHECK(inputPtr[0] == 3);
    CHECK(inputPtr[1] == 3);

    auto& backendRegistry = armnn::BackendRegistryInstance();
    backendRegistry.DeregisterAllocator(ClBackend::GetIdStatic());
}

TEST_CASE("DefaultAllocatorTestMock")
{
    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    IRuntimePtr run = IRuntime::Create(options);

    // Initialize Mock Backend
    MockBackendInitialiser initialiser;
    auto factoryFun = BackendRegistryInstance().GetFactory(MockBackend().GetIdStatic());
    ARMNN_ASSERT(factoryFun != nullptr);
    auto backend = factoryFun();
    auto defaultAllocator = backend->GetDefaultAllocator();

    // GetMemorySourceType
    CHECK(defaultAllocator->GetMemorySourceType() == MemorySource::Malloc);

    size_t totalBytes = 1 * sizeof(float);
    // Allocate
    void* ptr = defaultAllocator->allocate(totalBytes, 0);

    // GetMemoryRegionAtOffset
    CHECK(defaultAllocator->GetMemoryRegionAtOffset(ptr, 0, 0));

    // Free
    defaultAllocator->free(ptr);

    // Clean up
    auto& backendRegistry = armnn::BackendRegistryInstance();
    backendRegistry.Deregister(MockBackend().GetIdStatic());
    backendRegistry.DeregisterAllocator(ClBackend::GetIdStatic());
}

}


TEST_SUITE("ClDefaultAllocatorTests")
{

TEST_CASE("ClDefaultAllocatorTest")
{
    float number = 3;

    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<ClBackendDefaultAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", std::move(customAllocator)}};
    IRuntimePtr run = IRuntime::Create(options);

    // Creates structures for input & output
    unsigned int numElements = inputTensorInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    void* alignedInputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, 0);

    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    std::fill_n(inputPtr, numElements, number);
    CHECK(inputPtr[0] == 3);

    auto& backendRegistry = armnn::BackendRegistryInstance();
    backendRegistry.DeregisterAllocator(ClBackend::GetIdStatic());
}

TEST_CASE("ClDefaultAllocatorTestMulti")
{
    float number = 3;

    TensorInfo inputTensorInfo(TensorShape({2, 1}), DataType::Float32);

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<ClBackendDefaultAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", std::move(customAllocator)}};
    IRuntimePtr run = IRuntime::Create(options);

    // Creates structures for input & output
    unsigned int numElements = inputTensorInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    void* alignedInputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, 0);
    void* alignedInputPtr2 = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, 0);

    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    std::fill_n(inputPtr, numElements, number);
    CHECK(inputPtr[0] == 3);
    CHECK(inputPtr[1] == 3);

    auto* inputPtr2 = reinterpret_cast<float*>(alignedInputPtr2);
    std::fill_n(inputPtr2, numElements, number);
    CHECK(inputPtr2[0] == 3);
    CHECK(inputPtr2[1] == 3);

    // No overlap
    CHECK(inputPtr[0] == 3);
    CHECK(inputPtr[1] == 3);

    auto& backendRegistry = armnn::BackendRegistryInstance();
    backendRegistry.DeregisterAllocator(ClBackend::GetIdStatic());
}

}

} // namespace armnn