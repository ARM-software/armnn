//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>

#include <cl/ClImportTensorHandle.hpp>
#include <cl/ClImportTensorHandleFactory.hpp>
#include <cl/test/ClContextControlFixture.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(ClImportTensorHandleTests)

BOOST_FIXTURE_TEST_CASE(ClMallocImport, ClContextControlFixture)
{
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));

    TensorInfo info({ 1, 24, 16, 3 }, DataType::Float32);
    unsigned int numElements = info.GetNumElements();

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info);

    // Get CLtensor
    arm_compute::CLTensor& tensor = PolymorphicDowncast<ClImportTensorHandle*>(handle.get())->GetTensor();

    // Create and configure activation function
    const arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::CLActivationLayer act_func;
    act_func.configure(&tensor, nullptr, act_info);

    // Allocate user memory
    const size_t totalBytes = tensor.info()->total_size();
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment;
    auto testData = std::make_unique<uint8_t[]>(space);
    void* alignedPtr = testData.get();
    BOOST_CHECK(std::align(alignment, totalBytes, alignedPtr, space));

    // Import memory
    BOOST_CHECK(handle->Import(alignedPtr, armnn::MemorySource::Malloc));

    // Input with negative values
    auto* typedPtr = reinterpret_cast<float*>(alignedPtr);
    std::fill_n(typedPtr, numElements, -5.0f);

    // Execute function and sync
    act_func.run();
    arm_compute::CLScheduler::get().sync();

    // Validate result by checking that the output has no negative values
    for(unsigned int i = 0; i < numElements; ++i)
    {
        BOOST_ASSERT(typedPtr[i] >= 0);
    }
}

BOOST_FIXTURE_TEST_CASE(ClIncorrectMemorySourceImport, ClContextControlFixture)
{
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));

    TensorInfo info({ 1, 24, 16, 3 }, DataType::Float32);

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info);

    // Get CLtensor
    arm_compute::CLTensor& tensor = PolymorphicDowncast<ClImportTensorHandle*>(handle.get())->GetTensor();

    // Allocate user memory
    const size_t totalBytes = tensor.info()->total_size();
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment;
    auto testData = std::make_unique<uint8_t[]>(space);
    void* alignedPtr = testData.get();
    BOOST_CHECK(std::align(alignment, totalBytes, alignedPtr, space));

    // Import memory
    BOOST_CHECK_THROW(handle->Import(alignedPtr, armnn::MemorySource::Undefined), MemoryImportException);
}

BOOST_FIXTURE_TEST_CASE(ClInvalidMemorySourceImport, ClContextControlFixture)
{
    MemorySource invalidMemSource = static_cast<MemorySource>(256);
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(invalidMemSource),
                                              static_cast<MemorySourceFlags>(invalidMemSource));

    TensorInfo info({ 1, 2, 2, 1 }, DataType::Float32);

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info);

    // Allocate user memory
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    // Import non-support memory
    BOOST_CHECK_THROW(handle->Import(inputData.data(), invalidMemSource), MemoryImportException);
}

BOOST_AUTO_TEST_SUITE_END()