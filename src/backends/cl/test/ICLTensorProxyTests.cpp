//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>

#include <armnnTestUtils/TensorCopyUtils.hpp>

#include <cl/ClImportTensorHandle.hpp>
#include <cl/ClImportTensorHandleFactory.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClTensorHandleFactory.hpp>
#include <cl/ICLTensorProxy.hpp>
#include <cl/test/ClContextControlFixture.hpp>
#include <cl/test/ClWorkloadFactoryHelper.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("ICLTensorProxyTests")
{

TEST_CASE_FIXTURE(ClContextControlFixture, "ICLTensorProxyTest")
{
    ClTensorHandleFactory handleFactory =
        ClWorkloadFactoryHelper::GetTensorHandleFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    TensorInfo info({ 1, 3, 4, 1 }, DataType::Float32);

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info, true);

    std::vector<float> inputData
    {
        -5, -2, 1, 2,
        3, 10, -20, 8,
        0, -12, 7, -9
    };

    handle->Allocate();

    CopyDataToITensorHandle(handle.get(), inputData.data());

    // Get CLtensor
    arm_compute::CLTensor& tensor = PolymorphicDowncast<ClTensorHandle*>(handle.get())->GetTensor();
    ICLTensorProxy iclTensorProxy(&tensor);

    // Check that the ICLTensorProxy get correct information from the delegate tensor
    CHECK((iclTensorProxy.info() == tensor.info()));
    CHECK((iclTensorProxy.buffer() == tensor.buffer()));
    CHECK((iclTensorProxy.cl_buffer() == tensor.cl_buffer()));
    CHECK((iclTensorProxy.quantization().scale == tensor.quantization().scale));
    CHECK((iclTensorProxy.quantization().offset == tensor.quantization().offset));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ChangeICLTensorProxyExecutionTest")
{
    // Start execution with with copied tensor
    ClTensorHandleFactory handleFactory =
        ClWorkloadFactoryHelper::GetTensorHandleFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    TensorInfo info({ 1, 3, 4, 1 }, DataType::Float32);
    unsigned int numElements = info.GetNumElements();

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info, true);

    std::vector<float> inputData
    {
        -5, -2, 1, 2,
        3, 10, -20, 8,
        0, -12, 7, -9
    };

    std::vector<float> ExpectedOutput
    {
        0, 0, 1, 2,
        3, 10, 0, 8,
        0, 0, 7, 0
    };

    handle->Allocate();

    CopyDataToITensorHandle(handle.get(), inputData.data());

    // Get CLtensor
    arm_compute::CLTensor& tensor = PolymorphicDowncast<ClTensorHandle*>(handle.get())->GetTensor();

    // Set a proxy tensor to allocated tensor
    std::unique_ptr<ICLTensorProxy> iclTensorProxy;
    iclTensorProxy = std::make_unique<ICLTensorProxy>(&tensor);

    // Create and configure activation function
    const arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::CLActivationLayer act_func;
    act_func.configure(iclTensorProxy.get(), nullptr, act_info);

    act_func.run();
    arm_compute::CLScheduler::get().sync();

    std::vector<float> actualOutput(info.GetNumElements());

    CopyDataFromITensorHandle(actualOutput.data(), handle.get());

    // Validate result as expected output
    for(unsigned int i = 0; i < numElements; ++i)
    {
        CHECK((actualOutput[i] == ExpectedOutput[i]));
    }

    // Change to execute with imported tensor
    ClImportTensorHandleFactory importHandleFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));
    // create TensorHandle for memory import
    auto importHandle = importHandleFactory.CreateTensorHandle(info);

    // Get CLtensor
    arm_compute::CLTensor& importTensor = PolymorphicDowncast<ClImportTensorHandle*>(importHandle.get())->GetTensor();

    // Allocate user memory
    const size_t totalBytes = importTensor.info()->total_size();
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto testData = std::make_unique<uint8_t[]>(space);
    void* alignedPtr = testData.get();
    CHECK(std::align(alignment, totalBytes, alignedPtr, space));

    // Import memory
    CHECK(importHandle->Import(alignedPtr, armnn::MemorySource::Malloc));

    // Input with negative values
    auto* typedPtr = reinterpret_cast<float*>(alignedPtr);
    std::fill_n(typedPtr, numElements, -5.0f);

    // Set the import Tensor to TensorProxy to change Tensor in the CLActivationLayer without calling configure function
    iclTensorProxy->set(&importTensor);

    // Execute function and sync
    act_func.run();
    arm_compute::CLScheduler::get().sync();

    // Validate result by checking that the output has no negative values
    for(unsigned int i = 0; i < numElements; ++i)
    {
        CHECK(typedPtr[i] == 0);
    }
}
}