//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#if (defined(__aarch64__)) || (defined(__x86_64__)) // disable test failing on FireFly/Armv7

#include "ClWorkloadFactoryHelper.hpp"

#include <armnnTestUtils/TensorHelpers.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <cl/ClContextControl.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/OpenClTimer.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <arm_compute/runtime/CL/CLScheduler.h>

#include <doctest/doctest.h>

#include <iostream>

using namespace armnn;

struct OpenClFixture
{
    // Initialising ClContextControl to ensure OpenCL is loaded correctly for each test case.
    // NOTE: Profiling needs to be enabled in ClContextControl to be able to obtain execution
    // times from OpenClTimer.
    OpenClFixture() : m_ClContextControl(nullptr, nullptr, true) {}
    ~OpenClFixture() {}

    ClContextControl m_ClContextControl;
};

TEST_CASE_FIXTURE(OpenClFixture, "OpenClTimerBatchNorm")
{
//using FactoryType = ClWorkloadFactory;

    auto memoryManager = ClWorkloadFactoryHelper::GetMemoryManager();
    ClWorkloadFactory workloadFactory = ClWorkloadFactoryHelper::GetFactory(memoryManager);

    const unsigned int width    = 2;
    const unsigned int height   = 3;
    const unsigned int channels = 2;
    const unsigned int num      = 1;

    TensorInfo inputTensorInfo( {num, channels, height, width}, DataType::Float32);
    TensorInfo outputTensorInfo({num, channels, height, width}, DataType::Float32);
    TensorInfo tensorInfo({channels}, DataType::Float32);

    std::vector<float> input =
    {
         1.f, 4.f,
         4.f, 2.f,
         1.f, 6.f,

         1.f, 1.f,
         4.f, 1.f,
        -2.f, 4.f
    };

    // these values are per-channel of the input
    std::vector<float> mean     = { 3.f, -2.f };
    std::vector<float> variance = { 4.f,  9.f };
    std::vector<float> beta     = { 3.f,  2.f };
    std::vector<float> gamma    = { 2.f,  1.f };

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    std::unique_ptr<ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);
    ARMNN_NO_DEPRECATE_WARN_END

    BatchNormalizationQueueDescriptor data;
    WorkloadInfo info;
    ScopedTensorHandle meanTensor(tensorInfo);
    ScopedTensorHandle varianceTensor(tensorInfo);
    ScopedTensorHandle betaTensor(tensorInfo);
    ScopedTensorHandle gammaTensor(tensorInfo);

    AllocateAndCopyDataToITensorHandle(&meanTensor, mean.data());
    AllocateAndCopyDataToITensorHandle(&varianceTensor, variance.data());
    AllocateAndCopyDataToITensorHandle(&betaTensor, beta.data());
    AllocateAndCopyDataToITensorHandle(&gammaTensor, gamma.data());

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Mean             = &meanTensor;
    data.m_Variance         = &varianceTensor;
    data.m_Beta             = &betaTensor;
    data.m_Gamma            = &gammaTensor;
    data.m_Parameters.m_Eps = 0.0f;

    // for each channel:
    // substract mean, divide by standard deviation (with an epsilon to avoid div by 0)
    // multiply by gamma and add beta
    std::unique_ptr<IWorkload> workload = workloadFactory.CreateWorkload(LayerType::BatchNormalization, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    OpenClTimer openClTimer;

    CHECK_EQ(openClTimer.GetName(), "OpenClKernelTimer");

    //Start the timer
    openClTimer.Start();

    //Execute the workload
    workload->Execute();

    //Stop the timer
    openClTimer.Stop();

    CHECK_EQ(openClTimer.GetMeasurements().size(), 1);

    CHECK_EQ(openClTimer.GetMeasurements().front().m_Name,
                      "OpenClKernelTimer/0: batchnormalization_layer_nchw GWS[1,3,2]");

    CHECK(openClTimer.GetMeasurements().front().m_Value > 0);

}

#endif //aarch64 or x86_64
