//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#if (defined(__aarch64__)) || (defined(__x86_64__)) // disable test failing on FireFly/Armv7

#include "ClWorkloadFactoryHelper.hpp"

#include <test/TensorHelpers.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <cl/ClContextControl.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/OpenClTimer.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <arm_compute/runtime/CL/CLScheduler.h>

#include <boost/test/unit_test.hpp>

#include <iostream>

using namespace armnn;

struct OpenClFixture
{
    // Initialising ClContextControl to ensure OpenCL is loaded correctly for each test case.
    // NOTE: Profiling needs to be enabled in ClContextControl to be able to obtain execution
    // times from OpenClTimer.
    OpenClFixture() : m_ClContextControl(nullptr, true) {}
    ~OpenClFixture() {}

    ClContextControl m_ClContextControl;
};

BOOST_FIXTURE_TEST_SUITE(OpenClTimerBatchNorm, OpenClFixture)
using FactoryType = ClWorkloadFactory;

BOOST_AUTO_TEST_CASE(OpenClTimerBatchNorm)
{
    auto memoryManager = ClWorkloadFactoryHelper::GetMemoryManager();
    ClWorkloadFactory workloadFactory = ClWorkloadFactoryHelper::GetFactory(memoryManager);

    const unsigned int width    = 2;
    const unsigned int height   = 3;
    const unsigned int channels = 2;
    const unsigned int num      = 1;

    TensorInfo inputTensorInfo( {num, channels, height, width}, DataType::Float32);
    TensorInfo outputTensorInfo({num, channels, height, width}, DataType::Float32);
    TensorInfo tensorInfo({channels}, DataType::Float32);

    auto input = MakeTensor<float, 4>(inputTensorInfo,
        {
             1.f, 4.f,
             4.f, 2.f,
             1.f, 6.f,

             1.f, 1.f,
             4.f, 1.f,
            -2.f, 4.f
        });

    // these values are per-channel of the input
    auto mean     = MakeTensor<float, 1>(tensorInfo, { 3.f, -2.f });
    auto variance = MakeTensor<float, 1>(tensorInfo, { 4.f,  9.f });
    auto beta     = MakeTensor<float, 1>(tensorInfo, { 3.f,  2.f });
    auto gamma    = MakeTensor<float, 1>(tensorInfo, { 2.f,  1.f });

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    std::unique_ptr<ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);
    ARMNN_NO_DEPRECATE_WARN_END

    BatchNormalizationQueueDescriptor data;
    WorkloadInfo info;
    ScopedCpuTensorHandle meanTensor(tensorInfo);
    ScopedCpuTensorHandle varianceTensor(tensorInfo);
    ScopedCpuTensorHandle betaTensor(tensorInfo);
    ScopedCpuTensorHandle gammaTensor(tensorInfo);

    AllocateAndCopyDataToITensorHandle(&meanTensor, &mean[0]);
    AllocateAndCopyDataToITensorHandle(&varianceTensor, &variance[0]);
    AllocateAndCopyDataToITensorHandle(&betaTensor, &beta[0]);
    AllocateAndCopyDataToITensorHandle(&gammaTensor, &gamma[0]);

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
    std::unique_ptr<IWorkload> workload = workloadFactory.CreateBatchNormalization(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    OpenClTimer openClTimer;

    BOOST_CHECK_EQUAL(openClTimer.GetName(), "OpenClKernelTimer");

    //Start the timer
    openClTimer.Start();

    //Execute the workload
    workload->Execute();

    //Stop the timer
    openClTimer.Stop();

    BOOST_CHECK_EQUAL(openClTimer.GetMeasurements().size(), 1);

    BOOST_CHECK_EQUAL(openClTimer.GetMeasurements().front().m_Name,
                      "OpenClKernelTimer/0: batchnormalization_layer_nchw GWS[1,3,2]");

    BOOST_CHECK(openClTimer.GetMeasurements().front().m_Value > 0);

}

BOOST_AUTO_TEST_SUITE_END()

#endif //aarch64 or x86_64
