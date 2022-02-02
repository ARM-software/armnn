//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"


#include <armnnTestUtils/TensorHelpers.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <neon/NeonTimer.hpp>
#include <neon/NeonWorkloadFactory.hpp>

#include <backendsCommon/test/LayerTests.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <doctest/doctest.h>

#include <cstdlib>
#include <algorithm>

using namespace armnn;

TEST_SUITE("NeonTimerInstrument")
{

TEST_CASE("NeonTimerGetName")
{
    NeonTimer neonTimer;
    CHECK_EQ(std::string(neonTimer.GetName()), "NeonKernelTimer");
}

TEST_CASE("NeonTimerMeasure")
{
    NeonWorkloadFactory workloadFactory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    unsigned int inputWidth = 2000u;
    unsigned int inputHeight = 2000u;
    unsigned int inputChannels = 1u;
    unsigned int inputBatchSize = 1u;

    float upperBound = 1.0f;
    float lowerBound = -1.0f;

    size_t inputSize = inputWidth * inputHeight * inputChannels * inputBatchSize;
    std::vector<float> inputData(inputSize, 0.f);
    std::generate(inputData.begin(), inputData.end(), [](){
        return (static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 3)) + 1.f; });

    unsigned int outputWidth = inputWidth;
    unsigned int outputHeight = inputHeight;
    unsigned int outputChannels = inputChannels;
    unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
        armnn::DataType::Float32);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
        armnn::DataType::Float32);

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);
    ARMNN_NO_DEPRECATE_WARN_END

    // Setup bounded ReLu
    armnn::ActivationQueueDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(descriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    descriptor.m_Parameters.m_Function = armnn::ActivationFunction::BoundedReLu;
    descriptor.m_Parameters.m_A = upperBound;
    descriptor.m_Parameters.m_B = lowerBound;

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(LayerType::Activation, descriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    NeonTimer neonTimer;
    // Start the timer.
    neonTimer.Start();
    // Execute the workload.
    workload->Execute();
    // Stop the timer.
    neonTimer.Stop();

    std::vector<Measurement> measurements = neonTimer.GetMeasurements();

    CHECK(measurements.size() <= 2);
    if (measurements.size() > 1)
    {
        CHECK_EQ(measurements[0].m_Name, "NeonKernelTimer/0: NEFillBorderKernel");
        CHECK(measurements[0].m_Value > 0.0);
    }
    std::ostringstream oss_neon;
    std::ostringstream oss_cpu;
    oss_neon << "NeonKernelTimer/" << measurements.size()-1 << ": NEActivationLayerKernel";
    oss_cpu << "NeonKernelTimer/" << measurements.size()-1 << ": CpuActivationKernel";
    bool kernelCheck = ((measurements[measurements.size()-1].m_Name.find(oss_neon.str()) != std::string::npos)
                        || (measurements[measurements.size()-1].m_Name.find(oss_cpu.str()) != std::string::npos));
    CHECK(kernelCheck);
    CHECK(measurements[measurements.size()-1].m_Value > 0.0);
}

}
