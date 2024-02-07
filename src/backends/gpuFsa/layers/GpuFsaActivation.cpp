//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaActivation.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuTanh.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuSigmoid.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

using namespace arm_compute::experimental::dynamic_fusion;
using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status GpuFsaActivationValidate(const TensorInfo& input,
                                             const ActivationDescriptor& descriptor)
{
    // Create a new workload sketch, for validation purposes
    auto compileCtx         = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto workloadContext    = GpuWorkloadContext(&compileCtx);
    GpuWorkloadSketch sketch{ &workloadContext };

    arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, input.GetNumDimensions());
    aclInputInfo.set_are_values_constant(input.IsConstant());

    arm_compute::ITensorInfo* inputInfo = workloadContext.create_tensor_info(aclInputInfo);

    switch (descriptor.m_Function)
    {
        case ActivationFunction::TanH:
        {
            if ( descriptor.m_A != 1 || descriptor.m_B != 1)
            {
                 return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                                            "Activation function TanH only works with a=1 and b=1");
            }
            return GpuTanh::validate_op(sketch, inputInfo);
        }
        case ActivationFunction::Sigmoid:
        {
            return GpuSigmoid::validate_op(sketch, inputInfo);
        }
        default:
            return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                                       std::string("Activation function currently not supported in GpuFsa: ")
                                           + GetActivationFunctionAsCString(descriptor.m_Function));
    }

}

void GpuFsaActivationCreateOp(GpuFsaPreCompiledBlob* blob,
                              const TensorInfo& input,
                              const ActivationDescriptor& descriptor)
{
    GpuWorkloadSketch* sketch           = blob->sketch.get();
    GpuWorkloadContext* workloadContext = blob->workloadContext.get();
    std::vector<arm_compute::ITensorInfo*> inputTensorInfos  = {};
    std::vector<arm_compute::ITensorInfo*> outputTensorInfos = {};

    arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input, input.GetNumDimensions());

    aclInput0Info.set_are_values_constant(input.IsConstant());

    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclInput0Info));

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus{};
    switch (descriptor.m_Function)
    {
        case ActivationFunction::TanH:
        {
            aclStatus = GpuTanh::validate_op(*sketch, inputTensorInfos[0]);
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            aclStatus = GpuSigmoid::validate_op(*sketch, inputTensorInfos[0]);
            break;
        }
        default:
            throw InvalidArgumentException(std::string("Activation function currently not supported in GpuFsa: ")
                                           + GetActivationFunctionAsCString(descriptor.m_Function));

    }
    const bool supported = aclStatus.error_code() == arm_compute::ErrorCode::OK;
    if (!supported)
    {
        throw BackendCapabilityException("\"GpuFsa\" backend failed during Activation layer validation");
    }

    arm_compute::ITensorInfo* activationOutputInfo{};
    switch (descriptor.m_Function)
    {
        case ActivationFunction::TanH:
        {
            activationOutputInfo = GpuTanh::create_op(*sketch, inputTensorInfos[0]);
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            activationOutputInfo = GpuSigmoid::create_op(*sketch, inputTensorInfos[0]);
            break;
        }
        default:
            throw InvalidArgumentException(std::string("Activation function currently not supported in GpuFsa: ")
                                           + GetActivationFunctionAsCString(descriptor.m_Function));

    }

    // Temporary fix until fusing attempt is make for GpuFsa backend and Output layer workload is created.
    outputTensorInfos.emplace_back(workloadContext->create_tensor_info());
    GpuOutput::create_op(*sketch, activationOutputInfo, outputTensorInfos[0]);

    // Store the TensorInfos within the blob as unique_ptrs to be used later
    blob->inputTensorInfos  = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

} // namespace armnn