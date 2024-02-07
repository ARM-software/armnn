//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaSoftmax.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuSoftmax.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

using namespace arm_compute::experimental::dynamic_fusion;
using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status GpuFsaSoftmaxValidate(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const SoftmaxDescriptor& descriptor)
{
    // Create a new workload sketch, for validation purposes
    auto compileCtx = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto workloadContext = GpuWorkloadContext(&compileCtx);
    GpuWorkloadSketch sketch{ &workloadContext };

    // Build and create tensor infos using the sketch
    arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, input.GetNumDimensions());
    arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, output.GetNumDimensions());
    aclInputInfo.set_are_values_constant(input.IsConstant());
    aclOutputInfo.set_are_values_constant(output.IsConstant());
    arm_compute::ITensorInfo*  inputInfo = workloadContext.create_tensor_info(aclInputInfo);
    arm_compute::ITensorInfo*  outputInfo = workloadContext.create_tensor_info(aclOutputInfo);

    // Set Softmax attributes using descriptor
    SoftmaxAttributes softmaxAttributes{};
    softmaxAttributes.beta(descriptor.m_Beta);
    softmaxAttributes.is_log_softmax(false); // Use Softmax not LogSoftmax
    int aclAxis = ComputeAclAxis(descriptor.m_Axis, input);
    softmaxAttributes.axis(aclAxis);

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus = GpuSoftmax::validate_op(sketch,
                                                           inputInfo,
                                                           outputInfo,
                                                           softmaxAttributes);

#ifndef NDEBUG
    const bool validated = aclStatus.error_code() == arm_compute::ErrorCode::OK;
    if (!validated)
    {
        std::cout << "GpuFsaSoftmaxValidate failed: " << aclStatus.error_description() << std::endl;
    }
#endif

    return aclStatus;
}

void GpuFsaSoftmaxCreateOp(GpuFsaPreCompiledBlob* blob,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           const SoftmaxDescriptor& descriptor)
{
    GpuWorkloadSketch* sketch           = blob->sketch.get();
    GpuWorkloadContext* workloadContext = blob->workloadContext.get();
    std::vector<arm_compute::ITensorInfo*> inputTensorInfos  = {};
    std::vector<arm_compute::ITensorInfo*> outputTensorInfos  = {};

    arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, input.GetNumDimensions());
    arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, output.GetNumDimensions());
    aclInputInfo.set_are_values_constant(input.IsConstant());
    aclOutputInfo.set_are_values_constant(output.IsConstant());

    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclInputInfo));
    outputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclOutputInfo));

    // Set Softmax attributes using descriptor
    SoftmaxAttributes softmaxAttributes{};
    softmaxAttributes.beta(descriptor.m_Beta); // Only used for LogSoftmax else default
    softmaxAttributes.is_log_softmax(false); // Use Softmax not LogSoftmax
    int aclAxis = ComputeAclAxis(descriptor.m_Axis, input);
    softmaxAttributes.axis(aclAxis);

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus = GpuSoftmax::validate_op(*sketch,
                                                            inputTensorInfos[0],
                                                            outputTensorInfos[0],
                                                            softmaxAttributes);
    const bool supported = aclStatus.error_code() == arm_compute::ErrorCode::OK;
    if (!supported)
    {
        throw BackendCapabilityException("\"GpuFsa\" backend failed during softmax validation");
    }

    GpuSoftmax::create_op(*sketch, inputTensorInfos[0], outputTensorInfos[0], softmaxAttributes);

    // Store the TensorInfos within the blob as unique_ptrs to be used later
    blob->inputTensorInfos  = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

}