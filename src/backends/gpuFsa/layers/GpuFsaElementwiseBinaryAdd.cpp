//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaElementwiseBinaryAdd.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuAdd.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

using namespace arm_compute::experimental::dynamic_fusion;
using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status GpuFsaElementwiseBinaryAddValidate(const TensorInfo& input0,
                                                       const TensorInfo& input1)
{
    // Create a new workload sketch, for validation purposes
    auto compileCtx         = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto workloadContext    = GpuWorkloadContext(&compileCtx);
    GpuWorkloadSketch sketch{ &workloadContext };

    arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0, input0.GetNumDimensions());
    arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1, input1.GetNumDimensions());

    aclInput0Info.set_are_values_constant(input0.IsConstant());
    aclInput1Info.set_are_values_constant(input1.IsConstant());

    arm_compute::ITensorInfo*  inputInfo0 = workloadContext.create_tensor_info(aclInput0Info);
    arm_compute::ITensorInfo*  inputInfo1 = workloadContext.create_tensor_info(aclInput1Info);

    return GpuAdd::validate_op(sketch, inputInfo0, inputInfo1);
}

void GpuFsaElementwiseBinaryAddCreateOp(GpuFsaPreCompiledBlob* blob,
                                        const TensorInfo& input0,
                                        const TensorInfo& input1)
{
    GpuWorkloadSketch* sketch           = blob->sketch.get();
    GpuWorkloadContext* workloadContext = blob->workloadContext.get();
    std::vector<arm_compute::ITensorInfo*> inputTensorInfos  = {};
    std::vector<arm_compute::ITensorInfo*> outputTensorInfos = {};

    arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0, input0.GetNumDimensions());
    arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1, input1.GetNumDimensions());

    aclInput0Info.set_are_values_constant(input0.IsConstant());
    aclInput1Info.set_are_values_constant(input1.IsConstant());

    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclInput0Info));
    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclInput1Info));

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus = GpuAdd::validate_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);
    const bool supported = aclStatus.error_code() == arm_compute::ErrorCode::OK;
    if (!supported)
    {
        throw BackendCapabilityException("\"GpuFsa\" backend failed during elementwise binary add validation");
    }

    arm_compute::ITensorInfo* addOutputInfo =
            GpuAdd::create_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);

    // Temporary fix until fusing attempt is make for GpuFsa backend and Output layer workload is created.
    outputTensorInfos.emplace_back(workloadContext->create_tensor_info());
    GpuOutput::create_op(*sketch, addOutputInfo, outputTensorInfos[0]);

    // Store the TensorInfos within the blob as unique_ptrs to be used later
    blob->inputTensorInfos  = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

} // namespace armnn