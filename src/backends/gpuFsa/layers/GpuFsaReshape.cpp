//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaReshape.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuReshape.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>

using namespace arm_compute::experimental::dynamic_fusion;

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status GpuFsaReshapeValidate(const TensorInfo& input, const ReshapeDescriptor& descriptor)
{
    auto compileContext  = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto workloadContext = GpuWorkloadContext(&compileContext);

    GpuWorkloadSketch sketch(&workloadContext);

    arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, input.GetNumDimensions());
    aclInputInfo.set_are_values_constant(input.IsConstant());

    arm_compute::ITensorInfo* inputInfo = workloadContext.create_tensor_info(aclInputInfo);

    ReshapeAttributes attributes;
    attributes.shape(BuildArmComputeTensorShape(descriptor.m_TargetShape));

    arm_compute::Status aclStatus = GpuReshape::validate_op(sketch, inputInfo, attributes);

#ifndef NDEBUG
    if (aclStatus.error_code() != arm_compute::ErrorCode::OK)
    {
        std::cout << "GpuFsaReshapeValidate failed: " << aclStatus.error_description() << std::endl;
    }
#endif

    return aclStatus;
}

void GpuFsaReshapeCreateOp(GpuFsaPreCompiledBlob* blob, const TensorInfo& input, const ReshapeDescriptor& descriptor)
{
    GpuWorkloadSketch*  sketch          = blob->sketch.get();
    GpuWorkloadContext* workloadContext = blob->workloadContext.get();

    std::vector<arm_compute::ITensorInfo*> inputTensorInfos;
    std::vector<arm_compute::ITensorInfo*> outputTensorInfos;

    arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, input.GetNumDimensions());

    aclInputInfo.set_are_values_constant(input.IsConstant());

    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclInputInfo));

    ReshapeAttributes attributes;
    attributes.shape(BuildArmComputeTensorShape(descriptor.m_TargetShape));

    arm_compute::ITensorInfo* addOutputInfo = GpuReshape::create_op(*sketch, inputTensorInfos[0], attributes);

    // Temporary fix until fusing attempt is made for GpuFsa backend and outputLayer workoad is created
    outputTensorInfos.emplace_back(workloadContext->create_tensor_info());
    GpuOutput::create_op(*sketch, addOutputInfo, outputTensorInfos[0]);

    // Store the tensorInfos within the blob as std::unique_ptr<> so they can be used later
    blob->inputTensorInfos  = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

} // namespace armnn

