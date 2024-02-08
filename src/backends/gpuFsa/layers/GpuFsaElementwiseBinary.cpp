//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaElementwiseBinary.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuAdd.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuAdd.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuMul.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuSub.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

using namespace arm_compute::experimental::dynamic_fusion;
using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status GpuFsaElementwiseBinaryValidate(const TensorInfo& input0,
                                                    const TensorInfo& input1,
                                                    const ElementwiseBinaryDescriptor& descriptor)
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

    switch (descriptor.m_Operation)
    {
        case BinaryOperation::Add:
        {
            return GpuAdd::validate_op(sketch, inputInfo0, inputInfo1);
        }
        case BinaryOperation::Mul:
        {
            return GpuMul::validate_op(sketch, inputInfo0, inputInfo1);
        }
        case BinaryOperation::Sub:
        {
            return GpuSub::validate_op(sketch, inputInfo0, inputInfo1);
        }
        default:
            return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                                       std::string("Elementwise Binary operation not supported in GpuFsa: ")
                                       + GetBinaryOperationAsCString(descriptor.m_Operation));
    }
}

void GpuFsaElementwiseBinaryCreateOp(GpuFsaPreCompiledBlob* blob,
                                     const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const ElementwiseBinaryDescriptor& descriptor)
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
    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus{};
    switch (descriptor.m_Operation)
    {
        case BinaryOperation::Add:
        {
            aclStatus = GpuAdd::validate_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);
            break;
        }
        case BinaryOperation::Mul:
        {
            aclStatus = GpuMul::validate_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);
            break;
        }
        case BinaryOperation::Sub:
        {
            aclStatus = GpuSub::validate_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);
            break;
        }
        default:
            throw InvalidArgumentException(std::string("Elementwise Binary operation not supported in GpuFsa: ")
                                           + GetBinaryOperationAsCString(descriptor.m_Operation));
    }

    const bool supported = aclStatus.error_code() == arm_compute::ErrorCode::OK;
    if (!supported)
    {
        throw BackendCapabilityException("\"GpuFsa\" backend failed during elementwise binary add validation");
    }

    arm_compute::ITensorInfo* elementwiseBinaryOutputInfo{};
    switch (descriptor.m_Operation)
    {
        case BinaryOperation::Add:
        {
            elementwiseBinaryOutputInfo = GpuAdd::create_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);
            break;
        }
        case BinaryOperation::Mul:
        {
            elementwiseBinaryOutputInfo = GpuMul::create_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);
            break;
        }
        case BinaryOperation::Sub:
        {
            elementwiseBinaryOutputInfo = GpuSub::create_op(*sketch, inputTensorInfos[0], inputTensorInfos[1]);
            break;
        }
        default:
            throw InvalidArgumentException(std::string("Elementwise Binary operation not supported in GpuFsa: ")
                                           + GetBinaryOperationAsCString(descriptor.m_Operation));
    }

    // Temporary fix until fusing attempt is make for GpuFsa backend and Output layer workload is created.
    outputTensorInfos.emplace_back(workloadContext->create_tensor_info());
    GpuOutput::create_op(*sketch, elementwiseBinaryOutputInfo, outputTensorInfos[0]);

    // Store the TensorInfos within the blob as unique_ptrs to be used later
    blob->inputTensorInfos  = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

} // namespace armnn