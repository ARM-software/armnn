//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaConvolution2dValidate.hpp"

#include <armnn/Types.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/core/ITensorInfo.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/core/CL/CLCompileContext.h>

#include <arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

#include <vector>
#include <iostream>

namespace armnn
{

using namespace armcomputetensorutils;

inline arm_compute::Status ValidateAndCreateOp(const TensorInfo& input,
                                               const Convolution2dDescriptor& descriptor,
                                               const TensorInfo& weights,
                                               const Optional<TensorInfo>& biases,
                                               const bool createOp = false)
{
    // Create a new workload sketch, for validation purposes
    auto compileCtx = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto gpuCtx     = GpuWorkloadContext(&compileCtx);
    GpuWorkloadSketch sketch{ &gpuCtx };

    // Build and create tensor infos using the sketch
    const arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    arm_compute::TensorInfo       aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());

    auto inputInfo  = gpuCtx.create_tensor_info(aclInputInfo);
    auto weightInfo = gpuCtx.create_tensor_info(aclWeightsInfo);

    // Only create the bias tensor info if enabled, otherwise pass nullptr to validate_op
    arm_compute::TensorInfo aclBiasInfo;
    arm_compute::TensorInfo biasSketchInfo;
    arm_compute::TensorInfo* biasSketchInfoPtr = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        if(!biases.has_value())
        {
            throw InvalidArgumentException("GpuFsaConvolution2dValidate: No biases set when biases are enabled");
        }
        aclBiasInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasInfo.set_are_values_constant(biases.value().IsConstant());

        biasSketchInfo    = gpuCtx.create_tensor_info(aclBiasInfo);
        biasSketchInfoPtr = &biasSketchInfo;
    }

    // Set Conv2d attributes using descriptor
    const arm_compute::Size2D    aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                         descriptor.m_DilationY);
    const arm_compute::Padding2D aclPadInfo      = BuildArmComputePaddingInfo(descriptor);
    const arm_compute::Size2D    aclStrideInfo   = BuildArmComputeSize2D(descriptor.m_StrideX, descriptor.m_StrideY);

    Conv2dAttributes conv2DAttributes{};
    conv2DAttributes.dilation(aclDilationInfo);
    conv2DAttributes.pad(aclPadInfo);
    conv2DAttributes.stride(aclStrideInfo);

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus = GpuConv2d::validate_op(sketch,
                                                           &inputInfo,
                                                           &weightInfo,
                                                           biasSketchInfoPtr,
                                                           conv2DAttributes);

    if (createOp)
    {
        const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
        if (!supported)
        {
            throw BackendCapabilityException("\"GpuFsa\" backend failed during operation validation when attempting "
                                             "to fuse a GpuConv2d operator into the existing workload sketch.");
        }

        arm_compute::ITensorInfo* convOutInfo = GpuConv2d::create_op(sketch,
                                                                     &inputInfo,
                                                                     &weightInfo,
                                                                     biasSketchInfoPtr,
                                                                     conv2DAttributes);

        // Temporary fix until fusing attempt is make for GpuFsa backend and Output layer workload is created.
        auto outputInfo = gpuCtx.create_tensor_info();
        GpuOutput::create_op(sketch, convOutInfo, &outputInfo);
    }

    return aclStatus;
}

arm_compute::Status GpuFsaConvolution2dValidate(const TensorInfo& input,
                                                const Convolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases)
{
    return ValidateAndCreateOp(input, descriptor, weights, biases);
}

void GpuFsaConvolution2dCreateOp(const TensorInfo& input,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 const Optional<TensorInfo>& biases)
{
    ValidateAndCreateOp(input, descriptor, weights, biases, true);
}

} // namespace armnn