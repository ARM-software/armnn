//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaConvolution2d.hpp"

#include <armnn/Types.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/core/ITensorInfo.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/core/CL/CLCompileContext.h>

#include <arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <src/dynamic_fusion/sketch/gpu/GpuWorkloadContextImpl.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

#include <vector>
#include <iostream>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status GpuFsaConvolution2dValidate(const TensorInfo& input,
                                                const Convolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases)
{
    // Create a new workload sketch, for validation purposes
    auto compileCtx = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto workloadContext = GpuWorkloadContext(&compileCtx);
    GpuWorkloadSketch sketch{ &workloadContext };

    // Build and create tensor infos using the sketch
    const arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    arm_compute::TensorInfo       aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());

    auto inputInfo  = workloadContext.create_tensor_info(aclInputInfo);
    auto weightInfo = workloadContext.create_tensor_info(aclWeightsInfo);

    // Only create the bias tensor info if enabled, otherwise pass nullptr to validate_op
    arm_compute::TensorInfo aclBiasInfo;
    arm_compute::TensorInfo biasSketchInfo;
    arm_compute::TensorInfo* biasSketchInfoPtr = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        if(!biases.has_value())
        {
            throw InvalidArgumentException("GpuFsaConvolution2d::ValidateOp: No biases set when biases are enabled");
        }
        aclBiasInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasInfo.set_are_values_constant(biases.value().IsConstant());

        biasSketchInfo    = workloadContext.create_tensor_info(aclBiasInfo);
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

    return aclStatus;
}

void GpuFsaConvolution2dCreateOp(GpuFsaPreCompiledBlob* blob,
                                 const TensorInfo& input,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 const Optional<TensorInfo>& biases)
{
/*
 * Creating an Op for the GpuFds backend requires us to create and maintain quite a bit of data, which is then stored
 * in a GpuFsaPreCompiledBlob for execution later. Specifically we need:
 * GpuWorkloadContext, this contains the TensorInfos and is unique to the Graph being executed
 * Sketch, this is similar to a subgraph and can contain one or more operations. Multiple ops can be "fused" together
 * using a single sketch.
 * The TensorInfoIds, these are the ids of the TensorInfos used when creating the sketch. They refer to the TensorInfos
 * stored within the GpuWorkloadContext and are used to fetch them later when executing the sketch.
 */
    using namespace arm_compute::experimental::dynamic_fusion;
    GpuWorkloadSketch* sketch = blob->sketch.get();
    GpuWorkloadContext* workloadContext = blob->workloadContext.get();
    std::vector<int32_t> inputIds = {};
    std::vector<int32_t> outputIds = {};

    // Build and create tensor infos using the sketch
    const arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    arm_compute::TensorInfo       aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());
    auto inputInfo = workloadContext->create_tensor_info(aclInputInfo);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());
    inputIds.emplace_back(inputInfo.id());

    auto weightInfo = workloadContext->create_tensor_info(aclWeightsInfo);
    inputIds.emplace_back(weightInfo.id());

    // Only create the bias tensor info if enabled, otherwise pass nullptr to validate_op
    arm_compute::TensorInfo aclBiasInfo;
    arm_compute::TensorInfo biasSketchInfo;
    arm_compute::ITensorInfo* biasSketchInfoPtr = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        if(!biases.has_value())
        {
            throw InvalidArgumentException("GpuFsaConvolution2d::CreateOp: No biases set when biases are enabled");
        }
        aclBiasInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasInfo.set_are_values_constant(biases.value().IsConstant());

        biasSketchInfo    = workloadContext->create_tensor_info(aclBiasInfo);
        inputIds.emplace_back(biasSketchInfo.id());
        biasSketchInfoPtr = workloadContext->implementation().get_tensor_info(biasSketchInfo.id());
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
    arm_compute::Status aclStatus =
        GpuConv2d::validate_op(*sketch,
                               workloadContext->implementation().get_tensor_info(inputInfo.id()),
                               workloadContext->implementation().get_tensor_info(weightInfo.id()),
                               biasSketchInfoPtr,
                               conv2DAttributes);

    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported)
    {
        throw BackendCapabilityException("\"GpuFsa\" backend failed during Convolution2D operation validation");
    }

    arm_compute::ITensorInfo* convOutInfo =
        GpuConv2d::create_op(*sketch,
                             workloadContext->implementation().get_tensor_info(inputInfo.id()),
                             workloadContext->implementation().get_tensor_info(weightInfo.id()),
                             biasSketchInfoPtr,
                             conv2DAttributes);

    arm_compute::TensorInfo outputDstInfo = workloadContext->create_tensor_info();
    outputIds.emplace_back(outputDstInfo.id());

    GpuOutput::create_op(*sketch, convOutInfo, workloadContext->implementation().get_tensor_info(outputDstInfo.id()));
    blob->inputIds = std::make_unique<std::vector<int32_t>>(inputIds);
    blob->outputIds = std::make_unique<std::vector<int32_t>>(outputIds);
}

} // namespace armnn
