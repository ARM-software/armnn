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
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

#include <vector>

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
    arm_compute::ITensorInfo* biasSketchInfoPtr = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        if(!biases.has_value())
        {
            throw InvalidArgumentException("GpuFsaConvolution2d::ValidateOp: No biases set when biases are enabled");
        }
        aclBiasInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasInfo.set_are_values_constant(biases.value().IsConstant());

        biasSketchInfoPtr = workloadContext.create_tensor_info(aclBiasInfo);
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
                                                           inputInfo,
                                                           weightInfo,
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
 * Creating an Op for the GpuFsa backend requires us to create and maintain quite a bit of data, which is then stored
 * in a GpuFsaPreCompiledBlob for execution later. Specifically we need:
 * GpuWorkloadContext, this contains the TensorInfos and is unique to the Graph being executed
 * Sketch, this is similar to a subgraph and can contain one or more operations. Multiple ops can be "fused" together
 * using a single sketch.
 * The inputTensorinfos / outputTensorInfos, these are pointers to the TensorInfos used when creating the sketch.
 * They refer to the TensorInfos stored within the GpuWorkloadContext and are needed when executing the sketch
 * as the TensorInfos used when creating the Tensors must match those used to create the Sketch. Otherwise the runtime
 * doesn't know which Tensors to use.
 */
    using namespace arm_compute::experimental::dynamic_fusion;
    GpuWorkloadSketch* sketch = blob->sketch.get();
    GpuWorkloadContext* workloadContext = blob->workloadContext.get();
    std::vector<arm_compute::ITensorInfo*> inputTensorInfos = {};
    std::vector<arm_compute::ITensorInfo*> outputTensorInfos = {};

    // Build and create tensor infos using the sketch
    const arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    arm_compute::TensorInfo       aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());

    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclInputInfo));
    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclWeightsInfo));

    // Only create the bias tensor info if enabled, otherwise pass nullptr to validate_op / create_op
    arm_compute::TensorInfo aclBiasInfo;
    arm_compute::ITensorInfo* biasSketchInfoPtr = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        if(!biases.has_value())
        {
            throw InvalidArgumentException("GpuFsaConvolution2d::CreateOp: No biases set when biases are enabled");
        }
        aclBiasInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasInfo.set_are_values_constant(biases.value().IsConstant());

        inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclBiasInfo));
        biasSketchInfoPtr = inputTensorInfos[2];
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
    arm_compute::Status aclStatus = GpuConv2d::validate_op(*sketch,
                                                           inputTensorInfos[0],
                                                           inputTensorInfos[1],
                                                           biasSketchInfoPtr,
                                                           conv2DAttributes);

    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported)
    {
        throw BackendCapabilityException("\"GpuFsa\" backend failed during Convolution2D operation validation");
    }

    // Create the Op within the Sketch using the TensorInfos we have stored
    arm_compute::ITensorInfo* convOutInfo = GpuConv2d::create_op(*sketch,
                                                                 inputTensorInfos[0],
                                                                 inputTensorInfos[1],
                                                                 biasSketchInfoPtr,
                                                                 conv2DAttributes);

    // Create the Output
    outputTensorInfos.emplace_back(workloadContext->create_tensor_info());
    GpuOutput::create_op(*sketch, convOutInfo, outputTensorInfos[0]);

    // Store the TensorInfos within the blob as unique_ptrs to be used later
    blob->inputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

} // namespace armnn
