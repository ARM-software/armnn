//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaDepthwiseConvolution2d.hpp"
#include <backendsCommon/WorkloadUtils.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuDepthwiseConv2d.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

#include <vector>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status GpuFsaDepthwiseConvolution2dValidate(const TensorInfo& input,
                                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                                         const TensorInfo& weights,
                                                         const Optional<TensorInfo>& biases)
{
    // Create a new workload sketch, for validation purposes
    auto compileCtx = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto workloadContext = GpuWorkloadContext(&compileCtx);
    GpuWorkloadSketch sketch{ &workloadContext };

    // Build and create tensor infos using the sketch
    const arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);

    // ArmNN format for weights for depthwise is [1, H, W, C] independently of the input/output layout
    //
    // ACL format for weights for depthwise is:
    // - [1, H, W, C] for [N, H, W, C] input/output layout (matches with ArmNN)
    // - [1, C, H, W] for [N, C, H, W] input/output layout
    //
    // Therefore ArmNN weights have to be permuted when input/output layout is [N, C, H, W] to pass them to ACL.
    // The PermuteDepthwiseConv2dWeights backend optimization takes care of this, but it has not been performed yet,
    // so we do the permute here for the TensorInfo weights.
    unsigned int aclDepthMultiplier;
    TensorInfo weightsPermuted;
    std::tie(weightsPermuted, aclDepthMultiplier) = Convert1HWOTensorInfoToAcl(weights, input,descriptor.m_DataLayout);
    auto weightsShape = weightsPermuted.GetShape();
    weightsPermuted.SetShape({weightsShape[1], weightsShape[2], weightsShape[3]});

    arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weightsPermuted, descriptor.m_DataLayout);
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
            throw InvalidArgumentException(
                "GpuFsaDepthwiseConvolution2dValidate: No biases set when biases are enabled");
        }
        aclBiasInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasInfo.set_are_values_constant(biases.value().IsConstant());

        biasSketchInfoPtr = workloadContext.create_tensor_info(aclBiasInfo);
    }

    // Set DepthwiseConv2d attributes using descriptor
    const arm_compute::Size2D    aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                         descriptor.m_DilationY);
    const arm_compute::Padding2D aclPadInfo      = BuildArmComputePaddingInfo(descriptor);
    const arm_compute::Size2D    aclStrideInfo   = BuildArmComputeSize2D(descriptor.m_StrideX, descriptor.m_StrideY);

    DepthwiseConv2dAttributes depthwiseConv2dAttributes{};
    depthwiseConv2dAttributes.pad(aclPadInfo);
    depthwiseConv2dAttributes.stride(aclStrideInfo);
    depthwiseConv2dAttributes.dilation(aclDilationInfo);
    depthwiseConv2dAttributes.depth_multiplier(aclDepthMultiplier);

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus = GpuDepthwiseConv2d::validate_op(sketch,
                                                                    inputInfo,
                                                                    weightInfo,
                                                                    biasSketchInfoPtr,
                                                                    depthwiseConv2dAttributes);

    return aclStatus;
}

void GpuFsaDepthwiseConvolution2dCreateOp(GpuFsaPreCompiledBlob* blob,
                                          const TensorInfo& input,
                                          const DepthwiseConvolution2dDescriptor& descriptor,
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

    // ArmNN format for weights for depthwise is [1, H, W, C] independently of the input/output layout
    //
    // ACL format for weights for depthwise is:
    // - [1, H, W, C] for [N, H, W, C] input/output layout (matches with ArmNN)
    // - [1, C, H, W] for [N, C, H, W] input/output layout
    //
    // Therefore ArmNN weights have to be permuted when input/output layout is [N, C, H, W] to pass them to ACL.
    // The PermuteDepthwiseConv2dWeights backend optimization takes care of this, but it has not been performed yet,
    // so we do the permute here for the TensorInfo weights.
    unsigned int aclDepthMultiplier;
    TensorInfo weightsPermuted;
    std::tie(weightsPermuted, aclDepthMultiplier) = Convert1HWOTensorInfoToAcl(weights, input,descriptor.m_DataLayout);
    auto weightsShape = weightsPermuted.GetShape();
    weightsPermuted.SetShape({weightsShape[1], weightsShape[2], weightsShape[3]});

    arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weightsPermuted, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());

    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclInputInfo));
    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclWeightsInfo));

    // Only create the bias tensor info if enabled, otherwise pass nullptr to validate_op
    arm_compute::TensorInfo aclBiasInfo;
    arm_compute::ITensorInfo* biasSketchInfoPtr = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        if(!biases.has_value())
        {
            throw InvalidArgumentException("GpuFsaConvolution2dValidate: No biases set when biases are enabled");
        }
        aclBiasInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasInfo.set_are_values_constant(biases.value().IsConstant());

        inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclBiasInfo));
        biasSketchInfoPtr = inputTensorInfos[2];
    }

    // Set DepthwiseConv2d attributes using descriptor
    const arm_compute::Size2D    aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                         descriptor.m_DilationY);
    const arm_compute::Padding2D aclPadInfo      = BuildArmComputePaddingInfo(descriptor);
    const arm_compute::Size2D    aclStrideInfo   = BuildArmComputeSize2D(descriptor.m_StrideX, descriptor.m_StrideY);

    DepthwiseConv2dAttributes depthwiseConv2dAttributes{};
    depthwiseConv2dAttributes.pad(aclPadInfo);
    depthwiseConv2dAttributes.stride(aclStrideInfo);
    depthwiseConv2dAttributes.dilation(aclDilationInfo);
    depthwiseConv2dAttributes.depth_multiplier(aclDepthMultiplier);

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus = GpuDepthwiseConv2d::validate_op(*sketch,
                                                                    inputTensorInfos[0],
                                                                    inputTensorInfos[1],
                                                                    biasSketchInfoPtr,
                                                                    depthwiseConv2dAttributes);

    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported)
    {
        throw BackendCapabilityException(
            "\"GpuFsa\" backend failed during DepthwiseConvolution2D operation validation");
    }

    // Create the Op within the Sketch using the TensorInfos we have stored
    arm_compute::ITensorInfo* convOutInfo = GpuDepthwiseConv2d::create_op(*sketch,
                                                                          inputTensorInfos[0],
                                                                          inputTensorInfos[1],
                                                                          biasSketchInfoPtr,
                                                                          depthwiseConv2dAttributes);

    outputTensorInfos.emplace_back(workloadContext->create_tensor_info());
    GpuOutput::create_op(*sketch, convOutInfo, outputTensorInfos[0]);

    // Store the TensorInfos within the blob as unique_ptrs to be used later
    blob->inputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

} // namespace armnn
