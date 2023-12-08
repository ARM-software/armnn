//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaPreCompiledWorkload.hpp"
#include "GpuFsaWorkloadUtils.hpp"
#include "armnn/utility/PolymorphicDowncast.hpp"

#include <gpuFsa/GpuFsaTensorHandle.hpp>
#include <gpuFsa/GpuFsaBackend.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <arm_compute/runtime/CL/CLTensor.h>
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
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>

namespace armnn {

GpuFsaPreCompiledWorkload::GpuFsaPreCompiledWorkload(const PreCompiledQueueDescriptor &descriptor,
                                                     const WorkloadInfo &info)
        : BaseWorkload<PreCompiledQueueDescriptor>(descriptor, info), m_workloadInfo(info)
{
    // Check that the workload is holding a pointer to a valid pre-compiled object
    if (m_Data.m_PreCompiledObject == nullptr)
    {
        throw InvalidArgumentException(
                "GpuFsaPrecompiledWorkload requires a valid pre-compiled object (GpuWorkloadSketch).");
    }
}

void GpuFsaPreCompiledWorkload::Execute() const
{
/*
 * The Execute function of the GpuFsa Backends PreCompiled workload needs to jump through various hoops in order to
 * create a valid sketch and runtime that can execute the kernel
 * First we need all of the data stored within the PreCompiled blob which was used to setup the workload, namely:
 * The GpuWorkloadContext, this is a context which contains the TensorInfos and is unique to the graph being run
 * The Sketch, this can contain one or many ops and acts as a subgraph within the context
 * The TensorInfoIds, These are the ids of the TensorInfos used during the creation of the Sketch and stored within
 * the context.
 * It is very important that the Tensors passed into the Runtime being used to execute this sketch are created with
 * the same TensorInfos as used when creating the sketch. We do this by creating new tensors, getting the original
 * TensorInfos from the GpuWorkloadContext via their ids, and then importing the buffers from our own TensorHandles
 * directly into these newly created Tensors. This allows us to link the externally visible Tensors from ArmNN to the
 * Tensors which are needed to execute with the Sketch.
 *
 */
    using namespace arm_compute::experimental::dynamic_fusion;
    // Get the runtime and configure it with the precompiled sketch
    ClWorkloadRuntime runtime;
    GpuFsaPreCompiledBlob *preCompiledBlob = static_cast<GpuFsaPreCompiledBlob*>(m_Data.m_PreCompiledObject);
    auto workloadContext =
            &(preCompiledBlob->workloadContext->implementation());
    auto sketch = preCompiledBlob->sketch.release();
    std::vector<int32_t> inputIds = *(preCompiledBlob->inputIds.get());
    std::vector<int32_t> outputIds = *(preCompiledBlob->outputIds.get());
    auto status = runtime.configure(*sketch);

    // (Important) Allocate auxiliary tensor memory if there are any
    for(auto &data : runtime.get_auxiliary_tensors())
    {
        arm_compute::CLTensor*     tensor      = std::get<0>(data);
        arm_compute::TensorInfo    info        = std::get<1>(data);
        arm_compute::experimental::dynamic_fusion::AuxMemoryInfo aux_mem_req = std::get<2>(data);
        tensor->allocator()->init(info, aux_mem_req.alignment);
        tensor->allocator()->allocate(); // Use ACL allocated memory
    }

    // Create and initialize user tensors
    std::vector<arm_compute::CLTensor*> inputsWeightsOutputs;
    inputsWeightsOutputs.reserve(m_Data.m_Inputs.size() + m_Data.m_Outputs.size());

    for (uint32_t inputSlotIdx = 0; inputSlotIdx < m_Data.m_Inputs.size(); ++inputSlotIdx)
    {
        arm_compute::CLTensor* input = new arm_compute::CLTensor{};
        input->allocator()->init(*(dynamic_cast<arm_compute::TensorInfo*>(
                workloadContext->get_tensor_info(inputIds[inputSlotIdx]))));
        auto* inputHandle = PolymorphicDowncast<GpuFsaTensorHandle*>(m_Data.m_Inputs[inputSlotIdx]);
        input->allocator()->import_memory(inputHandle->GetTensor().cl_buffer());
        inputsWeightsOutputs.emplace_back(std::move(input));
    }
    // Set the outputs
    for (uint32_t outputSlotIdx = 0; outputSlotIdx < m_Data.m_Outputs.size(); ++outputSlotIdx)
    {
        arm_compute::CLTensor* output = new arm_compute::CLTensor{};
        output->allocator()->init(*(dynamic_cast<arm_compute::TensorInfo*>(
                workloadContext->get_tensor_info(outputIds[outputSlotIdx]))));
        auto* outputHandle = PolymorphicDowncast<GpuFsaTensorHandle*>(m_Data.m_Outputs[outputSlotIdx]);
        output->allocator()->import_memory(outputHandle->GetTensor().cl_buffer());
        inputsWeightsOutputs.emplace_back(std::move(output));
    }
    runtime.run(inputsWeightsOutputs);
}
} // namespace armnn