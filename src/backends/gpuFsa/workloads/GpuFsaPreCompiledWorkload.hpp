//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/backends/Workload.hpp"

#include <arm_compute/core/ITensorInfo.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/core/CL/CLCompileContext.h>

#include <arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>

#include <memory>
#include <string>
#include <vector>

namespace armnn
{

bool GpuFsaPreCompiledWorkloadValidate(std::string* reasonIfUnsupported);

class GpuFsaPreCompiledWorkload : public BaseWorkload<PreCompiledQueueDescriptor>
{
public:
    GpuFsaPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                              const WorkloadInfo& info);
    void Execute() const override;

private:
    bool SupportsTensorHandleReplacement() const override
    {
        return true;
    }

    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Inputs[slot] = tensorHandle;
    }

    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Outputs[slot] = tensorHandle;
    }

    WorkloadInfo m_workloadInfo;
};

}    //namespace armnn