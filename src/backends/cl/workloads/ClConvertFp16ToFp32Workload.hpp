//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLDepthConvertLayer.h>

#include <cl/ICLTensorProxy.hpp>

namespace armnn
{

class ClConvertFp16ToFp32Workload : public Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>
{
public:

    ClConvertFp16ToFp32Workload(const ConvertFp16ToFp32QueueDescriptor& descriptor,
                                const WorkloadInfo& info,
                                const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

    bool SupportsTensorHandleReplacement() const override { return true;};

    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;
private:
    mutable arm_compute::CLDepthConvertLayer m_Layer;
    virtual void Reconfigure();

    std::unique_ptr<ICLTensorProxy> m_InputProxy;
    std::unique_ptr<ICLTensorProxy> m_OutputProxy;
};

arm_compute::Status ClConvertFp16ToFp32WorkloadValidate(const TensorInfo& input, const TensorInfo& output);

} //namespace armnn
