//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

namespace armnn
{
template <typename QueueDescriptor>
class NeonBaseWorkload : public BaseWorkload<QueueDescriptor>
{
public:
    NeonBaseWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
            : BaseWorkload<QueueDescriptor>(descriptor, info)
    {}

    // Replace input tensor handle with the given TensorHandle and call Reconfigure()
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Inputs[slot] = tensorHandle;
        Reconfigure();
    }

    // Replace output tensor handle with the given TensorHandle and call Reconfigure()
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Outputs[slot] = tensorHandle;
        Reconfigure();
    }

    // Reconfigure the workload configuration. Throw armnn::UnimplementedException by default.
    virtual void Reconfigure()
    {
        throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
    }
};
} //namespace armnn
