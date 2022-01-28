//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

namespace armnn
{
template <typename QueueDescriptor>
class ClBaseWorkload : public BaseWorkload<QueueDescriptor>
{
public:
    ClBaseWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
            : BaseWorkload<QueueDescriptor>(descriptor, info)
    {}

    // Replace input tensor handle with the given TensorHandle and call Reconfigure()
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
        this->m_Data.m_Inputs[slot] = tensorHandle;
        try
        {
            Reconfigure();
        }
        catch(armnn::UnimplementedException& e)
        {
            // Cannot reconfigure, revert the slot back and throw the exception.
            this->m_Data.m_Inputs[slot] = backupHandle;
            throw e;
        }
    }

    // Replace output tensor handle with the given TensorHandle and call Reconfigure()
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        ITensorHandle* backupHandle = this->m_Data.m_Outputs[slot];
        this->m_Data.m_Outputs[slot] = tensorHandle;
        try
        {
            Reconfigure();
        }
        catch(armnn::UnimplementedException& e)
        {
            // Cannot reconfigure, revert the slot back and throw the exception.
            this->m_Data.m_Inputs[slot] = backupHandle;
            throw e;
        }
    }

protected:
    // Reconfigure the workload configuration. Throw armnn::UnimplementedException by default.
    virtual void Reconfigure()
    {
        throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
    }
};
} //namespace armnn
