//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

namespace armnn
{

template <typename QueueDescriptor>
class TosaRefBaseWorkload : public BaseWorkload<QueueDescriptor>
{
public:
    TosaRefBaseWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
            : BaseWorkload<QueueDescriptor>(descriptor, info)
    {}

    virtual bool SupportsTensorHandleReplacement()  const override
    {
        return true;
    }

    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Inputs[slot] = tensorHandle;
    }

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Outputs[slot] = tensorHandle;
    }
};

} //namespace armnn