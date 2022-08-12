//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/backends/Workload.hpp"

#include <memory>
#include <string>
#include <vector>

namespace armnn
{

bool TosaRefPreCompiledWorkloadValidate(std::string* reasonIfUnsupported);

class TosaRefPreCompiledWorkload : public BaseWorkload<PreCompiledQueueDescriptor>
{
public:
    TosaRefPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
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
};

}    //namespace armnn
