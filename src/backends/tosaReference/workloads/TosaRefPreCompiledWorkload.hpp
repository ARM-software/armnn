//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/backends/Workload.hpp"

#include <graph_status.h>
#include <model_runner.h>

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

    template <typename T>
    void SetInput(TosaReference::IModelRunner& runner, std::string inputName, uint32_t inputIndex) const;

    template <typename T>
    void GetOutput(TosaReference::IModelRunner& runner, std::string outputName, uint32_t outputIndex) const;

    WorkloadInfo m_workloadInfo;
};

}    //namespace armnn
