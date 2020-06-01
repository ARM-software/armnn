//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Workload.hpp"
#include "WorkloadUtils.hpp"

#include <armnn/backends/CpuTensorHandleFwd.hpp>

#include <utility>

namespace armnn
{

class ImportMemGenericWorkload : public BaseWorkload<MemImportQueueDescriptor>
{
public:
    ImportMemGenericWorkload(const MemImportQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TensorHandlePair = std::pair<const ITensorHandle*, ITensorHandle*>;
    TensorHandlePair m_TensorHandlePairs;
};

} //namespace armnn
