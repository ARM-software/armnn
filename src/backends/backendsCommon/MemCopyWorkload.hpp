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

class CopyMemGenericWorkload : public BaseWorkload<MemCopyQueueDescriptor>
{
public:
    CopyMemGenericWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TensorHandlePair = std::pair<const ITensorHandle*, ITensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

} //namespace armnn
