//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "CpuTensorHandleFwd.hpp"
#include "backends/Workload.hpp"
#include "WorkloadUtils.hpp"
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
