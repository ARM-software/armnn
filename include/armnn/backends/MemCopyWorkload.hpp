//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorHandle.hpp"
#include "Workload.hpp"

#include <utility>

namespace armnn
{

class CopyMemGenericWorkload : public BaseWorkload<MemCopyQueueDescriptor>
{
public:
    CopyMemGenericWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& descriptor) override;

private:
    using TensorHandlePair = std::pair<const ITensorHandle*, ITensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

} //namespace armnn
