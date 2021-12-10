//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "WorkloadUtils.hpp"

#include <armnn/backends/TensorHandleFwd.hpp>
#include <armnn/backends/Workload.hpp>

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
