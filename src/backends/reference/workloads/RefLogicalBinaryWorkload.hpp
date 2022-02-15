//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefLogicalBinaryWorkload : public RefBaseWorkload<LogicalBinaryQueueDescriptor>
{
public:
    using RefBaseWorkload<LogicalBinaryQueueDescriptor>::m_Data;

    RefLogicalBinaryWorkload(const LogicalBinaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    using InType  = bool;
    using OutType = bool;
};

} // namespace armnn
