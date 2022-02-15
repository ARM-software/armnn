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

class RefLogicalUnaryWorkload : public RefBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    using RefBaseWorkload<ElementwiseUnaryQueueDescriptor>::m_Data;

    RefLogicalUnaryWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    using InType  = bool;
    using OutType = bool;
};

} // namespace armnn
