//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefElementwiseUnaryWorkload : public RefBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    using RefBaseWorkload<ElementwiseUnaryQueueDescriptor>::m_Data;

    RefElementwiseUnaryWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    using InType  = float;
    using OutType = float;
};

} // namespace armnn
