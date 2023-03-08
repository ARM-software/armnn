//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefElementwiseBinaryWorkload : public RefBaseWorkload<ElementwiseBinaryQueueDescriptor>
{
public:
    using RefBaseWorkload<ElementwiseBinaryQueueDescriptor>::m_Data;

    RefElementwiseBinaryWorkload(const ElementwiseBinaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} // namespace armnn
