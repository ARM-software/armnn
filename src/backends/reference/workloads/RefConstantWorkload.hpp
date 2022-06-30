//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include <armnn/Types.hpp>

namespace armnn
{

// Base class template providing an implementation of the Constant layer common to all data types.
class RefConstantWorkload : public RefBaseWorkload<ConstantQueueDescriptor>
{
public:
    RefConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;
private:
    void Execute(std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
