//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <armnn/Types.hpp>

namespace armnn
{

// Base class template providing an implementation of the Constant layer common to all data types.
class RefConstantWorkload : public BaseWorkload<ConstantQueueDescriptor>
{
public:
    RefConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;
private:
    void Execute(std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
