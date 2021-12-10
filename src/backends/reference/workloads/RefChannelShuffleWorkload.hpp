//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefChannelShuffleWorkload : public BaseWorkload<ChannelShuffleQueueDescriptor>
{
public:
    using BaseWorkload<ChannelShuffleQueueDescriptor>::BaseWorkload;
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor) override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} // namespace armnn