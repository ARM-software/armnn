//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{
class RefPooling2dWorkload : public BaseWorkload<Pooling2dQueueDescriptor>
{
public:
    using BaseWorkload<Pooling2dQueueDescriptor>::BaseWorkload;

    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};
} //namespace armnn
