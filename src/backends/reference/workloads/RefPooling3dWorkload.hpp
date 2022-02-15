//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{
class RefPooling3dWorkload : public RefBaseWorkload<Pooling3dQueueDescriptor>
{
public:
    using RefBaseWorkload<Pooling3dQueueDescriptor>::RefBaseWorkload;

    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};
} //namespace armnn
