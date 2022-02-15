//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "RefBaseWorkload.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class RefSpaceToDepthWorkload : public RefBaseWorkload<SpaceToDepthQueueDescriptor>
{
public:
    using RefBaseWorkload<SpaceToDepthQueueDescriptor>::RefBaseWorkload;
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
