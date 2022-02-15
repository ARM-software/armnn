//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"

namespace armnn
{

class RefDepthToSpaceWorkload : public RefBaseWorkload<DepthToSpaceQueueDescriptor>
{
public:
    using RefBaseWorkload<DepthToSpaceQueueDescriptor>::RefBaseWorkload;
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} // namespace armnn
