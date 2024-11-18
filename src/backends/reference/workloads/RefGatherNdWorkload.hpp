//
// Copyright © 2022, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"

namespace armnn
{

class RefGatherNdWorkload : public RefBaseWorkload<GatherNdQueueDescriptor>
{
public:
    using RefBaseWorkload<GatherNdQueueDescriptor>::RefBaseWorkload;
    void Execute() const override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

};

} // namespace armnn
