//
// Copyright © 2018-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "RefBaseWorkload.hpp"

namespace armnn
{

class RefSpaceToBatchNdWorkload : public RefBaseWorkload<SpaceToBatchNdQueueDescriptor>
{
public:
    using RefBaseWorkload<SpaceToBatchNdQueueDescriptor>::RefBaseWorkload;

    void Execute() const override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
