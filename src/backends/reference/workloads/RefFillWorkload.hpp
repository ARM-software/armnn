//
// Copyright © 2022, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefFillWorkload : public RefBaseWorkload<FillQueueDescriptor>
{
public:
    using RefBaseWorkload<FillQueueDescriptor>::RefBaseWorkload;
    void Execute() const override;

private:
    void Execute(std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
