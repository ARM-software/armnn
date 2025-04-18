//
// Copyright © 2022, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include <armnn/TypesUtils.hpp>
#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

class RefGatherWorkload : public RefBaseWorkload<GatherQueueDescriptor>
{
public:
    using RefBaseWorkload<GatherQueueDescriptor>::RefBaseWorkload;
    void Execute() const override;

private:
    template <typename T>
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} // namespace armnn
