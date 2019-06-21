//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <armnn/TypesUtils.hpp>
#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

class RefGatherWorkload : public BaseWorkload<GatherQueueDescriptor>
{
public:
    using BaseWorkload<GatherQueueDescriptor>::BaseWorkload;
    void Execute() const override;
};

} // namespace armnn
