//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/Workload.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class RefSpaceToBatchNdWorkload : public BaseWorkload<SpaceToBatchNdQueueDescriptor>
{
public:
    using BaseWorkload<SpaceToBatchNdQueueDescriptor>::BaseWorkload;
    void Execute() const override;
};

} //namespace armnn
