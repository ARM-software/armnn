//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/Workload.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class RefSpaceToDepthWorkload : public BaseWorkload<SpaceToDepthQueueDescriptor>
{
public:
    using BaseWorkload<SpaceToDepthQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
