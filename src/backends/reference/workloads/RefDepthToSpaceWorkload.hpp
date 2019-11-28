//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

namespace armnn
{

class RefDepthToSpaceWorkload : public BaseWorkload<DepthToSpaceQueueDescriptor>
{
public:
    using BaseWorkload<DepthToSpaceQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} // namespace armnn
