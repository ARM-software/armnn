//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

namespace armnn
{

class RefActivationFloat32Workload : public Float32Workload<ActivationQueueDescriptor>
{
public:
    using Float32Workload<ActivationQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
