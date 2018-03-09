//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"

namespace armnn
{

class RefActivationFloat32Workload : public Float32Workload<ActivationQueueDescriptor>
{
public:
    using Float32Workload<ActivationQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
