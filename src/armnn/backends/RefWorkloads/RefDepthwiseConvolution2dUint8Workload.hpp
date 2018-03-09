//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefDepthwiseConvolution2dUint8Workload : public Uint8Workload<DepthwiseConvolution2dQueueDescriptor>
{
public:
    using Uint8Workload<DepthwiseConvolution2dQueueDescriptor>::Uint8Workload;
    virtual void Execute() const override;
};

} //namespace armnn
