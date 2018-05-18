//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "NeonConvolution2dBaseWorkload.hpp"

#include "arm_compute/runtime/MemoryManagerOnDemand.h"

#include <memory>

namespace armnn
{

class NeonConvolution2dUint8Workload : public NeonConvolution2dBaseWorkload<DataType::QuantisedAsymm8>
{
public:
    NeonConvolution2dUint8Workload(const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info,
                                   std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);

    virtual void ValidateData() const override;
    virtual void Execute() const override;
private:
};

} //namespace armnnn

