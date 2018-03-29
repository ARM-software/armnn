//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "NeonConvolution2dBaseWorkload.hpp"

namespace armnn
{

class NeonConvolution2dUint8Workload : public NeonConvolution2dBaseWorkload<DataType::QuantisedAsymm8>
{
public:
    NeonConvolution2dUint8Workload(const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void ValidateData() const override;
    virtual void Execute() const override;
private:
};

} //namespace armnnn




