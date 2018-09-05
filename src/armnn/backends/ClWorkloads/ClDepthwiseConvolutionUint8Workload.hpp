//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClDepthwiseConvolutionBaseWorkload.hpp"

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClDepthwiseConvolutionUint8Workload : public ClDepthwiseConvolutionBaseWorkload<DataType::QuantisedAsymm8>
{
public:
    ClDepthwiseConvolutionUint8Workload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                        const WorkloadInfo& info);
    void Execute() const override;
};

} //namespace armnn


