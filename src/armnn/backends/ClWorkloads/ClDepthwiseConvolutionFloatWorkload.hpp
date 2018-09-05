//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClDepthwiseConvolutionBaseWorkload.hpp"

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClDepthwiseConvolutionFloatWorkload : public ClDepthwiseConvolutionBaseWorkload<DataType::Float16,
                                                                                      DataType::Float32>
{
public:
    ClDepthwiseConvolutionFloatWorkload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                        const WorkloadInfo& info);
    void Execute() const override;
};

} //namespace armnn




