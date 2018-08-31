//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "ClDepthwiseConvolutionBaseWorkload.hpp"

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClDepthwiseConvolutionFloat32Workload : public ClDepthwiseConvolutionBaseWorkload<DataType::Float16,
                                                                                        DataType::Float32>
{
public:
    ClDepthwiseConvolutionFloat32Workload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                          const WorkloadInfo& info);
    void Execute() const override;
};

} //namespace armnn




