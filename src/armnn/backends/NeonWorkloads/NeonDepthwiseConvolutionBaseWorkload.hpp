//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/NeonWorkloadUtils.hpp"

#include <boost/optional.hpp>

namespace armnn
{

arm_compute::Status NeonDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
                                                             const TensorInfo& output,
                                                             const DepthwiseConvolution2dDescriptor& descriptor,
                                                             const TensorInfo& weights,
                                                             const boost::optional<TensorInfo>& biases);

} // namespace armnn
