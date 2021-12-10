//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Encoders.hpp"
#include "Decoders.hpp"

#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

void InstanceNorm(const InstanceNormalizationQueueDescriptor& data,
                  const TensorInfo& inputInfo,
                  Decoder<float>& inputData,
                  Encoder<float>& outputData);

} // namespace armnn
