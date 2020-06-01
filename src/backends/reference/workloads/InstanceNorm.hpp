//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Encoders.hpp"
#include "Decoders.hpp"

#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

void InstanceNorm(const InstanceNormalizationQueueDescriptor& data,
                  Decoder<float>& inputData,
                  Encoder<float>& outputData);

} // namespace armnn
