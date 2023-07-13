//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Encoders.hpp"
#include "Decoders.hpp"

#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

void ReverseV2(const TensorInfo& inputInfo,
               const TensorInfo& axisInfo,
               Decoder<float>& inputDecoder,
               Decoder<int>& axisDecoder,
               Encoder<float>& outputEncoder);

} // namespace armnn
