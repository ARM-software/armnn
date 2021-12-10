//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Encoders.hpp"
#include "Decoders.hpp"

#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

void PreluImpl(const TensorInfo& inputInfo,
               const TensorInfo& alphaInfo,
               const TensorInfo& outputInfo,
               Decoder<float>& inputData,
               Decoder<float>& alphaData,
               Encoder<float>& outputData);

} // namespace armnn
