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

void BatchNormImpl(const BatchNormalizationQueueDescriptor& data,
                   Decoder<float>& meanIn,
                   Decoder<float>& varIn,
                   Decoder<float>& betaIn,
                   Decoder<float>& gammaIn,
                   Decoder<float>& inputData,
                   Encoder<float>& outputData);

} // namespace armnn
