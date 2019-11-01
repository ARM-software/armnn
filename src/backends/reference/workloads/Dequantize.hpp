//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include "Encoders.hpp"
#include "Decoders.hpp"

namespace armnn
{

void Dequantize(Decoder<float>& inputDecoder,
                Encoder<float>& outputEncoder,
                const TensorInfo& inputInfo,
                const TensorInfo& outputInfo);

} //namespace armnn
