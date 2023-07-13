//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include "Encoders.hpp"
#include "Decoders.hpp"
#include "armnn/Descriptors.hpp"

namespace armnn
{

void Tile(const TileDescriptor& params,
          const TensorInfo& inputInfo,
          Decoder<float>& inputDecoder,
          Encoder<float>& outputEncoder);

} // namespace armnn