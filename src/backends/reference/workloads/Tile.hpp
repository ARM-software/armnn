//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include "Encoders.hpp"
#include "Decoders.hpp"
#include "armnn/Descriptors.hpp"

namespace armnn
{

template<typename I, typename O>
void Tile(const TileDescriptor& params,
          const TensorInfo& inputInfo,
          Decoder<I>& inputDecoder,
          Encoder<O>& outputEncoder);

} // namespace armnn
