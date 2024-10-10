//
// Copyright © 2017, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma  once

#include "armnn/Tensor.hpp"

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{
template<typename I, typename O>
void Gather(const TensorInfo& paramsInfo,
            const TensorInfo& indicesInfo,
            const TensorInfo& outputInfo,
            Decoder<I>& params,
            const int32_t* indices,
            Encoder<O>& output,
            const int32_t = 0);

} //namespace armnn
