//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma  once

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/Tensor.hpp>

namespace armnn
{

void Reduce(const TensorInfo& inputInfo,
            const TensorInfo& outputInfo,
            Decoder<float>& input,
            Encoder<float>& output,
            const std::vector<uint32_t> axis,
            const ReduceOperation reduceOperation);

} //namespace armnn
