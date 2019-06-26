//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

void TransposeConvolution2dImpl(const TransposeConvolution2dDescriptor& descriptor,
                                const TensorShape& inputShape,
                                Decoder<float>& inputDecoder,
                                const TensorShape& outputShape,
                                Encoder<float>& outputEncoder,
                                const TensorShape& weightsShape,
                                Decoder<float>& weightsDecoder,
                                Decoder<float>* biasesDecoder);

} // namespace armnn