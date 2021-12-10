//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"
#include <armnn/Tensor.hpp>
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

/// Performs a matrix multiplication and optionally adds a bias.
void FullyConnected(const TensorShape& rInputShape,
                    Decoder<float>& rInputDecoder,
                    const TensorShape& rOutputShape,
                    Encoder<float>& rOutputEncoder,
                    const TensorShape& rWeightsShape,
                    Decoder<float>& rWeightDecoder,
                    Decoder<float>* rBiasDecoder,
                    bool biasEnabled,
                    unsigned int K,
                    bool transposeWeights);

} //namespace armnn
