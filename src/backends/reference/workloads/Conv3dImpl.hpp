//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/Tensor.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

namespace armnn
{

void Convolve3d(const TensorShape& rInputShape,
                Decoder<float>& rInputDecoder,
                const TensorShape& rOutputShape,
                Encoder<float>& rOutputEncoder,
                const TensorShape& rFilterShape,
                Decoder<float>& rFilterDecoder,
                bool biasEnabled,
                Decoder<float>* pBiasDecoder,
                DataLayout dataLayout,
                unsigned int paddingTop,
                unsigned int paddingLeft,
                unsigned int paddingFront,
                unsigned int xStride,
                unsigned int yStride,
                unsigned int zStride,
                unsigned int xDilation,
                unsigned int yDilation,
                unsigned int zDilation);

} //namespace armnn
