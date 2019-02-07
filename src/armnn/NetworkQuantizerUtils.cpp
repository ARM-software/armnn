//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkQuantizerUtils.hpp"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace armnn
{

std::pair<int, float> ComputeQAsymmParams(int numBits, double min, double max)
{
    BOOST_ASSERT_MSG(min < max, "min >= max will result in invalid quantization.");
    double highest = (1 << numBits) - 1;

    min = std::min(0.0, min); // min <= 0.0
    max = std::max(0.0, max); // max >= 0.0

    // Assumes quantization range [0-highest]
    double scale = (max-min) / highest;
    double offset = -min / scale;

    // Clamp offset [0-highest]
    offset = std::max(0.0, std::min(highest, offset));

    return std::make_pair(static_cast<int>(std::round(offset)), static_cast<float>(scale));
}

ConstTensor CreateQuantizedConst(const ConstTensor& tensor, std::vector<uint8_t>& backing)
{
    float scale = 0.0f;
    int offset = 0;

    // Reserve the backing memory
    backing.resize(tensor.GetInfo().GetNumElements());

    DataType type = tensor.GetInfo().GetDataType();
    switch(type)
    {
        case DataType::Float32:
        {
            Quantize(static_cast<const float*>(tensor.GetMemoryArea()),
                     backing.data(),
                     backing.size(),
                     scale,
                     offset);
        }
            break;
        default:
            BOOST_ASSERT_MSG(false, "Can't quantize unsupported data type");
    }

    TensorInfo qInfo(tensor.GetInfo().GetShape(), DataType::QuantisedAsymm8, scale, offset);
    return ConstTensor(qInfo, backing);
}

} // namespace armnn
