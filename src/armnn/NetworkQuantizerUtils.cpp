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
            QuantizeConstant(static_cast<const float*>(tensor.GetMemoryArea()),
                             backing.data(),
                             backing.size(),
                             scale,
                             offset);
        }
            break;
        default:
            ARMNN_ASSERT_MSG(false, "Can't quantize unsupported data type");
    }

    TensorInfo qInfo(tensor.GetInfo().GetShape(), DataType::QAsymmU8, scale, offset);
    return ConstTensor(qInfo, backing);
}

} // namespace armnn
