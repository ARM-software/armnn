//
// Copyright © 2019, 2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Dequantize.hpp"

namespace armnn
{

void Dequantize(Decoder<float>& inputDecoder,
                Encoder<float>& outputEncoder,
                const TensorInfo& inputInfo,
                const TensorInfo& outputInfo)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(
        inputInfo.GetNumElements() == outputInfo.GetNumElements(),
        "Dequantize: The number of elements in the input and output tensors must be the same.");
    for (unsigned int i = 0; i < inputInfo.GetNumElements(); i++)
    {
        // inputDecoder.Get() dequantizes the data element from whatever
        // type is given by inputInfo to fp32 (If MakeDecoder supports that dequantization)
        // outputEncoder.Set() transforms the data element to whatever type is
        // given by outputInfo (if MakeEncoder supports that transformation)
        outputEncoder.Set(inputDecoder.Get());
        ++outputEncoder;
        ++inputDecoder;
    }
}

} // armnn namespace