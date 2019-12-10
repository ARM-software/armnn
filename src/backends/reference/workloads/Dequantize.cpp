//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Dequantize.hpp"

#include <boost/core/ignore_unused.hpp>
namespace armnn
{

void Dequantize(Decoder<float>& inputDecoder,
                Encoder<float>& outputEncoder,
                const TensorInfo& inputInfo,
                const TensorInfo& outputInfo)
{
    boost::ignore_unused(outputInfo);
    BOOST_ASSERT(inputInfo.GetNumElements() == outputInfo.GetNumElements());
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