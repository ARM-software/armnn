//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include <boost/assert.hpp>

namespace armnn
{

template<typename T>
inline std::unique_ptr<Decoder<T>> MakeDecoder(const TensorInfo& info, const void* data = nullptr);

template<>
inline std::unique_ptr<Decoder<float>> MakeDecoder(const TensorInfo& info, const void* data)
{
    switch(info.GetDataType())
    {
        case armnn::DataType::QuantisedAsymm8:
        {
            return std::make_unique<QASymm8Decoder>(
                static_cast<const uint8_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case armnn::DataType::QuantisedSymm16:
        {
            return std::make_unique<QSymm16Decoder>(
                static_cast<const int16_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case armnn::DataType::Float32:
        {
            return std::make_unique<FloatDecoder>(static_cast<const float*>(data));
        }
        case armnn::DataType::Signed32:
        {
            const float scale = info.GetQuantizationScale();
            if (scale == 0.f)
            {
                return std::make_unique<Int32Decoder>(static_cast<const int32_t*>(data));
            }
            // NOTE: ScaledInt32Decoder is used for quantized convolution biases
            return std::make_unique<ScaledInt32Decoder>(static_cast<const int32_t*>(data), scale);
        }
        default:
        {
            BOOST_ASSERT_MSG(false, "Not supported Data Type!");
            break;
        }
    }
    return nullptr;
}

} //namespace armnn
