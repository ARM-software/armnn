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
inline std::unique_ptr<Encoder<T>> MakeEncoder(const TensorInfo& info, void* data);

template<>
inline std::unique_ptr<Encoder<float>> MakeEncoder(const TensorInfo& info, void* data)
{
    switch(info.GetDataType())
    {
        case armnn::DataType::QuantisedAsymm8:
        {
            return std::make_unique<QASymm8Encoder>(
                static_cast<uint8_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case armnn::DataType::QuantisedSymm16:
        {
            return std::make_unique<QSymm16Encoder>(
                static_cast<int16_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case armnn::DataType::Float32:
        {
            return std::make_unique<FloatEncoder>(static_cast<float*>(data));
        }
        default:
        {
            BOOST_ASSERT_MSG(false, "Cannot encode from float. Not supported target Data Type!");
            break;
        }
    }
    return nullptr;
}

template<>
inline std::unique_ptr<Encoder<bool>> MakeEncoder(const TensorInfo& info, void* data)
{
    switch(info.GetDataType())
    {
        case armnn::DataType::Boolean:
        {
            return std::make_unique<BooleanEncoder>(static_cast<uint8_t*>(data));
        }
        default:
        {
            BOOST_ASSERT_MSG(false, "Cannot encode from boolean. Not supported target Data Type!");
            break;
        }
    }
    return nullptr;
}

} //namespace armnn
