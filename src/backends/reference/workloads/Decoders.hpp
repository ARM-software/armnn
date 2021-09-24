//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>
#include <armnnUtils/TensorUtils.hpp>

#include <armnn/utility/Assert.hpp>

namespace armnn
{

namespace
{

inline std::unique_ptr<Decoder<float>> MakeSigned32PerAxisDecoder(const TensorInfo& info, const void* data)
{
    return std::make_unique<ScaledInt32PerAxisDecoder>(static_cast<const int32_t*>(data), info);
}

inline std::unique_ptr<Decoder<float>> MakeSigned32Decoder(const TensorInfo& info, const void* data)
{
    if(info.HasMultipleQuantizationScales())
    {
        // NOTE: If we have multiple quantization scales, we create a ScaledInt32PerAxisDecoder.
        // This will be used to decode per-axis quantized convolution biases.
        return MakeSigned32PerAxisDecoder(info, data);
    }
    else
    {
        if (info.GetQuantizationDim().has_value())
        {
            // NOTE: Even though we only have a single quantization scale, if the quantization
            // dimension is set, the tensor has per-axis quantization and we need to create a
            // ScaledInt32PerAxisDecoder
            return MakeSigned32PerAxisDecoder(info, data);
        }

        const float scale = info.GetQuantizationScale();
        if (scale == 0.f)
        {
            // NOTE:: If no quantization scale is set, we create an Int32Decoder, which simply
            // casts the int value to float. This will be used for any INT32 data other than
            // convolution biases.
            return std::make_unique<Int32Decoder>(static_cast<const int32_t*>(data));
        }

        // NOTE: If we only have a single (non-zero) quantization scale and no quantization
        // dimension is specified, we need to create a ScaledInt32Decoder. This will be used
        // to decode per-tensor quantized convolution biases.
        return std::make_unique<ScaledInt32Decoder>(static_cast<const int32_t*>(data), scale);
    }
}

} // anonymous namespace

template<typename T>
inline std::unique_ptr<Decoder<T>> MakeDecoder(const TensorInfo& info, const void* data = nullptr);

template<>
inline std::unique_ptr<Decoder<float>> MakeDecoder(const TensorInfo& info, const void* data)
{
    switch(info.GetDataType())
    {
        case DataType::QAsymmS8:
        {
            return std::make_unique<QASymmS8Decoder>(
                static_cast<const int8_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case DataType::QAsymmU8:
        {
            return std::make_unique<QASymm8Decoder>(
                static_cast<const uint8_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case DataType::QSymmS16:
        {
            return std::make_unique<QSymm16Decoder>(
                static_cast<const int16_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case DataType::BFloat16:
        {
            return std::make_unique<BFloat16Decoder>(static_cast<const BFloat16*>(data));
        }
        case DataType::Float16:
        {
            return std::make_unique<Float16Decoder>(static_cast<const Half*>(data));
        }
        case DataType::Float32:
        {
            return std::make_unique<Float32Decoder>(static_cast<const float*>(data));
        }
        case DataType::Signed32:
        {
            return MakeSigned32Decoder(info, data);
        }
        case DataType::QSymmS8:
        {
            if (info.HasPerAxisQuantization())
            {
                std::pair<unsigned int, std::vector<float>> params = armnnUtils::GetPerAxisParams(info);
                return std::make_unique<QSymm8PerAxisDecoder>(static_cast<const int8_t*>(data), info);
            }
            else
            {
                return std::make_unique<QSymmS8Decoder>(
                    static_cast<const int8_t*>(data),
                    info.GetQuantizationScale(),
                    info.GetQuantizationOffset());
            }
        }
        case armnn::DataType::Boolean:
        {
            return std::make_unique<BooleanDecoder>(static_cast<const uint8_t*>(data));
        }
        default:
        {
            ARMNN_ASSERT_MSG(false, "Unsupported Data Type!");
            break;
        }
    }
    return nullptr;
}

template<>
inline std::unique_ptr<Decoder<bool>> MakeDecoder(const TensorInfo& info, const void* data)
{
    switch(info.GetDataType())
    {
        case DataType::Boolean:
        {
            return std::make_unique<BooleanDecoderBool>(static_cast<const uint8_t*>(data));
        }
        default:
        {
            ARMNN_ASSERT_MSG(false, "Unsupported Data Type!");
            break;
        }
    }
    return nullptr;
}

template<>
inline std::unique_ptr<Decoder<int32_t>> MakeDecoder(const TensorInfo& info, const void* data)
{
    switch(info.GetDataType())
    {
        case DataType::Signed32:
        {
            return std::make_unique<Int32ToInt32tDecoder>(static_cast<const int32_t*>(data));
        }
        default:
        {
            ARMNN_ASSERT_MSG(false, "Unsupported Data Type!");
            break;
        }
    }
    return nullptr;
}

} //namespace armnn
