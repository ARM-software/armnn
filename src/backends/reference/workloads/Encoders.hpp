//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include <armnnUtils/TensorUtils.hpp>

namespace armnn
{

template<typename T>
inline std::unique_ptr<Encoder<T>> MakeEncoder(const TensorInfo& info, void* data = nullptr);

template<>
inline std::unique_ptr<Encoder<float>> MakeEncoder(const TensorInfo& info, void* data)
{
    switch(info.GetDataType())
    {
        case armnn::DataType::QAsymmS8:
        {
            return std::make_unique<QASymmS8Encoder>(
                static_cast<int8_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case armnn::DataType::QAsymmU8:
        {
            return std::make_unique<QASymm8Encoder>(
                static_cast<uint8_t*>(data),
                info.GetQuantizationScale(),
                info.GetQuantizationOffset());
        }
        case DataType::QSymmS8:
        {
            if (info.HasPerAxisQuantization())
            {
                std::pair<unsigned int, std::vector<float>> params = armnnUtils::GetPerAxisParams(info);
                return std::make_unique<QSymm8PerAxisEncoder>(
                        static_cast<int8_t*>(data),
                        params.second,
                        params.first);
            }
            else
            {
                return std::make_unique<QSymmS8Encoder>(
                        static_cast<int8_t*>(data),
                        info.GetQuantizationScale(),
                        info.GetQuantizationOffset());
            }
        }
        case armnn::DataType::QSymmS16:
        {
            if (info.HasPerAxisQuantization())
            {
                unsigned int axis = info.GetQuantizationDim().value();
                auto axisDimensionality = info.GetShape()[axis];
                std::pair<unsigned int, std::vector<float>> params = armnnUtils::GetPerAxisParams(info);
                return std::make_unique<QSymm16PerAxisEncoder>(
                        static_cast<int16_t*>(data),
                        params.second,
                        params.first,
                        axisDimensionality);
            }
            else
            {
                return std::make_unique<QSymm16Encoder>(
                        static_cast<int16_t *>(data),
                        info.GetQuantizationScale(),
                        info.GetQuantizationOffset());
            }
        }
        case armnn::DataType::Signed32:
        {
            return std::make_unique<Int32Encoder>(static_cast<int32_t*>(data));
        }
        case armnn::DataType::Float16:
        {
            return std::make_unique<Float16Encoder>(static_cast<Half*>(data));
        }
        case armnn::DataType::Float32:
        {
            return std::make_unique<Float32Encoder>(static_cast<float*>(data));
        }
        default:
        {
            throw InvalidArgumentException("Unsupported target Data Type!");
            break;
        }
    }
    return nullptr;
}

template<>
inline std::unique_ptr<Encoder<double_t>> MakeEncoder(const TensorInfo& info, void* data)
{
    switch(info.GetDataType())
    {
        case armnn::DataType::Signed64:
        {
            return std::make_unique<Int64Encoder>(static_cast<int64_t*>(data));
        }
        default:
        {
            throw InvalidArgumentException("Cannot encode from double. Unsupported target Data Type!");
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
            throw InvalidArgumentException("Cannot encode from boolean. Unsupported target Data Type!");
            break;
        }
    }
    return nullptr;
}

template<>
inline std::unique_ptr<Encoder<int32_t>> MakeEncoder(const TensorInfo& info, void* data)
{
    switch(info.GetDataType())
    {
        case DataType::Signed32:
        {
            return std::make_unique<Int32ToInt32tEncoder>(static_cast<int32_t*>(data));
        }
        default:
        {
            throw InvalidArgumentException("Cannot encode from int32. Unsupported Data Type!");
            break;
        }
    }
    return nullptr;
}

} //namespace armnn
