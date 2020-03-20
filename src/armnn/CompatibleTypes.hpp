//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>

#include <BFloat16.hpp>
#include <Half.hpp>

namespace armnn
{

template<typename T>
bool CompatibleTypes(DataType)
{
    return false;
}

template<>
inline bool CompatibleTypes<float>(DataType dataType)
{
    return dataType == DataType::Float32;
}

template<>
inline bool CompatibleTypes<Half>(DataType dataType)
{
    return dataType == DataType::Float16;
}

template<>
inline bool CompatibleTypes<BFloat16>(DataType dataType)
{
    return dataType == DataType::BFloat16;
}

template<>
inline bool CompatibleTypes<uint8_t>(DataType dataType)
{
    return dataType == DataType::Boolean || dataType == DataType::QAsymmU8;
}

template<>
inline bool CompatibleTypes<int8_t>(DataType dataType)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return dataType == DataType::QSymmS8
        || dataType == DataType::QuantizedSymm8PerAxis
        || dataType == DataType::QAsymmS8;
    ARMNN_NO_DEPRECATE_WARN_END
}

template<>
inline bool CompatibleTypes<int16_t>(DataType dataType)
{
    return dataType == DataType::QSymmS16;
}

template<>
inline bool CompatibleTypes<int32_t>(DataType dataType)
{
    return dataType == DataType::Signed32;
}

} //namespace armnn
