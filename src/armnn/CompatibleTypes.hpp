//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Types.hpp"
#include "Half.hpp"

namespace armnn
{

template<typename T>
bool CompatibleTypes(DataType dataType)
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
inline bool CompatibleTypes<uint8_t>(DataType dataType)
{
    return dataType == DataType::Boolean || dataType == DataType::QuantisedAsymm8;
}

template<>
inline bool CompatibleTypes<int16_t>(DataType dataType)
{
    return dataType == DataType::QuantisedSymm16;
}

template<>
inline bool CompatibleTypes<int32_t>(DataType dataType)
{
    return dataType == DataType::Signed32;
}

} //namespace armnn
