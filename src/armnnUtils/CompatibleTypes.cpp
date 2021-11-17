//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/Types.hpp>
#include <armnnUtils/CompatibleTypes.hpp>

#include "BFloat16.hpp"
#include "Half.hpp"

using namespace armnn;

namespace armnnUtils
{

template<typename T>
bool CompatibleTypes(DataType)
{
    return false;
}

template<>
bool CompatibleTypes<float>(DataType dataType)
{
    return dataType == DataType::Float32;
}

template<>
bool CompatibleTypes<Half>(DataType dataType)
{
    return dataType == DataType::Float16;
}

template<>
bool CompatibleTypes<BFloat16>(DataType dataType)
{
    return dataType == DataType::BFloat16;
}

template<>
bool CompatibleTypes<uint8_t>(DataType dataType)
{
    return dataType == DataType::Boolean || dataType == DataType::QAsymmU8;
}

template<>
bool CompatibleTypes<int8_t>(DataType dataType)
{
    return dataType == DataType::QSymmS8
        || dataType == DataType::QAsymmS8;
}

template<>
bool CompatibleTypes<int16_t>(DataType dataType)
{
    return dataType == DataType::QSymmS16;
}

template<>
bool CompatibleTypes<int32_t>(DataType dataType)
{
    return dataType == DataType::Signed32;
}

} //namespace armnnUtils
