//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Types.hpp"
#include "BFloat16.hpp"
#include "Half.hpp"

namespace armnn
{

template<DataType DT>
struct ResolveTypeImpl;

template <>
struct ResolveTypeImpl<DataType::Float16>
{
    using Type = Half;
};

template<>
struct ResolveTypeImpl<DataType::Float32>
{
    using Type = float;
};

template<>
struct ResolveTypeImpl<DataType::QAsymmU8>
{
    using Type = uint8_t;
};

template<>
struct ResolveTypeImpl<DataType::QAsymmS8>
{
    using Type = int8_t;
};

template<>
struct ResolveTypeImpl<DataType::QSymmS8>
{
    using Type = int8_t;
};

template<>
struct ResolveTypeImpl<DataType::QSymmS16>
{
    using Type = int16_t;
};

template<>
struct ResolveTypeImpl<DataType::Signed32>
{
    using Type = int32_t;
};

template<>
struct ResolveTypeImpl<DataType::Signed64>
{
    using Type = int64_t;
};

template<>
struct ResolveTypeImpl<DataType::Boolean>
{
    using Type = uint8_t;
};

template<>
struct ResolveTypeImpl<DataType::BFloat16>
{
    using Type = BFloat16;
};

template<DataType DT>
using ResolveType = typename ResolveTypeImpl<DT>::Type;

} //namespace armnn
