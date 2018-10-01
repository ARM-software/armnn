//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Types.hpp"
#include "armnnUtils/Half.hpp"

namespace armnn
{


template<DataType DT>
struct ResolveTypeImpl;

template<>
struct ResolveTypeImpl<DataType::QuantisedAsymm8>
{
    using Type = uint8_t;
};

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

template<DataType DT>
using ResolveType = typename ResolveTypeImpl<DT>::Type;


} //namespace armnn
