//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>

#include <type_traits>

namespace armnn
{

using MemorySourceFlags = unsigned int;

template<typename T>
struct IsMemorySource
{
    static const bool value = false;
};

template<>
struct IsMemorySource<MemorySource>
{
    static const bool value = true;
};

template <typename Arg, typename std::enable_if<IsMemorySource<Arg>::value>::type* = nullptr>
MemorySourceFlags Combine(Arg sourceA, Arg sourceB)
{
    return static_cast<MemorySourceFlags>(sourceA) | static_cast<MemorySourceFlags>(sourceB);
}

template <typename Arg, typename ... Args, typename std::enable_if<IsMemorySource<Arg>::value>::type* = nullptr>
MemorySourceFlags Combine(Arg source, Args... rest)
{
    return static_cast<MemorySourceFlags>(source) | Combine(rest...);
}

inline bool CheckFlag(MemorySourceFlags flags, MemorySource source)
{
    return (static_cast<MemorySourceFlags>(source) & flags) != 0;
}

} //namespace armnn