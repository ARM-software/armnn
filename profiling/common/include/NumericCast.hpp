//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Assert.hpp"

#include <type_traits>
#include <limits>

namespace arm
{

namespace pipe
{

#if !defined(NDEBUG) || defined(ARM_PIPE_NUMERIC_CAST_TESTABLE)
#define ENABLE_NUMERIC_CAST_CHECKS 1
#else
#define ENABLE_NUMERIC_CAST_CHECKS 0
#endif

#if defined(ARM_PIPE_NUMERIC_CAST_TESTABLE)
#   define ARM_PIPE_NUMERIC_CAST_CHECK(cond, msg) ConditionalThrow<std::bad_cast>(cond)
#else
#   define ARM_PIPE_NUMERIC_CAST_CHECK(cond, msg) ARM_PIPE_ASSERT_MSG(cond, msg)
#endif

template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_unsigned<Source>::value &&
    std::is_unsigned<Dest>::value
    , Dest>
numeric_cast(Source source)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (source > std::numeric_limits<Dest>::max())
    {
        ARM_PIPE_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting unsigned type to "
                                        "narrower unsigned type. Overflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(source);
}

template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_signed<Source>::value &&
    std::is_signed<Dest>::value
    , Dest>
numeric_cast(Source source)
{
    static_assert(!std::is_floating_point<Source>::value && !std::is_floating_point<Dest>::value,
        "numeric_cast doesn't cast float.");

#if ENABLE_NUMERIC_CAST_CHECKS
    if (source > std::numeric_limits<Dest>::max())
    {
        ARM_PIPE_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed type to narrower signed type. "
                                        "Overflow detected.");
    }

    if (source < std::numeric_limits<Dest>::lowest())
    {
        ARM_PIPE_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed type to narrower signed type. "
                                        "Underflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(source);
}

// numeric cast from unsigned to signed checked for narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_signed<Dest>::value &&
    std::is_unsigned<Source>::value
    , Dest>
numeric_cast(Source sValue)
{
    static_assert(!std::is_floating_point<Dest>::value, "numeric_cast doesn't cast to float.");

#if ENABLE_NUMERIC_CAST_CHECKS
    if (sValue > static_cast< typename std::make_unsigned<Dest>::type >(std::numeric_limits<Dest>::max()))
    {
        ARM_PIPE_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting unsigned type to signed type. "
                                        "Overflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(sValue);
}

// numeric cast from signed to unsigned checked for underflows and narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_unsigned<Dest>::value &&
    std::is_signed<Source>::value
    , Dest>
numeric_cast(Source sValue)
{
    static_assert(!std::is_floating_point<Source>::value && !std::is_floating_point<Dest>::value,
        "numeric_cast doesn't cast floats.");

#if ENABLE_NUMERIC_CAST_CHECKS
    if (sValue < 0)
    {
        ARM_PIPE_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting negative value to unsigned type. "
                                        "Underflow detected.");
    }

    if (static_cast< typename std::make_unsigned<Source>::type >(sValue) > std::numeric_limits<Dest>::max())
    {
        ARM_PIPE_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed type to unsigned type. "
                                        "Overflow detected.");
    }

#endif // ENABLE_NUMERIC_CAST_CHECKS
    return static_cast<Dest>(sValue);
}

#undef ENABLE_NUMERIC_CAST_CHECKS

} // namespace pipe
} // namespace arm
