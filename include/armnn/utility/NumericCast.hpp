//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Assert.hpp"

#include <type_traits>
#include <limits>

namespace armnn
{

#if !defined(NDEBUG) || defined(ARMNN_NUMERIC_CAST_TESTABLE)
#define ENABLE_NUMERIC_CAST_CHECKS 1
#else
#define ENABLE_NUMERIC_CAST_CHECKS 0
#endif

#if defined(ARMNN_NUMERIC_CAST_TESTABLE)
#   define ARMNN_NUMERIC_CAST_CHECK(cond, msg) ConditionalThrow<std::bad_cast>(cond)
#else
#define ARMNN_NUMERIC_CAST_CHECK(cond, msg) ARMNN_ASSERT_MSG(cond, msg)
#endif

// Unsigned to Unsigned

template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_unsigned<Source>::value &&
    std::is_unsigned<Dest>::value,
    Dest>
numeric_cast(Source source)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (source > std::numeric_limits<Dest>::max())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting unsigned type to "
                                        "narrower unsigned type. Overflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(source);
}

// Signed to Signed

// numeric cast from signed integral to signed integral types, checked for narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_signed<Source>::value &&
    std::is_integral<Source>::value &&
    std::is_signed<Dest>::value &&
    std::is_integral<Dest>::value,
    Dest>
numeric_cast(Source source)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (source > std::numeric_limits<Dest>::max())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed integral type to narrower signed type. "
                                        "Overflow detected.");
    }

    if (source < std::numeric_limits<Dest>::lowest())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed integral type to narrower signed type. "
                                        "Underflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(source);
}

// numeric cast from floating point to floating point types, checked for narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_floating_point<Source>::value &&
    std::is_floating_point<Dest>::value,
    Dest>
numeric_cast(Source source)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (source > std::numeric_limits<Dest>::max())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting floating point type to narrower signed type. "
                                        "Overflow detected.");
    }

    if (source < std::numeric_limits<Dest>::lowest())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting floating point type to narrower signed type. "
                                        "Underflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(source);
}

// numeric cast from floating point types (signed) to signed integral types, checked for narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_floating_point<Source>::value &&
    std::is_signed<Dest>::value &&
    std::is_integral<Dest>::value,
    Dest>
numeric_cast(Source source)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (source > static_cast<Source>(std::numeric_limits<Dest>::max()))
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting floating point type to narrower signed type. "
                                        "Overflow detected.");
    }

    if (source < static_cast<Source>(std::numeric_limits<Dest>::lowest()))
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting floating point type to narrower signed type. "
                                        "Underflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(source);
}

// numeric cast from signed integral types to floating point types (signed), checked for narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_signed<Source>::value &&
    std::is_integral<Source>::value &&
    std::is_floating_point<Dest>::value,
    Dest>
numeric_cast(Source source)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    Dest sourceConverted = static_cast<Dest>(source);

    if (sourceConverted > std::numeric_limits<Dest>::max())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed type to narrower floating point type. "
                                        "Overflow detected.");
    }

    if (sourceConverted < std::numeric_limits<Dest>::lowest())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed type to narrower floating point type. "
                                        "Underflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(source);
}

// Unsigned to Signed

// numeric cast from unsigned integral type to signed integral type, checked for narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_signed<Dest>::value &&
    std::is_integral<Dest>::value &&
    std::is_unsigned<Source>::value,
    Dest>
numeric_cast(Source sValue)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (sValue > static_cast< typename std::make_unsigned<Dest>::type >(std::numeric_limits<Dest>::max()))
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting unsigned type to signed type. "
                                        "Overflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(sValue);
}

// numeric cast from unsigned integral type to floating point (signed), checked for narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_floating_point<Dest>::value &&
    std::is_unsigned<Source>::value,
    Dest>
numeric_cast(Source sValue)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (static_cast<Dest>(sValue) > std::numeric_limits<Dest>::max())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting unsigned type to floating point type. "
                                        "Overflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS

    return static_cast<Dest>(sValue);
}

// Signed to Unsigned

// numeric cast from signed integral types to unsigned integral type, checked for underflows and narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_unsigned<Dest>::value &&
    std::is_signed<Source>::value &&
    std::is_integral<Source>::value,
    Dest>
numeric_cast(Source sValue)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (sValue < 0)
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting negative value to unsigned type. "
                                        "Underflow detected.");
    }

    if (static_cast< typename std::make_unsigned<Source>::type >(sValue) > std::numeric_limits<Dest>::max())
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting signed type to unsigned type. "
                                        "Overflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS
    return static_cast<Dest>(sValue);
}

// numeric cast from floating point (signed) to unsigned integral type, checked for underflows and narrowing overflows
template<typename Dest, typename Source>
typename std::enable_if_t<
    std::is_unsigned<Dest>::value &&
    std::is_floating_point<Source>::value,
    Dest>
numeric_cast(Source sValue)
{
#if ENABLE_NUMERIC_CAST_CHECKS
    if (sValue < 0)
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting negative value to unsigned type. "
                                        "Underflow detected.");
    }

    if (sValue > static_cast<Source>(std::numeric_limits<Dest>::max()))
    {
        ARMNN_NUMERIC_CAST_CHECK(false, "numeric_cast failed casting floating point type to unsigned type. "
                                        "Overflow detected.");
    }
#endif // ENABLE_NUMERIC_CAST_CHECKS
    return static_cast<Dest>(sValue);
}

#undef ENABLE_NUMERIC_CAST_CHECKS

} //namespace armnn