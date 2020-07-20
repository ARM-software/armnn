//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#if __GNUC__
#   define ARM_PIPE_NO_CONVERSION_WARN_BEGIN \
    _Pragma("GCC diagnostic push")  \
    _Pragma("GCC diagnostic ignored \"-Wconversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"")

#   define ARM_PIPE_NO_CONVERSION_WARN_END \
    _Pragma("GCC diagnostic pop")

#elif __clang__
#   define ARM_PIPE_NO_CONVERSION_WARN_BEGIN \
    _Pragma("clang diagnostic push")  \
    _Pragma("clang diagnostic ignored \"-Wconversion\"") \
    _Pragma("clang diagnostic ignored \"-Wsign-conversion\"")

#   define ARM_PIPE_NO_CONVERSION_WARN_END \
    _Pragma("clang diagnostic pop")

#elif defined (_MSC_VER)
#   define ARM_PIPE_NO_CONVERSION_WARN_BEGIN \
    __pragma(warning( push )) \
    __pragma(warning(disable : 4101)) \
    __pragma(warning(disable : 4267))

#   define ARM_PIPE_NO_CONVERSION_WARN_END \
    __pragma(warning( pop ))

#else
#   define ARM_PIPE_NO_CONVERSION_WARN_BEGIN
#   define ARM_PIPE_NO_CONVERSION_WARN_END
#endif

#define ARM_PIPE_SUPRESS_CONVERSION_WARNING(func) \
ARM_PIPE_NO_CONVERSION_WARN_BEGIN \
func; \
ARM_PIPE_NO_CONVERSION_WARN_END
