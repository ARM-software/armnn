//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#if __GNUC__
#   define ARMNN_NO_DEPRECATE_WARN_BEGIN \
    _Pragma("GCC diagnostic push")  \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#   define ARMNN_NO_DEPRECATE_WARN_END \
    _Pragma("GCC diagnostic pop")

#elif __clang__
#   define ARMNN_NO_DEPRECATE_WARN_BEGIN \
    _Pragma("clang diagnostic push")  \
    _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")

#   define ARMNN_NO_DEPRECATE_WARN_END \
    _Pragma("clang diagnostic pop")

#elif defined (_MSC_VER)
#   define ARMNN_NO_DEPRECATE_WARN_BEGIN \
    __pragma(warning( push )) \
    __pragma(warning(disable : 4996))

#   define ARMNN_NO_DEPRECATE_WARN_END \
    __pragma(warning( pop ))

#else
#   define ARMNN_NO_DEPRECATE_WARN_BEGIN
#   define ARMNN_NO_DEPRECATE_WARN_END
#endif

#define ARMNN_SUPRESS_DEPRECATE_WARNING(func) \
ARMNN_NO_DEPRECATE_WARN_BEGIN \
func; \
ARMNN_NO_DEPRECATE_WARN_END

#define ARMNN_DEPRECATED [[deprecated]]
#define ARMNN_DEPRECATED_MSG(message) [[deprecated(message)]]

#if defined(__GNUC__) && (__GNUC__ < 6)
#   define ARMNN_DEPRECATED_ENUM
#   define ARMNN_DEPRECATED_ENUM_MSG(message)
#else
#   define ARMNN_DEPRECATED_ENUM ARMNN_DEPRECATED
#   define ARMNN_DEPRECATED_ENUM_MSG(message) ARMNN_DEPRECATED_MSG(message)
#endif