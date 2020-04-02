//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Assert.hpp"

#include <armnn/Exceptions.hpp>

#include <type_traits>

namespace armnn
{

// If we are testing then throw an exception, otherwise regular assert
#if defined(ARMNN_POLYMORPHIC_CAST_TESTABLE)
#   define ARMNN_POLYMORPHIC_CAST_CHECK_METHOD(cond) ConditionalThrow<std::bad_cast>(cond)
#else
#   define ARMNN_POLYMORPHIC_CAST_CHECK_METHOD(cond) ARMNN_ASSERT(cond)
#endif

//Only check the condition if debug build or during testing
#if !defined(NDEBUG) || defined(ARMNN_POLYMORPHIC_CAST_TESTABLE)
#   define ARMNN_POLYMORPHIC_CAST_CHECK(cond)  ARMNN_POLYMORPHIC_CAST_CHECK_METHOD(cond)
#else
#   define ARMNN_POLYMORPHIC_CAST_CHECK(cond) // release builds dont check the cast
#endif


template<typename DestType, typename SourceType>
DestType PolymorphicDowncast(SourceType value)
{
    static_assert(std::is_pointer<SourceType>::value &&
                  std::is_pointer<DestType>::value,
                  "PolymorphicDowncast only works with pointer types.");

    ARMNN_POLYMORPHIC_CAST_CHECK(dynamic_cast<DestType>(value) == static_cast<DestType>(value));
    return static_cast<DestType>(value);
}

} //namespace armnn