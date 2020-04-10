//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Assert.hpp"

#include <armnn/Exceptions.hpp>

#include <memory>
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


namespace utility
{
// static_pointer_cast overload for std::shared_ptr
template <class T1, class T2>
std::shared_ptr<T1> StaticPointerCast (const std::shared_ptr<T2>& sp)
{
    return std::static_pointer_cast<T1>(sp);
}

// dynamic_pointer_cast overload for std::shared_ptr
template <class T1, class T2>
std::shared_ptr<T1> DynamicPointerCast (const std::shared_ptr<T2>& sp)
{
    return std::dynamic_pointer_cast<T1>(sp);
}

// static_pointer_cast overload for raw pointers
template<class T1, class T2>
inline T1* StaticPointerCast(T2 *ptr)
{
    return static_cast<T1*>(ptr);
}

// dynamic_pointer_cast overload for raw pointers
template<class T1, class T2>
inline T1* DynamicPointerCast(T2 *ptr)
{
    return dynamic_cast<T1*>(ptr);
}

} // namespace utility

/// Polymorphic downcast for build in pointers only
///
/// Usage: Child* pChild = PolymorphicDowncast<Child*>(pBase);
///
/// \tparam DestType    Pointer type to the target object (Child pointer type)
/// \tparam SourceType  Pointer type to the source object (Base pointer type)
/// \param value        Pointer to the source object
/// \return             Pointer of type DestType (Pointer of type child)
template<typename DestType, typename SourceType>
DestType PolymorphicDowncast(SourceType value)
{
    static_assert(std::is_pointer<SourceType>::value &&
                  std::is_pointer<DestType>::value,
                  "PolymorphicDowncast only works with pointer types.");

    ARMNN_POLYMORPHIC_CAST_CHECK(dynamic_cast<DestType>(value) == static_cast<DestType>(value));
    return static_cast<DestType>(value);
}


/// Polymorphic downcast for shared pointers and build in pointers
///
/// Usage: auto pChild = PolymorphicPointerDowncast<Child>(pBase)
///
/// \tparam DestType    Type of the target object (Child type)
/// \tparam SourceType  Pointer type to the source object (Base (shared) pointer type)
/// \param value        Pointer to the source object
/// \return             Pointer of type DestType ((Shared) pointer of type child)
template<typename DestType, typename SourceType>
auto PolymorphicPointerDowncast(const SourceType& value)
{
    ARMNN_POLYMORPHIC_CAST_CHECK(utility::DynamicPointerCast<DestType>(value)
                                 == utility::StaticPointerCast<DestType>(value));
    return utility::StaticPointerCast<DestType>(value);
}

} //namespace armnn