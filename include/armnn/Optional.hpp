//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Exceptions.hpp"

#include <cstring>
#include <type_traits>

/// Optional is a drop in replacement for std::optional until we migrate
/// to c++-17. Only a subset of the optional features are implemented that
/// we intend to use in ArmNN.

/// There are two distinct implementations here:
///
///   1, for normal constructable/destructable types and reference types
///   2, for reference types

/// The std::optional features we support are:
///
/// - has_value() and operator bool() to tell if the optional has a value
/// - value() returns a reference to the held object
///

namespace armnn
{

/// EmptyOptional is used to initialize the Optional class in case we want
/// to have default value for an Optional in a function declaration.
struct EmptyOptional {};

/// Disambiguation tag that can be passed to the constructor to indicate that
/// the contained object should be constructed in-place
struct ConstructInPlace
{
    explicit ConstructInPlace() = default;
};

#define CONSTRUCT_IN_PLACE armnn::ConstructInPlace{}

/// OptionalBase is the common functionality between reference and non-reference
/// optional types.
class OptionalBase
{
public:
    OptionalBase() noexcept
        : m_HasValue{false}
    {
    }

    bool has_value() const noexcept
    {
        return m_HasValue;
    }

    /// Conversion to bool, so can be used in if-statements and similar contexts expecting a bool.
    /// Note this is explicit so that it doesn't get implicitly converted to a bool in unwanted cases,
    /// for example "Optional<TypeA> == Optional<TypeB>" should not compile.
    explicit operator bool() const noexcept
    {
        return has_value();
    }

protected:
    OptionalBase(bool hasValue) noexcept
        : m_HasValue{hasValue}
    {
    }

    bool m_HasValue;
};

///
/// The default implementation is the non-reference case. This
/// has an unsigned char array for storing the optional value which
/// is in-place constructed there.
///
template <bool IsReference, typename T>
class OptionalReferenceSwitch : public OptionalBase
{
public:
    using Base = OptionalBase;

    OptionalReferenceSwitch() noexcept : Base{} {}
    OptionalReferenceSwitch(EmptyOptional) noexcept : Base{} {}

    OptionalReferenceSwitch(const T& value)
        : Base{}
    {
        Construct(value);
    }

    template<class... Args>
    OptionalReferenceSwitch(ConstructInPlace, Args&&... args)
        : Base{}
    {
        Construct(CONSTRUCT_IN_PLACE, std::forward<Args>(args)...);
    }

    OptionalReferenceSwitch(const OptionalReferenceSwitch& other)
        : Base{}
    {
        *this = other;
    }

    OptionalReferenceSwitch& operator=(const T& value)
    {
        reset();
        Construct(value);
        return *this;
    }

    OptionalReferenceSwitch& operator=(const OptionalReferenceSwitch& other)
    {
        reset();
        if (other.has_value())
        {
            Construct(other.value());
        }

        return *this;
    }

    OptionalReferenceSwitch& operator=(EmptyOptional)
    {
        reset();
        return *this;
    }

    ~OptionalReferenceSwitch()
    {
        reset();
    }

    void reset()
    {
        if (Base::has_value())
        {
            value().T::~T();
            Base::m_HasValue = false;
        }
    }

    const T& value() const
    {
        if (!Base::has_value())
        {
            throw BadOptionalAccessException("Optional has no value");
        }

        auto valuePtr = reinterpret_cast<const T*>(m_Storage);
        return *valuePtr;
    }

    T& value()
    {
        if (!Base::has_value())
        {
            throw BadOptionalAccessException("Optional has no value");
        }

        auto valuePtr = reinterpret_cast<T*>(m_Storage);
        return *valuePtr;
    }

private:
    void Construct(const T& value)
    {
        new (m_Storage) T(value);
        m_HasValue = true;
    }

    template<class... Args>
    void Construct(ConstructInPlace, Args&&... args)
    {
        new (m_Storage) T(std::forward<Args>(args)...);
        m_HasValue = true;
    }

    alignas(alignof(T)) unsigned char m_Storage[sizeof(T)];
};

///
/// This is the special case for reference types. This holds a pointer
/// to the referenced type. This doesn't own the referenced memory and
/// it never calls delete on the pointer.
///
template <typename T>
class OptionalReferenceSwitch<true, T> : public OptionalBase
{
public:
    using Base = OptionalBase;
    using NonRefT = typename std::remove_reference<T>::type;

    OptionalReferenceSwitch() noexcept : Base{}, m_Storage{nullptr} {}
    OptionalReferenceSwitch(EmptyOptional) noexcept : Base{}, m_Storage{nullptr} {}

    OptionalReferenceSwitch(const OptionalReferenceSwitch& other) : Base{}
    {
        *this = other;
    }

    OptionalReferenceSwitch(T value)
        : Base{true}
        , m_Storage{&value}
    {
    }

    template<class... Args>
    OptionalReferenceSwitch(ConstructInPlace, Args&&... args) = delete;

    OptionalReferenceSwitch& operator=(const T value)
    {
        m_Storage = &value;
        Base::m_HasValue = true;
        return *this;
    }

    OptionalReferenceSwitch& operator=(const OptionalReferenceSwitch& other)
    {
        m_Storage = other.m_Storage;
        Base::m_HasValue = other.has_value();
        return *this;
    }

    OptionalReferenceSwitch& operator=(EmptyOptional)
    {
        reset();
        return *this;
    }

    ~OptionalReferenceSwitch()
    {
        reset();
    }

    void reset()
    {
        Base::m_HasValue = false;
        m_Storage = nullptr;
    }

    const T value() const
    {
        if (!Base::has_value())
        {
            throw BadOptionalAccessException("Optional has no value");
        }

        return *m_Storage;
    }

    T value()
    {
        if (!Base::has_value())
        {
            throw BadOptionalAccessException("Optional has no value");
        }

        return *m_Storage;
    }

private:
    NonRefT* m_Storage;
};

template <typename T>
class Optional final : public OptionalReferenceSwitch<std::is_reference<T>::value, T>
{
public:
    using BaseSwitch = OptionalReferenceSwitch<std::is_reference<T>::value, T>;

    Optional() noexcept : BaseSwitch{} {}
    Optional(const T& value) : BaseSwitch{value} {}
    Optional& operator=(const Optional& other) = default;
    Optional(EmptyOptional empty) : BaseSwitch{empty} {}
    Optional(const Optional& other) : BaseSwitch{other} {}
    Optional(const BaseSwitch& other) : BaseSwitch{other} {}

    template<class... Args>
    explicit Optional(ConstructInPlace, Args&&... args) :
        BaseSwitch(CONSTRUCT_IN_PLACE, std::forward<Args>(args)...) {}

    /// Two optionals are considered equal if they are both empty or both contain values which
    /// themselves are considered equal (via their own == operator).
    bool operator==(const Optional<T>& rhs) const
    {
        if (!this->has_value() && !rhs.has_value())
        {
            return true;
        }
        if (this->has_value() && rhs.has_value() && this->value() == rhs.value())
        {
            return true;
        }
        return false;
    }
};

/// Utility template that constructs an object of type T in-place and wraps
/// it inside an Optional<T> object
template<typename T, class... Args>
Optional<T> MakeOptional(Args&&... args)
{
    return Optional<T>(CONSTRUCT_IN_PLACE, std::forward<Args>(args)...);
}

}
