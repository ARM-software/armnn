//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Exceptions.hpp"
#include <type_traits>
#include <cstring>

// Optional is a drop in replacement for std::optional until we migrate
// to c++-17. Only a subset of the optional features are implemented that
// we intend to use in ArmNN.

// There are two distinct implementations here:
//
//   1, for normal constructable/destructable types and reference types
//   2, for reference types

// The std::optional features we support are:
//
// - has_value() and operator bool() to tell if the optional has a value
// - value() returns a reference to the held object
//

namespace armnn
{

// EmptyOptional is used to initialize the Optional class in case we want
// to have default value for an Optional in a function declaration.
struct EmptyOptional {};

// OptionalBase is the common functionality between reference and non-reference
// optional types.
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

    operator bool() const noexcept
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

//
// The default implementation is the non-reference case. This
// has an unsigned char array for storing the optional value which
// is in-place constructed there.
//
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

    alignas(alignof(T)) unsigned char m_Storage[sizeof(T)];
};

//
// This is the special case for reference types. This holds a pointer
// to the referenced type. This doesn't own the referenced memory and
// it never calls delete on the pointer.
//
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
    Optional(EmptyOptional empty) : BaseSwitch{empty} {}
    Optional(const Optional& other) : BaseSwitch{other} {}
    Optional(const BaseSwitch& other) : BaseSwitch{other} {}
};

}
