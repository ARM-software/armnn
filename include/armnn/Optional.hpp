//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Exceptions.hpp"

namespace armnn
{

// NOTE: the members of the Optional class don't follow the ArmNN
//       coding convention because the interface to be close to
//       the C++-17 interface so we can easily migrate to std::optional
//       later.

template <typename T>
class Optional final
{
public:
    Optional(T&& value)
        : m_HasValue{true}
    {
        new (m_Storage) T(value);
    }

    Optional(const T& value)
        : m_HasValue{true}
    {
        new (m_Storage) T(value);
    }

    Optional(const Optional& other)
        : m_HasValue{false}
    {
        *this = other;
    }

    Optional() noexcept
        : m_HasValue{false}
    {
    }

    ~Optional()
    {
        reset();
    }

    operator bool() const noexcept
    {
        return has_value();
    }

    Optional& operator=(T&& value)
    {
        reset();
        new (m_Storage) T(value);
        m_HasValue = true;
        return *this;
    }

    Optional& operator=(const T& value)
    {
        reset();
        new(m_Storage) T(value);
        m_HasValue = true;
        return *this;
    }

    Optional& operator=(const Optional& other)
    {
        reset();
        if (other.has_value())
        {
            new (m_Storage) T(other.value());
            m_HasValue = true;
        }

        return *this;
    }

    const T& value() const
    {
        if (!has_value())
        {
            throw BadOptionalAccessException("Optional has no value");
        }

        auto valuePtr = reinterpret_cast<const T*>(m_Storage);
        return *valuePtr;
    }

    T& value()
    {
        if (!has_value())
        {
            throw BadOptionalAccessException("Optional has no value");
        }

        auto valuePtr = reinterpret_cast<T*>(m_Storage);
        return *valuePtr;
    }

    bool has_value() const noexcept
    {
        return m_HasValue;
    }

    void reset()
    {
        if (has_value())
        {
            value().T::~T();
            m_HasValue = false;
        }
    }

private:
    alignas(alignof(T)) unsigned char m_Storage[sizeof(T)];
    bool m_HasValue;
};

}
