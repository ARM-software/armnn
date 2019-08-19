//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstddef>

namespace mlutil
{

namespace Impl
{

    constexpr uint32_t EncodeVersion(uint32_t major, uint32_t minor, uint32_t patch)
    {
        return (major << 22) | (minor << 12) | patch;
    }

} // namespace Impl

// Encodes a semantic version https://semver.org/ into a 32 bit integer in the following fashion
//
// bits 22:31 major: Unsigned 10-bit integer. Major component of the schema version number.
// bits 12:21 minor: Unsigned 10-bit integer. Minor component of the schema version number.
// bits 0:11  patch: Unsigned 12-bit integer. Patch component of the schema version number.
//
class Version
{
public:
    Version(uint32_t encodedValue)
    {
        m_Major = (encodedValue >> 22) & 1023;
        m_Minor = (encodedValue >> 12) & 1023;
        m_Patch = encodedValue & 4095;
    }

    Version(uint32_t major, uint32_t minor, uint32_t patch)
    : m_Major(major), m_Minor(minor), m_Patch(patch) {}

    uint32_t GetEncodedValue()
    {
        return mlutil::Impl::EncodeVersion(m_Major, m_Minor, m_Patch);
    }

    uint32_t GetMajor() {return m_Major;}
    uint32_t GetMinor() {return m_Minor;}
    uint32_t GetPatch() {return m_Patch;}

private:
    uint32_t m_Major;
    uint32_t m_Minor;
    uint32_t m_Patch;
};

} // namespace mlutil
