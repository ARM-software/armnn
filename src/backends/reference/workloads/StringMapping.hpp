//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

///
/// StringMapping is helper class to be able to use strings as template
/// parameters, so this allows simplifying code which only differs in
/// a string, such as a debug string literal.
///
struct StringMapping
{
public:
    enum Id {
        RefAdditionWorkload_Execute,
        RefDivisionWorkload_Execute,
        RefMaximumWorkload_Execute,
        RefMinimumWorkload_Execute,
        RefMultiplicationWorkload_Execute,
        RefSubtractionWorkload_Execute,
        MAX_STRING_ID
    };

    const char * Get(Id id) const
    {
        return m_Strings[id];
    }

    static const StringMapping& Instance();

private:
    StringMapping()
    {
        m_Strings[RefAdditionWorkload_Execute] = "RefAdditionWorkload_Execute";
        m_Strings[RefDivisionWorkload_Execute] = "RefDivisionWorkload_Execute";
        m_Strings[RefMaximumWorkload_Execute] = "RefMaximumWorkload_Execute";
        m_Strings[RefMinimumWorkload_Execute] = "RefMinimumWorkload_Execute";
        m_Strings[RefMultiplicationWorkload_Execute] = "RefMultiplicationWorkload_Execute";
        m_Strings[RefSubtractionWorkload_Execute] = "RefSubtractionWorkload_Execute";
    }

    StringMapping(const StringMapping &) = delete;
    StringMapping& operator=(const StringMapping &) = delete;

    const char * m_Strings[MAX_STRING_ID];
};

} //namespace armnn
