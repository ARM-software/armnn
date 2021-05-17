//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <sstream>

namespace armnn
{

class PredicateResult
{
public:
    explicit PredicateResult(bool result)
        : m_Result(result)
    {}

    PredicateResult(const PredicateResult& predicateResult)
        : m_Result(predicateResult.m_Result)
        , m_Message(predicateResult.m_Message.str())
    {}

    void SetResult(bool newResult)
    {
        m_Result = newResult;
    }

    std::stringstream& Message()
    {
        return m_Message;
    }

    bool operator!() const
    {
        return !m_Result;
    }

    void operator=(PredicateResult otherPredicateResult)
    {
        otherPredicateResult.m_Result = m_Result;
    }

    bool m_Result;
    std::stringstream m_Message;
};

}    // namespace armnn