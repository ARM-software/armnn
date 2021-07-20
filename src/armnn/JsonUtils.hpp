//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iomanip>

#include "armnn/Types.hpp"
#include "armnn/backends/WorkloadInfo.hpp"

namespace armnn
{

class JsonUtils
{
public:
    JsonUtils(std::ostream& outputStream)
        : m_NumTabs(0), m_OutputStream(outputStream)
    {}

    void PrintTabs()
    {
        unsigned int numTabs = m_NumTabs;
        while ( numTabs-- > 0 )
        {
            m_OutputStream << "\t";
        }
    }

    void DecrementNumberOfTabs()
    {
        if ( m_NumTabs == 0 )
        {
            return;
        }
        --m_NumTabs;
    }

    void IncrementNumberOfTabs()
    {
        ++m_NumTabs;
    }

    void PrintNewLine()
    {
        m_OutputStream << std::endl;
    }

    void PrintFooter()
    {
        DecrementNumberOfTabs();
        PrintTabs();
        m_OutputStream << "}";
    }

    void PrintHeader()
    {
        m_OutputStream << "{" << std::endl;
        IncrementNumberOfTabs();
    }

    void PrintArmNNHeader()
    {
        PrintTabs();
        m_OutputStream << R"("ArmNN": {)" << std::endl;
        IncrementNumberOfTabs();
    }
    void PrintSeparator()
    {
        m_OutputStream << ",";
    }

private:
    unsigned int m_NumTabs;
    std::ostream& m_OutputStream;
};

} // namespace armnn