//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "JsonPrinter.hpp"

#include <iomanip>
#include <iostream>

namespace armnn
{

void JsonPrinter::PrintJsonChildObject(const JsonChildObject& object)
{
    PrintLabel(object.m_Label);
    PrintMeasurementsList(object.m_Measurements);
    PrintSeparator();
    PrintNewLine();
    PrintUnit(object.m_Unit);

    if (!object.m_Children.empty())
    {
        PrintSeparator();
        PrintNewLine();
        for (unsigned int childIndex = 0; childIndex < object.m_Children.size(); ++childIndex)
        {
            PrintJsonChildObject(object.m_Children[childIndex]);
            // Only print separator and new line if current child is not the last element.
            if (&object.m_Children[childIndex] != &object.m_Children.back())
            {
                PrintSeparator();
                PrintNewLine();
            }
        }
    }
    PrintNewLine();
    PrintFooter();
}

void JsonPrinter::PrintHeader()
{
    m_OutputStream << "{" << std::endl;
    IncrementNumberOfTabs();
}

void JsonPrinter::PrintArmNNHeader()
{
    PrintTabs();
    m_OutputStream << R"("ArmNN": {)" << std::endl;
    IncrementNumberOfTabs();
}

void JsonPrinter::PrintLabel(const std::string& label)
{
    PrintTabs();
    m_OutputStream << R"(")" << label << R"(": {)" << std::endl;
    IncrementNumberOfTabs();
}

void JsonPrinter::PrintUnit(armnn::Measurement::Unit unit)
{
    PrintTabs();
    m_OutputStream << R"("unit": ")";
    m_OutputStream << armnn::Measurement::ToString(unit);
    m_OutputStream << R"(")";
}

void JsonPrinter::PrintMeasurementsList(const std::vector<double>& measurementsVector)
{
    if (measurementsVector.empty())
    {
        return;
    }

    PrintTabs();
    m_OutputStream << R"("raw": [)" << std::endl;
    IncrementNumberOfTabs();
    PrintTabs();
    auto iter = measurementsVector.begin();
    m_OutputStream << *iter;
    for (iter = std::next(iter); iter != measurementsVector.end(); ++iter)
    {
        m_OutputStream << "," << std::endl;
        PrintTabs();
        m_OutputStream << *iter;
    }
    m_OutputStream << std::endl;
    DecrementNumberOfTabs();
    PrintTabs();
    m_OutputStream << "]";
}

void JsonPrinter::PrintTabs()
{
    unsigned int numTabs = m_NumTabs;
    while (numTabs-- > 0)
    {
        m_OutputStream << "\t";
    }
}

void JsonPrinter::PrintSeparator()
{
    m_OutputStream << ",";
}

void JsonPrinter::PrintNewLine()
{
    m_OutputStream << std::endl;
}

void JsonPrinter::PrintFooter()
{
    DecrementNumberOfTabs();
    PrintTabs();
    m_OutputStream << "}";
}

void JsonPrinter::DecrementNumberOfTabs()
{
    if (m_NumTabs == 0)
    {
        return;
    }
    --m_NumTabs;
}

void JsonPrinter::IncrementNumberOfTabs()
{
    ++m_NumTabs;
}

} // namespace armnn