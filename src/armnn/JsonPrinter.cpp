//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "JsonPrinter.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace armnn
{

void JsonPrinter::PrintJsonChildObject(const JsonChildObject& object, size_t& id)
{
    if (object.GetType() == JsonObjectType::Event)
    {
        // Increase the Id for new events. This ensures a new event has a unique ID and any measurements belonging
        // to the event have the same id. This id is appended to the name during the call to PrintLabel() below.
        id++;
    }

    if (object.GetType() != JsonObjectType::ExecObjectDesc)
    {
        PrintLabel(object.m_Label, id);
        if (object.m_Guid.has_value())
        {
            PrintGuid(object.m_Guid.value());
        }
        PrintType(object.m_Type);
    }

    if (!object.m_Measurements.empty() || !object.m_Children.empty())
    {
        PrintSeparator();
        PrintNewLine();
    }
    if (object.GetType() == JsonObjectType::Measurement)
    {
        PrintMeasurementsList(object.m_Measurements);
        PrintSeparator();
        PrintNewLine();
        PrintUnit(object.m_Unit);
    }
    else if (object.GetType() == JsonObjectType::ExecObjectDesc)
    {
        // Add details opening
        DecrementNumberOfTabs();
        PrintTabs();
        m_OutputStream << std::quoted("Graph") << ":[";
        PrintNewLine();

        // Fill details body
        for (std::string stringLine : object.m_LayerDetailsList)
        {
           PrintTabs();
           m_OutputStream << stringLine;
           PrintNewLine();
        }

        // Close out details
        PrintTabs();
        object.IsDetailsOnlyEnabled() ? m_OutputStream << "]" : m_OutputStream << "],";

        PrintNewLine();
        IncrementNumberOfTabs();
    }
    if (!object.m_Children.empty())
    {
        for (unsigned int childIndex = 0; childIndex < object.m_Children.size(); ++childIndex)
        {
            PrintJsonChildObject(object.m_Children[childIndex], id);
            // Only print separator and new line if current child is not the last element.
            if (&object.m_Children[childIndex] != &object.m_Children.back())
            {
                PrintSeparator();
                PrintNewLine();
            }
        }
    }
    if (object.GetType() != JsonObjectType::ExecObjectDesc)
    {
        PrintNewLine();
        PrintFooter();
    }
}

std::string JsonPrinter::MakeKey(const std::string& label, size_t id)
{
    std::stringstream ss;
    ss << label << std::string("_#") << id;
    return ss.str();
}

void JsonPrinter::PrintLabel(const std::string& label, size_t id)
{
    PrintTabs();
    m_OutputStream << R"(")" << MakeKey(label, id) << R"(": {)" << std::endl;
    IncrementNumberOfTabs();
}

void JsonPrinter::PrintUnit(armnn::Measurement::Unit unit)
{
    PrintTabs();
    m_OutputStream << R"("unit": ")";
    m_OutputStream << armnn::Measurement::ToString(unit);
    m_OutputStream << R"(")";
}

void JsonPrinter::PrintType(armnn::JsonObjectType type)
{
    auto ToString = [](armnn::JsonObjectType type)
        {
            switch (type)
            {
                case JsonObjectType::Measurement:
                {
                    return "Measurement";
                }
                case JsonObjectType::Event:
                {
                    return "Event";
                }
                case JsonObjectType::ExecObjectDesc:
                {
                    return "Operator Description";
                }
                default:
                {
                    return "Unknown";
                }
            }
        };
    PrintTabs();
    m_OutputStream << R"("type": ")";
    m_OutputStream << ToString(type);
    m_OutputStream << R"(")";
}

void JsonPrinter::PrintGuid(arm::pipe::ProfilingGuid guid)
{
    PrintTabs();
    m_OutputStream << std::quoted("GUID") << ": " << std::quoted(std::to_string(guid))  << "," << std::endl;
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

} // namespace armnn