//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ostream>
#include <string.h>
#include <map>

#include "Instrument.hpp"

namespace armnn
{

struct JsonChildObject
{
    JsonChildObject(const std::string& label)
            : m_Label(label), m_Unit(Measurement::Unit::TIME_MS)
    {}
    JsonChildObject(const JsonChildObject&) = default;

    void AddMeasurement(const double measurement)
    {
        m_Measurements.push_back(measurement);
    }

    void AddChild(const JsonChildObject& childObject)
    {
        m_Children.push_back(childObject);
    }

    JsonChildObject GetChild(const unsigned int index)
    {
        return m_Children[index];
    }

    void SetUnit(const Measurement::Unit unit)
    {
        m_Unit = unit;
    }

    ~JsonChildObject() = default;

    std::string m_Label;
    Measurement::Unit m_Unit;
    std::vector<double> m_Measurements;
    std::vector<JsonChildObject> m_Children;

private:
    JsonChildObject() = delete;
};

class JsonPrinter
{
public:
    void PrintJsonChildObject(const JsonChildObject& object);
    void PrintHeader();
    void PrintArmNNHeader();
    void PrintFooter();
    void PrintSeparator();
    void PrintNewLine();
    void PrintLabel(const std::string& label);
    void PrintUnit(armnn::Measurement::Unit unit);
    void PrintMeasurementsList(const std::vector<double>& measurementsVector);

public:
    JsonPrinter(std::ostream &outputStream)
        : m_OutputStream(outputStream), m_NumTabs(0)
    {}

private:
    void PrintTabs();
    void DecrementNumberOfTabs();
    void IncrementNumberOfTabs();

    std::ostream &m_OutputStream;
    unsigned int m_NumTabs;
};

} // namespace armnn