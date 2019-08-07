//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ostream>
#include <string.h>
#include <map>
#include <set>

#include "Instrument.hpp"

namespace armnn
{

enum class JsonObjectType
{
    Measurement,
    Event
};

struct JsonChildObject
{
    JsonChildObject(const std::string& label)
            : m_Label(label), m_Unit(Measurement::Unit::TIME_MS), m_Type(JsonObjectType::Event)
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

    JsonChildObject& GetChild(const unsigned int index)
    {
        return m_Children[index];
    }

    void SetUnit(const Measurement::Unit unit)
    {
        m_Unit = unit;
    }

    size_t NumChildren() const
    {
        return m_Children.size();
    }

    void SetType(JsonObjectType type)
    {
        m_Type = type;
    }

    JsonObjectType GetType() const
    {
        return m_Type;
    }

    ~JsonChildObject() = default;

    std::string m_Label;
    Measurement::Unit m_Unit;
    JsonObjectType m_Type;
    std::vector<double> m_Measurements;
    std::vector<JsonChildObject> m_Children;

private:
    JsonChildObject() = delete;
};

class JsonPrinter
{
public:
    void PrintJsonChildObject(const JsonChildObject& object, size_t& id);
    void PrintHeader();
    void PrintArmNNHeader();
    void PrintFooter();
    void PrintSeparator();
    void PrintNewLine();
    void PrintLabel(const std::string& label, size_t id);
    void PrintUnit(armnn::Measurement::Unit unit);
    void PrintType(armnn::JsonObjectType type);
    void PrintMeasurementsList(const std::vector<double>& measurementsVector);

public:
    JsonPrinter(std::ostream &outputStream)
        : m_OutputStream(outputStream), m_NumTabs(0)
    {}

private:
    std::string MakeKey(const std::string& label, size_t id);
    void PrintTabs();
    void DecrementNumberOfTabs();
    void IncrementNumberOfTabs();

    std::ostream &m_OutputStream;
    unsigned int m_NumTabs;
};

} // namespace armnn