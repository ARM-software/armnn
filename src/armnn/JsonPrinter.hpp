//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>
#include <map>
#include <set>
#include <sstream>

#include <ProfilingGuid.hpp>
#include "Instrument.hpp"
#include "JsonUtils.hpp"

namespace armnn
{

enum class JsonObjectType
{
    Measurement,
    Event,
    ExecObjectDesc
};

struct JsonChildObject
{
    // Object type changes according to the JsonObjectType specified in enum
    JsonChildObject(const std::string& label)
        : m_Label(label),
          m_Unit(Measurement::Unit::TIME_MS),
          m_Type(JsonObjectType::Event),
          m_Guid(armnn::EmptyOptional()),
          m_DetailsOnly(false)
    {}
    JsonChildObject(const JsonChildObject&) = default;

    void AddMeasurement(const double measurement)
    {
        m_Measurements.push_back(measurement);
    }

    void SetAndParseDetails(std::string layerDetailsStr)
    {
        std::stringstream layerDetails(layerDetailsStr);
        std::string stringLine;
        while (std::getline(layerDetails, stringLine, '\n'))
        {
            m_LayerDetailsList.push_back(stringLine);
        }
    }

    void SetGuid(profiling::ProfilingGuid guid)
    {
        m_Guid = Optional<profiling::ProfilingGuid>(guid);
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

    void EnableDetailsOnly()
    {
        m_DetailsOnly = true;
    }

    bool IsDetailsOnlyEnabled() const
    {
        return m_DetailsOnly;
    }

    ~JsonChildObject() = default;

    std::string m_Label;
    Measurement::Unit m_Unit;
    JsonObjectType m_Type;
    Optional<profiling::ProfilingGuid> m_Guid;
    std::vector<double> m_Measurements;
    std::vector<std::string> m_LayerDetailsList;
    std::vector<JsonChildObject> m_Children;

private:
    bool m_DetailsOnly;
    JsonChildObject() = delete;
};

class JsonPrinter : public JsonUtils
{
public:
    void PrintJsonChildObject(const JsonChildObject& object, size_t& id);
    void PrintLabel(const std::string& label, size_t id);
    void PrintUnit(armnn::Measurement::Unit unit);
    void PrintType(armnn::JsonObjectType type);
    void PrintGuid(armnn::profiling::ProfilingGuid guid);
    void PrintMeasurementsList(const std::vector<double>& measurementsVector);

public:
    JsonPrinter(std::ostream& outputStream)
        : JsonUtils(outputStream), m_OutputStream(outputStream)
    {}

private:
    std::string MakeKey(const std::string& label, size_t id);

    std::ostream& m_OutputStream;
};

} // namespace armnn