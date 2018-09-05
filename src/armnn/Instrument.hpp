//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>
#include <vector>

namespace armnn
{

struct Measurement
{
    enum Unit
    {
        TIME_NS,
        TIME_US,
        TIME_MS,
    };

    inline static const char* ToString(Unit unit)
    {
        switch (unit)
        {
            case TIME_NS: return "ns";
            case TIME_US: return "us";
            case TIME_MS: return "ms";
            default:      return "";
        }
    }

    Measurement(const std::string& name, double value, Unit unit)
        : m_Name(name)
        , m_Value(value)
        , m_Unit(unit)
    {}
    Measurement(const Measurement&) = default;
    ~Measurement() = default;

    std::string m_Name;
    double m_Value;
    Unit m_Unit;

private:
    // please don't default construct, otherwise Units will be wrong
    Measurement() = delete;
};

class Instrument
{
public:
    virtual ~Instrument() {}

    virtual void Start() = 0;

    virtual void Stop() = 0;

    virtual std::vector<Measurement> GetMeasurements() const = 0;

    virtual const char* GetName() const = 0;

};

} //namespace armnn
