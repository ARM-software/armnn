//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "WallClockTimer.hpp"
#include "armnn/Exceptions.hpp"

namespace armnn
{

const std::string WallClockTimer::WALL_CLOCK_TIME      ("Wall clock time");
const std::string WallClockTimer::WALL_CLOCK_TIME_START(WallClockTimer::WALL_CLOCK_TIME + " (Start)");
const std::string WallClockTimer::WALL_CLOCK_TIME_STOP (WallClockTimer::WALL_CLOCK_TIME + " (Stop)");

const char* WallClockTimer::GetName() const
{
    return "WallClockTimer";
}

void WallClockTimer::Start()
{
    m_Start = clock::now();
}

void WallClockTimer::Stop()
{
    m_Stop = clock::now();
}

void WallClockTimer::SetScaleFactor(Measurement::Unit measurementUnit)
{
    switch(measurementUnit)
    {
        case Measurement::TIME_MS:
            m_ScaleFactor = 1.f;
            break;
        case Measurement::TIME_US:
            m_ScaleFactor = 1000.f;
            break;
        case Measurement::TIME_NS:
            m_ScaleFactor = 1000000.f;
            break;
        default:
            throw InvalidArgumentException("Invalid scale used");
    }
    m_Unit = measurementUnit;
}

std::vector<Measurement> WallClockTimer::GetMeasurements() const
{
    const auto delta       = std::chrono::duration<double, std::milli>(m_Stop - m_Start);
    const auto startTimeMs = std::chrono::duration<double, std::milli>(m_Start.time_since_epoch());
    const auto stopTimeMs  = std::chrono::duration<double, std::milli>(m_Stop.time_since_epoch());

    return { { WALL_CLOCK_TIME,       delta.count() * m_ScaleFactor,       m_Unit },
             { WALL_CLOCK_TIME_START, startTimeMs.count() * m_ScaleFactor, m_Unit },
             { WALL_CLOCK_TIME_STOP,  stopTimeMs.count() * m_ScaleFactor,  m_Unit } };
}

} //namespace armnn
