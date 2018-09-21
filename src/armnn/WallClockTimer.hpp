//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Instrument.hpp"
#include <chrono>

namespace armnn
{

// Clock class that uses the same timestamp function as the Mali DDK.
class monotonic_clock_raw {
public:
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<monotonic_clock_raw, duration>;

    static std::chrono::time_point<monotonic_clock_raw, std::chrono::nanoseconds> now() noexcept
    {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return time_point(std::chrono::nanoseconds(ts.tv_sec*1000000000 + ts.tv_nsec));
    }
};

// Implementation of an instrument to measure elapsed wall-clock time in milliseconds.
class WallClockTimer : public Instrument
{
public:
    // Construct a Wall Clock Timer
    WallClockTimer() = default;
    ~WallClockTimer() = default;

    // Start the Wall clock timer
    void Start() override;

    // Stop the Wall clock timer
    void Stop() override;

    // Get the name of the timer
    const char* GetName() const override;

    void SetScaleFactor(Measurement::Unit measurementUnit);

    // Get the recorded measurements
    std::vector<Measurement> GetMeasurements() const override;

#if defined(CLOCK_MONOTONIC_RAW)
    using clock = monotonic_clock_raw;
#else
    using clock = std::chrono::steady_clock;
#endif

    static const std::string WALL_CLOCK_TIME;
    static const std::string WALL_CLOCK_TIME_START;
    static const std::string WALL_CLOCK_TIME_STOP;

private:
    clock::time_point m_Start;
    clock::time_point m_Stop;
    float m_ScaleFactor = 1.f;
    Measurement::Unit m_Unit = Measurement::Unit::TIME_MS;
};

} //namespace armnn
