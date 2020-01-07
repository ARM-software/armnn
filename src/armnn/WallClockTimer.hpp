//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Instrument.hpp"
#include <chrono>
#include "DllExport.hpp"

namespace armnn
{

#if defined(CLOCK_MONOTONIC_RAW) && defined(__unix__)
#define USE_CLOCK_MONOTONIC_RAW 1
#else
#define USE_CLOCK_MONOTONIC_RAW 0
#endif

#if USE_CLOCK_MONOTONIC_RAW
class MonotonicClockRaw
{
public:
    using duration   = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<MonotonicClockRaw, duration>;

    static std::chrono::time_point<MonotonicClockRaw, std::chrono::nanoseconds> now() noexcept
    {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return time_point(std::chrono::nanoseconds(ts.tv_sec * 1000000000 + ts.tv_nsec));
    }
};
#endif

// Implementation of an instrument to measure elapsed wall-clock time in microseconds.
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

    // Get the recorded measurements
    std::vector<Measurement> GetMeasurements() const override;

#if USE_CLOCK_MONOTONIC_RAW
    using clock = MonotonicClockRaw;
#else
    using clock = std::chrono::steady_clock;
#endif

    ARMNN_DLLEXPORT static const std::string WALL_CLOCK_TIME;
    static const std::string WALL_CLOCK_TIME_START;
    static const std::string WALL_CLOCK_TIME_STOP;

private:
    clock::time_point m_Start;
    clock::time_point m_Stop;
};

} //namespace armnn
