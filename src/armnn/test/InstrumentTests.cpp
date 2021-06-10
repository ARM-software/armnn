//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <doctest/doctest.h>

#include "WallClockTimer.hpp"

#include <chrono>
#include <thread>

using namespace armnn;

TEST_SUITE("Instruments")
{
TEST_CASE("WallClockTimerInMicroseconds")
{
    WallClockTimer wallClockTimer;

    CHECK((std::string(wallClockTimer.GetName()) == std::string("WallClockTimer")));

    // start the timer
    wallClockTimer.Start();

    // wait for 10 microseconds
    std::this_thread::sleep_for(std::chrono::microseconds(10));

   // stop the timer
    wallClockTimer.Stop();

    CHECK((wallClockTimer.GetMeasurements().front().m_Name == WallClockTimer::WALL_CLOCK_TIME));

    // check that WallClockTimer measurement should be >= 10 microseconds
    CHECK_GE(wallClockTimer.GetMeasurements().front().m_Value, std::chrono::microseconds(10).count());
}

TEST_CASE("WallClockTimerInNanoseconds")
{
    WallClockTimer wallClockTimer;

    CHECK((std::string(wallClockTimer.GetName()) == std::string("WallClockTimer")));

    // start the timer
    wallClockTimer.Start();

    // wait for 500 nanoseconds - 0.5 microseconds
    std::this_thread::sleep_for(std::chrono::nanoseconds(500));

    // stop the timer
    wallClockTimer.Stop();

    CHECK((wallClockTimer.GetMeasurements().front().m_Name == WallClockTimer::WALL_CLOCK_TIME));

    // delta is 0.5 microseconds
    const auto delta =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(std::chrono::nanoseconds(500));

    // check that WallClockTimer measurement should be >= 0.5 microseconds
    CHECK_GE(wallClockTimer.GetMeasurements().front().m_Value, delta.count());
}

}
