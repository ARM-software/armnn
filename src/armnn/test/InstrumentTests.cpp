//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include "WallClockTimer.hpp"

#include <chrono>
#include <thread>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(Instruments)

BOOST_AUTO_TEST_CASE(WallClockTimerInMilliseconds)
{
    WallClockTimer wallClockTimer;

    BOOST_CHECK_EQUAL(wallClockTimer.GetName(), "WallClockTimer");

    // start the timer
    wallClockTimer.Start();

    // wait for 10 milliseconds
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

   // stop the timer
    wallClockTimer.Stop();

    BOOST_CHECK_EQUAL(wallClockTimer.GetMeasurements().front().m_Name, WallClockTimer::WALL_CLOCK_TIME);

    // check that WallClockTimer measurement should be >= 10 milliseconds
    BOOST_CHECK_GE(wallClockTimer.GetMeasurements().front().m_Value, std::chrono::milliseconds(10).count());
}

BOOST_AUTO_TEST_CASE(WallClockTimerInNanoseconds)
{
    WallClockTimer wallClockTimer;

    BOOST_CHECK_EQUAL(wallClockTimer.GetName(), "WallClockTimer");

    // start the timer
    wallClockTimer.Start();

    // wait for 500 nanoseconds - 0.0005 milliseconds
    std::this_thread::sleep_for(std::chrono::nanoseconds(500));

    // stop the timer
    wallClockTimer.Stop();

    BOOST_CHECK_EQUAL(wallClockTimer.GetMeasurements().front().m_Name, WallClockTimer::WALL_CLOCK_TIME);

    // delta is 0.0005 milliseconds
    const auto delta =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::nanoseconds(500));

    // check that WallClockTimer measurement should be >= 0.0005 milliseconds
    BOOST_CHECK_GE(wallClockTimer.GetMeasurements().front().m_Value, delta.count());
}

BOOST_AUTO_TEST_SUITE_END()
