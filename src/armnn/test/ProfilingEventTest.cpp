//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include "ProfilingEvent.hpp"
#include "Profiling.hpp"
#include <thread>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(ProfilingEvent)

BOOST_AUTO_TEST_CASE(ProfilingEventTest)
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profileManager = armnn::ProfilerManager::GetInstance();

    const char* eventName = "EventName";

    Event::Instruments insts1;
    insts1.emplace_back(std::make_unique<WallClockTimer>());
    Event testEvent(eventName,
                    nullptr,
                    nullptr,
                    armnn::Compute::Undefined,
                    std::move(insts1));

    BOOST_CHECK_EQUAL(testEvent.GetName(), "EventName");

    // start the timer - outer
    testEvent.Start();

    // wait for 10 milliseconds
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // stop the timer - outer
    testEvent.Stop();

    BOOST_CHECK_GE(testEvent.GetMeasurements().front().m_Value, 10.0);

    // create a sub event with CpuAcc
    Event::Instruments insts2;
    insts2.emplace_back(std::make_unique<WallClockTimer>());
    Event testEvent2(eventName,
                     profileManager.GetProfiler(),
                     &testEvent,
                     Compute::CpuAcc,
                     std::move(insts2));

    BOOST_CHECK_EQUAL(&testEvent, testEvent2.GetParentEvent());
    BOOST_CHECK_EQUAL(profileManager.GetProfiler(), testEvent2.GetProfiler());
    BOOST_CHECK_EQUAL(Compute::CpuAcc, testEvent2.GetComputeDevice());
}

BOOST_AUTO_TEST_CASE(ProfilingEventTestOnGpuAcc)
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profileManager = armnn::ProfilerManager::GetInstance();

    const char* eventName = "GPUEvent";

    Event::Instruments insts1;
    insts1.emplace_back(std::make_unique<WallClockTimer>());
    Event testEvent(eventName,
                    nullptr,
                    nullptr,
                    armnn::Compute::Undefined,
                    std::move(insts1));

    BOOST_CHECK_EQUAL(testEvent.GetName(), "GPUEvent");

    // start the timer - outer
    testEvent.Start();

    // wait for 10 milliseconds
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // stop the timer - outer
    testEvent.Stop();

    BOOST_CHECK_GE(testEvent.GetMeasurements().front().m_Value, 10.0);

    // create a sub event
    Event::Instruments insts2;
    insts2.emplace_back(std::make_unique<WallClockTimer>());
    Event testEvent2(eventName, profileManager.GetProfiler(), &testEvent, Compute::GpuAcc, std::move(insts2));

    BOOST_CHECK_EQUAL(&testEvent, testEvent2.GetParentEvent());
    BOOST_CHECK_EQUAL(profileManager.GetProfiler(), testEvent2.GetProfiler());
    BOOST_CHECK_EQUAL(Compute::GpuAcc, testEvent2.GetComputeDevice());
}

BOOST_AUTO_TEST_SUITE_END()
