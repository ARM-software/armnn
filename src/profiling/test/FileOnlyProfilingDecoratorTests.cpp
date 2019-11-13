//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../FileOnlyProfilingConnection.hpp"

#include <ProfilingService.hpp>
#include <Runtime.hpp>

#include <boost/core/ignore_unused.hpp>
#include <boost/filesystem.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/test/unit_test.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

using namespace armnn::profiling;
using namespace armnn;

using namespace std::chrono_literals;

BOOST_AUTO_TEST_SUITE(FileOnlyProfilingDecoratorTests)

BOOST_AUTO_TEST_CASE(DumpOutgoingValidFileEndToEnd)
{
    // Create a temporary file name.
    boost::filesystem::path tempPath = boost::filesystem::temp_directory_path();
    boost::filesystem::path tempFile = boost::filesystem::unique_path();
    tempPath                         = tempPath / tempFile;
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling     = true;
    options.m_FileOnly            = true;
    options.m_IncomingCaptureFile = "";
    options.m_OutgoingCaptureFile = tempPath.string();
    options.m_CapturePeriod       = 100;

    // Enable the profiling service
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);
    // Bring the profiling service to the "WaitingForAck" state
    profilingService.Update();
    profilingService.Update();

    uint32_t timeout   = 2000;
    uint32_t sleepTime = 50;
    uint32_t timeSlept = 0;

    // Give the profiling service sending thread time start executing and send the stream metadata.
    while (profilingService.GetCurrentState() != ProfilingState::WaitingForAck)
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: Profiling service did not switch to WaitingForAck state");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        timeSlept += sleepTime;
    }

    profilingService.Update();

    timeSlept = 0;

    while (profilingService.GetCurrentState() != profiling::ProfilingState::Active)
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: Profiling service did not switch to Active state");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        timeSlept += sleepTime;
    }

    // Minimum test here is to check that the file was created.
    BOOST_CHECK(boost::filesystem::exists(tempPath.c_str()) == true);

    // Increment a counter.
    BOOST_CHECK(profilingService.IsCounterRegistered(0) == true);
    profilingService.IncrementCounterValue(0);
    BOOST_CHECK(profilingService.GetCounterValue(0) > 0);

    // At this point the profiling service is active and we've activated all the counters. Waiting a collection
    // period should be enough to have some data in the file.

    // Wait for 1 collection period plus a bit of overhead..
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // In order to flush the files we need to gracefully close the profiling service.
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
    // Wait a short time to allow the threads to clean themselves up.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // The output file size should be greater than 0.
    struct stat statusBuffer;
    BOOST_CHECK(stat(tempPath.c_str(), &statusBuffer) == 0);
    BOOST_CHECK(statusBuffer.st_size > 0);

    // Delete the tmp file.
    BOOST_CHECK(remove(tempPath.c_str()) == 0);
}

BOOST_AUTO_TEST_SUITE_END()
