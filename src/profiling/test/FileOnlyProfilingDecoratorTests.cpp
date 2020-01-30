//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../FileOnlyProfilingConnection.hpp"

#include <ProfilingService.hpp>
#include <Runtime.hpp>
#include <Filesystem.hpp>

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

class FileOnlyHelperService : public ProfilingService
{
    public:
    // Wait for a notification from the send thread
    bool WaitForPacketsSent(uint32_t timeout = 1000)
    {
        return ProfilingService::WaitForPacketSent(ProfilingService::Instance(), timeout);
    }
};

BOOST_AUTO_TEST_SUITE(FileOnlyProfilingDecoratorTests)

BOOST_AUTO_TEST_CASE(DumpOutgoingValidFileEndToEnd, * boost::unit_test::disabled())
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

    FileOnlyHelperService helper;

    // Enable the profiling service
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);
    // Bring the profiling service to the "WaitingForAck" state
    profilingService.Update();
    profilingService.Update();


    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);

    profilingService.Update();
    // First packet sent will be the SendStreamMetaDataPacket, it's possible though unlikely that it will be sent twice
    // The second or possibly third packet will be the CounterDirectoryPacket which means the
    // ConnectionAcknowledgedCommandHandler has set the state to active
    uint32_t packetCount = 0;
    while(profilingService.GetCurrentState() != ProfilingState::Active && packetCount < 3)
    {
        if(!helper.WaitForPacketsSent())
        {
            BOOST_FAIL("Timeout waiting for packets");
        }
        packetCount++;
    }

    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);
    // Minimum test here is to check that the file was created.
    BOOST_CHECK(boost::filesystem::exists(tempPath.c_str()) == true);

    // Increment a counter.
    BOOST_CHECK(profilingService.IsCounterRegistered(0) == true);
    profilingService.IncrementCounterValue(0);
    BOOST_CHECK(profilingService.GetCounterValue(0) > 0);

    // At this point the profiling service is active and we've activated all the counters. Waiting a collection
    // period should be enough to have some data in the file.

    // Wait for 1 collection period plus a bit of overhead..
    helper.WaitForPacketsSent();

    // In order to flush the files we need to gracefully close the profiling service.
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    // The output file size should be greater than 0.
    BOOST_CHECK(armnnUtils::Filesystem::GetFileSize(tempPath.string().c_str()) > 0);

    // Delete the tmp file.
    BOOST_CHECK(armnnUtils::Filesystem::Remove(tempPath.string().c_str()));
}

BOOST_AUTO_TEST_SUITE_END()
