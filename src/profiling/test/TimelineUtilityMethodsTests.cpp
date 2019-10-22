//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SendCounterPacketTests.hpp"

#include <SendTimelinePacket.hpp>
#include <TimelineUtilityMethods.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;
using namespace armnn::profiling;

BOOST_AUTO_TEST_SUITE(TimelineUtilityMethodsTests)

BOOST_AUTO_TEST_CASE(DeclareLabelTest1)
{
    MockBufferManager mockBufferManager(1024);
    SendTimelinePacket sendTimelinePacket(mockBufferManager);
    TimelineUtilityMethods timelineUtilityMethods(sendTimelinePacket);

    // Try declaring an invalid (empty) label
    BOOST_CHECK_THROW(timelineUtilityMethods.DeclareLabel(""), InvalidArgumentException);

    // Try declaring an invalid (wrong SWTrace format) label
    BOOST_CHECK_THROW(timelineUtilityMethods.DeclareLabel("inv@lid lab€l"), RuntimeException);

    // Declare a valid label
    const std::string labelName = "valid label";
    ProfilingGuid labelGuid = 0;
    BOOST_CHECK_NO_THROW(labelGuid = timelineUtilityMethods.DeclareLabel(labelName));
    // TODO when the implementation of the profiling GUID generator is done, enable the following test
    //BOOST_CHECK(labelGuid != ProfilingGuid(0));

    // TODO when the implementation of the profiling GUID generator is done, enable the following tests
    // Try adding the same label as before
    //ProfilingGuid newLabelGuid = 0;
    //BOOST_CHECK_NO_THROW(labelGuid = timelineUtilityMethods.DeclareLabel(labelName));
    //BOOST_CHECK(newLabelGuid != ProfilingGuid(0));
    //BOOST_CHECK(newLabelGuid == labelGuid);
}

BOOST_AUTO_TEST_SUITE_END()
