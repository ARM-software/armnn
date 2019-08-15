//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../CommandHandlerKey.hpp"
#include "../Packet.hpp"

#include <cstdint>
#include <cstring>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ExternalProfiling)

BOOST_AUTO_TEST_CASE(CheckCommandHandlerKeyComparisons)
{
    CommandHandlerKey testKey0(1, 1);
    CommandHandlerKey testKey1(1, 1);
    CommandHandlerKey testKey2(1, 1);
    CommandHandlerKey testKey3(0, 0);
    CommandHandlerKey testKey4(2, 2);
    CommandHandlerKey testKey5(0, 2);

    BOOST_CHECK(testKey1<testKey4);
    BOOST_CHECK(testKey1>testKey3);
    BOOST_CHECK(testKey1<=testKey4);
    BOOST_CHECK(testKey1>=testKey3);
    BOOST_CHECK(testKey1<=testKey2);
    BOOST_CHECK(testKey1>=testKey2);
    BOOST_CHECK(testKey1==testKey2);
    BOOST_CHECK(testKey1==testKey1);

    BOOST_CHECK(!(testKey1==testKey5));
    BOOST_CHECK(!(testKey1!=testKey1));
    BOOST_CHECK(testKey1!=testKey5);

    BOOST_CHECK(testKey1==testKey2 && testKey2==testKey1);
    BOOST_CHECK(testKey0==testKey1 && testKey1==testKey2 && testKey0==testKey2);

    BOOST_CHECK(testKey1.GetPacketId()==1);
    BOOST_CHECK(testKey1.GetVersion()==1);

    std::vector<CommandHandlerKey> vect =
        {
            CommandHandlerKey(0,1), CommandHandlerKey(2,0), CommandHandlerKey(1,0),
            CommandHandlerKey(2,1), CommandHandlerKey(1,1), CommandHandlerKey(0,1),
            CommandHandlerKey(2,0), CommandHandlerKey(0,0)
        };

    std::sort(vect.begin(), vect.end());

    std::vector<CommandHandlerKey> expectedVect =
        {
            CommandHandlerKey(0,0), CommandHandlerKey(0,1), CommandHandlerKey(0,1),
            CommandHandlerKey(1,0), CommandHandlerKey(1,1), CommandHandlerKey(2,0),
            CommandHandlerKey(2,0), CommandHandlerKey(2,1)
        };

    BOOST_CHECK(vect == expectedVect);
}

BOOST_AUTO_TEST_CASE(CheckPacketClass)
{
    const char* data = "test";
    unsigned int length = static_cast<unsigned int>(std::strlen(data));

    Packet packetTest1(472580096,length,data);
    BOOST_CHECK_THROW(Packet packetTest2(472580096,0,""), armnn::Exception);

    Packet packetTest3(472580096,0, nullptr);

    BOOST_CHECK(packetTest1.GetLength() == length);
    BOOST_CHECK(packetTest1.GetData() == data);

    BOOST_CHECK(packetTest1.GetPacketFamily() == 7);
    BOOST_CHECK(packetTest1.GetPacketId() == 43);
    BOOST_CHECK(packetTest1.GetPacketType() == 3);
    BOOST_CHECK(packetTest1.GetPacketClass() == 5);
}

BOOST_AUTO_TEST_SUITE_END()
