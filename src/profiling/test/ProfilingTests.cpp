//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../CommandHandlerKey.hpp"
#include "../CommandHandlerFunctor.hpp"
#include "../Packet.hpp"

#include <cstdint>
#include <cstring>
#include <boost/test/unit_test.hpp>
#include <map>

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

BOOST_AUTO_TEST_CASE(CheckCommandHandlerFunctor)
{
    // Create Derived Classes
    class TestFunctorA : public CommandHandlerFunctor
    {
    public:
        using CommandHandlerFunctor::CommandHandlerFunctor;

        int GetCount() { return m_Count; }

        void operator()(const Packet& packet) override
        {
            m_Count++;
        }

    private:
        int m_Count = 0;
    };

    class TestFunctorB : public TestFunctorA
    {
        using TestFunctorA::TestFunctorA;
    };

    class TestFunctorC : public TestFunctorA
    {
        using TestFunctorA::TestFunctorA;
    };

    // Hard code the version as it will be the same during a single profiling session
    uint32_t version = 1;

    TestFunctorA testFunctorA(461, version);
    TestFunctorB testFunctorB(963, version);
    TestFunctorC testFunctorC(983, version);

    CommandHandlerKey keyA(testFunctorA.GetPacketId(), testFunctorA.GetVersion());
    CommandHandlerKey keyB(testFunctorB.GetPacketId(), testFunctorB.GetVersion());
    CommandHandlerKey keyC(testFunctorC.GetPacketId(), testFunctorC.GetVersion());

    // Create the unwrapped map to simulate the Command Handler Registry
    std::map<CommandHandlerKey, CommandHandlerFunctor*> registry;

    registry.insert(std::make_pair(keyB, &testFunctorB));
    registry.insert(std::make_pair(keyA, &testFunctorA));
    registry.insert(std::make_pair(keyC, &testFunctorC));

    // Check the order of the map is correct
    auto it = registry.begin();
    BOOST_CHECK(it->first==keyA);
    it++;
    BOOST_CHECK(it->first==keyB);
    it++;
    BOOST_CHECK(it->first==keyC);

    Packet packetA(500000000, 0, nullptr);
    Packet packetB(600000000, 0, nullptr);
    Packet packetC(400000000, 0, nullptr);

    // Check the correct operator of derived class is called
    registry.at(CommandHandlerKey(packetA.GetPacketId(), version))->operator()(packetA);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 0);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.at(CommandHandlerKey(packetB.GetPacketId(), version))->operator()(packetB);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.at(CommandHandlerKey(packetC.GetPacketId(), version))->operator()(packetC);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 1);
}

BOOST_AUTO_TEST_SUITE_END()
