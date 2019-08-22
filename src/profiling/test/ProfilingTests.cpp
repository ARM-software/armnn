//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../CommandHandlerKey.hpp"
#include "../CommandHandlerFunctor.hpp"
#include "../CommandHandlerRegistry.hpp"
#include "../EncodeVersion.hpp"
#include "../Packet.hpp"
#include "../PacketVersionResolver.hpp"

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <random>

BOOST_AUTO_TEST_SUITE(ExternalProfiling)

using namespace armnn::profiling;

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

BOOST_AUTO_TEST_CASE(CheckEncodeVersion)
{
    Version version1(12);

    BOOST_CHECK(version1.GetMajor() == 0);
    BOOST_CHECK(version1.GetMinor() == 0);
    BOOST_CHECK(version1.GetPatch() == 12);

    Version version2(4108);

    BOOST_CHECK(version2.GetMajor() == 0);
    BOOST_CHECK(version2.GetMinor() == 1);
    BOOST_CHECK(version2.GetPatch() == 12);

    Version version3(4198412);

    BOOST_CHECK(version3.GetMajor() == 1);
    BOOST_CHECK(version3.GetMinor() == 1);
    BOOST_CHECK(version3.GetPatch() == 12);

    Version version4(0);

    BOOST_CHECK(version4.GetMajor() == 0);
    BOOST_CHECK(version4.GetMinor() == 0);
    BOOST_CHECK(version4.GetPatch() == 0);

    Version version5(1, 0, 0);
    BOOST_CHECK(version5.GetEncodedValue() == 4194304);
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

BOOST_AUTO_TEST_CASE(CheckCommandHandlerFunctor)
{
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

BOOST_AUTO_TEST_CASE(CheckCommandHandlerRegistry)
{
    // Hard code the version as it will be the same during a single profiling session
    uint32_t version = 1;

    TestFunctorA testFunctorA(461, version);
    TestFunctorB testFunctorB(963, version);
    TestFunctorC testFunctorC(983, version);

    // Create the Command Handler Registry
    CommandHandlerRegistry registry;

    // Register multiple different derived classes
    registry.RegisterFunctor(&testFunctorA, testFunctorA.GetPacketId(), testFunctorA.GetVersion());
    registry.RegisterFunctor(&testFunctorB, testFunctorB.GetPacketId(), testFunctorB.GetVersion());
    registry.RegisterFunctor(&testFunctorC, testFunctorC.GetPacketId(), testFunctorC.GetVersion());

    Packet packetA(500000000, 0, nullptr);
    Packet packetB(600000000, 0, nullptr);
    Packet packetC(400000000, 0, nullptr);

    // Check the correct operator of derived class is called
    registry.GetFunctor(packetA.GetPacketId(), version)->operator()(packetA);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 0);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.GetFunctor(packetB.GetPacketId(), version)->operator()(packetB);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.GetFunctor(packetC.GetPacketId(), version)->operator()(packetC);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 1);

    // Re-register an existing key with a new function
    registry.RegisterFunctor(&testFunctorC, testFunctorA.GetPacketId(), version);
    registry.GetFunctor(packetA.GetPacketId(), version)->operator()(packetC);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 2);

    // Check that non-existent key returns nullptr for its functor
    BOOST_CHECK_THROW(registry.GetFunctor(0, 0), armnn::Exception);
}

BOOST_AUTO_TEST_CASE(CheckPacketVersionResolver)
{
    // Set up random number generator for generating packetId values
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<uint32_t> distribution(std::numeric_limits<uint32_t>::min(),
                                                         std::numeric_limits<uint32_t>::max());

    // NOTE: Expected version is always 1.0.0, regardless of packetId
    const Version expectedVersion(1, 0, 0);

    PacketVersionResolver packetVersionResolver;

    constexpr unsigned int numTests = 10u;

    for (unsigned int i = 0u; i < numTests; ++i)
    {
        const uint32_t packetId = distribution(generator);
        Version resolvedVersion = packetVersionResolver.ResolvePacketVersion(packetId);

        BOOST_TEST(resolvedVersion == expectedVersion);
    }
}

BOOST_AUTO_TEST_SUITE_END()
