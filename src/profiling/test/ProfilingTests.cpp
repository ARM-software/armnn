//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingTests.hpp"
#include "ProfilingTestUtils.hpp"

#include <backends/BackendProfiling.hpp>
#include <common/include/EncodeVersion.hpp>
#include <common/include/PacketVersionResolver.hpp>
#include <common/include/SwTrace.hpp>
#include <CommandHandler.hpp>
#include <ConnectionAcknowledgedCommandHandler.hpp>
#include <CounterDirectory.hpp>
#include <CounterIdMap.hpp>
#include <Holder.hpp>
#include <ICounterValues.hpp>
#include <PeriodicCounterCapture.hpp>
#include <PeriodicCounterSelectionCommandHandler.hpp>
#include <ProfilingStateMachine.hpp>
#include <ProfilingUtils.hpp>
#include <RegisterBackendCounters.hpp>
#include <RequestCounterDirectoryCommandHandler.hpp>
#include <Runtime.hpp>
#include <SocketProfilingConnection.hpp>
#include <SendCounterPacket.hpp>
#include <SendThread.hpp>
#include <SendTimelinePacket.hpp>

#include <armnn/Conversion.hpp>
#include <armnn/Types.hpp>

#include <armnn/Utils.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <common/include/CommandHandlerKey.hpp>
#include <common/include/CommandHandlerRegistry.hpp>
#include <common/include/SocketConnectionException.hpp>
#include <common/include/Packet.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <random>


using namespace armnn::profiling;
using PacketType = MockProfilingConnection::PacketType;

BOOST_AUTO_TEST_SUITE(ExternalProfiling)

BOOST_AUTO_TEST_CASE(CheckCommandHandlerKeyComparisons)
{
    arm::pipe::CommandHandlerKey testKey1_0(1, 1, 1);
    arm::pipe::CommandHandlerKey testKey1_1(1, 1, 1);
    arm::pipe::CommandHandlerKey testKey1_2(1, 2, 1);

    arm::pipe::CommandHandlerKey testKey0(0, 1, 1);
    arm::pipe::CommandHandlerKey testKey1(0, 1, 1);
    arm::pipe::CommandHandlerKey testKey2(0, 1, 1);
    arm::pipe::CommandHandlerKey testKey3(0, 0, 0);
    arm::pipe::CommandHandlerKey testKey4(0, 2, 2);
    arm::pipe::CommandHandlerKey testKey5(0, 0, 2);

    BOOST_CHECK(testKey1_0 > testKey0);
    BOOST_CHECK(testKey1_0 == testKey1_1);
    BOOST_CHECK(testKey1_0 < testKey1_2);

    BOOST_CHECK(testKey1 < testKey4);
    BOOST_CHECK(testKey1 > testKey3);
    BOOST_CHECK(testKey1 <= testKey4);
    BOOST_CHECK(testKey1 >= testKey3);
    BOOST_CHECK(testKey1 <= testKey2);
    BOOST_CHECK(testKey1 >= testKey2);
    BOOST_CHECK(testKey1 == testKey2);
    BOOST_CHECK(testKey1 == testKey1);

    BOOST_CHECK(!(testKey1 == testKey5));
    BOOST_CHECK(!(testKey1 != testKey1));
    BOOST_CHECK(testKey1 != testKey5);

    BOOST_CHECK(testKey1 == testKey2 && testKey2 == testKey1);
    BOOST_CHECK(testKey0 == testKey1 && testKey1 == testKey2 && testKey0 == testKey2);

    BOOST_CHECK(testKey1.GetPacketId() == 1);
    BOOST_CHECK(testKey1.GetVersion() == 1);

    std::vector<arm::pipe::CommandHandlerKey> vect = {
        arm::pipe::CommandHandlerKey(0, 0, 1), arm::pipe::CommandHandlerKey(0, 2, 0),
        arm::pipe::CommandHandlerKey(0, 1, 0), arm::pipe::CommandHandlerKey(0, 2, 1),
        arm::pipe::CommandHandlerKey(0, 1, 1), arm::pipe::CommandHandlerKey(0, 0, 1),
        arm::pipe::CommandHandlerKey(0, 2, 0), arm::pipe::CommandHandlerKey(0, 0, 0) };

    std::sort(vect.begin(), vect.end());

    std::vector<arm::pipe::CommandHandlerKey> expectedVect = {
        arm::pipe::CommandHandlerKey(0, 0, 0), arm::pipe::CommandHandlerKey(0, 0, 1),
        arm::pipe::CommandHandlerKey(0, 0, 1), arm::pipe::CommandHandlerKey(0, 1, 0),
        arm::pipe::CommandHandlerKey(0, 1, 1), arm::pipe::CommandHandlerKey(0, 2, 0),
        arm::pipe::CommandHandlerKey(0, 2, 0), arm::pipe::CommandHandlerKey(0, 2, 1) };

    BOOST_CHECK(vect == expectedVect);
}

BOOST_AUTO_TEST_CASE(CheckPacketKeyComparisons)
{
    arm::pipe::PacketKey key0(0, 0);
    arm::pipe::PacketKey key1(0, 0);
    arm::pipe::PacketKey key2(0, 1);
    arm::pipe::PacketKey key3(0, 2);
    arm::pipe::PacketKey key4(1, 0);
    arm::pipe::PacketKey key5(1, 0);
    arm::pipe::PacketKey key6(1, 1);

    BOOST_CHECK(!(key0 < key1));
    BOOST_CHECK(!(key0 > key1));
    BOOST_CHECK(key0 <= key1);
    BOOST_CHECK(key0 >= key1);
    BOOST_CHECK(key0 == key1);
    BOOST_CHECK(key0 < key2);
    BOOST_CHECK(key2 < key3);
    BOOST_CHECK(key3 > key0);
    BOOST_CHECK(key4 == key5);
    BOOST_CHECK(key4 > key0);
    BOOST_CHECK(key5 < key6);
    BOOST_CHECK(key5 <= key6);
    BOOST_CHECK(key5 != key6);
}

BOOST_AUTO_TEST_CASE(CheckCommandHandler)
{
    arm::pipe::PacketVersionResolver packetVersionResolver;
    ProfilingStateMachine profilingStateMachine;

    TestProfilingConnectionBase testProfilingConnectionBase;
    TestProfilingConnectionTimeoutError testProfilingConnectionTimeOutError;
    TestProfilingConnectionArmnnError testProfilingConnectionArmnnError;
    CounterDirectory counterDirectory;
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer);
    SendThread sendThread(profilingStateMachine, mockBuffer, sendCounterPacket);
    SendTimelinePacket sendTimelinePacket(mockBuffer);
    MockProfilingServiceStatus mockProfilingServiceStatus;

    ConnectionAcknowledgedCommandHandler connectionAcknowledgedCommandHandler(0, 1, 4194304, counterDirectory,
                                                                              sendCounterPacket, sendTimelinePacket,
                                                                              profilingStateMachine,
                                                                              mockProfilingServiceStatus);
    arm::pipe::CommandHandlerRegistry commandHandlerRegistry;

    commandHandlerRegistry.RegisterFunctor(&connectionAcknowledgedCommandHandler);

    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);

    CommandHandler commandHandler0(1, true, commandHandlerRegistry, packetVersionResolver);

    // This should start the command handler thread return the connection ack and put the profiling
    // service into active state.
    commandHandler0.Start(testProfilingConnectionBase);
    // Try to start the send thread many times, it must only start once
    commandHandler0.Start(testProfilingConnectionBase);

    // This could take up to 20mSec but we'll check often.
    for (int i = 0; i < 10; i++)
    {
        if (profilingStateMachine.GetCurrentState() == ProfilingState::Active)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    BOOST_CHECK(profilingStateMachine.GetCurrentState() == ProfilingState::Active);

    // Close the thread again.
    commandHandler0.Stop();

    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);

    // In this test we'll simulate a timeout without a connection ack packet being received.
    // Stop after timeout is set so we expect the command handler to stop almost immediately.
    CommandHandler commandHandler1(1, true, commandHandlerRegistry, packetVersionResolver);

    commandHandler1.Start(testProfilingConnectionTimeOutError);
    // Wait until we know a timeout exception has been sent at least once.
    for (int i = 0; i < 10; i++)
    {
        if (testProfilingConnectionTimeOutError.ReadCalledCount())
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // The command handler loop should have stopped after the timeout.
    // wait for the timeout exception to be processed and the loop to break.
    uint32_t timeout   = 50;
    uint32_t timeSlept = 0;
    while (commandHandler1.IsRunning())
    {
        if (timeSlept >= timeout)
        {
            BOOST_FAIL("Timeout: The command handler loop did not stop after the timeout");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        timeSlept ++;
    }

    commandHandler1.Stop();
    // The state machine should never have received the ack so will still be in WaitingForAck.
    BOOST_CHECK(profilingStateMachine.GetCurrentState() == ProfilingState::WaitingForAck);

    // Now try sending a bad connection acknowledged packet
    TestProfilingConnectionBadAckPacket testProfilingConnectionBadAckPacket;
    commandHandler1.Start(testProfilingConnectionBadAckPacket);
    commandHandler1.Stop();
    // This should also not change the state machine
    BOOST_CHECK(profilingStateMachine.GetCurrentState() == ProfilingState::WaitingForAck);

    // Disable stop after timeout and now commandHandler1 should persist after a timeout
    commandHandler1.SetStopAfterTimeout(false);
    // Restart the thread.
    commandHandler1.Start(testProfilingConnectionTimeOutError);

    // Wait for at the three timeouts and the ack to be sent.
    for (int i = 0; i < 10; i++)
    {
        if (testProfilingConnectionTimeOutError.ReadCalledCount() > 3)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    commandHandler1.Stop();

    // Even after the 3 exceptions the ack packet should have transitioned the command handler to active.
    BOOST_CHECK(profilingStateMachine.GetCurrentState() == ProfilingState::Active);

    // A command handler that gets exceptions other than timeouts should keep going.
    CommandHandler commandHandler2(1, false, commandHandlerRegistry, packetVersionResolver);

    commandHandler2.Start(testProfilingConnectionArmnnError);

    // Wait for two exceptions to be thrown.
    for (int i = 0; i < 10; i++)
    {
        if (testProfilingConnectionTimeOutError.ReadCalledCount() >= 2)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    BOOST_CHECK(commandHandler2.IsRunning());
    commandHandler2.Stop();
}

BOOST_AUTO_TEST_CASE(CheckEncodeVersion)
{
    arm::pipe::Version version1(12);

    BOOST_CHECK(version1.GetMajor() == 0);
    BOOST_CHECK(version1.GetMinor() == 0);
    BOOST_CHECK(version1.GetPatch() == 12);

    arm::pipe::Version version2(4108);

    BOOST_CHECK(version2.GetMajor() == 0);
    BOOST_CHECK(version2.GetMinor() == 1);
    BOOST_CHECK(version2.GetPatch() == 12);

    arm::pipe::Version version3(4198412);

    BOOST_CHECK(version3.GetMajor() == 1);
    BOOST_CHECK(version3.GetMinor() == 1);
    BOOST_CHECK(version3.GetPatch() == 12);

    arm::pipe::Version version4(0);

    BOOST_CHECK(version4.GetMajor() == 0);
    BOOST_CHECK(version4.GetMinor() == 0);
    BOOST_CHECK(version4.GetPatch() == 0);

    arm::pipe::Version version5(1, 0, 0);
    BOOST_CHECK(version5.GetEncodedValue() == 4194304);
}

BOOST_AUTO_TEST_CASE(CheckPacketClass)
{
    uint32_t length                              = 4;
    std::unique_ptr<unsigned char[]> packetData0 = std::make_unique<unsigned char[]>(length);
    std::unique_ptr<unsigned char[]> packetData1 = std::make_unique<unsigned char[]>(0);
    std::unique_ptr<unsigned char[]> nullPacketData;

    arm::pipe::Packet packetTest0(472580096, length, packetData0);

    BOOST_CHECK(packetTest0.GetHeader() == 472580096);
    BOOST_CHECK(packetTest0.GetPacketFamily() == 7);
    BOOST_CHECK(packetTest0.GetPacketId() == 43);
    BOOST_CHECK(packetTest0.GetLength() == length);
    BOOST_CHECK(packetTest0.GetPacketType() == 3);
    BOOST_CHECK(packetTest0.GetPacketClass() == 5);

    BOOST_CHECK_THROW(arm::pipe::Packet packetTest1(472580096, 0, packetData1), arm::pipe::InvalidArgumentException);
    BOOST_CHECK_NO_THROW(arm::pipe::Packet packetTest2(472580096, 0, nullPacketData));

    arm::pipe::Packet packetTest3(472580096, 0, nullPacketData);
    BOOST_CHECK(packetTest3.GetLength() == 0);
    BOOST_CHECK(packetTest3.GetData() == nullptr);

    const unsigned char* packetTest0Data = packetTest0.GetData();
    arm::pipe::Packet packetTest4(std::move(packetTest0));

    BOOST_CHECK(packetTest0.GetData() == nullptr);
    BOOST_CHECK(packetTest4.GetData() == packetTest0Data);

    BOOST_CHECK(packetTest4.GetHeader() == 472580096);
    BOOST_CHECK(packetTest4.GetPacketFamily() == 7);
    BOOST_CHECK(packetTest4.GetPacketId() == 43);
    BOOST_CHECK(packetTest4.GetLength() == length);
    BOOST_CHECK(packetTest4.GetPacketType() == 3);
    BOOST_CHECK(packetTest4.GetPacketClass() == 5);
}

BOOST_AUTO_TEST_CASE(CheckCommandHandlerFunctor)
{
    // Hard code the version as it will be the same during a single profiling session
    uint32_t version = 1;

    TestFunctorA testFunctorA(7, 461, version);
    TestFunctorB testFunctorB(8, 963, version);
    TestFunctorC testFunctorC(5, 983, version);

    arm::pipe::CommandHandlerKey keyA(
        testFunctorA.GetFamilyId(), testFunctorA.GetPacketId(), testFunctorA.GetVersion());
    arm::pipe::CommandHandlerKey keyB(
        testFunctorB.GetFamilyId(), testFunctorB.GetPacketId(), testFunctorB.GetVersion());
    arm::pipe::CommandHandlerKey keyC(
        testFunctorC.GetFamilyId(), testFunctorC.GetPacketId(), testFunctorC.GetVersion());

    // Create the unwrapped map to simulate the Command Handler Registry
    std::map<arm::pipe::CommandHandlerKey, arm::pipe::CommandHandlerFunctor*> registry;

    registry.insert(std::make_pair(keyB, &testFunctorB));
    registry.insert(std::make_pair(keyA, &testFunctorA));
    registry.insert(std::make_pair(keyC, &testFunctorC));

    // Check the order of the map is correct
    auto it = registry.begin();
    BOOST_CHECK(it->first == keyC);    // familyId == 5
    it++;
    BOOST_CHECK(it->first == keyA);    // familyId == 7
    it++;
    BOOST_CHECK(it->first == keyB);    // familyId == 8

    std::unique_ptr<unsigned char[]> packetDataA;
    std::unique_ptr<unsigned char[]> packetDataB;
    std::unique_ptr<unsigned char[]> packetDataC;

    arm::pipe::Packet packetA(500000000, 0, packetDataA);
    arm::pipe::Packet packetB(600000000, 0, packetDataB);
    arm::pipe::Packet packetC(400000000, 0, packetDataC);

    // Check the correct operator of derived class is called
    registry.at(arm::pipe::CommandHandlerKey(
        packetA.GetPacketFamily(), packetA.GetPacketId(), version))->operator()(packetA);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 0);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.at(arm::pipe::CommandHandlerKey(
        packetB.GetPacketFamily(), packetB.GetPacketId(), version))->operator()(packetB);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.at(arm::pipe::CommandHandlerKey(
        packetC.GetPacketFamily(), packetC.GetPacketId(), version))->operator()(packetC);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 1);
}

BOOST_AUTO_TEST_CASE(CheckCommandHandlerRegistry)
{
    // Hard code the version as it will be the same during a single profiling session
    uint32_t version = 1;

    TestFunctorA testFunctorA(7, 461, version);
    TestFunctorB testFunctorB(8, 963, version);
    TestFunctorC testFunctorC(5, 983, version);

    // Create the Command Handler Registry
    arm::pipe::CommandHandlerRegistry registry;

    // Register multiple different derived classes
    registry.RegisterFunctor(&testFunctorA);
    registry.RegisterFunctor(&testFunctorB);
    registry.RegisterFunctor(&testFunctorC);

    std::unique_ptr<unsigned char[]> packetDataA;
    std::unique_ptr<unsigned char[]> packetDataB;
    std::unique_ptr<unsigned char[]> packetDataC;

    arm::pipe::Packet packetA(500000000, 0, packetDataA);
    arm::pipe::Packet packetB(600000000, 0, packetDataB);
    arm::pipe::Packet packetC(400000000, 0, packetDataC);

    // Check the correct operator of derived class is called
    registry.GetFunctor(packetA.GetPacketFamily(), packetA.GetPacketId(), version)->operator()(packetA);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 0);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.GetFunctor(packetB.GetPacketFamily(), packetB.GetPacketId(), version)->operator()(packetB);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 0);

    registry.GetFunctor(packetC.GetPacketFamily(), packetC.GetPacketId(), version)->operator()(packetC);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 1);

    // Re-register an existing key with a new function
    registry.RegisterFunctor(&testFunctorC, testFunctorA.GetFamilyId(), testFunctorA.GetPacketId(), version);
    registry.GetFunctor(packetA.GetPacketFamily(), packetA.GetPacketId(), version)->operator()(packetC);
    BOOST_CHECK(testFunctorA.GetCount() == 1);
    BOOST_CHECK(testFunctorB.GetCount() == 1);
    BOOST_CHECK(testFunctorC.GetCount() == 2);

    // Check that non-existent key returns nullptr for its functor
    BOOST_CHECK_THROW(registry.GetFunctor(0, 0, 0), arm::pipe::ProfilingException);
}

BOOST_AUTO_TEST_CASE(CheckPacketVersionResolver)
{
    // Set up random number generator for generating packetId values
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<uint32_t> distribution(std::numeric_limits<uint32_t>::min(),
                                                         std::numeric_limits<uint32_t>::max());

    // NOTE: Expected version is always 1.0.0, regardless of packetId
    const arm::pipe::Version expectedVersion(1, 0, 0);

    arm::pipe::PacketVersionResolver packetVersionResolver;

    constexpr unsigned int numTests = 10u;

    for (unsigned int i = 0u; i < numTests; ++i)
    {
        const uint32_t familyId = distribution(generator);
        const uint32_t packetId = distribution(generator);
        arm::pipe::Version resolvedVersion = packetVersionResolver.ResolvePacketVersion(familyId, packetId);

        BOOST_TEST(resolvedVersion == expectedVersion);
    }
}

void ProfilingCurrentStateThreadImpl(ProfilingStateMachine& states)
{
    ProfilingState newState = ProfilingState::NotConnected;
    states.GetCurrentState();
    states.TransitionToState(newState);
}

BOOST_AUTO_TEST_CASE(CheckProfilingStateMachine)
{
    ProfilingStateMachine profilingState1(ProfilingState::Uninitialised);
    profilingState1.TransitionToState(ProfilingState::Uninitialised);
    BOOST_CHECK(profilingState1.GetCurrentState() == ProfilingState::Uninitialised);

    ProfilingStateMachine profilingState2(ProfilingState::Uninitialised);
    profilingState2.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK(profilingState2.GetCurrentState() == ProfilingState::NotConnected);

    ProfilingStateMachine profilingState3(ProfilingState::NotConnected);
    profilingState3.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK(profilingState3.GetCurrentState() == ProfilingState::NotConnected);

    ProfilingStateMachine profilingState4(ProfilingState::NotConnected);
    profilingState4.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK(profilingState4.GetCurrentState() == ProfilingState::WaitingForAck);

    ProfilingStateMachine profilingState5(ProfilingState::WaitingForAck);
    profilingState5.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK(profilingState5.GetCurrentState() == ProfilingState::WaitingForAck);

    ProfilingStateMachine profilingState6(ProfilingState::WaitingForAck);
    profilingState6.TransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingState6.GetCurrentState() == ProfilingState::Active);

    ProfilingStateMachine profilingState7(ProfilingState::Active);
    profilingState7.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK(profilingState7.GetCurrentState() == ProfilingState::NotConnected);

    ProfilingStateMachine profilingState8(ProfilingState::Active);
    profilingState8.TransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingState8.GetCurrentState() == ProfilingState::Active);

    ProfilingStateMachine profilingState9(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(profilingState9.TransitionToState(ProfilingState::WaitingForAck), armnn::Exception);

    ProfilingStateMachine profilingState10(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(profilingState10.TransitionToState(ProfilingState::Active), armnn::Exception);

    ProfilingStateMachine profilingState11(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(profilingState11.TransitionToState(ProfilingState::Uninitialised), armnn::Exception);

    ProfilingStateMachine profilingState12(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(profilingState12.TransitionToState(ProfilingState::Active), armnn::Exception);

    ProfilingStateMachine profilingState13(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(profilingState13.TransitionToState(ProfilingState::Uninitialised), armnn::Exception);

    ProfilingStateMachine profilingState14(ProfilingState::WaitingForAck);
    profilingState14.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK(profilingState14.GetCurrentState() == ProfilingState::NotConnected);

    ProfilingStateMachine profilingState15(ProfilingState::Active);
    BOOST_CHECK_THROW(profilingState15.TransitionToState(ProfilingState::Uninitialised), armnn::Exception);

    ProfilingStateMachine profilingState16(armnn::profiling::ProfilingState::Active);
    BOOST_CHECK_THROW(profilingState16.TransitionToState(ProfilingState::WaitingForAck), armnn::Exception);

    ProfilingStateMachine profilingState17(ProfilingState::Uninitialised);

    std::thread thread1(ProfilingCurrentStateThreadImpl, std::ref(profilingState17));
    std::thread thread2(ProfilingCurrentStateThreadImpl, std::ref(profilingState17));
    std::thread thread3(ProfilingCurrentStateThreadImpl, std::ref(profilingState17));
    std::thread thread4(ProfilingCurrentStateThreadImpl, std::ref(profilingState17));
    std::thread thread5(ProfilingCurrentStateThreadImpl, std::ref(profilingState17));

    thread1.join();
    thread2.join();
    thread3.join();
    thread4.join();
    thread5.join();

    BOOST_TEST((profilingState17.GetCurrentState() == ProfilingState::NotConnected));
}

void CaptureDataWriteThreadImpl(Holder& holder, uint32_t capturePeriod, const std::vector<uint16_t>& counterIds)
{
    holder.SetCaptureData(capturePeriod, counterIds, {});
}

void CaptureDataReadThreadImpl(const Holder& holder, CaptureData& captureData)
{
    captureData = holder.GetCaptureData();
}

BOOST_AUTO_TEST_CASE(CheckCaptureDataHolder)
{
    std::map<uint32_t, std::vector<uint16_t>> periodIdMap;
    std::vector<uint16_t> counterIds;
    uint32_t numThreads = 10;
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        counterIds.emplace_back(i);
        periodIdMap.insert(std::make_pair(i, counterIds));
    }

    // Verify the read and write threads set the holder correctly
    // and retrieve the expected values
    Holder holder;
    BOOST_CHECK((holder.GetCaptureData()).GetCapturePeriod() == 0);
    BOOST_CHECK(((holder.GetCaptureData()).GetCounterIds()).empty());

    // Check Holder functions
    std::thread thread1(CaptureDataWriteThreadImpl, std::ref(holder), 2, std::ref(periodIdMap[2]));
    thread1.join();
    BOOST_CHECK((holder.GetCaptureData()).GetCapturePeriod() == 2);
    BOOST_CHECK((holder.GetCaptureData()).GetCounterIds() == periodIdMap[2]);
    // NOTE: now that we have some initial values in the holder we don't have to worry
    //       in the multi-threaded section below about a read thread accessing the holder
    //       before any write thread has gotten to it so we read period = 0, counterIds empty
    //       instead of period = 0, counterIds = {0} as will the case when write thread 0
    //       has executed.

    CaptureData captureData;
    std::thread thread2(CaptureDataReadThreadImpl, std::ref(holder), std::ref(captureData));
    thread2.join();
    BOOST_CHECK(captureData.GetCapturePeriod() == 2);
    BOOST_CHECK(captureData.GetCounterIds() == periodIdMap[2]);

    std::map<uint32_t, CaptureData> captureDataIdMap;
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        CaptureData perThreadCaptureData;
        captureDataIdMap.insert(std::make_pair(i, perThreadCaptureData));
    }

    std::vector<std::thread> threadsVect;
    std::vector<std::thread> readThreadsVect;
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        threadsVect.emplace_back(
            std::thread(CaptureDataWriteThreadImpl, std::ref(holder), i, std::ref(periodIdMap[i])));

        // Verify that the CaptureData goes into the thread in a virgin state
        BOOST_CHECK(captureDataIdMap.at(i).GetCapturePeriod() == 0);
        BOOST_CHECK(captureDataIdMap.at(i).GetCounterIds().empty());
        readThreadsVect.emplace_back(
            std::thread(CaptureDataReadThreadImpl, std::ref(holder), std::ref(captureDataIdMap.at(i))));
    }

    for (uint32_t i = 0; i < numThreads; ++i)
    {
        threadsVect[i].join();
        readThreadsVect[i].join();
    }

    // Look at the CaptureData that each read thread has filled
    // the capture period it read should match the counter ids entry
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        CaptureData perThreadCaptureData = captureDataIdMap.at(i);
        BOOST_CHECK(perThreadCaptureData.GetCounterIds() == periodIdMap.at(perThreadCaptureData.GetCapturePeriod()));
    }
}

BOOST_AUTO_TEST_CASE(CaptureDataMethods)
{
    // Check CaptureData setter and getter functions
    std::vector<uint16_t> counterIds = { 42, 29, 13 };
    CaptureData captureData;
    BOOST_CHECK(captureData.GetCapturePeriod() == 0);
    BOOST_CHECK((captureData.GetCounterIds()).empty());
    captureData.SetCapturePeriod(150);
    captureData.SetCounterIds(counterIds);
    BOOST_CHECK(captureData.GetCapturePeriod() == 150);
    BOOST_CHECK(captureData.GetCounterIds() == counterIds);

    // Check assignment operator
    CaptureData secondCaptureData;

    secondCaptureData = captureData;
    BOOST_CHECK(secondCaptureData.GetCapturePeriod() == 150);
    BOOST_CHECK(secondCaptureData.GetCounterIds() == counterIds);

    // Check copy constructor
    CaptureData copyConstructedCaptureData(captureData);

    BOOST_CHECK(copyConstructedCaptureData.GetCapturePeriod() == 150);
    BOOST_CHECK(copyConstructedCaptureData.GetCounterIds() == counterIds);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceDisabled)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceCounterDirectory)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    const ICounterDirectory& counterDirectory0 = profilingService.GetCounterDirectory();
    BOOST_CHECK(counterDirectory0.GetCounterCount() == 0);
    profilingService.Update();
    BOOST_CHECK(counterDirectory0.GetCounterCount() == 0);

    options.m_EnableProfiling = true;
    profilingService.ResetExternalProfilingOptions(options);

    const ICounterDirectory& counterDirectory1 = profilingService.GetCounterDirectory();
    BOOST_CHECK(counterDirectory1.GetCounterCount() == 0);
    profilingService.Update();
    BOOST_CHECK(counterDirectory1.GetCounterCount() != 0);
    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceCounterValues)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    profilingService.Update();
    const ICounterDirectory& counterDirectory = profilingService.GetCounterDirectory();
    const Counters& counters                  = counterDirectory.GetCounters();
    BOOST_CHECK(!counters.empty());

    std::vector<std::thread> writers;

    BOOST_CHECK(!counters.empty());

    // Test GetAbsoluteCounterValue
    for (int i = 0; i < 4; ++i)
    {
        // Increment and decrement the INFERENCES_RUN counter 250 times
        writers.push_back(std::thread([&profilingService]()
                                      {
                                          for (int i = 0; i < 250; ++i)
                                          {
                                              profilingService.IncrementCounterValue(INFERENCES_RUN);
                                          }
                                      }));
        // Add 10 to the INFERENCES_RUN counter 200 times
        writers.push_back(std::thread([&profilingService]()
                                      {
                                          for (int i = 0; i < 200; ++i)
                                          {
                                              profilingService.AddCounterValue(INFERENCES_RUN, 10);
                                          }
                                      }));
        // Subtract 5 from the INFERENCES_RUN counter 200 times
        writers.push_back(std::thread([&profilingService]()
                                      {
                                          for (int i = 0; i < 200; ++i)
                                          {
                                              profilingService.SubtractCounterValue(INFERENCES_RUN, 5);
                                          }
                                      }));
    }

    std::for_each(writers.begin(), writers.end(), mem_fn(&std::thread::join));

    uint32_t absoluteCounterValue = 0;

    BOOST_CHECK_NO_THROW(absoluteCounterValue = profilingService.GetAbsoluteCounterValue(INFERENCES_RUN));
    BOOST_CHECK(absoluteCounterValue = 5000);

    // Test SetCounterValue
    BOOST_CHECK_NO_THROW(profilingService.SetCounterValue(INFERENCES_RUN, 0));
    BOOST_CHECK_NO_THROW(absoluteCounterValue = profilingService.GetAbsoluteCounterValue(INFERENCES_RUN));
    BOOST_CHECK(absoluteCounterValue == 0);

    // Test GetDeltaCounterValue
    writers.clear();
    uint32_t deltaCounterValue = 0;
    //Start a reading thread to randomly read the INFERENCES_RUN counter value
    std::thread reader([&profilingService](uint32_t& deltaCounterValue)
                       {
                           for (int i = 0; i < 300; ++i)
                           {
                               deltaCounterValue += profilingService.GetDeltaCounterValue(INFERENCES_RUN);
                           }
                       }, std::ref(deltaCounterValue));

    for (int i = 0; i < 4; ++i)
    {
        // Increment and decrement the INFERENCES_RUN counter 250 times
        writers.push_back(std::thread([&profilingService]()
                                      {
                                          for (int i = 0; i < 250; ++i)
                                          {
                                              profilingService.IncrementCounterValue(INFERENCES_RUN);
                                          }
                                      }));
        // Add 10 to the INFERENCES_RUN counter 200 times
        writers.push_back(std::thread([&profilingService]()
                                      {
                                          for (int i = 0; i < 200; ++i)
                                          {
                                              profilingService.AddCounterValue(INFERENCES_RUN, 10);
                                          }
                                      }));
        // Subtract 5 from the INFERENCES_RUN counter 200 times
        writers.push_back(std::thread([&profilingService]()
                                      {
                                          for (int i = 0; i < 200; ++i)
                                          {
                                              profilingService.SubtractCounterValue(INFERENCES_RUN, 5);
                                          }
                                      }));
    }

    std::for_each(writers.begin(), writers.end(), mem_fn(&std::thread::join));
    reader.join();

    // Do one last read in case the reader stopped early
    deltaCounterValue += profilingService.GetDeltaCounterValue(INFERENCES_RUN);
    BOOST_CHECK(deltaCounterValue == 5000);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingObjectUids)
{
    uint16_t uid = 0;
    BOOST_CHECK_NO_THROW(uid = GetNextUid());
    BOOST_CHECK(uid >= 1);

    uint16_t nextUid = 0;
    BOOST_CHECK_NO_THROW(nextUid = GetNextUid());
    BOOST_CHECK(nextUid > uid);

    std::vector<uint16_t> counterUids;
    BOOST_CHECK_NO_THROW(counterUids = GetNextCounterUids(uid,0));
    BOOST_CHECK(counterUids.size() == 1);

    std::vector<uint16_t> nextCounterUids;
    BOOST_CHECK_NO_THROW(nextCounterUids = GetNextCounterUids(nextUid, 2));
    BOOST_CHECK(nextCounterUids.size() == 2);
    BOOST_CHECK(nextCounterUids[0] > counterUids[0]);

    std::vector<uint16_t> counterUidsMultiCore;
    uint16_t thirdUid = 4;
    uint16_t numberOfCores = 13;
    BOOST_CHECK_NO_THROW(counterUidsMultiCore = GetNextCounterUids(thirdUid, numberOfCores));
    BOOST_CHECK(counterUidsMultiCore.size() == numberOfCores);
    BOOST_CHECK(counterUidsMultiCore.front() >= nextCounterUids[0]);
    for (size_t i = 1; i < numberOfCores; i++)
    {
        BOOST_CHECK(counterUidsMultiCore[i] == counterUidsMultiCore[i - 1] + 1);
    }
    BOOST_CHECK(counterUidsMultiCore.back() == counterUidsMultiCore.front() + numberOfCores - 1);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterCategory)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);

    // Register a category with an invalid name
    const Category* noCategory = nullptr;
    BOOST_CHECK_THROW(noCategory = counterDirectory.RegisterCategory(""), armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 0);
    BOOST_CHECK(!noCategory);

    // Register a category with an invalid name
    BOOST_CHECK_THROW(noCategory = counterDirectory.RegisterCategory("invalid category"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 0);
    BOOST_CHECK(!noCategory);

    // Register a new category
    const std::string categoryName = "some_category";
    const Category* category       = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());

    // Get the registered category
    const Category* registeredCategory = counterDirectory.GetCategory(categoryName);
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(registeredCategory);
    BOOST_CHECK(registeredCategory == category);

    // Try to get a category not registered
    const Category* notRegisteredCategory = counterDirectory.GetCategory("not_registered_category");
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(!notRegisteredCategory);

    // Register a category already registered
    const Category* anotherCategory = nullptr;
    BOOST_CHECK_THROW(anotherCategory = counterDirectory.RegisterCategory(categoryName),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(!anotherCategory);

    // Register a device for testing
    const std::string deviceName = "some_device";
    const Device* device         = nullptr;
    BOOST_CHECK_NO_THROW(device = counterDirectory.RegisterDevice(deviceName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 1);
    BOOST_CHECK(device);
    BOOST_CHECK(device->m_Uid >= 1);
    BOOST_CHECK(device->m_Name == deviceName);
    BOOST_CHECK(device->m_Cores == 0);

    // Register a new category not associated to any device
    const std::string categoryWoDeviceName = "some_category_without_device";
    const Category* categoryWoDevice       = nullptr;
    BOOST_CHECK_NO_THROW(categoryWoDevice = counterDirectory.RegisterCategory(categoryWoDeviceName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 2);
    BOOST_CHECK(categoryWoDevice);
    BOOST_CHECK(categoryWoDevice->m_Name == categoryWoDeviceName);
    BOOST_CHECK(categoryWoDevice->m_Counters.empty());

    // Register a new category associated to an invalid device name (already exist)
    const Category* categoryInvalidDeviceName = nullptr;
    BOOST_CHECK_THROW(categoryInvalidDeviceName =
                          counterDirectory.RegisterCategory(categoryWoDeviceName),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 2);
    BOOST_CHECK(!categoryInvalidDeviceName);

    // Register a new category associated to a valid device
    const std::string categoryWValidDeviceName = "some_category_with_valid_device";
    const Category* categoryWValidDevice       = nullptr;
    BOOST_CHECK_NO_THROW(categoryWValidDevice =
                             counterDirectory.RegisterCategory(categoryWValidDeviceName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 3);
    BOOST_CHECK(categoryWValidDevice);
    BOOST_CHECK(categoryWValidDevice != category);
    BOOST_CHECK(categoryWValidDevice->m_Name == categoryWValidDeviceName);

    // Register a counter set for testing
    const std::string counterSetName = "some_counter_set";
    const CounterSet* counterSet     = nullptr;
    BOOST_CHECK_NO_THROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 1);
    BOOST_CHECK(counterSet);
    BOOST_CHECK(counterSet->m_Uid >= 1);
    BOOST_CHECK(counterSet->m_Name == counterSetName);
    BOOST_CHECK(counterSet->m_Count == 0);

    // Register a new category not associated to any counter set
    const std::string categoryWoCounterSetName = "some_category_without_counter_set";
    const Category* categoryWoCounterSet       = nullptr;
    BOOST_CHECK_NO_THROW(categoryWoCounterSet =
                             counterDirectory.RegisterCategory(categoryWoCounterSetName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 4);
    BOOST_CHECK(categoryWoCounterSet);
    BOOST_CHECK(categoryWoCounterSet->m_Name == categoryWoCounterSetName);

    // Register a new category associated to a valid counter set
    const std::string categoryWValidCounterSetName = "some_category_with_valid_counter_set";
    const Category* categoryWValidCounterSet       = nullptr;
    BOOST_CHECK_NO_THROW(categoryWValidCounterSet = counterDirectory.RegisterCategory(categoryWValidCounterSetName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 5);
    BOOST_CHECK(categoryWValidCounterSet);
    BOOST_CHECK(categoryWValidCounterSet != category);
    BOOST_CHECK(categoryWValidCounterSet->m_Name == categoryWValidCounterSetName);

    // Register a new category associated to a valid device and counter set
    const std::string categoryWValidDeviceAndValidCounterSetName = "some_category_with_valid_device_and_counter_set";
    const Category* categoryWValidDeviceAndValidCounterSet       = nullptr;
    BOOST_CHECK_NO_THROW(categoryWValidDeviceAndValidCounterSet = counterDirectory.RegisterCategory(
                             categoryWValidDeviceAndValidCounterSetName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 6);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet != category);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet->m_Name == categoryWValidDeviceAndValidCounterSetName);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterDevice)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);

    // Register a device with an invalid name
    const Device* noDevice = nullptr;
    BOOST_CHECK_THROW(noDevice = counterDirectory.RegisterDevice(""), armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 0);
    BOOST_CHECK(!noDevice);

    // Register a device with an invalid name
    BOOST_CHECK_THROW(noDevice = counterDirectory.RegisterDevice("inv@lid nam€"), armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 0);
    BOOST_CHECK(!noDevice);

    // Register a new device with no cores or parent category
    const std::string deviceName = "some_device";
    const Device* device         = nullptr;
    BOOST_CHECK_NO_THROW(device = counterDirectory.RegisterDevice(deviceName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 1);
    BOOST_CHECK(device);
    BOOST_CHECK(device->m_Name == deviceName);
    BOOST_CHECK(device->m_Uid >= 1);
    BOOST_CHECK(device->m_Cores == 0);

    // Try getting an unregistered device
    const Device* unregisteredDevice = counterDirectory.GetDevice(9999);
    BOOST_CHECK(!unregisteredDevice);

    // Get the registered device
    const Device* registeredDevice = counterDirectory.GetDevice(device->m_Uid);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 1);
    BOOST_CHECK(registeredDevice);
    BOOST_CHECK(registeredDevice == device);

    // Register a device with the name of a device already registered
    const Device* deviceSameName = nullptr;
    BOOST_CHECK_THROW(deviceSameName = counterDirectory.RegisterDevice(deviceName), armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 1);
    BOOST_CHECK(!deviceSameName);

    // Register a new device with cores and no parent category
    const std::string deviceWCoresName = "some_device_with_cores";
    const Device* deviceWCores         = nullptr;
    BOOST_CHECK_NO_THROW(deviceWCores = counterDirectory.RegisterDevice(deviceWCoresName, 2));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(deviceWCores);
    BOOST_CHECK(deviceWCores->m_Name == deviceWCoresName);
    BOOST_CHECK(deviceWCores->m_Uid >= 1);
    BOOST_CHECK(deviceWCores->m_Uid > device->m_Uid);
    BOOST_CHECK(deviceWCores->m_Cores == 2);

    // Get the registered device
    const Device* registeredDeviceWCores = counterDirectory.GetDevice(deviceWCores->m_Uid);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(registeredDeviceWCores);
    BOOST_CHECK(registeredDeviceWCores == deviceWCores);
    BOOST_CHECK(registeredDeviceWCores != device);

    // Register a new device with cores and invalid parent category
    const std::string deviceWCoresWInvalidParentCategoryName = "some_device_with_cores_with_invalid_parent_category";
    const Device* deviceWCoresWInvalidParentCategory         = nullptr;
    BOOST_CHECK_THROW(deviceWCoresWInvalidParentCategory =
                          counterDirectory.RegisterDevice(deviceWCoresWInvalidParentCategoryName, 3, std::string("")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(!deviceWCoresWInvalidParentCategory);

    // Register a new device with cores and invalid parent category
    const std::string deviceWCoresWInvalidParentCategoryName2 = "some_device_with_cores_with_invalid_parent_category2";
    const Device* deviceWCoresWInvalidParentCategory2         = nullptr;
    BOOST_CHECK_THROW(deviceWCoresWInvalidParentCategory2 = counterDirectory.RegisterDevice(
                          deviceWCoresWInvalidParentCategoryName2, 3, std::string("invalid_parent_category")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(!deviceWCoresWInvalidParentCategory2);

    // Register a category for testing
    const std::string categoryName = "some_category";
    const Category* category       = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());

    // Register a new device with cores and valid parent category
    const std::string deviceWCoresWValidParentCategoryName = "some_device_with_cores_with_valid_parent_category";
    const Device* deviceWCoresWValidParentCategory         = nullptr;
    BOOST_CHECK_NO_THROW(deviceWCoresWValidParentCategory =
                             counterDirectory.RegisterDevice(deviceWCoresWValidParentCategoryName, 4, categoryName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 3);
    BOOST_CHECK(deviceWCoresWValidParentCategory);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Name == deviceWCoresWValidParentCategoryName);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Uid >= 1);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Uid > device->m_Uid);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Uid > deviceWCores->m_Uid);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Cores == 4);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterCounterSet)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);

    // Register a counter set with an invalid name
    const CounterSet* noCounterSet = nullptr;
    BOOST_CHECK_THROW(noCounterSet = counterDirectory.RegisterCounterSet(""), armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(!noCounterSet);

    // Register a counter set with an invalid name
    BOOST_CHECK_THROW(noCounterSet = counterDirectory.RegisterCounterSet("invalid name"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(!noCounterSet);

    // Register a new counter set with no count or parent category
    const std::string counterSetName = "some_counter_set";
    const CounterSet* counterSet     = nullptr;
    BOOST_CHECK_NO_THROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 1);
    BOOST_CHECK(counterSet);
    BOOST_CHECK(counterSet->m_Name == counterSetName);
    BOOST_CHECK(counterSet->m_Uid >= 1);
    BOOST_CHECK(counterSet->m_Count == 0);

    // Try getting an unregistered counter set
    const CounterSet* unregisteredCounterSet = counterDirectory.GetCounterSet(9999);
    BOOST_CHECK(!unregisteredCounterSet);

    // Get the registered counter set
    const CounterSet* registeredCounterSet = counterDirectory.GetCounterSet(counterSet->m_Uid);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 1);
    BOOST_CHECK(registeredCounterSet);
    BOOST_CHECK(registeredCounterSet == counterSet);

    // Register a counter set with the name of a counter set already registered
    const CounterSet* counterSetSameName = nullptr;
    BOOST_CHECK_THROW(counterSetSameName = counterDirectory.RegisterCounterSet(counterSetName),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 1);
    BOOST_CHECK(!counterSetSameName);

    // Register a new counter set with count and no parent category
    const std::string counterSetWCountName = "some_counter_set_with_count";
    const CounterSet* counterSetWCount     = nullptr;
    BOOST_CHECK_NO_THROW(counterSetWCount = counterDirectory.RegisterCounterSet(counterSetWCountName, 37));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 2);
    BOOST_CHECK(counterSetWCount);
    BOOST_CHECK(counterSetWCount->m_Name == counterSetWCountName);
    BOOST_CHECK(counterSetWCount->m_Uid >= 1);
    BOOST_CHECK(counterSetWCount->m_Uid > counterSet->m_Uid);
    BOOST_CHECK(counterSetWCount->m_Count == 37);

    // Get the registered counter set
    const CounterSet* registeredCounterSetWCount = counterDirectory.GetCounterSet(counterSetWCount->m_Uid);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 2);
    BOOST_CHECK(registeredCounterSetWCount);
    BOOST_CHECK(registeredCounterSetWCount == counterSetWCount);
    BOOST_CHECK(registeredCounterSetWCount != counterSet);

    // Register a new counter set with count and invalid parent category
    const std::string counterSetWCountWInvalidParentCategoryName = "some_counter_set_with_count_"
                                                                   "with_invalid_parent_category";
    const CounterSet* counterSetWCountWInvalidParentCategory = nullptr;
    BOOST_CHECK_THROW(counterSetWCountWInvalidParentCategory = counterDirectory.RegisterCounterSet(
                          counterSetWCountWInvalidParentCategoryName, 42, std::string("")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 2);
    BOOST_CHECK(!counterSetWCountWInvalidParentCategory);

    // Register a new counter set with count and invalid parent category
    const std::string counterSetWCountWInvalidParentCategoryName2 = "some_counter_set_with_count_"
                                                                    "with_invalid_parent_category2";
    const CounterSet* counterSetWCountWInvalidParentCategory2 = nullptr;
    BOOST_CHECK_THROW(counterSetWCountWInvalidParentCategory2 = counterDirectory.RegisterCounterSet(
                          counterSetWCountWInvalidParentCategoryName2, 42, std::string("invalid_parent_category")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 2);
    BOOST_CHECK(!counterSetWCountWInvalidParentCategory2);

    // Register a category for testing
    const std::string categoryName = "some_category";
    const Category* category       = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());

    // Register a new counter set with count and valid parent category
    const std::string counterSetWCountWValidParentCategoryName = "some_counter_set_with_count_"
                                                                 "with_valid_parent_category";
    const CounterSet* counterSetWCountWValidParentCategory = nullptr;
    BOOST_CHECK_NO_THROW(counterSetWCountWValidParentCategory = counterDirectory.RegisterCounterSet(
                             counterSetWCountWValidParentCategoryName, 42, categoryName));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 3);
    BOOST_CHECK(counterSetWCountWValidParentCategory);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Name == counterSetWCountWValidParentCategoryName);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Uid >= 1);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Uid > counterSet->m_Uid);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Uid > counterSetWCount->m_Uid);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Count == 42);

    // Register a counter set associated to a category with invalid name
    const std::string counterSetSameCategoryName = "some_counter_set_with_invalid_parent_category";
    const std::string invalidCategoryName = "";
    const CounterSet* counterSetSameCategory     = nullptr;
    BOOST_CHECK_THROW(counterSetSameCategory =
                          counterDirectory.RegisterCounterSet(counterSetSameCategoryName, 0, invalidCategoryName),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 3);
    BOOST_CHECK(!counterSetSameCategory);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterCounter)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);

    // Register a counter with an invalid parent category name
    const Counter* noCounter = nullptr;
    BOOST_CHECK_THROW(noCounter =
                          counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                           0,
                                                           "",
                                                           0,
                                                           1,
                                                           123.45f,
                                                           "valid ",
                                                           "name"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid parent category name
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   1,
                                                                   "invalid parent category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid class
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   2,
                                                                   "valid_parent_category",
                                                                   2,
                                                                   1,
                                                                   123.45f,
                                                                   "valid "
                                                                   "name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid interpolation
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   4,
                                                                   "valid_parent_category",
                                                                   0,
                                                                   3,
                                                                   123.45f,
                                                                   "valid "
                                                                   "name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid multiplier
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   5,
                                                                   "valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   .0f,
                                                                   "valid "
                                                                   "name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid name
    BOOST_CHECK_THROW(
        noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                     6,
                                                     "valid_parent_category",
                                                     0,
                                                     1,
                                                     123.45f,
                                                     "",
                                                     "valid description"),
        armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid name
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   7,
                                                                   "valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "invalid nam€",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid description
    BOOST_CHECK_THROW(noCounter =
                          counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                           8,
                                                           "valid_parent_category",
                                                           0,
                                                           1,
                                                           123.45f,
                                                           "valid name",
                                                           ""),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid description
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   9,
                                                                   "valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid "
                                                                   "name",
                                                                   "inv@lid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid unit2
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   10,
                                                                   "valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "valid description",
                                                                   std::string("Mb/s2")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with a non-existing parent category name
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   11,
                                                                   "invalid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Try getting an unregistered counter
    const Counter* unregisteredCounter = counterDirectory.GetCounter(9999);
    BOOST_CHECK(!unregisteredCounter);

    // Register a category for testing
    const std::string categoryName = "some_category";
    const Category* category       = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());

    // Register a counter with a valid parent category name
    const Counter* counter = nullptr;
    BOOST_CHECK_NO_THROW(
        counter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                   12,
                                                   categoryName,
                                                   0,
                                                   1,
                                                   123.45f,
                                                   "valid name",
                                                   "valid description"));
    BOOST_CHECK(counterDirectory.GetCounterCount() == 1);
    BOOST_CHECK(counter);
    BOOST_CHECK(counter->m_MaxCounterUid == counter->m_Uid);
    BOOST_CHECK(counter->m_Class == 0);
    BOOST_CHECK(counter->m_Interpolation == 1);
    BOOST_CHECK(counter->m_Multiplier == 123.45f);
    BOOST_CHECK(counter->m_Name == "valid name");
    BOOST_CHECK(counter->m_Description == "valid description");
    BOOST_CHECK(counter->m_Units == "");
    BOOST_CHECK(counter->m_DeviceUid == 0);
    BOOST_CHECK(counter->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 1);
    BOOST_CHECK(category->m_Counters.back() == counter->m_Uid);

    // Register a counter with a name of a counter already registered for the given parent category name
    const Counter* counterSameName = nullptr;
    BOOST_CHECK_THROW(counterSameName =
                          counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                           13,
                                                           categoryName,
                                                           0,
                                                           0,
                                                           1.0f,
                                                           "valid name",
                                                           "valid description",
                                                           std::string("description")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 1);
    BOOST_CHECK(!counterSameName);

    // Register a counter with a valid parent category name and units
    const Counter* counterWUnits = nullptr;
    BOOST_CHECK_NO_THROW(counterWUnits = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                             14,
                                                                             categoryName,
                                                                             0,
                                                                             1,
                                                                             123.45f,
                                                                             "valid name 2",
                                                                             "valid description",
                                                                             std::string("Mnnsq2")));    // Units
    BOOST_CHECK(counterDirectory.GetCounterCount() == 2);
    BOOST_CHECK(counterWUnits);
    BOOST_CHECK(counterWUnits->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWUnits->m_MaxCounterUid == counterWUnits->m_Uid);
    BOOST_CHECK(counterWUnits->m_Class == 0);
    BOOST_CHECK(counterWUnits->m_Interpolation == 1);
    BOOST_CHECK(counterWUnits->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWUnits->m_Name == "valid name 2");
    BOOST_CHECK(counterWUnits->m_Description == "valid description");
    BOOST_CHECK(counterWUnits->m_Units == "Mnnsq2");
    BOOST_CHECK(counterWUnits->m_DeviceUid == 0);
    BOOST_CHECK(counterWUnits->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 2);
    BOOST_CHECK(category->m_Counters.back() == counterWUnits->m_Uid);

    // Register a counter with a valid parent category name and not associated with a device
    const Counter* counterWoDevice = nullptr;
    BOOST_CHECK_NO_THROW(counterWoDevice = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                            26,
                                                                            categoryName,
                                                                            0,
                                                                            1,
                                                                            123.45f,
                                                                            "valid name 3",
                                                                            "valid description",
                                                                            armnn::EmptyOptional(),// Units
                                                                            armnn::EmptyOptional(),// Number of cores
                                                                            0));                   // Device UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 3);
    BOOST_CHECK(counterWoDevice);
    BOOST_CHECK(counterWoDevice->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWoDevice->m_MaxCounterUid == counterWoDevice->m_Uid);
    BOOST_CHECK(counterWoDevice->m_Class == 0);
    BOOST_CHECK(counterWoDevice->m_Interpolation == 1);
    BOOST_CHECK(counterWoDevice->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWoDevice->m_Name == "valid name 3");
    BOOST_CHECK(counterWoDevice->m_Description == "valid description");
    BOOST_CHECK(counterWoDevice->m_Units == "");
    BOOST_CHECK(counterWoDevice->m_DeviceUid == 0);
    BOOST_CHECK(counterWoDevice->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 3);
    BOOST_CHECK(category->m_Counters.back() == counterWoDevice->m_Uid);

    // Register a counter with a valid parent category name and associated to an invalid device
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   15,
                                                                   categoryName,
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name 4",
                                                                   "valid description",
                                                                   armnn::EmptyOptional(),    // Units
                                                                   armnn::EmptyOptional(),    // Number of cores
                                                                   100),                      // Device UID
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 3);
    BOOST_CHECK(!noCounter);

    // Register a device for testing
    const std::string deviceName = "some_device";
    const Device* device         = nullptr;
    BOOST_CHECK_NO_THROW(device = counterDirectory.RegisterDevice(deviceName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 1);
    BOOST_CHECK(device);
    BOOST_CHECK(device->m_Name == deviceName);
    BOOST_CHECK(device->m_Uid >= 1);
    BOOST_CHECK(device->m_Cores == 0);

    // Register a counter with a valid parent category name and associated to a device
    const Counter* counterWDevice = nullptr;
    BOOST_CHECK_NO_THROW(counterWDevice = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                           16,
                                                                           categoryName,
                                                                           0,
                                                                           1,
                                                                           123.45f,
                                                                           "valid name 5",
                                                                           std::string("valid description"),
                                                                           armnn::EmptyOptional(), // Units
                                                                           armnn::EmptyOptional(), // Number of cores
                                                                           device->m_Uid));        // Device UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 4);
    BOOST_CHECK(counterWDevice);
    BOOST_CHECK(counterWDevice->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWDevice->m_MaxCounterUid == counterWDevice->m_Uid);
    BOOST_CHECK(counterWDevice->m_Class == 0);
    BOOST_CHECK(counterWDevice->m_Interpolation == 1);
    BOOST_CHECK(counterWDevice->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWDevice->m_Name == "valid name 5");
    BOOST_CHECK(counterWDevice->m_Description == "valid description");
    BOOST_CHECK(counterWDevice->m_Units == "");
    BOOST_CHECK(counterWDevice->m_DeviceUid == device->m_Uid);
    BOOST_CHECK(counterWDevice->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 4);
    BOOST_CHECK(category->m_Counters.back() == counterWDevice->m_Uid);

    // Register a counter with a valid parent category name and not associated with a counter set
    const Counter* counterWoCounterSet = nullptr;
    BOOST_CHECK_NO_THROW(counterWoCounterSet = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                                17,
                                                                                categoryName,
                                                                                0,
                                                                                1,
                                                                                123.45f,
                                                                                "valid name 6",
                                                                                "valid description",
                                                                                armnn::EmptyOptional(),// Units
                                                                                armnn::EmptyOptional(),// No of cores
                                                                                armnn::EmptyOptional(),// Device UID
                                                                                0));               // CounterSet UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 5);
    BOOST_CHECK(counterWoCounterSet);
    BOOST_CHECK(counterWoCounterSet->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWoCounterSet->m_MaxCounterUid == counterWoCounterSet->m_Uid);
    BOOST_CHECK(counterWoCounterSet->m_Class == 0);
    BOOST_CHECK(counterWoCounterSet->m_Interpolation == 1);
    BOOST_CHECK(counterWoCounterSet->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWoCounterSet->m_Name == "valid name 6");
    BOOST_CHECK(counterWoCounterSet->m_Description == "valid description");
    BOOST_CHECK(counterWoCounterSet->m_Units == "");
    BOOST_CHECK(counterWoCounterSet->m_DeviceUid == 0);
    BOOST_CHECK(counterWoCounterSet->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 5);
    BOOST_CHECK(category->m_Counters.back() == counterWoCounterSet->m_Uid);

    // Register a counter with a valid parent category name and associated to an invalid counter set
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID,
                                                                   18,
                                                                   categoryName,
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid ",
                                                                   "name 7",
                                                                   std::string("valid description"),
                                                                   armnn::EmptyOptional(),    // Units
                                                                   armnn::EmptyOptional(),    // Number of cores
                                                                   100),            // Counter set UID
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 5);
    BOOST_CHECK(!noCounter);

    // Register a counter with a valid parent category name and with a given number of cores
    const Counter* counterWNumberOfCores = nullptr;
    uint16_t numberOfCores               = 15;
    BOOST_CHECK_NO_THROW(counterWNumberOfCores = counterDirectory.RegisterCounter(
                             armnn::profiling::BACKEND_ID, 50,
                             categoryName, 0, 1, 123.45f, "valid name 8", "valid description",
                             armnn::EmptyOptional(),      // Units
                             numberOfCores,               // Number of cores
                             armnn::EmptyOptional(),      // Device UID
                             armnn::EmptyOptional()));    // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 20);
    BOOST_CHECK(counterWNumberOfCores);
    BOOST_CHECK(counterWNumberOfCores->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWNumberOfCores->m_MaxCounterUid == counterWNumberOfCores->m_Uid + numberOfCores - 1);
    BOOST_CHECK(counterWNumberOfCores->m_Class == 0);
    BOOST_CHECK(counterWNumberOfCores->m_Interpolation == 1);
    BOOST_CHECK(counterWNumberOfCores->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWNumberOfCores->m_Name == "valid name 8");
    BOOST_CHECK(counterWNumberOfCores->m_Description == "valid description");
    BOOST_CHECK(counterWNumberOfCores->m_Units == "");
    BOOST_CHECK(counterWNumberOfCores->m_DeviceUid == 0);
    BOOST_CHECK(counterWNumberOfCores->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 20);
    for (size_t i = 0; i < numberOfCores; i++)
    {
        BOOST_CHECK(category->m_Counters[category->m_Counters.size() - numberOfCores + i] ==
                    counterWNumberOfCores->m_Uid + i);
    }

    // Register a multi-core device for testing
    const std::string multiCoreDeviceName = "some_multi_core_device";
    const Device* multiCoreDevice         = nullptr;
    BOOST_CHECK_NO_THROW(multiCoreDevice = counterDirectory.RegisterDevice(multiCoreDeviceName, 4));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(multiCoreDevice);
    BOOST_CHECK(multiCoreDevice->m_Name == multiCoreDeviceName);
    BOOST_CHECK(multiCoreDevice->m_Uid >= 1);
    BOOST_CHECK(multiCoreDevice->m_Cores == 4);

    // Register a counter with a valid parent category name and associated to the multi-core device
    const Counter* counterWMultiCoreDevice = nullptr;
    BOOST_CHECK_NO_THROW(counterWMultiCoreDevice = counterDirectory.RegisterCounter(
                             armnn::profiling::BACKEND_ID, 19, categoryName, 0, 1,
                             123.45f, "valid name 9", "valid description",
                             armnn::EmptyOptional(),      // Units
                             armnn::EmptyOptional(),      // Number of cores
                             multiCoreDevice->m_Uid,      // Device UID
                             armnn::EmptyOptional()));    // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 24);
    BOOST_CHECK(counterWMultiCoreDevice);
    BOOST_CHECK(counterWMultiCoreDevice->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWMultiCoreDevice->m_MaxCounterUid ==
                counterWMultiCoreDevice->m_Uid + multiCoreDevice->m_Cores - 1);
    BOOST_CHECK(counterWMultiCoreDevice->m_Class == 0);
    BOOST_CHECK(counterWMultiCoreDevice->m_Interpolation == 1);
    BOOST_CHECK(counterWMultiCoreDevice->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWMultiCoreDevice->m_Name == "valid name 9");
    BOOST_CHECK(counterWMultiCoreDevice->m_Description == "valid description");
    BOOST_CHECK(counterWMultiCoreDevice->m_Units == "");
    BOOST_CHECK(counterWMultiCoreDevice->m_DeviceUid == multiCoreDevice->m_Uid);
    BOOST_CHECK(counterWMultiCoreDevice->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 24);
    for (size_t i = 0; i < 4; i++)
    {
        BOOST_CHECK(category->m_Counters[category->m_Counters.size() - 4 + i] == counterWMultiCoreDevice->m_Uid + i);
    }

    // Register a multi-core device associate to a parent category for testing
    const std::string multiCoreDeviceNameWParentCategory = "some_multi_core_device_with_parent_category";
    const Device* multiCoreDeviceWParentCategory         = nullptr;
    BOOST_CHECK_NO_THROW(multiCoreDeviceWParentCategory =
                             counterDirectory.RegisterDevice(multiCoreDeviceNameWParentCategory, 2, categoryName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 3);
    BOOST_CHECK(multiCoreDeviceWParentCategory);
    BOOST_CHECK(multiCoreDeviceWParentCategory->m_Name == multiCoreDeviceNameWParentCategory);
    BOOST_CHECK(multiCoreDeviceWParentCategory->m_Uid >= 1);
    BOOST_CHECK(multiCoreDeviceWParentCategory->m_Cores == 2);

    // Register a counter with a valid parent category name and getting the number of cores of the multi-core device
    // associated to that category
    const Counter* counterWMultiCoreDeviceWParentCategory = nullptr;
    uint16_t numberOfCourse = multiCoreDeviceWParentCategory->m_Cores;
    BOOST_CHECK_NO_THROW(counterWMultiCoreDeviceWParentCategory =
                                                counterDirectory.RegisterCounter(
                                                    armnn::profiling::BACKEND_ID,
                                                    100,
                                                    categoryName,
                                                    0,
                                                    1,
                                                    123.45f,
                                                    "valid name 10",
                                                    "valid description",
                                                    armnn::EmptyOptional(),  // Units
                                                    numberOfCourse,          // Number of cores
                                                    armnn::EmptyOptional(),  // Device UID
                                                    armnn::EmptyOptional()));// Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 26);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_MaxCounterUid ==
                counterWMultiCoreDeviceWParentCategory->m_Uid + multiCoreDeviceWParentCategory->m_Cores - 1);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Class == 0);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Interpolation == 1);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Name == "valid name 10");
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Description == "valid description");
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Units == "");
    BOOST_CHECK(category->m_Counters.size() == 26);
    for (size_t i = 0; i < 2; i++)
    {
        BOOST_CHECK(category->m_Counters[category->m_Counters.size() - 2 + i] ==
                    counterWMultiCoreDeviceWParentCategory->m_Uid + i);
    }

    // Register a counter set for testing
    const std::string counterSetName = "some_counter_set";
    const CounterSet* counterSet     = nullptr;
    BOOST_CHECK_NO_THROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 1);
    BOOST_CHECK(counterSet);
    BOOST_CHECK(counterSet->m_Name == counterSetName);
    BOOST_CHECK(counterSet->m_Uid >= 1);
    BOOST_CHECK(counterSet->m_Count == 0);

    // Register a counter with a valid parent category name and associated to a counter set
    const Counter* counterWCounterSet = nullptr;
    BOOST_CHECK_NO_THROW(counterWCounterSet = counterDirectory.RegisterCounter(
                             armnn::profiling::BACKEND_ID, 300,
                             categoryName, 0, 1, 123.45f, "valid name 11", "valid description",
                             armnn::EmptyOptional(),    // Units
                             0,                         // Number of cores
                             armnn::EmptyOptional(),    // Device UID
                             counterSet->m_Uid));       // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 27);
    BOOST_CHECK(counterWCounterSet);
    BOOST_CHECK(counterWCounterSet->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWCounterSet->m_MaxCounterUid == counterWCounterSet->m_Uid);
    BOOST_CHECK(counterWCounterSet->m_Class == 0);
    BOOST_CHECK(counterWCounterSet->m_Interpolation == 1);
    BOOST_CHECK(counterWCounterSet->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWCounterSet->m_Name == "valid name 11");
    BOOST_CHECK(counterWCounterSet->m_Description == "valid description");
    BOOST_CHECK(counterWCounterSet->m_Units == "");
    BOOST_CHECK(counterWCounterSet->m_DeviceUid == 0);
    BOOST_CHECK(counterWCounterSet->m_CounterSetUid == counterSet->m_Uid);
    BOOST_CHECK(category->m_Counters.size() == 27);
    BOOST_CHECK(category->m_Counters.back() == counterWCounterSet->m_Uid);

    // Register a counter with a valid parent category name and associated to a device and a counter set
    const Counter* counterWDeviceWCounterSet = nullptr;
    BOOST_CHECK_NO_THROW(counterWDeviceWCounterSet = counterDirectory.RegisterCounter(
                             armnn::profiling::BACKEND_ID, 23,
                             categoryName, 0, 1, 123.45f, "valid name 12", "valid description",
                             armnn::EmptyOptional(),    // Units
                             1,                         // Number of cores
                             device->m_Uid,             // Device UID
                             counterSet->m_Uid));       // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 28);
    BOOST_CHECK(counterWDeviceWCounterSet);
    BOOST_CHECK(counterWDeviceWCounterSet->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWDeviceWCounterSet->m_MaxCounterUid == counterWDeviceWCounterSet->m_Uid);
    BOOST_CHECK(counterWDeviceWCounterSet->m_Class == 0);
    BOOST_CHECK(counterWDeviceWCounterSet->m_Interpolation == 1);
    BOOST_CHECK(counterWDeviceWCounterSet->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWDeviceWCounterSet->m_Name == "valid name 12");
    BOOST_CHECK(counterWDeviceWCounterSet->m_Description == "valid description");
    BOOST_CHECK(counterWDeviceWCounterSet->m_Units == "");
    BOOST_CHECK(counterWDeviceWCounterSet->m_DeviceUid == device->m_Uid);
    BOOST_CHECK(counterWDeviceWCounterSet->m_CounterSetUid == counterSet->m_Uid);
    BOOST_CHECK(category->m_Counters.size() == 28);
    BOOST_CHECK(category->m_Counters.back() == counterWDeviceWCounterSet->m_Uid);

    // Register another category for testing
    const std::string anotherCategoryName = "some_other_category";
    const Category* anotherCategory       = nullptr;
    BOOST_CHECK_NO_THROW(anotherCategory = counterDirectory.RegisterCategory(anotherCategoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 2);
    BOOST_CHECK(anotherCategory);
    BOOST_CHECK(anotherCategory != category);
    BOOST_CHECK(anotherCategory->m_Name == anotherCategoryName);
    BOOST_CHECK(anotherCategory->m_Counters.empty());

    // Register a counter to the other category
    const Counter* anotherCounter = nullptr;
    BOOST_CHECK_NO_THROW(anotherCounter = counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID, 24,
                                                                           anotherCategoryName, 1, 0, .00043f,
                                                                           "valid name", "valid description",
                                                                           armnn::EmptyOptional(), // Units
                                                                           armnn::EmptyOptional(), // Number of cores
                                                                           device->m_Uid,          // Device UID
                                                                           counterSet->m_Uid));    // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 29);
    BOOST_CHECK(anotherCounter);
    BOOST_CHECK(anotherCounter->m_MaxCounterUid == anotherCounter->m_Uid);
    BOOST_CHECK(anotherCounter->m_Class == 1);
    BOOST_CHECK(anotherCounter->m_Interpolation == 0);
    BOOST_CHECK(anotherCounter->m_Multiplier == .00043f);
    BOOST_CHECK(anotherCounter->m_Name == "valid name");
    BOOST_CHECK(anotherCounter->m_Description == "valid description");
    BOOST_CHECK(anotherCounter->m_Units == "");
    BOOST_CHECK(anotherCounter->m_DeviceUid == device->m_Uid);
    BOOST_CHECK(anotherCounter->m_CounterSetUid == counterSet->m_Uid);
    BOOST_CHECK(anotherCategory->m_Counters.size() == 1);
    BOOST_CHECK(anotherCategory->m_Counters.back() == anotherCounter->m_Uid);
}

BOOST_AUTO_TEST_CASE(CounterSelectionCommandHandlerParseData)
{
    ProfilingStateMachine profilingStateMachine;

    class TestCaptureThread : public IPeriodicCounterCapture
    {
        void Start() override
        {}
        void Stop() override
        {}
    };

    class TestReadCounterValues : public IReadCounterValues
    {
        bool IsCounterRegistered(uint16_t counterUid) const override
        {
            armnn::IgnoreUnused(counterUid);
            return true;
        }
        uint16_t GetCounterCount() const override
        {
            return 0;
        }
        uint32_t GetAbsoluteCounterValue(uint16_t counterUid) const override
        {
            armnn::IgnoreUnused(counterUid);
            return 0;
        }
        uint32_t GetDeltaCounterValue(uint16_t counterUid) override
        {
            armnn::IgnoreUnused(counterUid);
            return 0;
        }
    };
    const uint32_t familyId = 0;
    const uint32_t packetId = 0x40000;

    uint32_t version = 1;
    const std::unordered_map<armnn::BackendId,
            std::shared_ptr<armnn::profiling::IBackendProfilingContext>> backendProfilingContext;
    CounterIdMap counterIdMap;
    Holder holder;
    TestCaptureThread captureThread;
    TestReadCounterValues readCounterValues;
    MockBufferManager mockBuffer(512);
    SendCounterPacket sendCounterPacket(mockBuffer);
    SendThread sendThread(profilingStateMachine, mockBuffer, sendCounterPacket);

    uint32_t sizeOfUint32 = armnn::numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = armnn::numeric_cast<uint32_t>(sizeof(uint16_t));

    // Data with period and counters
    uint32_t period1     = armnn::LOWEST_CAPTURE_PERIOD;
    uint32_t dataLength1 = 8;
    uint32_t offset      = 0;

    std::unique_ptr<unsigned char[]> uniqueData1 = std::make_unique<unsigned char[]>(dataLength1);
    unsigned char* data1                         = reinterpret_cast<unsigned char*>(uniqueData1.get());

    WriteUint32(data1, offset, period1);
    offset += sizeOfUint32;
    WriteUint16(data1, offset, 4000);
    offset += sizeOfUint16;
    WriteUint16(data1, offset, 5000);

    arm::pipe::Packet packetA(packetId, dataLength1, uniqueData1);

    PeriodicCounterSelectionCommandHandler commandHandler(familyId, packetId, version, backendProfilingContext,
                                                          counterIdMap, holder, 10000u, captureThread,
                                                          readCounterValues, sendCounterPacket, profilingStateMachine);

    profilingStateMachine.TransitionToState(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(commandHandler(packetA), armnn::RuntimeException);
    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(commandHandler(packetA), armnn::RuntimeException);
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(commandHandler(packetA), armnn::RuntimeException);
    profilingStateMachine.TransitionToState(ProfilingState::Active);
    BOOST_CHECK_NO_THROW(commandHandler(packetA));

    const std::vector<uint16_t> counterIdsA = holder.GetCaptureData().GetCounterIds();

    BOOST_TEST(holder.GetCaptureData().GetCapturePeriod() == period1);
    BOOST_TEST(counterIdsA.size() == 2);
    BOOST_TEST(counterIdsA[0] == 4000);
    BOOST_TEST(counterIdsA[1] == 5000);

    auto readBuffer = mockBuffer.GetReadableBuffer();

    offset = 0;

    uint32_t headerWord0 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    uint32_t headerWord1 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    uint32_t period = ReadUint32(readBuffer, offset);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 0);             // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x3FF) == 4);            // packet id
    BOOST_TEST(headerWord1 == 8);                              // data length
    BOOST_TEST(period ==  armnn::LOWEST_CAPTURE_PERIOD);       // capture period

    uint16_t counterId = 0;
    offset += sizeOfUint32;
    counterId = ReadUint16(readBuffer, offset);
    BOOST_TEST(counterId == 4000);
    offset += sizeOfUint16;
    counterId = ReadUint16(readBuffer, offset);
    BOOST_TEST(counterId == 5000);

    mockBuffer.MarkRead(readBuffer);

    // Data with period only
    uint32_t period2     = 9000; // We'll specify a value below LOWEST_CAPTURE_PERIOD. It should be pulled upwards.
    uint32_t dataLength2 = 4;

    std::unique_ptr<unsigned char[]> uniqueData2 = std::make_unique<unsigned char[]>(dataLength2);

    WriteUint32(reinterpret_cast<unsigned char*>(uniqueData2.get()), 0, period2);

    arm::pipe::Packet packetB(packetId, dataLength2, uniqueData2);

    commandHandler(packetB);

    const std::vector<uint16_t> counterIdsB = holder.GetCaptureData().GetCounterIds();

    // Value should have been pulled up from 9000 to LOWEST_CAPTURE_PERIOD.
    BOOST_TEST(holder.GetCaptureData().GetCapturePeriod() ==  armnn::LOWEST_CAPTURE_PERIOD);
    BOOST_TEST(counterIdsB.size() == 0);

    readBuffer = mockBuffer.GetReadableBuffer();

    offset = 0;

    headerWord0 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    headerWord1 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    period = ReadUint32(readBuffer, offset);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 0);         // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x3FF) == 4);        // packet id
    BOOST_TEST(headerWord1 == 4);                          // data length
    BOOST_TEST(period == armnn::LOWEST_CAPTURE_PERIOD);    // capture period
}

BOOST_AUTO_TEST_CASE(CheckTimelineActivationAndDeactivation)
{
    class TestReportStructure : public IReportStructure
    {
        public:
        virtual void ReportStructure() override
        {
            m_ReportStructureCalled = true;
        }

        bool m_ReportStructureCalled = false;
    };

    class TestNotifyBackends : public INotifyBackends
    {
        public:
        TestNotifyBackends() : m_timelineReporting(false) {}
        virtual void NotifyBackendsForTimelineReporting() override
        {
            m_TestNotifyBackendsCalled = m_timelineReporting.load();
        }

        bool m_TestNotifyBackendsCalled = false;
        std::atomic<bool> m_timelineReporting;
    };

    arm::pipe::PacketVersionResolver packetVersionResolver;

    BufferManager bufferManager(512);
    SendTimelinePacket sendTimelinePacket(bufferManager);
    ProfilingStateMachine stateMachine;
    TestReportStructure testReportStructure;
    TestNotifyBackends testNotifyBackends;

    profiling::ActivateTimelineReportingCommandHandler activateTimelineReportingCommandHandler(0,
                                                           6,
                                                           packetVersionResolver.ResolvePacketVersion(0, 6)
                                                           .GetEncodedValue(),
                                                           sendTimelinePacket,
                                                           stateMachine,
                                                           testReportStructure,
                                                           testNotifyBackends.m_timelineReporting,
                                                           testNotifyBackends);

    // Write an "ActivateTimelineReporting" packet into the mock profiling connection, to simulate an input from an
    // external profiling service
    const uint32_t packetFamily1 = 0;
    const uint32_t packetId1     = 6;
    uint32_t packetHeader1 = ConstructHeader(packetFamily1, packetId1);

    // Create the ActivateTimelineReportingPacket
    arm::pipe::Packet ActivateTimelineReportingPacket(packetHeader1); // Length == 0

    BOOST_CHECK_THROW(
            activateTimelineReportingCommandHandler.operator()(ActivateTimelineReportingPacket), armnn::Exception);

    stateMachine.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(
            activateTimelineReportingCommandHandler.operator()(ActivateTimelineReportingPacket), armnn::Exception);

    stateMachine.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(
            activateTimelineReportingCommandHandler.operator()(ActivateTimelineReportingPacket), armnn::Exception);

    stateMachine.TransitionToState(ProfilingState::Active);
    activateTimelineReportingCommandHandler.operator()(ActivateTimelineReportingPacket);

    BOOST_CHECK(testReportStructure.m_ReportStructureCalled);
    BOOST_CHECK(testNotifyBackends.m_TestNotifyBackendsCalled);
    BOOST_CHECK(testNotifyBackends.m_timelineReporting.load());

    DeactivateTimelineReportingCommandHandler deactivateTimelineReportingCommandHandler(0,
                                                  7,
                                                  packetVersionResolver.ResolvePacketVersion(0, 7).GetEncodedValue(),
                                                  testNotifyBackends.m_timelineReporting,
                                                  stateMachine,
                                                  testNotifyBackends);

    const uint32_t packetFamily2 = 0;
    const uint32_t packetId2     = 7;
    uint32_t packetHeader2 = ConstructHeader(packetFamily2, packetId2);

    // Create the DeactivateTimelineReportingPacket
    arm::pipe::Packet deactivateTimelineReportingPacket(packetHeader2); // Length == 0

    stateMachine.Reset();
    BOOST_CHECK_THROW(
            deactivateTimelineReportingCommandHandler.operator()(deactivateTimelineReportingPacket), armnn::Exception);

    stateMachine.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(
            deactivateTimelineReportingCommandHandler.operator()(deactivateTimelineReportingPacket), armnn::Exception);

    stateMachine.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(
            deactivateTimelineReportingCommandHandler.operator()(deactivateTimelineReportingPacket), armnn::Exception);

    stateMachine.TransitionToState(ProfilingState::Active);
    deactivateTimelineReportingCommandHandler.operator()(deactivateTimelineReportingPacket);

    BOOST_CHECK(!testNotifyBackends.m_TestNotifyBackendsCalled);
    BOOST_CHECK(!testNotifyBackends.m_timelineReporting.load());
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceNotActive)
{
    using namespace armnn;
    using namespace armnn::profiling;

    // Create runtime in which the test will run
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;

    armnn::Runtime runtime(options);
    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper(GetProfilingService(&runtime));
    profilingServiceHelper.ForceTransitionToState(ProfilingState::NotConnected);
    profilingServiceHelper.ForceTransitionToState(ProfilingState::WaitingForAck);
    profilingServiceHelper.ForceTransitionToState(ProfilingState::Active);

    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is enabled, the post-optimisation structure should be created
    BOOST_CHECK(readableBuffer == nullptr);
}

BOOST_AUTO_TEST_CASE(CheckConnectionAcknowledged)
{
    const uint32_t packetFamilyId     = 0;
    const uint32_t connectionPacketId = 0x10000;
    const uint32_t version            = 1;

    uint32_t sizeOfUint32 = armnn::numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = armnn::numeric_cast<uint32_t>(sizeof(uint16_t));

    // Data with period and counters
    uint32_t period1     = 10;
    uint32_t dataLength1 = 8;
    uint32_t offset      = 0;

    std::unique_ptr<unsigned char[]> uniqueData1 = std::make_unique<unsigned char[]>(dataLength1);
    unsigned char* data1                         = reinterpret_cast<unsigned char*>(uniqueData1.get());

    WriteUint32(data1, offset, period1);
    offset += sizeOfUint32;
    WriteUint16(data1, offset, 4000);
    offset += sizeOfUint16;
    WriteUint16(data1, offset, 5000);

    arm::pipe::Packet packetA(connectionPacketId, dataLength1, uniqueData1);

    ProfilingStateMachine profilingState(ProfilingState::Uninitialised);
    BOOST_CHECK(profilingState.GetCurrentState() == ProfilingState::Uninitialised);
    CounterDirectory counterDirectory;
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(mockBuffer);
    SendThread sendThread(profilingState, mockBuffer, sendCounterPacket);
    SendTimelinePacket sendTimelinePacket(mockBuffer);
    MockProfilingServiceStatus mockProfilingServiceStatus;

    ConnectionAcknowledgedCommandHandler commandHandler(packetFamilyId,
                                                        connectionPacketId,
                                                        version,
                                                        counterDirectory,
                                                        sendCounterPacket,
                                                        sendTimelinePacket,
                                                        profilingState,
                                                        mockProfilingServiceStatus);

    // command handler received packet on ProfilingState::Uninitialised
    BOOST_CHECK_THROW(commandHandler(packetA), armnn::Exception);

    profilingState.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK(profilingState.GetCurrentState() == ProfilingState::NotConnected);
    // command handler received packet on ProfilingState::NotConnected
    BOOST_CHECK_THROW(commandHandler(packetA), armnn::Exception);

    profilingState.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK(profilingState.GetCurrentState() == ProfilingState::WaitingForAck);
    // command handler received packet on ProfilingState::WaitingForAck
    BOOST_CHECK_NO_THROW(commandHandler(packetA));
    BOOST_CHECK(profilingState.GetCurrentState() == ProfilingState::Active);

    // command handler received packet on ProfilingState::Active
    BOOST_CHECK_NO_THROW(commandHandler(packetA));
    BOOST_CHECK(profilingState.GetCurrentState() == ProfilingState::Active);

    // command handler received different packet
    const uint32_t differentPacketId = 0x40000;
    arm::pipe::Packet packetB(differentPacketId, dataLength1, uniqueData1);
    profilingState.TransitionToState(ProfilingState::NotConnected);
    profilingState.TransitionToState(ProfilingState::WaitingForAck);
    ConnectionAcknowledgedCommandHandler differentCommandHandler(packetFamilyId,
                                                                 differentPacketId,
                                                                 version,
                                                                 counterDirectory,
                                                                 sendCounterPacket,
                                                                 sendTimelinePacket,
                                                                 profilingState,
                                                                 mockProfilingServiceStatus);
    BOOST_CHECK_THROW(differentCommandHandler(packetB), armnn::Exception);
}

BOOST_AUTO_TEST_CASE(CheckSocketConnectionException)
{
    // Check that creating a SocketProfilingConnection armnnProfiling in an exception as the Gator UDS doesn't exist.
    BOOST_CHECK_THROW(new SocketProfilingConnection(), arm::pipe::SocketConnectionException);
}

BOOST_AUTO_TEST_CASE(CheckSocketConnectionException2)
{
    try
    {
        new SocketProfilingConnection();
    }
    catch (const arm::pipe::SocketConnectionException& ex)
    {
        BOOST_CHECK(ex.GetSocketFd() == 0);
        BOOST_CHECK(ex.GetErrorNo() == 111);
        BOOST_CHECK(ex.what()
                    == std::string("SocketProfilingConnection: Cannot connect to stream socket: Connection refused"));
    }
}

BOOST_AUTO_TEST_CASE(SwTraceIsValidCharTest)
{
    // Only ASCII 7-bit encoding supported
    for (unsigned char c = 0; c < 128; c++)
    {
        BOOST_CHECK(arm::pipe::SwTraceCharPolicy::IsValidChar(c));
    }

    // Not ASCII
    for (unsigned char c = 255; c >= 128; c++)
    {
        BOOST_CHECK(!arm::pipe::SwTraceCharPolicy::IsValidChar(c));
    }
}

BOOST_AUTO_TEST_CASE(SwTraceIsValidNameCharTest)
{
    // Only alpha-numeric and underscore ASCII 7-bit encoding supported
    const unsigned char validChars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    for (unsigned char i = 0; i < sizeof(validChars) / sizeof(validChars[0]) - 1; i++)
    {
        BOOST_CHECK(arm::pipe::SwTraceNameCharPolicy::IsValidChar(validChars[i]));
    }

    // Non alpha-numeric chars
    for (unsigned char c = 0; c < 48; c++)
    {
        BOOST_CHECK(!arm::pipe::SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 58; c < 65; c++)
    {
        BOOST_CHECK(!arm::pipe::SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 91; c < 95; c++)
    {
        BOOST_CHECK(!arm::pipe::SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 96; c < 97; c++)
    {
        BOOST_CHECK(!arm::pipe::SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 123; c < 128; c++)
    {
        BOOST_CHECK(!arm::pipe::SwTraceNameCharPolicy::IsValidChar(c));
    }

    // Not ASCII
    for (unsigned char c = 255; c >= 128; c++)
    {
        BOOST_CHECK(!arm::pipe::SwTraceNameCharPolicy::IsValidChar(c));
    }
}

BOOST_AUTO_TEST_CASE(IsValidSwTraceStringTest)
{
    // Valid SWTrace strings
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>(""));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("_"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("0123"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("valid_string"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("VALID_string_456"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>(" "));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("valid string"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("!$%"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("valid|\\~string#123"));

    // Invalid SWTrace strings
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("€£"));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("invalid‡string"));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceCharPolicy>("12Ž34"));
}

BOOST_AUTO_TEST_CASE(IsValidSwTraceNameStringTest)
{
    // Valid SWTrace name strings
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>(""));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("_"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("0123"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("valid_string"));
    BOOST_CHECK(arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("VALID_string_456"));

    // Invalid SWTrace name strings
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>(" "));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("invalid string"));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("!$%"));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("invalid|\\~string#123"));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("€£"));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("invalid‡string"));
    BOOST_CHECK(!arm::pipe::IsValidSwTraceString<arm::pipe::SwTraceNameCharPolicy>("12Ž34"));
}

template <typename SwTracePolicy>
void StringToSwTraceStringTestHelper(const std::string& testString, std::vector<uint32_t> buffer, size_t expectedSize)
{
    // Convert the test string to a SWTrace string
    BOOST_CHECK(arm::pipe::StringToSwTraceString<SwTracePolicy>(testString, buffer));

    // The buffer must contain at least the length of the string
    BOOST_CHECK(!buffer.empty());

    // The buffer must be of the expected size (in words)
    BOOST_CHECK(buffer.size() == expectedSize);

    // The first word of the byte must be the length of the string including the null-terminator
    BOOST_CHECK(buffer[0] == testString.size() + 1);

    // The contents of the buffer must match the test string
    BOOST_CHECK(std::memcmp(testString.data(), buffer.data() + 1, testString.size()) == 0);

    // The buffer must include the null-terminator at the end of the string
    size_t nullTerminatorIndex = sizeof(uint32_t) + testString.size();
    BOOST_CHECK(reinterpret_cast<unsigned char*>(buffer.data())[nullTerminatorIndex] == '\0');
}

BOOST_AUTO_TEST_CASE(StringToSwTraceStringTest)
{
    std::vector<uint32_t> buffer;

    // Valid SWTrace strings (expected size in words)
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("", buffer, 2);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("_", buffer, 2);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("0123", buffer, 3);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("valid_string", buffer, 5);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("VALID_string_456", buffer, 6);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>(" ", buffer, 2);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("valid string", buffer, 5);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("!$%", buffer, 2);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceCharPolicy>("valid|\\~string#123", buffer, 6);

    // Invalid SWTrace strings
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceCharPolicy>("€£", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceCharPolicy>("invalid‡string", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceCharPolicy>("12Ž34", buffer));
    BOOST_CHECK(buffer.empty());
}

BOOST_AUTO_TEST_CASE(StringToSwTraceNameStringTest)
{
    std::vector<uint32_t> buffer;

    // Valid SWTrace namestrings (expected size in words)
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceNameCharPolicy>("", buffer, 2);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceNameCharPolicy>("_", buffer, 2);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceNameCharPolicy>("0123", buffer, 3);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceNameCharPolicy>("valid_string", buffer, 5);
    StringToSwTraceStringTestHelper<arm::pipe::SwTraceNameCharPolicy>("VALID_string_456", buffer, 6);

    // Invalid SWTrace namestrings
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>(" ", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>("invalid string", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>("!$%", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>("invalid|\\~string#123", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>("€£", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>("invalid‡string", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>("12Ž34", buffer));
    BOOST_CHECK(buffer.empty());
}

BOOST_AUTO_TEST_CASE(CheckPeriodicCounterCaptureThread)
{
    class CaptureReader : public IReadCounterValues
    {
    public:
        CaptureReader(uint16_t counterSize)
        {
            for (uint16_t i = 0; i < counterSize; ++i)
            {
                m_Data[i] = 0;
            }
            m_CounterSize = counterSize;
        }
        //not used
        bool IsCounterRegistered(uint16_t counterUid) const override
        {
            armnn::IgnoreUnused(counterUid);
            return false;
        }

        uint16_t GetCounterCount() const override
        {
            return m_CounterSize;
        }

        uint32_t GetAbsoluteCounterValue(uint16_t counterUid) const override
        {
            if (counterUid > m_CounterSize)
            {
                BOOST_FAIL("Invalid counter Uid");
            }
            return m_Data.at(counterUid).load();
        }

        uint32_t GetDeltaCounterValue(uint16_t counterUid)  override
        {
            if (counterUid > m_CounterSize)
            {
                BOOST_FAIL("Invalid counter Uid");
            }
            return m_Data.at(counterUid).load();
        }

        void SetCounterValue(uint16_t counterUid, uint32_t value)
        {
            if (counterUid > m_CounterSize)
            {
                BOOST_FAIL("Invalid counter Uid");
            }
            m_Data.at(counterUid).store(value);
        }

    private:
        std::unordered_map<uint16_t, std::atomic<uint32_t>> m_Data;
        uint16_t m_CounterSize;
    };

    ProfilingStateMachine profilingStateMachine;

    const std::unordered_map<armnn::BackendId,
            std::shared_ptr<armnn::profiling::IBackendProfilingContext>> backendProfilingContext;
    CounterIdMap counterIdMap;
    Holder data;
    std::vector<uint16_t> captureIds1 = { 0, 1 };
    std::vector<uint16_t> captureIds2;

    MockBufferManager mockBuffer(512);
    SendCounterPacket sendCounterPacket(mockBuffer);
    SendThread sendThread(profilingStateMachine, mockBuffer, sendCounterPacket);

    std::vector<uint16_t> counterIds;
    CaptureReader captureReader(2);

    unsigned int valueA   = 10;
    unsigned int valueB   = 15;
    unsigned int numSteps = 5;

    PeriodicCounterCapture periodicCounterCapture(std::ref(data), std::ref(sendCounterPacket), captureReader,
                                                  counterIdMap, backendProfilingContext);

    for (unsigned int i = 0; i < numSteps; ++i)
    {
        data.SetCaptureData(1, captureIds1, {});
        captureReader.SetCounterValue(0, valueA * (i + 1));
        captureReader.SetCounterValue(1, valueB * (i + 1));

        periodicCounterCapture.Start();
        periodicCounterCapture.Stop();
    }

    auto buffer = mockBuffer.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(buffer, 0);
    uint32_t headerWord1 = ReadUint32(buffer, 4);

    BOOST_TEST(((headerWord0 >> 26) & 0x0000003F) == 3);    // packet family
    BOOST_TEST(((headerWord0 >> 19) & 0x0000007F) == 0);    // packet class
    BOOST_TEST(((headerWord0 >> 16) & 0x00000007) == 0);    // packet type
    BOOST_TEST(headerWord1 == 20);

    uint32_t offset    = 16;
    uint16_t readIndex = ReadUint16(buffer, offset);
    BOOST_TEST(0 == readIndex);

    offset += 2;
    uint32_t readValue = ReadUint32(buffer, offset);
    BOOST_TEST((valueA * numSteps) == readValue);

    offset += 4;
    readIndex = ReadUint16(buffer, offset);
    BOOST_TEST(1 == readIndex);

    offset += 2;
    readValue = ReadUint32(buffer, offset);
    BOOST_TEST((valueB * numSteps) == readValue);
}

BOOST_AUTO_TEST_CASE(RequestCounterDirectoryCommandHandlerTest1)
{
    const uint32_t familyId = 0;
    const uint32_t packetId = 3;
    const uint32_t version  = 1;
    ProfilingStateMachine profilingStateMachine;
    CounterDirectory counterDirectory;
    MockBufferManager mockBuffer1(1024);
    SendCounterPacket sendCounterPacket(mockBuffer1);
    SendThread sendThread(profilingStateMachine, mockBuffer1, sendCounterPacket);
    MockBufferManager mockBuffer2(1024);
    SendTimelinePacket sendTimelinePacket(mockBuffer2);
    RequestCounterDirectoryCommandHandler commandHandler(familyId, packetId, version, counterDirectory,
                                                         sendCounterPacket, sendTimelinePacket, profilingStateMachine);

    const uint32_t wrongPacketId = 47;
    const uint32_t wrongHeader   = (wrongPacketId & 0x000003FF) << 16;

    arm::pipe::Packet wrongPacket(wrongHeader);

    profilingStateMachine.TransitionToState(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::Active);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::InvalidArgumentException); // Wrong packet

    const uint32_t rightHeader = (packetId & 0x000003FF) << 16;

    arm::pipe::Packet rightPacket(rightHeader);

    BOOST_CHECK_NO_THROW(commandHandler(rightPacket)); // Right packet

    auto readBuffer1 = mockBuffer1.GetReadableBuffer();

    uint32_t header1Word0 = ReadUint32(readBuffer1, 0);
    uint32_t header1Word1 = ReadUint32(readBuffer1, 4);

    // Counter directory packet
    BOOST_TEST(((header1Word0 >> 26) & 0x0000003F) == 0); // packet family
    BOOST_TEST(((header1Word0 >> 16) & 0x000003FF) == 2); // packet id
    BOOST_TEST(header1Word1 == 24);                       // data length

    uint32_t bodyHeader1Word0   = ReadUint32(readBuffer1, 8);
    uint16_t deviceRecordCount = armnn::numeric_cast<uint16_t>(bodyHeader1Word0 >> 16);
    BOOST_TEST(deviceRecordCount == 0); // device_records_count

    auto readBuffer2 = mockBuffer2.GetReadableBuffer();

    uint32_t header2Word0 = ReadUint32(readBuffer2, 0);
    uint32_t header2Word1 = ReadUint32(readBuffer2, 4);

    // Timeline message directory packet
    BOOST_TEST(((header2Word0 >> 26) & 0x0000003F) == 1); // packet family
    BOOST_TEST(((header2Word0 >> 16) & 0x000003FF) == 0); // packet id
    BOOST_TEST(header2Word1 == 443);                      // data length
}

BOOST_AUTO_TEST_CASE(RequestCounterDirectoryCommandHandlerTest2)
{
    const uint32_t familyId = 0;
    const uint32_t packetId = 3;
    const uint32_t version  = 1;
    ProfilingStateMachine profilingStateMachine;
    CounterDirectory counterDirectory;
    MockBufferManager mockBuffer1(1024);
    SendCounterPacket sendCounterPacket(mockBuffer1);
    SendThread sendThread(profilingStateMachine, mockBuffer1, sendCounterPacket);
    MockBufferManager mockBuffer2(1024);
    SendTimelinePacket sendTimelinePacket(mockBuffer2);
    RequestCounterDirectoryCommandHandler commandHandler(familyId, packetId, version, counterDirectory,
                                                         sendCounterPacket, sendTimelinePacket, profilingStateMachine);
    const uint32_t header = (packetId & 0x000003FF) << 16;
    const arm::pipe::Packet packet(header);

    const Device* device = counterDirectory.RegisterDevice("deviceA", 1);
    BOOST_CHECK(device != nullptr);
    const CounterSet* counterSet = counterDirectory.RegisterCounterSet("countersetA");
    BOOST_CHECK(counterSet != nullptr);
    counterDirectory.RegisterCategory("categoryA");
    counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID, 24,
                                     "categoryA", 0, 1, 2.0f, "counterA", "descA");
    counterDirectory.RegisterCounter(armnn::profiling::BACKEND_ID, 25,
                                     "categoryA", 1, 1, 3.0f, "counterB", "descB");

    profilingStateMachine.TransitionToState(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(commandHandler(packet), armnn::RuntimeException);    // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(commandHandler(packet), armnn::RuntimeException);    // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(commandHandler(packet), armnn::RuntimeException);    // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::Active);
    BOOST_CHECK_NO_THROW(commandHandler(packet));

    auto readBuffer1 = mockBuffer1.GetReadableBuffer();

    const uint32_t header1Word0 = ReadUint32(readBuffer1, 0);
    const uint32_t header1Word1 = ReadUint32(readBuffer1, 4);

    BOOST_TEST(((header1Word0 >> 26) & 0x0000003F) == 0); // packet family
    BOOST_TEST(((header1Word0 >> 16) & 0x000003FF) == 2); // packet id
    BOOST_TEST(header1Word1 == 236);                      // data length

    const uint32_t bodyHeaderSizeBytes = bodyHeaderSize * sizeof(uint32_t);

    const uint32_t bodyHeader1Word0      = ReadUint32(readBuffer1, 8);
    const uint32_t bodyHeader1Word1      = ReadUint32(readBuffer1, 12);
    const uint32_t bodyHeader1Word2      = ReadUint32(readBuffer1, 16);
    const uint32_t bodyHeader1Word3      = ReadUint32(readBuffer1, 20);
    const uint32_t bodyHeader1Word4      = ReadUint32(readBuffer1, 24);
    const uint32_t bodyHeader1Word5      = ReadUint32(readBuffer1, 28);
    const uint16_t deviceRecordCount     = armnn::numeric_cast<uint16_t>(bodyHeader1Word0 >> 16);
    const uint16_t counterSetRecordCount = armnn::numeric_cast<uint16_t>(bodyHeader1Word2 >> 16);
    const uint16_t categoryRecordCount   = armnn::numeric_cast<uint16_t>(bodyHeader1Word4 >> 16);
    BOOST_TEST(deviceRecordCount == 1);                      // device_records_count
    BOOST_TEST(bodyHeader1Word1 == 0 + bodyHeaderSizeBytes);      // device_records_pointer_table_offset
    BOOST_TEST(counterSetRecordCount == 1);                  // counter_set_count
    BOOST_TEST(bodyHeader1Word3 == 4 + bodyHeaderSizeBytes);      // counter_set_pointer_table_offset
    BOOST_TEST(categoryRecordCount == 1);                    // categories_count
    BOOST_TEST(bodyHeader1Word5 == 8 + bodyHeaderSizeBytes);      // categories_pointer_table_offset

    const uint32_t deviceRecordOffset = ReadUint32(readBuffer1, 32);
    BOOST_TEST(deviceRecordOffset == 12);

    const uint32_t counterSetRecordOffset = ReadUint32(readBuffer1, 36);
    BOOST_TEST(counterSetRecordOffset == 28);

    const uint32_t categoryRecordOffset = ReadUint32(readBuffer1, 40);
    BOOST_TEST(categoryRecordOffset == 48);

    auto readBuffer2 = mockBuffer2.GetReadableBuffer();

    const uint32_t header2Word0 = ReadUint32(readBuffer2, 0);
    const uint32_t header2Word1 = ReadUint32(readBuffer2, 4);

    // Timeline message directory packet
    BOOST_TEST(((header2Word0 >> 26) & 0x0000003F) == 1); // packet family
    BOOST_TEST(((header2Word0 >> 16) & 0x000003FF) == 0); // packet id
    BOOST_TEST(header2Word1 == 443);                      // data length
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodConnectionAcknowledgedPacket)
{
    unsigned int streamMetadataPacketsize = GetStreamMetaDataPacketSize();

    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "WaitingForAck" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Remove the packets received so far
    mockProfilingConnection->Clear();

    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Wait for the Stream Metadata packet to be sent
    BOOST_CHECK(helper.WaitForPacketsSent(
            mockProfilingConnection, PacketType::StreamMetaData, streamMetadataPacketsize) >= 1);

    // Write a valid "Connection Acknowledged" packet into the mock profiling connection, to simulate a valid
    // reply from an external profiling service

    // Connection Acknowledged Packet header (word 0, word 1 is always zero):
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000001
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 1;
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    // Create the Connection Acknowledged Packet
    arm::pipe::Packet connectionAcknowledgedPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(connectionAcknowledgedPacket));

    // Wait for the counter directory packet to ensure the ConnectionAcknowledgedCommandHandler has run.
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::CounterDirectory) == 1);

    // The Connection Acknowledged Command Handler should have updated the profiling state accordingly
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodRequestCounterDirectoryPacket)
{
    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Force the profiling service to the "Active" state
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Write a valid "Request Counter Directory" packet into the mock profiling connection, to simulate a valid
    // reply from an external profiling service

    // Request Counter Directory packet header (word 0, word 1 is always zero):
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000011
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 3;
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    // Create the Request Counter Directory packet
    arm::pipe::Packet requestCounterDirectoryPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(requestCounterDirectoryPacket));

    // Expecting one CounterDirectory Packet of length 652
    // and one TimelineMessageDirectory packet of length 451
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::CounterDirectory, 652) == 1);
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::TimelineMessageDirectory, 451) == 1);

    // The Request Counter Directory Command Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceBadPeriodicCounterSelectionPacketInvalidCounterUid)
{
    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Force the profiling service to the "Active" state
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Remove the packets received so far
    mockProfilingConnection->Clear();

    // Write a "Periodic Counter Selection" packet into the mock profiling connection, to simulate an input from an
    // external profiling service

    // Periodic Counter Selection packet header:
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000100
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 4;
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    uint32_t capturePeriod = 123456;    // Some capture period (microseconds)

    // Get the first valid counter UID
    const ICounterDirectory& counterDirectory = profilingService.GetCounterDirectory();
    const Counters& counters                  = counterDirectory.GetCounters();
    BOOST_CHECK(counters.size() > 1);
    uint16_t counterUidA = counters.begin()->first;    // First valid counter UID
    uint16_t counterUidB = 9999;                       // Second invalid counter UID

    uint32_t length = 8;

    auto data = std::make_unique<unsigned char[]>(length);
    WriteUint32(data.get(), 0, capturePeriod);
    WriteUint16(data.get(), 4, counterUidA);
    WriteUint16(data.get(), 6, counterUidB);

    // Create the Periodic Counter Selection packet
    // Length > 0, this will start the Period Counter Capture thread
    arm::pipe::Packet periodicCounterSelectionPacket(header, length, data);


    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(periodicCounterSelectionPacket));

    // Expecting one Periodic Counter Selection packet of length 14
    // and at least one Periodic Counter Capture packet of length 22
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterSelection, 14) == 1);
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterCapture, 22) >= 1);

    // The Periodic Counter Selection Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodPeriodicCounterSelectionPacketNoCounters)
{
    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Wait for the Stream Metadata packet the be sent
    // (we are not testing the connection acknowledgement here so it will be ignored by this test)
    helper.WaitForPacketsSent(mockProfilingConnection, PacketType::StreamMetaData);

    // Force the profiling service to the "Active" state
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Write a "Periodic Counter Selection" packet into the mock profiling connection, to simulate an input from an
    // external profiling service

    // Periodic Counter Selection packet header:
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000100
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 4;
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    // Create the Periodic Counter Selection packet
    // Length == 0, this will disable the collection of counters
    arm::pipe::Packet periodicCounterSelectionPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(periodicCounterSelectionPacket));

    // Wait for the Periodic Counter Selection packet of length 12 to be sent
    // The size of the expected Periodic Counter Selection (echos the sent one)
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterSelection, 12) == 1);

    // The Periodic Counter Selection Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // No Periodic Counter packets are expected
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterCapture, 0, 0) == 0);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodPeriodicCounterSelectionPacketSingleCounter)
{
    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Wait for the Stream Metadata packet to be sent
    // (we are not testing the connection acknowledgement here so it will be ignored by this test)
    helper.WaitForPacketsSent(mockProfilingConnection, PacketType::StreamMetaData);

    // Force the profiling service to the "Active" state
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Write a "Periodic Counter Selection" packet into the mock profiling connection, to simulate an input from an
    // external profiling service

    // Periodic Counter Selection packet header:
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000100
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 4;
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    uint32_t capturePeriod = 123456;    // Some capture period (microseconds)

    // Get the first valid counter UID
    const ICounterDirectory& counterDirectory = profilingService.GetCounterDirectory();
    const Counters& counters                  = counterDirectory.GetCounters();
    BOOST_CHECK(!counters.empty());
    uint16_t counterUid = counters.begin()->first;    // Valid counter UID

    uint32_t length = 6;

    auto data = std::make_unique<unsigned char[]>(length);
    WriteUint32(data.get(), 0, capturePeriod);
    WriteUint16(data.get(), 4, counterUid);

    // Create the Periodic Counter Selection packet
    // Length > 0, this will start the Period Counter Capture thread
    arm::pipe::Packet periodicCounterSelectionPacket(header, length, data);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(periodicCounterSelectionPacket));

    // Expecting one Periodic Counter Selection packet of length 14
    // and at least one Periodic Counter Capture packet of length 22
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterSelection, 14) == 1);
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterCapture, 22) >= 1);

    // The Periodic Counter Selection Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodPeriodicCounterSelectionPacketMultipleCounters)
{
    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Wait for the Stream Metadata packet the be sent
    // (we are not testing the connection acknowledgement here so it will be ignored by this test)
    helper.WaitForPacketsSent(mockProfilingConnection, PacketType::StreamMetaData);

    // Force the profiling service to the "Active" state
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Write a "Periodic Counter Selection" packet into the mock profiling connection, to simulate an input from an
    // external profiling service

    // Periodic Counter Selection packet header:
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000100
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 4;
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    uint32_t capturePeriod = 123456;    // Some capture period (microseconds)

    // Get the first valid counter UID
    const ICounterDirectory& counterDirectory = profilingService.GetCounterDirectory();
    const Counters& counters                  = counterDirectory.GetCounters();
    BOOST_CHECK(counters.size() > 1);
    uint16_t counterUidA = counters.begin()->first;        // First valid counter UID
    uint16_t counterUidB = (counters.begin()++)->first;    // Second valid counter UID

    uint32_t length = 8;

    auto data = std::make_unique<unsigned char[]>(length);
    WriteUint32(data.get(), 0, capturePeriod);
    WriteUint16(data.get(), 4, counterUidA);
    WriteUint16(data.get(), 6, counterUidB);

    // Create the Periodic Counter Selection packet
    // Length > 0, this will start the Period Counter Capture thread
    arm::pipe::Packet periodicCounterSelectionPacket(header, length, data);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(periodicCounterSelectionPacket));

    // Expecting one PeriodicCounterSelection Packet with a length of 16
    // And at least one PeriodicCounterCapture Packet with a length of 28
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterSelection, 16) == 1);
    BOOST_CHECK(helper.WaitForPacketsSent(mockProfilingConnection, PacketType::PeriodicCounterCapture, 28) >= 1);

    // The Periodic Counter Selection Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceDisconnect)
{
    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Try to disconnect the profiling service while in the "Uninitialised" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Disconnect();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);    // The state should not change

    // Try to disconnect the profiling service while in the "NotConnected" state
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Disconnect();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);    // The state should not change

    // Try to disconnect the profiling service while in the "WaitingForAck" state
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Disconnect();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);    // The state should not change

    // Try to disconnect the profiling service while in the "Active" state
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Wait for the Stream Metadata packet the be sent
    // (we are not testing the connection acknowledgement here so it will be ignored by this test)
    helper.WaitForPacketsSent(mockProfilingConnection, PacketType::StreamMetaData);

    // Force the profiling service to the "Active" state
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Check that the profiling connection is open
    BOOST_CHECK(mockProfilingConnection->IsOpen());

    profilingService.Disconnect();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);   // The state should have changed

    // Check that the profiling connection has been reset
    mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection == nullptr);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodPerJobCounterSelectionPacket)
{
    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Wait for the Stream Metadata packet the be sent
    // (we are not testing the connection acknowledgement here so it will be ignored by this test)
    helper.WaitForPacketsSent(mockProfilingConnection, PacketType::StreamMetaData);

    // Force the profiling service to the "Active" state
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Write a "Per-Job Counter Selection" packet into the mock profiling connection, to simulate an input from an
    // external profiling service

    // Per-Job Counter Selection packet header:
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000100
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 5;
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    // Create the Per-Job Counter Selection packet
    // Length == 0, this will disable the collection of counters
    arm::pipe::Packet periodicCounterSelectionPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(periodicCounterSelectionPacket));

    // Wait for a bit (must at least be the delay value of the mock profiling connection) to make sure that
    // the Per-Job Counter Selection packet gets processed by the profiling service
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // The Per-Job Counter Selection Command Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // The Per-Job Counter Selection packets are dropped silently, so there should be no reply coming
    // from the profiling service
    const auto StreamMetaDataSize = static_cast<unsigned long>(
            helper.WaitForPacketsSent(mockProfilingConnection, PacketType::StreamMetaData, 0, 0));
    BOOST_CHECK(StreamMetaDataSize == mockProfilingConnection->GetWrittenDataSize());

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckConfigureProfilingServiceOn)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.ConfigureProfilingService(options);
    // should get as far as NOT_CONNECTED
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckConfigureProfilingServiceOff)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    armnn::profiling::ProfilingService profilingService;
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.ConfigureProfilingService(options);
    // should not move from Uninitialised
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceEnabled)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);

    // Redirect the output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());
    profilingService.Update();

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    streamRedirector.CancelRedirect();

    // Check that the expected error has occurred and logged to the standard output
    if (ss.str().find("Cannot connect to stream socket: Connection refused") == std::string::npos)
    {
        std::cout << ss.str();
        BOOST_FAIL("Expected string not found.");
    }
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceEnabledRuntime)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    options.m_EnableProfiling = true;
    profilingService.ResetExternalProfilingOptions(options);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);

    // Redirect the output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());
    profilingService.Update();

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    streamRedirector.CancelRedirect();

    // Check that the expected error has occurred and logged to the standard output
    if (ss.str().find("Cannot connect to stream socket: Connection refused") == std::string::npos)
    {
        std::cout << ss.str();
        BOOST_FAIL("Expected string not found.");
    }
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceBadConnectionAcknowledgedPacket)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);


    // Redirect the standard output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());

    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "WaitingForAck" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);

    // Connection Acknowledged Packet header (word 0, word 1 is always zero):
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000001
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 37;    // Wrong packet id!!!
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    // Create the Connection Acknowledged Packet
    arm::pipe::Packet connectionAcknowledgedPacket(header);
    // Write an invalid "Connection Acknowledged" packet into the mock profiling connection, to simulate an invalid
    // reply from an external profiling service
    mockProfilingConnection->WritePacket(std::move(connectionAcknowledgedPacket));

    // Start the command thread
    profilingService.Update();

    // Wait for the command thread to join
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    streamRedirector.CancelRedirect();

    // Check that the expected error has occurred and logged to the standard output
    if (ss.str().find("Functor with requested PacketId=37 and Version=4194304 does not exist") == std::string::npos)
    {
        std::cout << ss.str();
        BOOST_FAIL("Expected string not found.");
    }
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceBadRequestCounterDirectoryPacket)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);

    // Redirect the standard output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());

    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    helper.ForceTransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Write a valid "Request Counter Directory" packet into the mock profiling connection, to simulate a valid
    // reply from an external profiling service

    // Request Counter Directory packet header (word 0, word 1 is always zero):
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000011
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 123;    // Wrong packet id!!!
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    // Create the Request Counter Directory packet
    arm::pipe::Packet requestCounterDirectoryPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(requestCounterDirectoryPacket));

    // Start the command handler and the send thread
    profilingService.Update();

    // Reset the profiling service to stop and join any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    streamRedirector.CancelRedirect();

    // Check that the expected error has occurred and logged to the standard output
    if (ss.str().find("Functor with requested PacketId=123 and Version=4194304 does not exist") == std::string::npos)
    {
        std::cout << ss.str();
        BOOST_FAIL("Expected string not found.");
    }
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceBadPeriodicCounterSelectionPacket)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);

    // Redirect the standard output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());

    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper(profilingService);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();    // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();    // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();    // Start the command handler and the send thread

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Write a "Periodic Counter Selection" packet into the mock profiling connection, to simulate an input from an
    // external profiling service

    // Periodic Counter Selection packet header:
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000100
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 999;    // Wrong packet id!!!
    uint32_t header       = ((packetFamily & 0x0000003F) << 26) | ((packetId & 0x000003FF) << 16);

    // Create the Periodic Counter Selection packet
    // Length == 0, this will disable the collection of counters
    arm::pipe::Packet periodicCounterSelectionPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(periodicCounterSelectionPacket));
    profilingService.Update();

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);

    // Check that the expected error has occurred and logged to the standard output
    streamRedirector.CancelRedirect();

    // Check that the expected error has occurred and logged to the standard output
    if (ss.str().find("Functor with requested PacketId=999 and Version=4194304 does not exist") == std::string::npos)
    {
        std::cout << ss.str();
        BOOST_FAIL("Expected string not found.");
    }
}

BOOST_AUTO_TEST_CASE(CheckCounterIdMap)
{
    CounterIdMap counterIdMap;
    BOOST_CHECK_THROW(counterIdMap.GetBackendId(0), armnn::Exception);
    BOOST_CHECK_THROW(counterIdMap.GetGlobalId(0, armnn::profiling::BACKEND_ID), armnn::Exception);

    uint16_t globalCounterIds = 0;

    armnn::BackendId cpuRefId(armnn::Compute::CpuRef);
    armnn::BackendId cpuAccId(armnn::Compute::CpuAcc);

    std::vector<uint16_t> cpuRefCounters = {0, 1, 2, 3};
    std::vector<uint16_t> cpuAccCounters = {0, 1};

    for (uint16_t backendCounterId : cpuRefCounters)
    {
        counterIdMap.RegisterMapping(globalCounterIds, backendCounterId, cpuRefId);
        ++globalCounterIds;
    }
    for (uint16_t backendCounterId : cpuAccCounters)
    {
        counterIdMap.RegisterMapping(globalCounterIds, backendCounterId, cpuAccId);
        ++globalCounterIds;
    }

    BOOST_CHECK(counterIdMap.GetBackendId(0) == (std::pair<uint16_t, armnn::BackendId>(0, cpuRefId)));
    BOOST_CHECK(counterIdMap.GetBackendId(1) == (std::pair<uint16_t, armnn::BackendId>(1, cpuRefId)));
    BOOST_CHECK(counterIdMap.GetBackendId(2) == (std::pair<uint16_t, armnn::BackendId>(2, cpuRefId)));
    BOOST_CHECK(counterIdMap.GetBackendId(3) == (std::pair<uint16_t, armnn::BackendId>(3, cpuRefId)));
    BOOST_CHECK(counterIdMap.GetBackendId(4) == (std::pair<uint16_t, armnn::BackendId>(0, cpuAccId)));
    BOOST_CHECK(counterIdMap.GetBackendId(5) == (std::pair<uint16_t, armnn::BackendId>(1, cpuAccId)));

    BOOST_CHECK(counterIdMap.GetGlobalId(0, cpuRefId) == 0);
    BOOST_CHECK(counterIdMap.GetGlobalId(1, cpuRefId) == 1);
    BOOST_CHECK(counterIdMap.GetGlobalId(2, cpuRefId) == 2);
    BOOST_CHECK(counterIdMap.GetGlobalId(3, cpuRefId) == 3);
    BOOST_CHECK(counterIdMap.GetGlobalId(0, cpuAccId) == 4);
    BOOST_CHECK(counterIdMap.GetGlobalId(1, cpuAccId) == 5);
}

BOOST_AUTO_TEST_CASE(CheckRegisterBackendCounters)
{
    uint16_t globalCounterIds = armnn::profiling::INFERENCES_RUN;
    armnn::BackendId cpuRefId(armnn::Compute::CpuRef);

    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling          = true;
    ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    RegisterBackendCounters registerBackendCounters(globalCounterIds, cpuRefId, profilingService);



    BOOST_CHECK(profilingService.GetCounterDirectory().GetCategories().empty());
    registerBackendCounters.RegisterCategory("categoryOne");
    auto categoryOnePtr = profilingService.GetCounterDirectory().GetCategory("categoryOne");
    BOOST_CHECK(categoryOnePtr);

    BOOST_CHECK(profilingService.GetCounterDirectory().GetDevices().empty());
    globalCounterIds = registerBackendCounters.RegisterDevice("deviceOne");
    auto deviceOnePtr = profilingService.GetCounterDirectory().GetDevice(globalCounterIds);
    BOOST_CHECK(deviceOnePtr);
    BOOST_CHECK(deviceOnePtr->m_Name == "deviceOne");

    BOOST_CHECK(profilingService.GetCounterDirectory().GetCounterSets().empty());
    globalCounterIds = registerBackendCounters.RegisterCounterSet("counterSetOne");
    auto counterSetOnePtr = profilingService.GetCounterDirectory().GetCounterSet(globalCounterIds);
    BOOST_CHECK(counterSetOnePtr);
    BOOST_CHECK(counterSetOnePtr->m_Name == "counterSetOne");

    uint16_t newGlobalCounterId = registerBackendCounters.RegisterCounter(0,
                                                                          "categoryOne",
                                                                          0,
                                                                          0,
                                                                          1.f,
                                                                          "CounterOne",
                                                                          "first test counter");
    BOOST_CHECK(newGlobalCounterId = armnn::profiling::INFERENCES_RUN + 1);
    uint16_t mappedGlobalId = profilingService.GetCounterMappings().GetGlobalId(0, cpuRefId);
    BOOST_CHECK(mappedGlobalId == newGlobalCounterId);
    auto backendMapping = profilingService.GetCounterMappings().GetBackendId(newGlobalCounterId);
    BOOST_CHECK(backendMapping.first == 0);
    BOOST_CHECK(backendMapping.second == cpuRefId);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckCounterStatusQuery)
{
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;

    // Reset the profiling service to the uninitialized state
    ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options.m_ProfilingOptions, true);

    const armnn::BackendId cpuRefId(armnn::Compute::CpuRef);
    const armnn::BackendId cpuAccId(armnn::Compute::CpuAcc);

    // Create BackendProfiling for each backend
    BackendProfiling backendProfilingCpuRef(options, profilingService, cpuRefId);
    BackendProfiling backendProfilingCpuAcc(options, profilingService, cpuAccId);

    uint16_t initialNumGlobalCounterIds = armnn::profiling::INFERENCES_RUN;

    // Create RegisterBackendCounters for CpuRef
    RegisterBackendCounters registerBackendCountersCpuRef(initialNumGlobalCounterIds, cpuRefId, profilingService);

    // Create 'testCategory' in CounterDirectory (backend agnostic)
    BOOST_CHECK(profilingService.GetCounterDirectory().GetCategories().empty());
    registerBackendCountersCpuRef.RegisterCategory("testCategory");
    auto categoryOnePtr = profilingService.GetCounterDirectory().GetCategory("testCategory");
    BOOST_CHECK(categoryOnePtr);

    // Counters:
    // Global | Local | Backend
    //    5   |   0   | CpuRef
    //    6   |   1   | CpuRef
    //    7   |   1   | CpuAcc

    std::vector<uint16_t> cpuRefCounters = {0, 1};
    std::vector<uint16_t> cpuAccCounters = {0};

    // Register the backend counters for CpuRef and validate GetGlobalId and GetBackendId
    uint16_t currentNumGlobalCounterIds = registerBackendCountersCpuRef.RegisterCounter(
            0, "testCategory", 0, 0, 1.f, "CpuRefCounter0", "Zeroth CpuRef Counter");
    BOOST_CHECK(currentNumGlobalCounterIds == initialNumGlobalCounterIds + 1);
    uint16_t mappedGlobalId = profilingService.GetCounterMappings().GetGlobalId(0, cpuRefId);
    BOOST_CHECK(mappedGlobalId == currentNumGlobalCounterIds);
    auto backendMapping = profilingService.GetCounterMappings().GetBackendId(currentNumGlobalCounterIds);
    BOOST_CHECK(backendMapping.first == 0);
    BOOST_CHECK(backendMapping.second == cpuRefId);

    currentNumGlobalCounterIds = registerBackendCountersCpuRef.RegisterCounter(
            1, "testCategory", 0, 0, 1.f, "CpuRefCounter1", "First CpuRef Counter");
    BOOST_CHECK(currentNumGlobalCounterIds == initialNumGlobalCounterIds + 2);
    mappedGlobalId = profilingService.GetCounterMappings().GetGlobalId(1, cpuRefId);
    BOOST_CHECK(mappedGlobalId == currentNumGlobalCounterIds);
    backendMapping = profilingService.GetCounterMappings().GetBackendId(currentNumGlobalCounterIds);
    BOOST_CHECK(backendMapping.first == 1);
    BOOST_CHECK(backendMapping.second == cpuRefId);

    // Create RegisterBackendCounters for CpuAcc
    RegisterBackendCounters registerBackendCountersCpuAcc(currentNumGlobalCounterIds, cpuAccId, profilingService);

    // Register the backend counter for CpuAcc and validate GetGlobalId and GetBackendId
    currentNumGlobalCounterIds = registerBackendCountersCpuAcc.RegisterCounter(
            0, "testCategory", 0, 0, 1.f, "CpuAccCounter0", "Zeroth CpuAcc Counter");
    BOOST_CHECK(currentNumGlobalCounterIds == initialNumGlobalCounterIds + 3);
    mappedGlobalId = profilingService.GetCounterMappings().GetGlobalId(0, cpuAccId);
    BOOST_CHECK(mappedGlobalId == currentNumGlobalCounterIds);
    backendMapping = profilingService.GetCounterMappings().GetBackendId(currentNumGlobalCounterIds);
    BOOST_CHECK(backendMapping.first == 0);
    BOOST_CHECK(backendMapping.second == cpuAccId);

    // Create vectors for active counters
    const std::vector<uint16_t> activeGlobalCounterIds = {5}; // CpuRef(0) activated
    const std::vector<uint16_t> newActiveGlobalCounterIds = {6, 7}; // CpuRef(0) and CpuAcc(1) activated

    const uint32_t capturePeriod = 200;
    const uint32_t newCapturePeriod = 100;

    // Set capture period and active counters in CaptureData
    profilingService.SetCaptureData(capturePeriod, activeGlobalCounterIds, {});

    // Get vector of active counters for CpuRef and CpuAcc backends
    std::vector<CounterStatus> cpuRefCounterStatus = backendProfilingCpuRef.GetActiveCounters();
    std::vector<CounterStatus> cpuAccCounterStatus = backendProfilingCpuAcc.GetActiveCounters();
    BOOST_CHECK_EQUAL(cpuRefCounterStatus.size(), 1);
    BOOST_CHECK_EQUAL(cpuAccCounterStatus.size(), 0);

    // Check active CpuRef counter
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_GlobalCounterId, activeGlobalCounterIds[0]);
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_BackendCounterId, cpuRefCounters[0]);
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_SamplingRateInMicroseconds, capturePeriod);
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_Enabled, true);

    // Check inactive CpuRef counter
    CounterStatus inactiveCpuRefCounter = backendProfilingCpuRef.GetCounterStatus(cpuRefCounters[1]);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_GlobalCounterId, 6);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_BackendCounterId, cpuRefCounters[1]);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_SamplingRateInMicroseconds, 0);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_Enabled, false);

    // Check inactive CpuAcc counter
    CounterStatus inactiveCpuAccCounter = backendProfilingCpuAcc.GetCounterStatus(cpuAccCounters[0]);
    BOOST_CHECK_EQUAL(inactiveCpuAccCounter.m_GlobalCounterId, 7);
    BOOST_CHECK_EQUAL(inactiveCpuAccCounter.m_BackendCounterId, cpuAccCounters[0]);
    BOOST_CHECK_EQUAL(inactiveCpuAccCounter.m_SamplingRateInMicroseconds, 0);
    BOOST_CHECK_EQUAL(inactiveCpuAccCounter.m_Enabled, false);

    // Set new capture period and new active counters in CaptureData
    profilingService.SetCaptureData(newCapturePeriod, newActiveGlobalCounterIds, {});

    // Get vector of active counters for CpuRef and CpuAcc backends
    cpuRefCounterStatus = backendProfilingCpuRef.GetActiveCounters();
    cpuAccCounterStatus = backendProfilingCpuAcc.GetActiveCounters();
    BOOST_CHECK_EQUAL(cpuRefCounterStatus.size(), 1);
    BOOST_CHECK_EQUAL(cpuAccCounterStatus.size(), 1);

    // Check active CpuRef counter
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_GlobalCounterId, newActiveGlobalCounterIds[0]);
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_BackendCounterId, cpuRefCounters[1]);
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_SamplingRateInMicroseconds, newCapturePeriod);
    BOOST_CHECK_EQUAL(cpuRefCounterStatus[0].m_Enabled, true);

    // Check active CpuAcc counter
    BOOST_CHECK_EQUAL(cpuAccCounterStatus[0].m_GlobalCounterId, newActiveGlobalCounterIds[1]);
    BOOST_CHECK_EQUAL(cpuAccCounterStatus[0].m_BackendCounterId, cpuAccCounters[0]);
    BOOST_CHECK_EQUAL(cpuAccCounterStatus[0].m_SamplingRateInMicroseconds, newCapturePeriod);
    BOOST_CHECK_EQUAL(cpuAccCounterStatus[0].m_Enabled, true);

    // Check inactive CpuRef counter
    inactiveCpuRefCounter = backendProfilingCpuRef.GetCounterStatus(cpuRefCounters[0]);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_GlobalCounterId, 5);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_BackendCounterId, cpuRefCounters[0]);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_SamplingRateInMicroseconds, 0);
    BOOST_CHECK_EQUAL(inactiveCpuRefCounter.m_Enabled, false);

    // Reset the profiling service to stop any running thread
    options.m_ProfilingOptions.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options.m_ProfilingOptions, true);
}

BOOST_AUTO_TEST_CASE(CheckRegisterCounters)
{
    armnn::Runtime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
    MockBufferManager mockBuffer(1024);

    CaptureData captureData;
    MockProfilingService mockProfilingService(mockBuffer, options.m_ProfilingOptions.m_EnableProfiling, captureData);
    armnn::BackendId cpuRefId(armnn::Compute::CpuRef);

    mockProfilingService.RegisterMapping(6, 0, cpuRefId);
    mockProfilingService.RegisterMapping(7, 1, cpuRefId);
    mockProfilingService.RegisterMapping(8, 2, cpuRefId);

    armnn::profiling::BackendProfiling backendProfiling(options,
                                                        mockProfilingService,
                                                        cpuRefId);

    armnn::profiling::Timestamp timestamp;
    timestamp.timestamp = 1000998;
    timestamp.counterValues.emplace_back(0, 700);
    timestamp.counterValues.emplace_back(2, 93);
    std::vector<armnn::profiling::Timestamp> timestamps;
    timestamps.push_back(timestamp);
    backendProfiling.ReportCounters(timestamps);

    auto readBuffer = mockBuffer.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(readBuffer, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer, 4);
    uint64_t readTimestamp = ReadUint64(readBuffer, 8);

    BOOST_TEST(((headerWord0 >> 26) & 0x0000003F) == 3); // packet family
    BOOST_TEST(((headerWord0 >> 19) & 0x0000007F) == 0); // packet class
    BOOST_TEST(((headerWord0 >> 16) & 0x00000007) == 0); // packet type
    BOOST_TEST(headerWord1 == 20);                       // data length
    BOOST_TEST(1000998 == readTimestamp);                // capture period

    uint32_t offset = 16;
    // Check Counter Index
    uint16_t readIndex = ReadUint16(readBuffer, offset);
    BOOST_TEST(6 == readIndex);

    // Check Counter Value
    offset += 2;
    uint32_t readValue = ReadUint32(readBuffer, offset);
    BOOST_TEST(700 == readValue);

    // Check Counter Index
    offset += 4;
    readIndex = ReadUint16(readBuffer, offset);
    BOOST_TEST(8 == readIndex);

    // Check Counter Value
    offset += 2;
    readValue = ReadUint32(readBuffer, offset);
    BOOST_TEST(93 == readValue);
}

BOOST_AUTO_TEST_CASE(CheckFileFormat) {
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);

    // Create profiling options.
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    // Check the default value set to binary
    BOOST_CHECK(options.m_FileFormat == "binary");

    // Change file format to an unsupported value
    options.m_FileFormat = "json";
    // Enable the profiling service
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);
    // Start the command handler and the send thread
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState()==ProfilingState::NotConnected);

    // Redirect the output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());

    // When Update is called and the current state is ProfilingState::NotConnected
    // an exception will be raised from GetProfilingConnection and displayed as warning in the output local stream
    profilingService.Update();

    streamRedirector.CancelRedirect();

    // Check that the expected error has occurred and logged to the standard output
    if (ss.str().find("Unsupported profiling file format, only binary is supported") == std::string::npos)
    {
        std::cout << ss.str();
        BOOST_FAIL("Expected string not found.");
    }
}

BOOST_AUTO_TEST_SUITE_END()
