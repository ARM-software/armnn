//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingTests.hpp"

#include <CommandHandler.hpp>
#include <CommandHandlerKey.hpp>
#include <CommandHandlerRegistry.hpp>
#include <ConnectionAcknowledgedCommandHandler.hpp>
#include <CounterDirectory.hpp>
#include <EncodeVersion.hpp>
#include <Holder.hpp>
#include <ICounterValues.hpp>
#include <Packet.hpp>
#include <PacketVersionResolver.hpp>
#include <PeriodicCounterCapture.hpp>
#include <PeriodicCounterSelectionCommandHandler.hpp>
#include <ProfilingStateMachine.hpp>
#include <ProfilingUtils.hpp>
#include <RequestCounterDirectoryCommandHandler.hpp>
#include <Runtime.hpp>
#include <SocketProfilingConnection.hpp>

#include <armnn/Conversion.hpp>

#include <armnn/Utils.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <random>

using namespace armnn::profiling;

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

BOOST_AUTO_TEST_CASE(CheckCommandHandler)
{
    PacketVersionResolver packetVersionResolver;
    ProfilingStateMachine profilingStateMachine;

    TestProfilingConnectionBase testProfilingConnectionBase;
    TestProfilingConnectionTimeoutError testProfilingConnectionTimeOutError;
    TestProfilingConnectionArmnnError testProfilingConnectionArmnnError;

    ConnectionAcknowledgedCommandHandler connectionAcknowledgedCommandHandler(1, 4194304, profilingStateMachine);
    CommandHandlerRegistry commandHandlerRegistry;

    commandHandlerRegistry.RegisterFunctor(&connectionAcknowledgedCommandHandler);

    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);

    CommandHandler commandHandler0(1,
                                   true,
                                   commandHandlerRegistry,
                                   packetVersionResolver);

    commandHandler0.Start(testProfilingConnectionBase);
    commandHandler0.Start(testProfilingConnectionBase);
    commandHandler0.Start(testProfilingConnectionBase);

    commandHandler0.Stop();

    BOOST_CHECK(profilingStateMachine.GetCurrentState() == ProfilingState::Active);

    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
    // commandHandler1 should give up after one timeout
    CommandHandler commandHandler1(10,
                                   true,
                                   commandHandlerRegistry,
                                   packetVersionResolver);

    commandHandler1.Start(testProfilingConnectionTimeOutError);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    BOOST_CHECK(!commandHandler1.IsRunning());
    commandHandler1.Stop();

    BOOST_CHECK(profilingStateMachine.GetCurrentState() == ProfilingState::WaitingForAck);
    // Now commandHandler1 should persist after a timeout
    commandHandler1.SetStopAfterTimeout(false);
    commandHandler1.Start(testProfilingConnectionTimeOutError);

    for (int i = 0; i < 100; i++)
    {
        if (profilingStateMachine.GetCurrentState() == ProfilingState::Active)
        {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    commandHandler1.Stop();

    BOOST_CHECK(profilingStateMachine.GetCurrentState() == ProfilingState::Active);

    CommandHandler commandHandler2(100,
                                   false,
                                   commandHandlerRegistry,
                                   packetVersionResolver);

    commandHandler2.Start(testProfilingConnectionArmnnError);

    // commandHandler2 should not stop once it encounters a non timing error
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    BOOST_CHECK(commandHandler2.IsRunning());
    commandHandler2.Stop();
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
    uint32_t length = 4;
    std::unique_ptr<char[]> packetData0 = std::make_unique<char[]>(length);
    std::unique_ptr<char[]> packetData1 = std::make_unique<char[]>(0);
    std::unique_ptr<char[]> nullPacketData;

    Packet packetTest0(472580096, length, packetData0);

    BOOST_CHECK(packetTest0.GetHeader() == 472580096);
    BOOST_CHECK(packetTest0.GetPacketFamily() == 7);
    BOOST_CHECK(packetTest0.GetPacketId() == 43);
    BOOST_CHECK(packetTest0.GetLength() == length);
    BOOST_CHECK(packetTest0.GetPacketType() == 3);
    BOOST_CHECK(packetTest0.GetPacketClass() == 5);

    BOOST_CHECK_THROW(Packet packetTest1(472580096, 0, packetData1), armnn::Exception);
    BOOST_CHECK_NO_THROW(Packet packetTest2(472580096, 0, nullPacketData));

    Packet packetTest3(472580096, 0, nullPacketData);
    BOOST_CHECK(packetTest3.GetLength() == 0);
    BOOST_CHECK(packetTest3.GetData() == nullptr);

    const char* packetTest0Data = packetTest0.GetData();
    Packet packetTest4(std::move(packetTest0));

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

    std::unique_ptr<char[]> packetDataA;
    std::unique_ptr<char[]> packetDataB;
    std::unique_ptr<char[]> packetDataC;

    Packet packetA(500000000, 0, packetDataA);
    Packet packetB(600000000, 0, packetDataB);
    Packet packetC(400000000, 0, packetDataC);

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
    registry.RegisterFunctor(&testFunctorA);
    registry.RegisterFunctor(&testFunctorB);
    registry.RegisterFunctor(&testFunctorC);

    std::unique_ptr<char[]> packetDataA;
    std::unique_ptr<char[]> packetDataB;
    std::unique_ptr<char[]> packetDataC;

    Packet packetA(500000000, 0, packetDataA);
    Packet packetB(600000000, 0, packetDataB);
    Packet packetC(400000000, 0, packetDataC);

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
    BOOST_CHECK(profilingState1.GetCurrentState() ==  ProfilingState::Uninitialised);

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
    BOOST_CHECK_THROW(profilingState9.TransitionToState(ProfilingState::WaitingForAck),
                      armnn::Exception);

    ProfilingStateMachine profilingState10(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(profilingState10.TransitionToState(ProfilingState::Active),
                      armnn::Exception);

    ProfilingStateMachine profilingState11(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(profilingState11.TransitionToState(ProfilingState::Uninitialised),
                      armnn::Exception);

    ProfilingStateMachine profilingState12(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(profilingState12.TransitionToState(ProfilingState::Active),
                      armnn::Exception);

    ProfilingStateMachine profilingState13(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(profilingState13.TransitionToState(ProfilingState::Uninitialised),
                      armnn::Exception);

    ProfilingStateMachine profilingState14(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(profilingState14.TransitionToState(ProfilingState::NotConnected),
                      armnn::Exception);

    ProfilingStateMachine profilingState15(ProfilingState::Active);
    BOOST_CHECK_THROW(profilingState15.TransitionToState(ProfilingState::Uninitialised),
                      armnn::Exception);

    ProfilingStateMachine profilingState16(armnn::profiling::ProfilingState::Active);
    BOOST_CHECK_THROW(profilingState16.TransitionToState(ProfilingState::WaitingForAck),
                      armnn::Exception);

    ProfilingStateMachine profilingState17(ProfilingState::Uninitialised);

    std::thread thread1 (ProfilingCurrentStateThreadImpl,std::ref(profilingState17));
    std::thread thread2 (ProfilingCurrentStateThreadImpl,std::ref(profilingState17));
    std::thread thread3 (ProfilingCurrentStateThreadImpl,std::ref(profilingState17));
    std::thread thread4 (ProfilingCurrentStateThreadImpl,std::ref(profilingState17));
    std::thread thread5 (ProfilingCurrentStateThreadImpl,std::ref(profilingState17));

    thread1.join();
    thread2.join();
    thread3.join();
    thread4.join();
    thread5.join();

    BOOST_TEST((profilingState17.GetCurrentState() == ProfilingState::NotConnected));
}

void CaptureDataWriteThreadImpl(Holder& holder, uint32_t capturePeriod, const std::vector<uint16_t>& counterIds)
{
    holder.SetCaptureData(capturePeriod, counterIds);
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
        threadsVect.emplace_back(std::thread(CaptureDataWriteThreadImpl,
                                 std::ref(holder),
                                 i,
                                 std::ref(periodIdMap[i])));

        // Verify that the CaptureData goes into the thread in a virgin state
        BOOST_CHECK(captureDataIdMap.at(i).GetCapturePeriod() == 0);
        BOOST_CHECK(captureDataIdMap.at(i).GetCounterIds().empty());
        readThreadsVect.emplace_back(std::thread(CaptureDataReadThreadImpl,
                                     std::ref(holder),
                                     std::ref(captureDataIdMap.at(i))));
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
    std::vector<uint16_t> counterIds = {42, 29, 13};
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
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceEnabled)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);

    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);

    // Redirect the output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());
    profilingService.Update();
    BOOST_CHECK(boost::contains(ss.str(), "Cannot connect to stream socket: Connection refused"));
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceEnabledRuntime)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);

    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    ProfilingService& profilingService = ProfilingService::Instance();
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
    BOOST_CHECK(boost::contains(ss.str(), "Cannot connect to stream socket: Connection refused"));
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceCounterDirectory)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    ProfilingService& profilingService = ProfilingService::Instance();
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
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceCounterValues)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);

    profilingService.Update();
    const ICounterDirectory& counterDirectory = profilingService.GetCounterDirectory();
    const Counters& counters = counterDirectory.GetCounters();
    BOOST_CHECK(!counters.empty());

    // Get the UID of the first counter for testing
    uint16_t counterUid = counters.begin()->first;

    ProfilingService* profilingServicePtr = &profilingService;
    std::vector<std::thread> writers;

    for (int i = 0; i < 100 ; ++i)
    {
        // Increment and decrement the first counter
        writers.push_back(std::thread(&ProfilingService::IncrementCounterValue, profilingServicePtr, counterUid));
        writers.push_back(std::thread(&ProfilingService::DecrementCounterValue, profilingServicePtr, counterUid));
        // Add 10 and subtract 5 from the first counter
        writers.push_back(std::thread(&ProfilingService::AddCounterValue,      profilingServicePtr, counterUid, 10));
        writers.push_back(std::thread(&ProfilingService::SubtractCounterValue, profilingServicePtr, counterUid, 5));
    }

    std::for_each(writers.begin(), writers.end(), mem_fn(&std::thread::join));

    uint32_t counterValue = 0;
    BOOST_CHECK_NO_THROW(counterValue = profilingService.GetCounterValue(counterUid));
    BOOST_CHECK(counterValue == 500);

    BOOST_CHECK_NO_THROW(profilingService.SetCounterValue(counterUid, 0));
    BOOST_CHECK_NO_THROW(counterValue = profilingService.GetCounterValue(counterUid));
    BOOST_CHECK(counterValue == 0);
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
    BOOST_CHECK_NO_THROW(counterUids = GetNextCounterUids(0));
    BOOST_CHECK(counterUids.size() == 1);
    BOOST_CHECK(counterUids[0] >= 0);

    std::vector<uint16_t> nextCounterUids;
    BOOST_CHECK_NO_THROW(nextCounterUids = GetNextCounterUids(1));
    BOOST_CHECK(nextCounterUids.size() == 1);
    BOOST_CHECK(nextCounterUids[0] > counterUids[0]);

    std::vector<uint16_t> counterUidsMultiCore;
    uint16_t numberOfCores = 13;
    BOOST_CHECK_NO_THROW(counterUidsMultiCore = GetNextCounterUids(numberOfCores));
    BOOST_CHECK(counterUidsMultiCore.size() == numberOfCores);
    BOOST_CHECK(counterUidsMultiCore.front() >= nextCounterUids[0]);
    for (size_t i = 1; i < numberOfCores; i ++)
    {
        BOOST_CHECK(counterUidsMultiCore[i] == counterUidsMultiCore[i - 1] + 1);
    }
    BOOST_CHECK(counterUidsMultiCore.back() == counterUidsMultiCore.front() + numberOfCores - 1);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterCategory)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount()   == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount()     == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount()    == 0);

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
    const Category* category = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());
    BOOST_CHECK(category->m_DeviceUid == 0);
    BOOST_CHECK(category->m_CounterSetUid == 0);

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
    const Device* device = nullptr;
    BOOST_CHECK_NO_THROW(device = counterDirectory.RegisterDevice(deviceName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 1);
    BOOST_CHECK(device);
    BOOST_CHECK(device->m_Uid >= 1);
    BOOST_CHECK(device->m_Name == deviceName);
    BOOST_CHECK(device->m_Cores == 0);

    // Register a new category not associated to any device
    const std::string categoryWoDeviceName = "some_category_without_device";
    const Category* categoryWoDevice = nullptr;
    BOOST_CHECK_NO_THROW(categoryWoDevice = counterDirectory.RegisterCategory(categoryWoDeviceName, 0));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 2);
    BOOST_CHECK(categoryWoDevice);
    BOOST_CHECK(categoryWoDevice->m_Name == categoryWoDeviceName);
    BOOST_CHECK(categoryWoDevice->m_Counters.empty());
    BOOST_CHECK(categoryWoDevice->m_DeviceUid == 0);
    BOOST_CHECK(categoryWoDevice->m_CounterSetUid == 0);

    // Register a new category associated to an invalid device
    const std::string categoryWInvalidDeviceName = "some_category_with_invalid_device";

    ARMNN_NO_CONVERSION_WARN_BEGIN
    uint16_t invalidDeviceUid = device->m_Uid + 10;
    ARMNN_NO_CONVERSION_WARN_END

    const Category* categoryWInvalidDevice = nullptr;
    BOOST_CHECK_THROW(categoryWInvalidDevice
                      = counterDirectory.RegisterCategory(categoryWInvalidDeviceName,
                                                          invalidDeviceUid),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 2);
    BOOST_CHECK(!categoryWInvalidDevice);

    // Register a new category associated to a valid device
    const std::string categoryWValidDeviceName = "some_category_with_valid_device";
    const Category* categoryWValidDevice = nullptr;
    BOOST_CHECK_NO_THROW(categoryWValidDevice
                         = counterDirectory.RegisterCategory(categoryWValidDeviceName,
                                                             device->m_Uid));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 3);
    BOOST_CHECK(categoryWValidDevice);
    BOOST_CHECK(categoryWValidDevice != category);
    BOOST_CHECK(categoryWValidDevice->m_Name == categoryWValidDeviceName);
    BOOST_CHECK(categoryWValidDevice->m_DeviceUid == device->m_Uid);
    BOOST_CHECK(categoryWValidDevice->m_CounterSetUid == 0);

    // Register a counter set for testing
    const std::string counterSetName = "some_counter_set";
    const CounterSet* counterSet = nullptr;
    BOOST_CHECK_NO_THROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 1);
    BOOST_CHECK(counterSet);
    BOOST_CHECK(counterSet->m_Uid >= 1);
    BOOST_CHECK(counterSet->m_Name == counterSetName);
    BOOST_CHECK(counterSet->m_Count == 0);

    // Register a new category not associated to any counter set
    const std::string categoryWoCounterSetName = "some_category_without_counter_set";
    const Category* categoryWoCounterSet = nullptr;
    BOOST_CHECK_NO_THROW(categoryWoCounterSet
                         = counterDirectory.RegisterCategory(categoryWoCounterSetName,
                                                             armnn::EmptyOptional(),
                                                             0));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 4);
    BOOST_CHECK(categoryWoCounterSet);
    BOOST_CHECK(categoryWoCounterSet->m_Name == categoryWoCounterSetName);
    BOOST_CHECK(categoryWoCounterSet->m_DeviceUid == 0);
    BOOST_CHECK(categoryWoCounterSet->m_CounterSetUid == 0);

    // Register a new category associated to an invalid counter set
    const std::string categoryWInvalidCounterSetName = "some_category_with_invalid_counter_set";

    ARMNN_NO_CONVERSION_WARN_BEGIN
    uint16_t invalidCunterSetUid = counterSet->m_Uid + 10;
    ARMNN_NO_CONVERSION_WARN_END

    const Category* categoryWInvalidCounterSet = nullptr;
    BOOST_CHECK_THROW(categoryWInvalidCounterSet
                      = counterDirectory.RegisterCategory(categoryWInvalidCounterSetName,
                                                          armnn::EmptyOptional(),
                                                          invalidCunterSetUid),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 4);
    BOOST_CHECK(!categoryWInvalidCounterSet);

    // Register a new category associated to a valid counter set
    const std::string categoryWValidCounterSetName = "some_category_with_valid_counter_set";
    const Category* categoryWValidCounterSet = nullptr;
    BOOST_CHECK_NO_THROW(categoryWValidCounterSet
                         = counterDirectory.RegisterCategory(categoryWValidCounterSetName,
                                                             armnn::EmptyOptional(),
                                                             counterSet->m_Uid));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 5);
    BOOST_CHECK(categoryWValidCounterSet);
    BOOST_CHECK(categoryWValidCounterSet != category);
    BOOST_CHECK(categoryWValidCounterSet->m_Name == categoryWValidCounterSetName);
    BOOST_CHECK(categoryWValidCounterSet->m_DeviceUid == 0);
    BOOST_CHECK(categoryWValidCounterSet->m_CounterSetUid == counterSet->m_Uid);

    // Register a new category associated to a valid device and counter set
    const std::string categoryWValidDeviceAndValidCounterSetName = "some_category_with_valid_device_and_counter_set";
    const Category* categoryWValidDeviceAndValidCounterSet = nullptr;
    BOOST_CHECK_NO_THROW(categoryWValidDeviceAndValidCounterSet
                         = counterDirectory.RegisterCategory(categoryWValidDeviceAndValidCounterSetName,
                                                             device->m_Uid,
                                                             counterSet->m_Uid));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 6);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet != category);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet->m_Name == categoryWValidDeviceAndValidCounterSetName);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet->m_DeviceUid == device->m_Uid);
    BOOST_CHECK(categoryWValidDeviceAndValidCounterSet->m_CounterSetUid == counterSet->m_Uid);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterDevice)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount()   == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount()     == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount()    == 0);

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
    const Device* device = nullptr;
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
    const Device* deviceWCores = nullptr;
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
    const Device* deviceWCoresWInvalidParentCategory = nullptr;
    BOOST_CHECK_THROW(deviceWCoresWInvalidParentCategory
                      = counterDirectory.RegisterDevice(deviceWCoresWInvalidParentCategoryName,
                                                        3,
                                                        std::string("")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(!deviceWCoresWInvalidParentCategory);

    // Register a new device with cores and invalid parent category
    const std::string deviceWCoresWInvalidParentCategoryName2 = "some_device_with_cores_with_invalid_parent_category2";
    const Device* deviceWCoresWInvalidParentCategory2 = nullptr;
    BOOST_CHECK_THROW(deviceWCoresWInvalidParentCategory2
                      = counterDirectory.RegisterDevice(deviceWCoresWInvalidParentCategoryName2,
                                                        3,
                                                        std::string("invalid_parent_category")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(!deviceWCoresWInvalidParentCategory2);

    // Register a category for testing
    const std::string categoryName = "some_category";
    const Category* category = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());
    BOOST_CHECK(category->m_DeviceUid == 0);
    BOOST_CHECK(category->m_CounterSetUid == 0);

    // Register a new device with cores and valid parent category
    const std::string deviceWCoresWValidParentCategoryName = "some_device_with_cores_with_valid_parent_category";
    const Device* deviceWCoresWValidParentCategory = nullptr;
    BOOST_CHECK_NO_THROW(deviceWCoresWValidParentCategory
                         = counterDirectory.RegisterDevice(deviceWCoresWValidParentCategoryName,
                                                           4,
                                                           categoryName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 3);
    BOOST_CHECK(deviceWCoresWValidParentCategory);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Name == deviceWCoresWValidParentCategoryName);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Uid >= 1);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Uid > device->m_Uid);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Uid > deviceWCores->m_Uid);
    BOOST_CHECK(deviceWCoresWValidParentCategory->m_Cores == 4);
    BOOST_CHECK(category->m_DeviceUid == deviceWCoresWValidParentCategory->m_Uid);

    // Register a device associated to a category already associated to a different device
    const std::string deviceSameCategoryName = "some_device_with_invalid_parent_category";
    const Device* deviceSameCategory = nullptr;
    BOOST_CHECK_THROW(deviceSameCategory = counterDirectory.RegisterDevice(deviceSameCategoryName, 0, categoryName),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 3);
    BOOST_CHECK(!deviceSameCategory);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterCounterSet)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount()   == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount()     == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount()    == 0);

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
    const CounterSet* counterSet = nullptr;
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
    const CounterSet* counterSetWCount = nullptr;
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
    BOOST_CHECK_THROW(counterSetWCountWInvalidParentCategory
                      = counterDirectory.RegisterCounterSet(counterSetWCountWInvalidParentCategoryName,
                                                            42,
                                                            std::string("")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 2);
    BOOST_CHECK(!counterSetWCountWInvalidParentCategory);

    // Register a new counter set with count and invalid parent category
    const std::string counterSetWCountWInvalidParentCategoryName2 = "some_counter_set_with_count_"
                                                                    "with_invalid_parent_category2";
    const CounterSet* counterSetWCountWInvalidParentCategory2 = nullptr;
    BOOST_CHECK_THROW(counterSetWCountWInvalidParentCategory2
                      = counterDirectory.RegisterCounterSet(counterSetWCountWInvalidParentCategoryName2,
                                                            42,
                                                            std::string("invalid_parent_category")),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 2);
    BOOST_CHECK(!counterSetWCountWInvalidParentCategory2);

    // Register a category for testing
    const std::string categoryName = "some_category";
    const Category* category = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());
    BOOST_CHECK(category->m_DeviceUid == 0);
    BOOST_CHECK(category->m_CounterSetUid == 0);

    // Register a new counter set with count and valid parent category
    const std::string counterSetWCountWValidParentCategoryName = "some_counter_set_with_count_"
                                                                 "with_valid_parent_category";
    const CounterSet* counterSetWCountWValidParentCategory = nullptr;
    BOOST_CHECK_NO_THROW(counterSetWCountWValidParentCategory
                         = counterDirectory.RegisterCounterSet(counterSetWCountWValidParentCategoryName,
                                                               42,
                                                               categoryName));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 3);
    BOOST_CHECK(counterSetWCountWValidParentCategory);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Name == counterSetWCountWValidParentCategoryName);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Uid >= 1);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Uid > counterSet->m_Uid);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Uid > counterSetWCount->m_Uid);
    BOOST_CHECK(counterSetWCountWValidParentCategory->m_Count == 42);
    BOOST_CHECK(category->m_CounterSetUid == counterSetWCountWValidParentCategory->m_Uid);

    // Register a counter set associated to a category already associated to a different counter set
    const std::string counterSetSameCategoryName = "some_counter_set_with_invalid_parent_category";
    const CounterSet* counterSetSameCategory = nullptr;
    BOOST_CHECK_THROW(counterSetSameCategory = counterDirectory.RegisterCounterSet(counterSetSameCategoryName,
                                                                                   0,
                                                                                   categoryName),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 3);
    BOOST_CHECK(!counterSetSameCategory);
}

BOOST_AUTO_TEST_CASE(CheckCounterDirectoryRegisterCounter)
{
    CounterDirectory counterDirectory;
    BOOST_CHECK(counterDirectory.GetCategoryCount()   == 0);
    BOOST_CHECK(counterDirectory.GetDeviceCount()     == 0);
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 0);
    BOOST_CHECK(counterDirectory.GetCounterCount()    == 0);

    // Register a counter with an invalid parent category name
    const Counter* noCounter = nullptr;
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid parent category name
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("invalid parent category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid class
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
                                                                   2,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid interpolation
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
                                                                   0,
                                                                   3,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid multiplier
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   .0f,
                                                                   "valid name",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid name
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid name
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "invalid nam€",
                                                                   "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid description
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   ""),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid description
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name",
                                                                   "inv@lid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 0);
    BOOST_CHECK(!noCounter);

    // Register a counter with an invalid unit2
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("valid_parent_category",
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
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter("invalid_parent_category",
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
    const Category* category = nullptr;
    BOOST_CHECK_NO_THROW(category = counterDirectory.RegisterCategory(categoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 1);
    BOOST_CHECK(category);
    BOOST_CHECK(category->m_Name == categoryName);
    BOOST_CHECK(category->m_Counters.empty());
    BOOST_CHECK(category->m_DeviceUid == 0);
    BOOST_CHECK(category->m_CounterSetUid == 0);

    // Register a counter with a valid parent category name
    const Counter* counter = nullptr;
    BOOST_CHECK_NO_THROW(counter = counterDirectory.RegisterCounter(categoryName,
                                                                    0,
                                                                    1,
                                                                    123.45f,
                                                                    "valid name",
                                                                    "valid description"));
    BOOST_CHECK(counterDirectory.GetCounterCount() == 1);
    BOOST_CHECK(counter);
    BOOST_CHECK(counter->m_Uid >= 0);
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
    BOOST_CHECK_THROW(counterSameName = counterDirectory.RegisterCounter(categoryName,
                                                                         0,
                                                                         0,
                                                                         1.0f,
                                                                         "valid name",
                                                                         "valid description"),
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 1);
    BOOST_CHECK(!counterSameName);

    // Register a counter with a valid parent category name and units
    const Counter* counterWUnits = nullptr;
    BOOST_CHECK_NO_THROW(counterWUnits = counterDirectory.RegisterCounter(categoryName,
                                                                          0,
                                                                          1,
                                                                          123.45f,
                                                                          "valid name 2",
                                                                          "valid description",
                                                                          std::string("Mnnsq2"))); // Units
    BOOST_CHECK(counterDirectory.GetCounterCount() == 2);
    BOOST_CHECK(counterWUnits);
    BOOST_CHECK(counterWUnits->m_Uid >= 0);
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
    BOOST_CHECK_NO_THROW(counterWoDevice = counterDirectory.RegisterCounter(categoryName,
                                                                            0,
                                                                            1,
                                                                            123.45f,
                                                                            "valid name 3",
                                                                            "valid description",
                                                                            armnn::EmptyOptional(), // Units
                                                                            armnn::EmptyOptional(), // Number of cores
                                                                            0));                    // Device UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 3);
    BOOST_CHECK(counterWoDevice);
    BOOST_CHECK(counterWoDevice->m_Uid >= 0);
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
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(categoryName,
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name 4",
                                                                   "valid description",
                                                                   armnn::EmptyOptional(), // Units
                                                                   armnn::EmptyOptional(), // Number of cores
                                                                   100),                   // Device UID
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 3);
    BOOST_CHECK(!noCounter);

    // Register a device for testing
    const std::string deviceName = "some_device";
    const Device* device = nullptr;
    BOOST_CHECK_NO_THROW(device = counterDirectory.RegisterDevice(deviceName));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 1);
    BOOST_CHECK(device);
    BOOST_CHECK(device->m_Name == deviceName);
    BOOST_CHECK(device->m_Uid >= 1);
    BOOST_CHECK(device->m_Cores == 0);

    // Register a counter with a valid parent category name and associated to a device
    const Counter* counterWDevice = nullptr;
    BOOST_CHECK_NO_THROW(counterWDevice = counterDirectory.RegisterCounter(categoryName,
                                                                           0,
                                                                           1,
                                                                           123.45f,
                                                                           "valid name 5",
                                                                           "valid description",
                                                                           armnn::EmptyOptional(), // Units
                                                                           armnn::EmptyOptional(), // Number of cores
                                                                           device->m_Uid));        // Device UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 4);
    BOOST_CHECK(counterWDevice);
    BOOST_CHECK(counterWDevice->m_Uid >= 0);
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
    BOOST_CHECK_NO_THROW(counterWoCounterSet
                         = counterDirectory.RegisterCounter(categoryName,
                                                            0,
                                                            1,
                                                            123.45f,
                                                            "valid name 6",
                                                            "valid description",
                                                            armnn::EmptyOptional(), // Units
                                                            armnn::EmptyOptional(), // Number of cores
                                                            armnn::EmptyOptional(), // Device UID
                                                            0));                    // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 5);
    BOOST_CHECK(counterWoCounterSet);
    BOOST_CHECK(counterWoCounterSet->m_Uid >= 0);
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
    BOOST_CHECK_THROW(noCounter = counterDirectory.RegisterCounter(categoryName,
                                                                   0,
                                                                   1,
                                                                   123.45f,
                                                                   "valid name 7",
                                                                   "valid description",
                                                                   armnn::EmptyOptional(), // Units
                                                                   armnn::EmptyOptional(), // Number of cores
                                                                   armnn::EmptyOptional(), // Device UID
                                                                   100),                   // Counter set UID
                      armnn::InvalidArgumentException);
    BOOST_CHECK(counterDirectory.GetCounterCount() == 5);
    BOOST_CHECK(!noCounter);

    // Register a counter with a valid parent category name and with a given number of cores
    const Counter* counterWNumberOfCores = nullptr;
    uint16_t numberOfCores = 15;
    BOOST_CHECK_NO_THROW(counterWNumberOfCores
                         = counterDirectory.RegisterCounter(categoryName,
                                                            0,
                                                            1,
                                                            123.45f,
                                                            "valid name 8",
                                                            "valid description",
                                                            armnn::EmptyOptional(),   // Units
                                                            numberOfCores,            // Number of cores
                                                            armnn::EmptyOptional(),   // Device UID
                                                            armnn::EmptyOptional())); // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 20);
    BOOST_CHECK(counterWNumberOfCores);
    BOOST_CHECK(counterWNumberOfCores->m_Uid >= 0);
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
    for (size_t i = 0; i < numberOfCores; i ++)
    {
        BOOST_CHECK(category->m_Counters[category->m_Counters.size() - numberOfCores + i] ==
                    counterWNumberOfCores->m_Uid + i);
    }

    // Register a multi-core device for testing
    const std::string multiCoreDeviceName = "some_multi_core_device";
    const Device* multiCoreDevice = nullptr;
    BOOST_CHECK_NO_THROW(multiCoreDevice = counterDirectory.RegisterDevice(multiCoreDeviceName, 4));
    BOOST_CHECK(counterDirectory.GetDeviceCount() == 2);
    BOOST_CHECK(multiCoreDevice);
    BOOST_CHECK(multiCoreDevice->m_Name == multiCoreDeviceName);
    BOOST_CHECK(multiCoreDevice->m_Uid >= 1);
    BOOST_CHECK(multiCoreDevice->m_Cores == 4);

    // Register a counter with a valid parent category name and associated to the multi-core device
    const Counter* counterWMultiCoreDevice = nullptr;
    BOOST_CHECK_NO_THROW(counterWMultiCoreDevice
                         = counterDirectory.RegisterCounter(categoryName,
                                                            0,
                                                            1,
                                                            123.45f,
                                                            "valid name 9",
                                                            "valid description",
                                                            armnn::EmptyOptional(),   // Units
                                                            armnn::EmptyOptional(),   // Number of cores
                                                            multiCoreDevice->m_Uid,   // Device UID
                                                            armnn::EmptyOptional())); // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 24);
    BOOST_CHECK(counterWMultiCoreDevice);
    BOOST_CHECK(counterWMultiCoreDevice->m_Uid >= 0);
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
    for (size_t i = 0; i < 4; i ++)
    {
        BOOST_CHECK(category->m_Counters[category->m_Counters.size() - 4 + i] == counterWMultiCoreDevice->m_Uid + i);
    }

    // Register a multi-core device associate to a parent category for testing
    const std::string multiCoreDeviceNameWParentCategory = "some_multi_core_device_with_parent_category";
    const Device* multiCoreDeviceWParentCategory = nullptr;
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
    BOOST_CHECK_NO_THROW(counterWMultiCoreDeviceWParentCategory
                         = counterDirectory.RegisterCounter(categoryName,
                                                            0,
                                                            1,
                                                            123.45f,
                                                            "valid name 10",
                                                            "valid description",
                                                            armnn::EmptyOptional(),   // Units
                                                            armnn::EmptyOptional(),   // Number of cores
                                                            armnn::EmptyOptional(),   // Device UID
                                                            armnn::EmptyOptional())); // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 26);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Uid >= 0);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Uid > counter->m_Uid);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_MaxCounterUid ==
                counterWMultiCoreDeviceWParentCategory->m_Uid + multiCoreDeviceWParentCategory->m_Cores - 1);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Class == 0);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Interpolation == 1);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Multiplier == 123.45f);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Name == "valid name 10");
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Description == "valid description");
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_Units == "");
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_DeviceUid == 0);
    BOOST_CHECK(counterWMultiCoreDeviceWParentCategory->m_CounterSetUid == 0);
    BOOST_CHECK(category->m_Counters.size() == 26);
    for (size_t i = 0; i < 2; i ++)
    {
        BOOST_CHECK(category->m_Counters[category->m_Counters.size() - 2 + i] ==
                    counterWMultiCoreDeviceWParentCategory->m_Uid + i);
    }

    // Register a counter set for testing
    const std::string counterSetName = "some_counter_set";
    const CounterSet* counterSet = nullptr;
    BOOST_CHECK_NO_THROW(counterSet = counterDirectory.RegisterCounterSet(counterSetName));
    BOOST_CHECK(counterDirectory.GetCounterSetCount() == 1);
    BOOST_CHECK(counterSet);
    BOOST_CHECK(counterSet->m_Name == counterSetName);
    BOOST_CHECK(counterSet->m_Uid >= 1);
    BOOST_CHECK(counterSet->m_Count == 0);

    // Register a counter with a valid parent category name and associated to a counter set
    const Counter* counterWCounterSet = nullptr;
    BOOST_CHECK_NO_THROW(counterWCounterSet
                         = counterDirectory.RegisterCounter(categoryName,
                                                            0,
                                                            1,
                                                            123.45f,
                                                            "valid name 11",
                                                            "valid description",
                                                            armnn::EmptyOptional(), // Units
                                                            0,                      // Number of cores
                                                            armnn::EmptyOptional(), // Device UID
                                                            counterSet->m_Uid));    // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 27);
    BOOST_CHECK(counterWCounterSet);
    BOOST_CHECK(counterWCounterSet->m_Uid >= 0);
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
    BOOST_CHECK_NO_THROW(counterWDeviceWCounterSet
                         = counterDirectory.RegisterCounter(categoryName,
                                                            0,
                                                            1,
                                                            123.45f,
                                                            "valid name 12",
                                                            "valid description",
                                                            armnn::EmptyOptional(), // Units
                                                            1,                      // Number of cores
                                                            device->m_Uid,          // Device UID
                                                            counterSet->m_Uid));    // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 28);
    BOOST_CHECK(counterWDeviceWCounterSet);
    BOOST_CHECK(counterWDeviceWCounterSet->m_Uid >= 0);
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
    const Category* anotherCategory = nullptr;
    BOOST_CHECK_NO_THROW(anotherCategory = counterDirectory.RegisterCategory(anotherCategoryName));
    BOOST_CHECK(counterDirectory.GetCategoryCount() == 2);
    BOOST_CHECK(anotherCategory);
    BOOST_CHECK(anotherCategory != category);
    BOOST_CHECK(anotherCategory->m_Name == anotherCategoryName);
    BOOST_CHECK(anotherCategory->m_Counters.empty());
    BOOST_CHECK(anotherCategory->m_DeviceUid == 0);
    BOOST_CHECK(anotherCategory->m_CounterSetUid == 0);

    // Register a counter to the other category
    const Counter* anotherCounter = nullptr;
    BOOST_CHECK_NO_THROW(anotherCounter = counterDirectory.RegisterCounter(anotherCategoryName,
                                                                           1,
                                                                           0,
                                                                           .00043f,
                                                                           "valid name",
                                                                           "valid description",
                                                                           armnn::EmptyOptional(), // Units
                                                                           armnn::EmptyOptional(), // Number of cores
                                                                           device->m_Uid,          // Device UID
                                                                           counterSet->m_Uid));    // Counter set UID
    BOOST_CHECK(counterDirectory.GetCounterCount() == 29);
    BOOST_CHECK(anotherCounter);
    BOOST_CHECK(anotherCounter->m_Uid >= 0);
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
    using boost::numeric_cast;

    ProfilingStateMachine profilingStateMachine;

    class TestCaptureThread : public IPeriodicCounterCapture
    {
        void Start() override {}
        void Stop() override {}
    };

    const uint32_t packetId = 0x40000;

    uint32_t version = 1;
    Holder holder;
    TestCaptureThread captureThread;
    MockBufferManager mockBuffer(512);
    SendCounterPacket sendCounterPacket(profilingStateMachine, mockBuffer);

    uint32_t sizeOfUint32 = numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = numeric_cast<uint32_t>(sizeof(uint16_t));

    // Data with period and counters
    uint32_t period1 = 10;
    uint32_t dataLength1 = 8;
    uint32_t offset = 0;

    std::unique_ptr<char[]> uniqueData1 = std::make_unique<char[]>(dataLength1);
    unsigned char* data1 = reinterpret_cast<unsigned char*>(uniqueData1.get());

    WriteUint32(data1, offset, period1);
    offset += sizeOfUint32;
    WriteUint16(data1, offset, 4000);
    offset += sizeOfUint16;
    WriteUint16(data1, offset, 5000);

    Packet packetA(packetId, dataLength1, uniqueData1);

    PeriodicCounterSelectionCommandHandler commandHandler(packetId, version, holder, captureThread,
                                                          sendCounterPacket);
    commandHandler(packetA);

    std::vector<uint16_t> counterIds = holder.GetCaptureData().GetCounterIds();

    BOOST_TEST(holder.GetCaptureData().GetCapturePeriod() == period1);
    BOOST_TEST(counterIds.size() == 2);
    BOOST_TEST(counterIds[0] == 4000);
    BOOST_TEST(counterIds[1] == 5000);

    auto readBuffer = mockBuffer.GetReadableBuffer();

    offset = 0;

    uint32_t headerWord0 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    uint32_t headerWord1 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    uint32_t period = ReadUint32(readBuffer, offset);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 0);  // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x3FF) == 4); // packet id
    BOOST_TEST(headerWord1 == 8);                   // data lenght
    BOOST_TEST(period == 10);                       // capture period

    uint16_t counterId = 0;
    offset += sizeOfUint32;
    counterId = ReadUint16(readBuffer, offset);
    BOOST_TEST(counterId == 4000);
    offset += sizeOfUint16;
    counterId = ReadUint16(readBuffer, offset);
    BOOST_TEST(counterId == 5000);

    mockBuffer.MarkRead(readBuffer);

    // Data with period only
    uint32_t period2 = 11;
    uint32_t dataLength2 = 4;

    std::unique_ptr<char[]> uniqueData2 = std::make_unique<char[]>(dataLength2);

    WriteUint32(reinterpret_cast<unsigned char*>(uniqueData2.get()), 0, period2);

    Packet packetB(packetId, dataLength2, uniqueData2);

    commandHandler(packetB);

    counterIds = holder.GetCaptureData().GetCounterIds();

    BOOST_TEST(holder.GetCaptureData().GetCapturePeriod() == period2);
    BOOST_TEST(counterIds.size() == 0);

    readBuffer = mockBuffer.GetReadableBuffer();

    offset = 0;

    headerWord0 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    headerWord1 = ReadUint32(readBuffer, offset);
    offset += sizeOfUint32;
    period = ReadUint32(readBuffer, offset);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 0);  // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x3FF) == 4); // packet id
    BOOST_TEST(headerWord1 == 4);                   // data lenght
    BOOST_TEST(period == 11);                       // capture period
}

BOOST_AUTO_TEST_CASE(CheckConnectionAcknowledged)
{
    using boost::numeric_cast;

    const uint32_t connectionPacketId = 0x10000;
    const uint32_t version = 1;

    uint32_t sizeOfUint32 = numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = numeric_cast<uint32_t>(sizeof(uint16_t));

    // Data with period and counters
    uint32_t period1 = 10;
    uint32_t dataLength1 = 8;
    uint32_t offset = 0;

    std::unique_ptr<char[]> uniqueData1 = std::make_unique<char[]>(dataLength1);
    unsigned char* data1 = reinterpret_cast<unsigned char*>(uniqueData1.get());

    WriteUint32(data1, offset, period1);
    offset += sizeOfUint32;
    WriteUint16(data1, offset, 4000);
    offset += sizeOfUint16;
    WriteUint16(data1, offset, 5000);

    Packet packetA(connectionPacketId, dataLength1, uniqueData1);

    ProfilingStateMachine profilingState(ProfilingState::Uninitialised);
    BOOST_CHECK(profilingState.GetCurrentState() == ProfilingState::Uninitialised);

    ConnectionAcknowledgedCommandHandler commandHandler(connectionPacketId, version, profilingState);

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
    Packet packetB(differentPacketId, dataLength1, uniqueData1);
    profilingState.TransitionToState(ProfilingState::NotConnected);
    profilingState.TransitionToState(ProfilingState::WaitingForAck);
    ConnectionAcknowledgedCommandHandler differentCommandHandler(differentPacketId, version, profilingState);
    BOOST_CHECK_THROW(differentCommandHandler(packetB), armnn::Exception);
}

BOOST_AUTO_TEST_CASE(CheckSocketProfilingConnection)
{
    // Check that creating a SocketProfilingConnection results in an exception as the Gator UDS doesn't exist.
    BOOST_CHECK_THROW(new SocketProfilingConnection(), armnn::Exception);
}

BOOST_AUTO_TEST_CASE(SwTraceIsValidCharTest)
{
    // Only ASCII 7-bit encoding supported
    for (unsigned char c = 0; c < 128; c++)
    {
        BOOST_CHECK(SwTraceCharPolicy::IsValidChar(c));
    }

    // Not ASCII
    for (unsigned char c = 255; c >= 128; c++)
    {
        BOOST_CHECK(!SwTraceCharPolicy::IsValidChar(c));
    }
}

BOOST_AUTO_TEST_CASE(SwTraceIsValidNameCharTest)
{
    // Only alpha-numeric and underscore ASCII 7-bit encoding supported
    const unsigned char validChars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    for (unsigned char i = 0; i < sizeof(validChars) / sizeof(validChars[0]) - 1; i++)
    {
        BOOST_CHECK(SwTraceNameCharPolicy::IsValidChar(validChars[i]));
    }

    // Non alpha-numeric chars
    for (unsigned char c = 0; c < 48; c++)
    {
        BOOST_CHECK(!SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 58; c < 65; c++)
    {
        BOOST_CHECK(!SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 91; c < 95; c++)
    {
        BOOST_CHECK(!SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 96; c < 97; c++)
    {
        BOOST_CHECK(!SwTraceNameCharPolicy::IsValidChar(c));
    }
    for (unsigned char c = 123; c < 128; c++)
    {
        BOOST_CHECK(!SwTraceNameCharPolicy::IsValidChar(c));
    }

    // Not ASCII
    for (unsigned char c = 255; c >= 128; c++)
    {
        BOOST_CHECK(!SwTraceNameCharPolicy::IsValidChar(c));
    }
}

BOOST_AUTO_TEST_CASE(IsValidSwTraceStringTest)
{
    // Valid SWTrace strings
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>(""));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>("_"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>("0123"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>("valid_string"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>("VALID_string_456"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>(" "));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>("valid string"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>("!$%"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceCharPolicy>("valid|\\~string#123"));

    // Invalid SWTrace strings
    BOOST_CHECK(!IsValidSwTraceString<SwTraceCharPolicy>("€£"));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceCharPolicy>("invalid‡string"));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceCharPolicy>("12Ž34"));
}

BOOST_AUTO_TEST_CASE(IsValidSwTraceNameStringTest)
{
    // Valid SWTrace name strings
    BOOST_CHECK(IsValidSwTraceString<SwTraceNameCharPolicy>(""));
    BOOST_CHECK(IsValidSwTraceString<SwTraceNameCharPolicy>("_"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceNameCharPolicy>("0123"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceNameCharPolicy>("valid_string"));
    BOOST_CHECK(IsValidSwTraceString<SwTraceNameCharPolicy>("VALID_string_456"));

    // Invalid SWTrace name strings
    BOOST_CHECK(!IsValidSwTraceString<SwTraceNameCharPolicy>(" "));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceNameCharPolicy>("invalid string"));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceNameCharPolicy>("!$%"));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceNameCharPolicy>("invalid|\\~string#123"));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceNameCharPolicy>("€£"));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceNameCharPolicy>("invalid‡string"));
    BOOST_CHECK(!IsValidSwTraceString<SwTraceNameCharPolicy>("12Ž34"));
}

template <typename SwTracePolicy>
void StringToSwTraceStringTestHelper(const std::string& testString, std::vector<uint32_t> buffer, size_t expectedSize)
{
    // Convert the test string to a SWTrace string
    BOOST_CHECK(StringToSwTraceString<SwTracePolicy>(testString, buffer));

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
    BOOST_CHECK(reinterpret_cast<unsigned char*>(buffer.data())[nullTerminatorIndex]  == '\0');
}

BOOST_AUTO_TEST_CASE(StringToSwTraceStringTest)
{
    std::vector<uint32_t> buffer;

    // Valid SWTrace strings (expected size in words)
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("", buffer, 2);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("_", buffer, 2);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("0123", buffer, 3);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("valid_string", buffer, 5);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("VALID_string_456", buffer, 6);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>(" ", buffer, 2);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("valid string", buffer, 5);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("!$%", buffer, 2);
    StringToSwTraceStringTestHelper<SwTraceCharPolicy>("valid|\\~string#123", buffer, 6);

    // Invalid SWTrace strings
    BOOST_CHECK(!StringToSwTraceString<SwTraceCharPolicy>("€£", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceCharPolicy>("invalid‡string", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceCharPolicy>("12Ž34", buffer));
    BOOST_CHECK(buffer.empty());
}

BOOST_AUTO_TEST_CASE(StringToSwTraceNameStringTest)
{
    std::vector<uint32_t> buffer;

    // Valid SWTrace namestrings (expected size in words)
    StringToSwTraceStringTestHelper<SwTraceNameCharPolicy>("", buffer, 2);
    StringToSwTraceStringTestHelper<SwTraceNameCharPolicy>("_", buffer, 2);
    StringToSwTraceStringTestHelper<SwTraceNameCharPolicy>("0123", buffer, 3);
    StringToSwTraceStringTestHelper<SwTraceNameCharPolicy>("valid_string", buffer, 5);
    StringToSwTraceStringTestHelper<SwTraceNameCharPolicy>("VALID_string_456", buffer, 6);

    // Invalid SWTrace namestrings
    BOOST_CHECK(!StringToSwTraceString<SwTraceNameCharPolicy>(" ", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceNameCharPolicy>("invalid string", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceNameCharPolicy>("!$%", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceNameCharPolicy>("invalid|\\~string#123", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceNameCharPolicy>("€£", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceNameCharPolicy>("invalid‡string", buffer));
    BOOST_CHECK(buffer.empty());
    BOOST_CHECK(!StringToSwTraceString<SwTraceNameCharPolicy>("12Ž34", buffer));
    BOOST_CHECK(buffer.empty());
}

BOOST_AUTO_TEST_CASE(CheckPeriodicCounterCaptureThread)
{
    class CaptureReader : public IReadCounterValues
    {
    public:
        CaptureReader() {}

        uint16_t GetCounterCount() const override
        {
            return boost::numeric_cast<uint16_t>(m_Data.size());
        }

        uint32_t GetCounterValue(uint16_t index) const override
        {
            if (m_Data.find(index) == m_Data.end())
            {
                return 0;
            }

            return m_Data.at(index);
        }

        void SetCounterValue(uint16_t index, uint32_t value)
        {
            if (m_Data.find(index) == m_Data.end())
            {
                m_Data.insert(std::pair<uint16_t, uint32_t>(index, value));
            }
            else
            {
                m_Data.at(index) = value;
            }
        }

    private:
        std::unordered_map<uint16_t, uint32_t> m_Data;
    };

    ProfilingStateMachine profilingStateMachine;

    Holder data;
    std::vector<uint16_t> captureIds1 = { 0, 1 };
    std::vector<uint16_t> captureIds2;

    MockBufferManager mockBuffer(512);
    SendCounterPacket sendCounterPacket(profilingStateMachine, mockBuffer);

    std::vector<uint16_t> counterIds;
    CaptureReader captureReader;

    unsigned int valueA = 10;
    unsigned int valueB = 15;
    unsigned int numSteps = 5;

    PeriodicCounterCapture periodicCounterCapture(std::ref(data), std::ref(sendCounterPacket), captureReader);

    for (unsigned int i = 0; i < numSteps; ++i)
    {
        data.SetCaptureData(1, captureIds1);
        captureReader.SetCounterValue(0, valueA * (i + 1));
        captureReader.SetCounterValue(1, valueB * (i + 1));

        periodicCounterCapture.Start();

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        periodicCounterCapture.Start();

        data.SetCaptureData(0, captureIds2);

        periodicCounterCapture.Start();
    }

    periodicCounterCapture.Stop();

    auto buffer = mockBuffer.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(buffer, 0);
    uint32_t headerWord1 = ReadUint32(buffer, 4);

    BOOST_TEST(((headerWord0 >> 26) & 0x3F) == 1);   // packet family
    BOOST_TEST(((headerWord0 >> 19) & 0x3F) == 0);   // packet class
    BOOST_TEST(((headerWord0 >> 16) & 0x3) == 0);    // packet type
    BOOST_TEST(headerWord1 == 20);                   // data length

    uint32_t offset = 16;
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
    using boost::numeric_cast;

    const uint32_t packetId = 3;
    const uint32_t version = 1;
    ProfilingStateMachine profilingStateMachine;
    CounterDirectory counterDirectory;
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(profilingStateMachine, mockBuffer);
    RequestCounterDirectoryCommandHandler commandHandler(packetId,
                                                         version,
                                                         counterDirectory,
                                                         sendCounterPacket,
                                                         profilingStateMachine);

    const uint32_t wrongPacketId = 47;
    const uint32_t wrongHeader = (wrongPacketId & 0x000003FF) << 16;

    Packet wrongPacket(wrongHeader);

    profilingStateMachine.TransitionToState(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::Active);
    BOOST_CHECK_THROW(commandHandler(wrongPacket), armnn::InvalidArgumentException); // Wrong packet

    const uint32_t rightHeader = (packetId & 0x000003FF) << 16;

    Packet rightPacket(rightHeader);

    BOOST_CHECK_NO_THROW(commandHandler(rightPacket)); // Right packet

    auto readBuffer = mockBuffer.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(readBuffer, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer, 4);

    BOOST_TEST(((headerWord0 >> 26) & 0x0000003F) == 0);  // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x000003FF) == 2);  // packet id
    BOOST_TEST(headerWord1 == 24);                        // data length

    uint32_t bodyHeaderWord0 = ReadUint32(readBuffer,  8);
    uint16_t deviceRecordCount = numeric_cast<uint16_t>(bodyHeaderWord0 >> 16);
    BOOST_TEST(deviceRecordCount == 0); // device_records_count
}

BOOST_AUTO_TEST_CASE(RequestCounterDirectoryCommandHandlerTest2)
{
    using boost::numeric_cast;

    const uint32_t packetId = 3;
    const uint32_t version = 1;
    ProfilingStateMachine profilingStateMachine;
    CounterDirectory counterDirectory;
    MockBufferManager mockBuffer(1024);
    SendCounterPacket sendCounterPacket(profilingStateMachine, mockBuffer);
    RequestCounterDirectoryCommandHandler commandHandler(packetId,
                                                         version,
                                                         counterDirectory,
                                                         sendCounterPacket,
                                                         profilingStateMachine);
    const uint32_t header = (packetId & 0x000003FF) << 16;
    Packet packet(header);

    const Device* device = counterDirectory.RegisterDevice("deviceA", 1);
    const CounterSet* counterSet = counterDirectory.RegisterCounterSet("countersetA");
    counterDirectory.RegisterCategory("categoryA", device->m_Uid, counterSet->m_Uid);
    counterDirectory.RegisterCounter("categoryA", 0, 1, 2.0f, "counterA", "descA");
    counterDirectory.RegisterCounter("categoryA", 1, 1, 3.0f, "counterB", "descB");

    profilingStateMachine.TransitionToState(ProfilingState::Uninitialised);
    BOOST_CHECK_THROW(commandHandler(packet), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK_THROW(commandHandler(packet), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::WaitingForAck);
    BOOST_CHECK_THROW(commandHandler(packet), armnn::RuntimeException); // Wrong profiling state
    profilingStateMachine.TransitionToState(ProfilingState::Active);
    BOOST_CHECK_NO_THROW(commandHandler(packet));

    auto readBuffer = mockBuffer.GetReadableBuffer();

    uint32_t headerWord0 = ReadUint32(readBuffer, 0);
    uint32_t headerWord1 = ReadUint32(readBuffer, 4);

    BOOST_TEST(((headerWord0 >> 26) & 0x0000003F) == 0);  // packet family
    BOOST_TEST(((headerWord0 >> 16) & 0x000003FF) == 2);  // packet id
    BOOST_TEST(headerWord1 == 240);                       // data length

    uint32_t bodyHeaderWord0 = ReadUint32(readBuffer,  8);
    uint32_t bodyHeaderWord1 = ReadUint32(readBuffer, 12);
    uint32_t bodyHeaderWord2 = ReadUint32(readBuffer, 16);
    uint32_t bodyHeaderWord3 = ReadUint32(readBuffer, 20);
    uint32_t bodyHeaderWord4 = ReadUint32(readBuffer, 24);
    uint32_t bodyHeaderWord5 = ReadUint32(readBuffer, 28);
    uint16_t deviceRecordCount = numeric_cast<uint16_t>(bodyHeaderWord0 >> 16);
    uint16_t counterSetRecordCount = numeric_cast<uint16_t>(bodyHeaderWord2 >> 16);
    uint16_t categoryRecordCount = numeric_cast<uint16_t>(bodyHeaderWord4 >> 16);
    BOOST_TEST(deviceRecordCount == 1);     // device_records_count
    BOOST_TEST(bodyHeaderWord1 == 0);       // device_records_pointer_table_offset
    BOOST_TEST(counterSetRecordCount == 1); // counter_set_count
    BOOST_TEST(bodyHeaderWord3 == 4);       // counter_set_pointer_table_offset
    BOOST_TEST(categoryRecordCount == 1);   // categories_count
    BOOST_TEST(bodyHeaderWord5 == 8);       // categories_pointer_table_offset

    uint32_t deviceRecordOffset = ReadUint32(readBuffer, 32);
    BOOST_TEST(deviceRecordOffset ==  0);

    uint32_t counterSetRecordOffset = ReadUint32(readBuffer, 36);
    BOOST_TEST(counterSetRecordOffset == 20);

    uint32_t categoryRecordOffset = ReadUint32(readBuffer, 40);
    BOOST_TEST(categoryRecordOffset ==  44);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceBadConnectionAcknowledgedPacket)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper;

    // Redirect the standard output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());

    // Calculate the size of a Stream Metadata packet
    std::string processName = GetProcessName().substr(0, 60);
    unsigned int processNameSize = processName.empty() ? 0 : boost::numeric_cast<unsigned int>(processName.size()) + 1;
    unsigned int streamMetadataPacketsize = 118 + processNameSize;

    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);

    // Bring the profiling service to the "WaitingForAck" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();

    // Wait for a bit to make sure that we get the packet
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Check that the mock profiling connection contains one Stream Metadata packet
    const std::vector<uint32_t> writtenData = mockProfilingConnection->GetWrittenData();
    BOOST_TEST(writtenData.size() == 1);
    BOOST_TEST(writtenData[0] == streamMetadataPacketsize);

    // Write a valid "Connection Acknowledged" packet into the mock profiling connection, to simulate a valid
    // reply from an external profiling service

    // Connection Acknowledged Packet header (word 0, word 1 is always zero):
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000001
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 37; // Wrong packet id!!!
    uint32_t header = ((packetFamily & 0x0000003F) << 26) |
                      ((packetId     & 0x000003FF) << 16);

    // Create the Connection Acknowledged Packet
    Packet connectionAcknowledgedPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(connectionAcknowledgedPacket));

    // Wait for a bit (must at least be the delay value of the mock profiling connection) to make sure that
    // the Connection Acknowledged packet gets processed by the profiling service
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Check that the expected error has occurred and logged to the standard output
    BOOST_CHECK(boost::contains(ss.str(), "Functor with requested PacketId=37 and Version=4194304 does not exist"));

    // The Connection Acknowledged Command Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodConnectionAcknowledgedPacket)
{
    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper;

    // Calculate the size of a Stream Metadata packet
    std::string processName = GetProcessName().substr(0, 60);
    unsigned int processNameSize = processName.empty() ? 0 : boost::numeric_cast<unsigned int>(processName.size()) + 1;
    unsigned int streamMetadataPacketsize = 118 + processNameSize;

    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);

    // Bring the profiling service to the "WaitingForAck" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update();
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update();

    // Wait for a bit to make sure that we get the packet
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Get the mock profiling connection
    MockProfilingConnection* mockProfilingConnection = helper.GetMockProfilingConnection();
    BOOST_CHECK(mockProfilingConnection);

    // Check that the mock profiling connection contains one Stream Metadata packet
    const std::vector<uint32_t> writtenData = mockProfilingConnection->GetWrittenData();
    BOOST_TEST(writtenData.size() == 1);
    BOOST_TEST(writtenData[0] == streamMetadataPacketsize);

    // Write a valid "Connection Acknowledged" packet into the mock profiling connection, to simulate a valid
    // reply from an external profiling service

    // Connection Acknowledged Packet header (word 0, word 1 is always zero):
    // 26:31 [6]  packet_family: Control Packet Family, value 0b000000
    // 16:25 [10] packet_id: Packet identifier, value 0b0000000001
    // 8:15  [8]  reserved: Reserved, value 0b00000000
    // 0:7   [8]  reserved: Reserved, value 0b00000000
    uint32_t packetFamily = 0;
    uint32_t packetId     = 1;
    uint32_t header = ((packetFamily & 0x0000003F) << 26) |
                      ((packetId     & 0x000003FF) << 16);

    // Create the Connection Acknowledged Packet
    Packet connectionAcknowledgedPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(connectionAcknowledgedPacket));

    // Wait for a bit (must at least be the delay value of the mock profiling connection) to make sure that
    // the Connection Acknowledged packet gets processed by the profiling service
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // The Connection Acknowledged Command Handler should have updated the profiling state accordingly
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceBadRequestCounterDirectoryPacket)
{
    // Locally reduce log level to "Warning", as this test needs to parse a warning message from the standard output
    LogLevelSwapper logLevelSwapper(armnn::LogSeverity::Warning);

    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper;

    // Redirect the standard output to a local stream so that we can parse the warning message
    std::stringstream ss;
    StreamRedirector streamRedirector(std::cout, ss.rdbuf());

    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    helper.ForceTransitionToState(ProfilingState::NotConnected);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update(); // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update(); // Start the threads
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

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
    uint32_t packetId     = 123; // Wrong packet id!!!
    uint32_t header = ((packetFamily & 0x0000003F) << 26) |
                      ((packetId     & 0x000003FF) << 16);

    // Create the Request Counter Directory packet
    Packet requestCounterDirectoryPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(requestCounterDirectoryPacket));

    // Wait for a bit (must at least be the delay value of the mock profiling connection) to make sure that
    // the Create the Request Counter packet gets processed by the profiling service
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Check that the expected error has occurred and logged to the standard output
    BOOST_CHECK(boost::contains(ss.str(), "Functor with requested PacketId=123 and Version=4194304 does not exist"));

    // The Connection Acknowledged Command Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_CASE(CheckProfilingServiceGoodRequestCounterDirectoryPacket)
{
    // Swap the profiling connection factory in the profiling service instance with our mock one
    SwapProfilingConnectionFactoryHelper helper;

    // Reset the profiling service to the uninitialized state
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    ProfilingService& profilingService = ProfilingService::Instance();
    profilingService.ResetExternalProfilingOptions(options, true);

    // Bring the profiling service to the "Active" state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Uninitialised);
    profilingService.Update(); // Initialize the counter directory
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::NotConnected);
    profilingService.Update(); // Create the profiling connection
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::WaitingForAck);
    profilingService.Update(); // Start the threads
    helper.ForceTransitionToState(ProfilingState::Active);
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

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
    uint32_t packetId     = 3;
    uint32_t header = ((packetFamily & 0x0000003F) << 26) |
                      ((packetId     & 0x000003FF) << 16);

    // Create the Request Counter Directory packet
    Packet requestCounterDirectoryPacket(header);

    // Write the packet to the mock profiling connection
    mockProfilingConnection->WritePacket(std::move(requestCounterDirectoryPacket));

    // Wait for a bit (must at least be the delay value of the mock profiling connection) to make sure that
    // the Create the Request Counter packet gets processed by the profiling service
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // The Connection Acknowledged Command Handler should not have updated the profiling state
    BOOST_CHECK(profilingService.GetCurrentState() == ProfilingState::Active);

    // Check that the mock profiling connection contains one Counter Directory packet
    const std::vector<uint32_t> writtenData = mockProfilingConnection->GetWrittenData();
    BOOST_TEST(writtenData.size() == 1);
    BOOST_TEST(writtenData[0] == 416); // The size of a valid Counter Directory packet

    // Reset the profiling service to stop any running thread
    options.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options, true);
}

BOOST_AUTO_TEST_SUITE_END()
