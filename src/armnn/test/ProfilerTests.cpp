//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include <boost/algorithm/string.hpp>

#include <memory>
#include <thread>

#include <armnn/TypesUtils.hpp>
#include <Profiling.hpp>

namespace armnn
{

size_t GetProfilerEventSequenceSize(armnn::Profiler* profiler)
{
    if (!profiler)
    {
        return static_cast<size_t>(-1);
    }

    return profiler->m_EventSequence.size();
}
} // namespace armnn

namespace
{

void RegisterUnregisterProfilerSingleThreadImpl()
{
    // Important! Regular assertions must be used in this function for testing (rather than
    // BOOST_TEST macros) otherwise multi-threading tests would randomly fail.

    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    // Check that there's no profiler registered for this thread.
    assert(!profilerManager.GetProfiler());

    // Create and register a profiler for this thread.
    std::unique_ptr<armnn::Profiler> profiler = std::make_unique<armnn::Profiler>();
    profilerManager.RegisterProfiler(profiler.get());

    // Check that on a single thread we get the same profiler we registered.
    assert(profiler.get() == profilerManager.GetProfiler());

    // Destroy the profiler.
    profiler.reset();

    // Check that the profiler has been un-registered for this thread.
    assert(!profilerManager.GetProfiler());
}

} // namespace

BOOST_AUTO_TEST_SUITE(Profiler)

BOOST_AUTO_TEST_CASE(EnableDisableProfiling)
{
    std::unique_ptr<armnn::Profiler> profiler = std::make_unique<armnn::Profiler>();

    // Check that profiling is disabled by default.
    BOOST_TEST(!profiler->IsProfilingEnabled());

    // Enable profiling.
    profiler->EnableProfiling(true);

    // Check that profiling is enabled.
    BOOST_TEST(profiler->IsProfilingEnabled());

    // Disable profiling.
    profiler->EnableProfiling(false);

    // Check that profiling is disabled.
    BOOST_TEST(!profiler->IsProfilingEnabled());
}

BOOST_AUTO_TEST_CASE(RegisterUnregisterProfilerSingleThread)
{
    RegisterUnregisterProfilerSingleThreadImpl();
}

BOOST_AUTO_TEST_CASE(RegisterUnregisterProfilerMultipleThreads)
{
    std::thread thread1([]() { RegisterUnregisterProfilerSingleThreadImpl(); });
    std::thread thread2([]() { RegisterUnregisterProfilerSingleThreadImpl(); });
    std::thread thread3([]() { RegisterUnregisterProfilerSingleThreadImpl(); });

    thread1.join();
    thread2.join();
    thread3.join();
}

BOOST_AUTO_TEST_CASE(ProfilingMacros)
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    { // --- No profiler ---

        // Check that there's no profiler registered for this thread.
        BOOST_TEST(!profilerManager.GetProfiler());

        // Test scoped event.
        { ARMNN_SCOPED_PROFILING_EVENT(armnn::Compute::CpuAcc, "test"); }

        // Check that we still cannot get a profiler for this thread.
        BOOST_TEST(!profilerManager.GetProfiler());
    }

    // Create and register a profiler for this thread.
    std::unique_ptr<armnn::Profiler> profiler = std::make_unique<armnn::Profiler>();
    profilerManager.RegisterProfiler(profiler.get());

    { // --- Profiler, but profiling disabled ---

        // Get current event sequence size.
        size_t eventSequenceSizeBefore = armnn::GetProfilerEventSequenceSize(profiler.get());

        // Test scoped macro.
        { ARMNN_SCOPED_PROFILING_EVENT(armnn::Compute::CpuAcc, "test"); }

        // Check that no profiling event has been added to the sequence.
        size_t eventSequenceSizeAfter = armnn::GetProfilerEventSequenceSize(profiler.get());
        BOOST_TEST(eventSequenceSizeBefore == eventSequenceSizeAfter);
    }

    // Enable profiling.
    profiler->EnableProfiling(true);

    { // --- Profiler, and profiling enabled ---

        // Get current event sequence size.
        size_t eventSequenceSizeBefore = armnn::GetProfilerEventSequenceSize(profiler.get());

        // Test scoped macro.
        { ARMNN_SCOPED_PROFILING_EVENT(armnn::Compute::CpuAcc, "test"); }

        // Check that a profiling event has been added to the sequence.
        size_t eventSequenceSizeAfter = armnn::GetProfilerEventSequenceSize(profiler.get());
        BOOST_TEST(eventSequenceSizeAfter == eventSequenceSizeBefore + 1);
    }

    // Disable profiling here to not print out anything on stdout.
    profiler->EnableProfiling(false);
}

BOOST_AUTO_TEST_CASE(RuntimeLoadNetwork)
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    // Check that there's no profiler registered for this thread.
    BOOST_TEST(!profilerManager.GetProfiler());

    // Build a mock-network and load it into the runtime.
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::NetworkId networkIdentifier = 1;
    armnn::INetworkPtr mockNetwork(armnn::INetwork::Create());
    mockNetwork->AddInputLayer(0, "test layer");
    std::vector<armnn::Compute> backends = { armnn::Compute::CpuRef };
    runtime->LoadNetwork(networkIdentifier, armnn::Optimize(*mockNetwork, backends, runtime->GetDeviceSpec()));

    // Check that now there's a profiler registered for this thread (created and registered by the loading the network).
    BOOST_TEST(profilerManager.GetProfiler());

    // Unload the network.
    runtime->UnloadNetwork(networkIdentifier);

    // Check that the profiler has been un-registered for this thread.
    BOOST_TEST(!profilerManager.GetProfiler());
}

BOOST_AUTO_TEST_CASE(WriteEventResults)
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profileManager = armnn::ProfilerManager::GetInstance();

    // Create and register a profiler for this thread.
    std::unique_ptr<armnn::Profiler> profiler = std::make_unique<armnn::Profiler>();
    profileManager.RegisterProfiler(profiler.get());

    // Enable profiling.
    profiler->EnableProfiling(true);

    { // --- Profiler, and profiling enabled ---

        // Get current event sequence size.
        size_t eventSequenceSizeBefore = armnn::GetProfilerEventSequenceSize(profiler.get());

        // Test scoped macro.
        {
            // Need to directly create a ScopedProfilingEvent as the one created by the macro falls out of scope
            // immediately causing the Event.Stop() function method to be called immediately after the Event.Start()
            // function resulting in periodic test failures on the Dent and Smith HiKeys
            armnn::ScopedProfilingEvent testEvent(armnn::Compute::CpuAcc, "test", armnn::WallClockTimer());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Check that a profiling event has been added to the sequence.
        size_t eventSequenceSizeAfter = armnn::GetProfilerEventSequenceSize(profiler.get());
        BOOST_TEST(eventSequenceSizeAfter == eventSequenceSizeBefore + 1);

        boost::test_tools::output_test_stream output;
        profiler->AnalyzeEventsAndWriteResults(output);
        BOOST_TEST(!output.is_empty(false));

        // output should contain event name 'test'
        BOOST_CHECK(boost::contains(output.str(), "test"));

        // output should contain headers
        BOOST_CHECK(boost::contains(output.str(), "Event Sequence - Name"));
        BOOST_CHECK(boost::contains(output.str(), "Event Stats - Name"));
        BOOST_CHECK(boost::contains(output.str(), "Total"));
        BOOST_CHECK(boost::contains(output.str(), "Device"));
        // output should contain compute device 'CpuAcc'
        BOOST_CHECK(boost::contains(output.str(), "CpuAcc"));
        // output should not contain un-readable numbers
        BOOST_CHECK(!(boost::contains(output.str(), "e+")));
        // output should not contain un-readable numbers
        BOOST_CHECK(!(boost::contains(output.str(), "+")));
        // output should not contain zero value
        BOOST_CHECK(!(boost::contains(output.str(), " 0 ")));
    }

    // Disable profiling here to not print out anything on stdout.
    profiler->EnableProfiling(false);
}

BOOST_AUTO_TEST_SUITE_END()
