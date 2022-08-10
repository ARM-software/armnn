//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/IRuntime.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <thread>

#include <Profiling.hpp>
#include <armnn/Optional.hpp>
#include <armnnUtils/TensorUtils.hpp>

namespace armnn
{

size_t GetProfilerEventSequenceSize(armnn::IProfiler* profiler)
{
    if (!profiler)
    {
        return static_cast<size_t>(-1);
    }

    return profiler->pProfilerImpl->m_EventSequence.size();
}
} // namespace armnn

namespace
{

void RegisterUnregisterProfilerSingleThreadImpl(bool &res)
{
    // Important! Don't use CHECK macros in this function as they
    // seem to have problems when used in threads

    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    // Check that there's no profiler registered for this thread.
    res = !profilerManager.GetProfiler();

    // Create and register a profiler for this thread.
    std::unique_ptr<armnn::IProfiler> profiler = std::make_unique<armnn::IProfiler>();
    profilerManager.RegisterProfiler(profiler.get());

    // Check that on a single thread we get the same profiler we registered.
    res &= profiler.get() == profilerManager.GetProfiler();

    // Destroy the profiler.
    profiler.reset();

    // Check that the profiler has been un-registered for this thread.
    res &= !profilerManager.GetProfiler();

    armnn::ProfilerManager::GetInstance().RegisterProfiler(nullptr);
}

} // namespace

TEST_SUITE("Profiler")
{
TEST_CASE("EnableDisableProfiling")
{
    std::unique_ptr<armnn::IProfiler> profiler = std::make_unique<armnn::IProfiler>();

    // Check that profiling is disabled by default.
    CHECK(!profiler->IsProfilingEnabled());

    // Enable profiling.
    profiler->EnableProfiling(true);

    // Check that profiling is enabled.
    CHECK(profiler->IsProfilingEnabled());

    // Disable profiling.
    profiler->EnableProfiling(false);

    // Check that profiling is disabled.
    CHECK(!profiler->IsProfilingEnabled());
}

TEST_CASE("RegisterUnregisterProfilerSingleThread")
{
    bool res = false;
    RegisterUnregisterProfilerSingleThreadImpl(res);
    CHECK(res);
}

TEST_CASE("RegisterUnregisterProfilerMultipleThreads")
{
    bool res[3] = {false, false, false};
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < 3; ++i)
    {
        threads.push_back(std::thread([&res, i]() { RegisterUnregisterProfilerSingleThreadImpl(res[i]); }));
    }
    std::for_each(threads.begin(), threads.end(), [](std::thread& theThread)
    {
        theThread.join();
    });

    for (int i = 0 ; i < 3 ; ++i)
    {
        CHECK(res[i]);
    }
}

TEST_CASE("LayerWorkloadConstructorWithEmptyWorkloadInfo")
{
    // Calling AddDetailsToString with a descriptor that contains no tensors or parameters results
    // in an invalid piece of JSON as the function assumes there will be something after the input
    // and output tensors are printed and leaves a hanging ','
    // This test validates that the fix for that continues to work.
    armnn::ProfilingDetails classOnTest;
    armnn::ArgMinMaxDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    arm::pipe::ProfilingGuid guid;
    classOnTest.AddDetailsToString("NeonArgMinMaxWorkload_Construct", descriptor, workloadInfo, guid);
    std::string result = classOnTest.GetProfilingDetails();
    // Make sure the string ends with a "GUID": "0"\n}"
    REQUIRE(result.find("GUID\": \"0\"\n}") != std::string::npos);
}

TEST_CASE("LayerWorkloadConstructorWithWorkloadInfoOnlyInputTensor")
{
    // Calling AddDetailsToString with a descriptor that contains a single inout tensor and no output
    // tensor or parameters results in an invalid piece of JSON as the function assumes there will be
    // something after the input and output tensors are printed and leaves a hanging ','
    // This test validates that the fix for that continues to work.

    armnn::ProfilingDetails classOnTest;
    armnn::ArgMinMaxDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    arm::pipe::ProfilingGuid guid;
    armnn::TensorInfo inputTensorInfo =
        armnnUtils::GetTensorInfo(1, 1, 1, 1, armnn::DataLayout::NCHW, armnn::DataType::Float32);
    workloadInfo.m_InputTensorInfos.push_back(inputTensorInfo);

    classOnTest.AddDetailsToString("NeonArgMinMaxWorkload_Construct", descriptor, workloadInfo, guid);
    std::string result = classOnTest.GetProfilingDetails();
    // Make sure the string ends with a "Num Dims\": \"4\"\n\t}\n}"
    REQUIRE(result.find("Num Dims\": \"4\"\n\t}\n}") != std::string::npos);
}

TEST_CASE("LayerWorkloadConstructorWithWorkloadInfoInputAndOutputTensorNoParameters")
{
    // Calling AddDetailsToString with a descriptor that contains no tensors or parameters results
    // in an invalid piece of JSON as the function assumes there will be something after the input
    // and output tensors are printed and leaves a hanging ','
    // This test validates that the fix for that continues to work.
    armnn::ProfilingDetails classOnTest;
    armnn::ArgMinMaxDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    arm::pipe::ProfilingGuid guid;
    armnn::TensorInfo inputTensorInfo =
        armnnUtils::GetTensorInfo(1, 1, 1, 1, armnn::DataLayout::NCHW, armnn::DataType::Float32);
    workloadInfo.m_InputTensorInfos.push_back(inputTensorInfo);

    // We'll make the output tensrinfo have 5 dimensions to make errors easier to detect.
    armnn::TensorInfo outputTensorInfo =
        armnnUtils::GetTensorInfo(1, 1, 1, 1, 1, armnn::DataLayout::NCDHW, armnn::DataType::Float32);
    workloadInfo.m_OutputTensorInfos.push_back(outputTensorInfo);

    classOnTest.AddDetailsToString("NeonArgMinMaxWorkload_Construct", descriptor, workloadInfo, guid);
    std::string result = classOnTest.GetProfilingDetails();
    // Make sure the string ends with a "Num Dims\": \"4\"\n\t}\n}"
    REQUIRE(result.find("Num Dims\": \"5\"\n\t}\n}") != std::string::npos);
}

TEST_CASE("LayerWorkloadConstructorWithWorkloadInfoMultipleInputAndMultipleOutputTensorNoParameters")
{
    // Calling AddDetailsToString with a descriptor that contains multiple input and output tensors. This is
    // specifically looking at the usage of "addSeparator" parameter in PrintInfos.
    armnn::ProfilingDetails classOnTest;
    armnn::ArgMinMaxDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    arm::pipe::ProfilingGuid guid;
    armnn::TensorInfo inputTensorInfo =
        armnnUtils::GetTensorInfo(1, 1, 1, 1, armnn::DataLayout::NCHW, armnn::DataType::Float32);
    // Add two inputs.
    workloadInfo.m_InputTensorInfos.push_back(inputTensorInfo);
    workloadInfo.m_InputTensorInfos.push_back(inputTensorInfo);

    // We'll make the output tensrinfo have 5 dimensions to make errors easier to detect.
    armnn::TensorInfo outputTensorInfo =
        armnnUtils::GetTensorInfo(1, 1, 1, 1, 1, armnn::DataLayout::NCDHW, armnn::DataType::Float32);
    // and two outputs.
    workloadInfo.m_OutputTensorInfos.push_back(outputTensorInfo);
    workloadInfo.m_OutputTensorInfos.push_back(outputTensorInfo);

    classOnTest.AddDetailsToString("NeonArgMinMaxWorkload_Construct", descriptor, workloadInfo, guid);
    std::string result = classOnTest.GetProfilingDetails();
    // Look for the piece in the middle between Input 0 and Input 1:
    REQUIRE(result.find("Dims\": \"4\"\n\t},\n\t\"Input 1\": {\n\t\t\"Shape\": \"[1,1,1,1]\"") != std::string::npos);
    // Look for the piece in the middle between Output 0and Output 1:
    REQUIRE(result.find("Dims\": \"5\"\n\t},\n\t\"Output 1\": {\n\t\t\"Shape\": \"[1,1,1,1,1]\"") != std::string::npos);
}

TEST_CASE("ProfilingMacros")
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    { // --- No profiler ---

        // Check that there's no profiler registered for this thread.
        CHECK(!profilerManager.GetProfiler());

        // Test scoped event.
        { ARMNN_SCOPED_PROFILING_EVENT(armnn::Compute::CpuAcc, "test"); }

        // Check that we still cannot get a profiler for this thread.
        CHECK(!profilerManager.GetProfiler());
    }

    // Create and register a profiler for this thread.
    std::unique_ptr<armnn::IProfiler> profiler = std::make_unique<armnn::IProfiler>();
    profilerManager.RegisterProfiler(profiler.get());

    { // --- Profiler, but profiling disabled ---

        // Get current event sequence size.
        size_t eventSequenceSizeBefore = armnn::GetProfilerEventSequenceSize(profiler.get());

        // Test scoped macro.
        { ARMNN_SCOPED_PROFILING_EVENT(armnn::Compute::CpuAcc, "test"); }

        // Check that no profiling event has been added to the sequence.
        size_t eventSequenceSizeAfter = armnn::GetProfilerEventSequenceSize(profiler.get());
        CHECK(eventSequenceSizeBefore == eventSequenceSizeAfter);
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
        CHECK(eventSequenceSizeAfter == eventSequenceSizeBefore + 1);
    }

    // Disable profiling here to not print out anything on stdout.
    profiler->EnableProfiling(false);
}

#if defined(ARMNNREF_ENABLED)

// This test unit needs the reference backend, it's not available if the reference backend is not built

TEST_CASE("RuntimeLoadNetwork")
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    // Check that there's no profiler registered for this thread.
    CHECK(!profilerManager.GetProfiler());

    // Build a mock-network and load it into the runtime.
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::NetworkId networkIdentifier = 1;
    armnn::INetworkPtr mockNetwork(armnn::INetwork::Create());
    mockNetwork->AddInputLayer(0, "test layer");
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    runtime->LoadNetwork(networkIdentifier, armnn::Optimize(*mockNetwork, backends, runtime->GetDeviceSpec()));

    // Check that now there's a profiler registered for this thread (created and registered by the loading the network).
    CHECK(profilerManager.GetProfiler());

    // Unload the network.
    runtime->UnloadNetwork(networkIdentifier);

    // Check that the profiler has been un-registered for this thread.
    CHECK(!profilerManager.GetProfiler());
}

#endif

TEST_CASE("WriteEventResults")
{
    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profileManager = armnn::ProfilerManager::GetInstance();

    // Create and register a profiler for this thread.
    std::unique_ptr<armnn::IProfiler> profiler = std::make_unique<armnn::IProfiler>();
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
            armnn::ScopedProfilingEvent testEvent(armnn::Compute::CpuAcc,
                                                  armnn::EmptyOptional(),
                                                  "test",
                                                  armnn::WallClockTimer());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Check that a profiling event has been added to the sequence.
        size_t eventSequenceSizeAfter = armnn::GetProfilerEventSequenceSize(profiler.get());
        CHECK(eventSequenceSizeAfter == eventSequenceSizeBefore + 1);

        std::ostringstream output;
        profiler->AnalyzeEventsAndWriteResults(output);
        CHECK(!output.str().empty());

        // output should contain event name 'test'
        CHECK(output.str().find("test") != std::string::npos);

        // output should contain headers
        CHECK(output.str().find("Event Sequence - Name") != std::string::npos);
        CHECK(output.str().find("Event Stats - Name") != std::string::npos);
        CHECK(output.str().find("Total") != std::string::npos);
        CHECK(output.str().find("Device") != std::string::npos);
        // output should contain compute device 'CpuAcc'
        CHECK(output.str().find("CpuAcc") != std::string::npos);
        // output should not contain un-readable numbers
        CHECK(output.str().find("e+") == std::string::npos);
        // output should not contain un-readable numbers
        CHECK(output.str().find("+") == std::string::npos);
        // output should not contain zero value
        CHECK(output.str().find(" 0 ") == std::string::npos);
    }

    // Disable profiling here to not print out anything on stdout.
    profiler->EnableProfiling(false);
}

TEST_CASE("ProfilerJsonPrinter")
{
    class TestInstrument : public armnn::Instrument
    {
    public:
        virtual ~TestInstrument() {}
        void Start() override {}
        void Stop() override {}

        std::vector<armnn::Measurement> GetMeasurements() const override
        {
            std::vector<armnn::Measurement> measurements;
            measurements.emplace_back(armnn::Measurement("Measurement1",
                                                         1.0,
                                                         armnn::Measurement::Unit::TIME_MS));
            measurements.emplace_back(armnn::Measurement("Measurement2",
                                                         2.0,
                                                         armnn::Measurement::Unit::TIME_US));
            return measurements;
        }

        const char* GetName() const override
        {
            return "TestInstrument";
        }
    };

    // Get a reference to the profiler manager.
    armnn::ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    // Create and register a profiler for this thread.
    std::unique_ptr<armnn::IProfiler> profiler = std::make_unique<armnn::IProfiler>();
    profilerManager.RegisterProfiler(profiler.get());

    profiler->EnableProfiling(true);

    {
        ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                      armnn::EmptyOptional(),
                                                      "Optimizer",
                                                      TestInstrument())
        ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                      armnn::EmptyOptional(),
                                                      "Level 0",
                                                      TestInstrument())
        {
            {
                ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                              armnn::EmptyOptional(),
                                                              "Level 1A",
                                                              TestInstrument())
            }
        }
    }
    {
        ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                      armnn::EmptyOptional(),
                                                      "LoadedNetwork",
                                                      TestInstrument())
        ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                      armnn::EmptyOptional(),
                                                      "Level 0",
                                                      TestInstrument())
        {
            {
                ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                              armnn::EmptyOptional(),
                                                              "Level 1A",
                                                              TestInstrument())
            }
        }
    }
    {
        // Test scoped macro.
            ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                          armnn::EmptyOptional(),
                                                          "EnqueueWorkload",
                                                          TestInstrument())
            ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                          armnn::EmptyOptional(),
                                                          "Level 0",
                                                          TestInstrument())
            {
                {
                    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                                  armnn::EmptyOptional(),
                                                                  "Level 1A",
                                                                  TestInstrument())
                }
                {
                    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                                  armnn::EmptyOptional(),
                                                                  "Level 1B",
                                                                  TestInstrument())

                    {
                        ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc,
                                                                      armnn::EmptyOptional(),
                                                                      "Level 2A",
                                                                      TestInstrument())
                    }
                }
            }
    }

    std::stringbuf buffer;
    std::ostream json(&buffer);
    profiler->Print(json);

    std::string output = buffer.str();
    armnn::IgnoreUnused(output);

    // Disable profiling here to not print out anything on stdout.
    profiler->EnableProfiling(false);

    // blessed output validated by a human eyeballing the output to make sure it's ok and then copying it here.
    // validation also included running the blessed output through an online json validation site
    std::string blessedOutput("{\n\t\"ArmNN\": {\n\t\t\"optimize_measurements_#1\": {\n\t\t\t\"type\": \"Event\""
                              ",\n\t\t\t\"Measurement1_#1\": {\n\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\"raw\""
                              ": [\n\t\t\t\t\t1.000000\n\t\t\t\t],\n\t\t\t\t\"unit\": \"ms\"\n\t\t\t},\n\t\t\t\""
                              "Measurement2_#1\": {\n\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\"raw\": [\n\t\t\t"
                              "\t\t2.000000\n\t\t\t\t],\n\t\t\t\t\"unit\": \"us\"\n\t\t\t},\n\t\t\t\"Level 0_#2\": {\n"
                              "\t\t\t\t\"type\": \"Event\",\n\t\t\t\t\"Measurement1_#2\": {\n\t\t\t\t\t\"type\": \""
                              "Measurement\",\n\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t1.000000\n\t\t\t\t\t],\n\t\t\t\t\t\""
                              "unit\": \"ms\"\n\t\t\t\t},\n\t\t\t\t\"Measurement2_#2\": {\n\t\t\t\t\t\"type\": \""
                              "Measurement\",\n\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t2.000000\n\t\t\t\t\t],\n\t\t\t\t\t\""
                              "unit\": \"us\"\n\t\t\t\t},\n\t\t\t\t\"Level 1A_#3\": {\n\t\t\t\t\t\"type\": \"Event\",\n"
                              "\t\t\t\t\t\"Measurement1_#3\": {\n\t\t\t\t\t\t\"type\": \"Measurement\",\n"
                              "\t\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t\t1.000000\n\t\t\t\t\t\t],\n\t\t\t\t\t\t\"unit\""
                              ": \"ms\"\n\t\t\t\t\t},\n\t\t\t\t\t\"Measurement2_#3\": {\n\t\t\t\t\t\t\"type\": \""
                              "Measurement\",\n\t\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t\t2.000000\n\t\t\t\t\t\t],\n\t\t\t"
                              "\t\t\t\"unit\": \"us\"\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t},\n\t\t\""
                              "loaded_network_measurements_#4\": {\n\t\t\t\"type\": \"Event\",\n\t\t\t\""
                              "Measurement1_#4\": {\n\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\"raw\": [\n\t\t\t\t"
                              "\t1.000000\n\t\t\t\t],\n\t\t\t\t\"unit\": \"ms\"\n\t\t\t},\n\t\t\t\"Measurement2_#4\""
                              ": {\n\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\"raw\": [\n\t\t\t\t\t2.000000\n"
                              "\t\t\t\t],\n\t\t\t\t\"unit\": \"us\"\n\t\t\t},\n\t\t\t\"Level 0_#5\": {\n\t\t\t\t\""
                              "type\": \"Event\",\n\t\t\t\t\"Measurement1_#5\": {\n\t\t\t\t\t\"type\": \"Measurement\""
                              ",\n\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t1.000000\n\t\t\t\t\t],\n\t\t\t\t\t\"unit\": \""
                              "ms\"\n\t\t\t\t},\n\t\t\t\t\"Measurement2_#5\": {\n\t\t\t\t\t\"type\": \"Measurement\""
                              ",\n\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t2.000000\n\t\t\t\t\t],\n\t\t\t\t\t\"unit\": \"us\""
                              "\n\t\t\t\t},\n\t\t\t\t\"Level 1A_#6\": {\n\t\t\t\t\t\"type\": \"Event\",\n\t\t\t\t\t\""
                              "Measurement1_#6\": {\n\t\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\t\"raw\": [\n"
                              "\t\t\t\t\t\t\t1.000000\n\t\t\t\t\t\t],\n\t\t\t\t\t\t\"unit\": \"ms\"\n\t\t\t\t\t},\n"
                              "\t\t\t\t\t\"Measurement2_#6\": {\n\t\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\t\""
                              "raw\": [\n\t\t\t\t\t\t\t2.000000\n\t\t\t\t\t\t],\n\t\t\t\t\t\t\"unit\": \"us\""
                              "\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t},\n\t\t\"inference_measurements_#7\": {\n"
                              "\t\t\t\"type\": \"Event\",\n\t\t\t\"Measurement1_#7\": {\n\t\t\t\t\"type\": \""
                              "Measurement\",\n\t\t\t\t\"raw\": [\n\t\t\t\t\t1.000000\n\t\t\t\t],\n\t\t\t\t\"unit\": \""
                              "ms\"\n\t\t\t},\n\t\t\t\"Measurement2_#7\": {\n\t\t\t\t\"type\": \"Measurement\",\n"
                              "\t\t\t\t\"raw\": [\n\t\t\t\t\t2.000000\n\t\t\t\t],\n\t\t\t\t\"unit\": \"us\"\n\t\t\t},\n"
                              "\t\t\t\"Level 0_#8\": {\n\t\t\t\t\"type\": \"Event\",\n\t\t\t\t\"Measurement1_#8\": {\n"
                              "\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t1.000000\n"
                              "\t\t\t\t\t],\n\t\t\t\t\t\"unit\": \"ms\"\n\t\t\t\t},\n\t\t\t\t\"Measurement2_#8\": {\n"
                              "\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t2.000000\n"
                              "\t\t\t\t\t],\n\t\t\t\t\t\"unit\": \"us\"\n\t\t\t\t},\n\t\t\t\t\"Level 1A_#9\": {\n"
                              "\t\t\t\t\t\"type\": \"Event\",\n\t\t\t\t\t\"Measurement1_#9\": {\n\t\t\t\t\t\t\"type\""
                              ": \"Measurement\",\n\t\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t\t1.000000\n\t\t\t\t\t\t],\n"
                              "\t\t\t\t\t\t\"unit\": \"ms\"\n\t\t\t\t\t},\n\t\t\t\t\t\"Measurement2_#9\": {\n"
                              "\t\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t\t2.000000\n"
                              "\t\t\t\t\t\t],\n\t\t\t\t\t\t\"unit\": \"us\"\n\t\t\t\t\t}\n\t\t\t\t},\n\t\t\t\t\""
                              "Level 1B_#10\": {\n\t\t\t\t\t\"type\": \"Event\",\n\t\t\t\t\t\"Measurement1_#10\""
                              ": {\n\t\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t\t"
                              "1.000000\n\t\t\t\t\t\t],\n\t\t\t\t\t\t\"unit\": \"ms\"\n\t\t\t\t\t},\n\t\t\t\t\t\""
                              "Measurement2_#10\": {\n\t\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\t\"raw\""
                              ": [\n\t\t\t\t\t\t\t2.000000\n\t\t\t\t\t\t],\n\t\t\t\t\t\t\"unit\": \"us\"\n"
                              "\t\t\t\t\t},\n\t\t\t\t\t\"Level 2A_#11\": {\n\t\t\t\t\t\t\"type\": \"Event\",\n\t\t\t"
                              "\t\t\t\"Measurement1_#11\": {\n\t\t\t\t\t\t\t\"type\": \"Measurement\",\n\t\t\t\t\t\t"
                              "\t\"raw\": [\n\t\t\t\t\t\t\t\t1.000000\n\t\t\t\t\t\t\t],\n\t\t\t\t\t\t\t\"unit\": \""
                              "ms\"\n\t\t\t\t\t\t},\n\t\t\t\t\t\t\"Measurement2_#11\": {\n\t\t\t\t\t\t\t\"type\": \""
                              "Measurement\",\n\t\t\t\t\t\t\t\"raw\": [\n\t\t\t\t\t\t\t\t2.000000\n\t\t\t\t\t\t\t],\n"
                              "\t\t\t\t\t\t\t\"unit\": \"us\"\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n"
                              "\t}\n}\n");

    CHECK(output == blessedOutput);
    armnn::ProfilerManager::GetInstance().RegisterProfiler(nullptr);
}

}
