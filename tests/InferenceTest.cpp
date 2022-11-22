//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "InferenceTest.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnnUtils/Filesystem.hpp>

#include "../src/armnn/Profiling.hpp"
#include <cxxopts/cxxopts.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <array>

using namespace std;
using namespace std::chrono;
using namespace armnn::test;

namespace armnn
{
namespace test
{
/// Parse the command line of an ArmNN inference test program.
/// \return false if any error occurred during options processing, otherwise true
bool ParseCommandLine(int argc, char** argv, IInferenceTestCaseProvider& testCaseProvider,
    InferenceTestOptions& outParams)
{
    cxxopts::Options options("InferenceTest", "Inference iteration parameters");

    try
    {
        // Adds generic options needed for all inference tests.
        options
            .allow_unrecognised_options()
            .add_options()
                ("h,help", "Display help messages")
                ("i,iterations", "Sets the number of inferences to perform. If unset, will only be run once.",
                 cxxopts::value<unsigned int>(outParams.m_IterationCount)->default_value("0"))
                ("inference-times-file",
                 "If non-empty, each individual inference time will be recorded and output to this file",
                 cxxopts::value<std::string>(outParams.m_InferenceTimesFile)->default_value(""))
                ("e,event-based-profiling", "Enables built in profiler. If unset, defaults to off.",
                 cxxopts::value<bool>(outParams.m_EnableProfiling)->default_value("0"));

        std::vector<std::string> required; //to be passed as reference to derived inference tests

        // Adds options specific to the ITestCaseProvider.
        testCaseProvider.AddCommandLineOptions(options, required);

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return false;
        }

        CheckRequiredOptions(result, required);

    }
    catch (const cxxopts::OptionException& e)
    {
        std::cerr << e.what() << std::endl << options.help() << std::endl;
        return false;
    }
    catch (const std::exception& e)
    {
        ARMNN_ASSERT_MSG(false, "Caught unexpected exception");
        std::cerr << "Fatal internal error: " << e.what() << std::endl;
        return false;
    }

    if (!testCaseProvider.ProcessCommandLineOptions(outParams))
    {
        return false;
    }

    return true;
}

bool ValidateDirectory(std::string& dir)
{
    if (dir.empty())
    {
        std::cerr << "No directory specified" << std::endl;
        return false;
    }

    if (dir[dir.length() - 1] != '/')
    {
        dir += "/";
    }

    if (!fs::exists(dir))
    {
        std::cerr << "Given directory " << dir << " does not exist" << std::endl;
        return false;
    }

    if (!fs::is_directory(dir))
    {
        std::cerr << "Given directory [" << dir << "] is not a directory" << std::endl;
        return false;
    }

    return true;
}

bool InferenceTest(const InferenceTestOptions& params,
    const std::vector<unsigned int>& defaultTestCaseIds,
    IInferenceTestCaseProvider& testCaseProvider)
{
#if !defined (NDEBUG)
    if (params.m_IterationCount > 0) // If just running a few select images then don't bother to warn.
    {
        ARMNN_LOG(warning) << "Performance test running in DEBUG build - results may be inaccurate.";
    }
#endif

    double totalTime = 0;
    unsigned int nbProcessed = 0;
    bool success = true;

    // Opens the file to write inference times too, if needed.
    ofstream inferenceTimesFile;
    const bool recordInferenceTimes = !params.m_InferenceTimesFile.empty();
    if (recordInferenceTimes)
    {
        inferenceTimesFile.open(params.m_InferenceTimesFile.c_str(), ios_base::trunc | ios_base::out);
        if (!inferenceTimesFile.good())
        {
            ARMNN_LOG(error) << "Failed to open inference times file for writing: "
                << params.m_InferenceTimesFile;
            return false;
        }
    }

    // Create a profiler and register it for the current thread.
    std::unique_ptr<IProfiler> profiler = std::make_unique<IProfiler>();
    ProfilerManager::GetInstance().RegisterProfiler(profiler.get());

    // Enable profiling if requested.
    profiler->EnableProfiling(params.m_EnableProfiling);

    // Run a single test case to 'warm-up' the model. The first one can sometimes take up to 10x longer
    std::unique_ptr<IInferenceTestCase> warmupTestCase = testCaseProvider.GetTestCase(0);
    if (warmupTestCase == nullptr)
    {
        ARMNN_LOG(error) << "Failed to load test case";
        return false;
    }

    try
    {
        warmupTestCase->Run();
    }
    catch (const TestFrameworkException& testError)
    {
        ARMNN_LOG(error) << testError.what();
        return false;
    }

    const unsigned int nbTotalToProcess = params.m_IterationCount > 0 ? params.m_IterationCount
        : static_cast<unsigned int>(defaultTestCaseIds.size());

    for (; nbProcessed < nbTotalToProcess; nbProcessed++)
    {
        const unsigned int testCaseId = params.m_IterationCount > 0 ? nbProcessed : defaultTestCaseIds[nbProcessed];
        std::unique_ptr<IInferenceTestCase> testCase = testCaseProvider.GetTestCase(testCaseId);

        if (testCase == nullptr)
        {
            ARMNN_LOG(error) << "Failed to load test case";
            return false;
        }

        time_point<high_resolution_clock> predictStart;
        time_point<high_resolution_clock> predictEnd;

        TestCaseResult result = TestCaseResult::Ok;

        try
        {
            predictStart = high_resolution_clock::now();

            testCase->Run();

            predictEnd = high_resolution_clock::now();

            // duration<double> will convert the time difference into seconds as a double by default.
            double timeTakenS = duration<double>(predictEnd - predictStart).count();
            totalTime += timeTakenS;

            // Outputss inference times, if needed.
            if (recordInferenceTimes)
            {
                inferenceTimesFile << testCaseId << " " << (timeTakenS * 1000.0) << std::endl;
            }

            result = testCase->ProcessResult(params);

        }
        catch (const TestFrameworkException& testError)
        {
            ARMNN_LOG(error) << testError.what();
            result = TestCaseResult::Abort;
        }

        switch (result)
        {
        case TestCaseResult::Ok:
            break;
        case TestCaseResult::Abort:
            return false;
        case TestCaseResult::Failed:
            // This test failed so we will fail the entire program eventually, but keep going for now.
            success = false;
            break;
        default:
            ARMNN_ASSERT_MSG(false, "Unexpected TestCaseResult");
            return false;
        }
    }

    const double averageTimePerTestCaseMs = totalTime / nbProcessed * 1000.0f;

    ARMNN_LOG(info) << std::fixed << std::setprecision(3) <<
        "Total time for " << nbProcessed << " test cases: " << totalTime << " seconds";
    ARMNN_LOG(info) << std::fixed << std::setprecision(3) <<
        "Average time per test case: " << averageTimePerTestCaseMs << " ms";

    // if profiling is enabled print out the results
    if (profiler && profiler->IsProfilingEnabled())
    {
        profiler->Print(std::cout);
    }

    if (!success)
    {
        ARMNN_LOG(error) << "One or more test cases failed";
        return false;
    }

    return testCaseProvider.OnInferenceTestFinished();
}

} // namespace test

} // namespace armnn
