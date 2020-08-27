//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Logging.hpp>
#include <armnn/Utils.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

#include <backendsCommon/test/LayerTests.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>
#include "TensorHelpers.hpp"
#include <boost/test/unit_test.hpp>

inline void ConfigureLoggingTest()
{
    // Configures logging for both the ARMNN library and this test program.
    armnn::ConfigureLogging(true, true, armnn::LogSeverity::Fatal);
}

// The following macros require the caller to have defined FactoryType, with one of the following using statements:
//
//      using FactoryType = armnn::RefWorkloadFactory;
//      using FactoryType = armnn::ClWorkloadFactory;
//      using FactoryType = armnn::NeonWorkloadFactory;

/// Executes BOOST_TEST on CompareTensors() return value so that the predicate_result message is reported.
/// If the test reports itself as not supported then the tensors are not compared.
/// Additionally this checks that the supportedness reported by the test matches the name of the test.
/// Unsupported tests must be 'tagged' by including "UNSUPPORTED" in their name.
/// This is useful because it clarifies that the feature being tested is not actually supported
/// (a passed test with the name of a feature would imply that feature was supported).
/// If support is added for a feature, the test case will fail because the name incorrectly contains UNSUPPORTED.
/// If support is removed for a feature, the test case will fail because the name doesn't contain UNSUPPORTED.
template <typename T, std::size_t n>
void CompareTestResultIfSupported(const std::string& testName, const LayerTestResult<T, n>& testResult)
{
    bool testNameIndicatesUnsupported = testName.find("UNSUPPORTED") != std::string::npos;
    BOOST_CHECK_MESSAGE(testNameIndicatesUnsupported != testResult.supported,
        "The test name does not match the supportedness it is reporting");
    if (testResult.supported)
    {
        BOOST_TEST(CompareTensors(testResult.output, testResult.outputExpected, testResult.compareBoolean));
    }
}

template <typename T, std::size_t n>
void CompareTestResultIfSupported(const std::string& testName, const std::vector<LayerTestResult<T, n>>& testResult)
{
    bool testNameIndicatesUnsupported = testName.find("UNSUPPORTED") != std::string::npos;
    for (unsigned int i = 0; i < testResult.size(); ++i)
    {
        BOOST_CHECK_MESSAGE(testNameIndicatesUnsupported != testResult[i].supported,
            "The test name does not match the supportedness it is reporting");
        if (testResult[i].supported)
        {
            BOOST_TEST(CompareTensors(testResult[i].output, testResult[i].outputExpected));
        }
    }
}

template<typename FactoryType, typename TFuncPtr, typename... Args>
void RunTestFunction(const char* testName, TFuncPtr testFunction, Args... args)
{
    std::unique_ptr<armnn::Profiler> profiler = std::make_unique<armnn::Profiler>();
    armnn::ProfilerManager::GetInstance().RegisterProfiler(profiler.get());

    auto memoryManager = WorkloadFactoryHelper<FactoryType>::GetMemoryManager();
    FactoryType workloadFactory = WorkloadFactoryHelper<FactoryType>::GetFactory(memoryManager);

    auto testResult = (*testFunction)(workloadFactory, memoryManager, args...);
    CompareTestResultIfSupported(testName, testResult);

    armnn::ProfilerManager::GetInstance().RegisterProfiler(nullptr);
}


template<typename FactoryType, typename TFuncPtr, typename... Args>
void RunTestFunctionUsingTensorHandleFactory(const char* testName, TFuncPtr testFunction, Args... args)
{
    std::unique_ptr<armnn::Profiler> profiler = std::make_unique<armnn::Profiler>();
    armnn::ProfilerManager::GetInstance().RegisterProfiler(profiler.get());

    auto memoryManager = WorkloadFactoryHelper<FactoryType>::GetMemoryManager();
    FactoryType workloadFactory = WorkloadFactoryHelper<FactoryType>::GetFactory(memoryManager);

    auto tensorHandleFactory = WorkloadFactoryHelper<FactoryType>::GetTensorHandleFactory(memoryManager);

    auto testResult = (*testFunction)(workloadFactory, memoryManager, tensorHandleFactory, args...);
    CompareTestResultIfSupported(testName, testResult);

    armnn::ProfilerManager::GetInstance().RegisterProfiler(nullptr);
}

#define ARMNN_SIMPLE_TEST_CASE(TestName, TestFunction) \
    BOOST_AUTO_TEST_CASE(TestName) \
    { \
        TestFunction(); \
    }

#define ARMNN_AUTO_TEST_CASE(TestName, TestFunction, ...) \
    BOOST_AUTO_TEST_CASE(TestName) \
    { \
        RunTestFunction<FactoryType>(#TestName, &TestFunction, ##__VA_ARGS__); \
    }

#define ARMNN_AUTO_TEST_CASE_WITH_THF(TestName, TestFunction, ...) \
    BOOST_AUTO_TEST_CASE(TestName) \
    { \
        RunTestFunctionUsingTensorHandleFactory<FactoryType>(#TestName, &TestFunction, ##__VA_ARGS__); \
    }

template<typename FactoryType, typename TFuncPtr, typename... Args>
void CompareRefTestFunction(const char* testName, TFuncPtr testFunction, Args... args)
{
    auto memoryManager = WorkloadFactoryHelper<FactoryType>::GetMemoryManager();
    FactoryType workloadFactory = WorkloadFactoryHelper<FactoryType>::GetFactory(memoryManager);

    armnn::RefWorkloadFactory refWorkloadFactory;

    auto testResult = (*testFunction)(workloadFactory, memoryManager, refWorkloadFactory, args...);
    CompareTestResultIfSupported(testName, testResult);
}

template<typename FactoryType, typename TFuncPtr, typename... Args>
void CompareRefTestFunctionUsingTensorHandleFactory(const char* testName, TFuncPtr testFunction, Args... args)
{
    auto memoryManager = WorkloadFactoryHelper<FactoryType>::GetMemoryManager();
    FactoryType workloadFactory = WorkloadFactoryHelper<FactoryType>::GetFactory(memoryManager);

    armnn::RefWorkloadFactory refWorkloadFactory;
    auto tensorHandleFactory = WorkloadFactoryHelper<FactoryType>::GetTensorHandleFactory(memoryManager);
    auto refTensorHandleFactory =
        RefWorkloadFactoryHelper::GetTensorHandleFactory(memoryManager);

    auto testResult = (*testFunction)(
        workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory, args...);
    CompareTestResultIfSupported(testName, testResult);
}

#define ARMNN_COMPARE_REF_AUTO_TEST_CASE(TestName, TestFunction, ...) \
    BOOST_AUTO_TEST_CASE(TestName) \
    { \
        CompareRefTestFunction<FactoryType>(#TestName, &TestFunction, ##__VA_ARGS__); \
    }

#define ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(TestName, TestFunction, ...) \
    BOOST_AUTO_TEST_CASE(TestName) \
    { \
        CompareRefTestFunctionUsingTensorHandleFactory<FactoryType>(#TestName, &TestFunction, ##__VA_ARGS__); \
    }

#define ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(TestName, Fixture, TestFunction, ...) \
    BOOST_FIXTURE_TEST_CASE(TestName, Fixture) \
    { \
        CompareRefTestFunction<FactoryType>(#TestName, &TestFunction, ##__VA_ARGS__); \
    }

#define ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(TestName, Fixture, TestFunction, ...) \
    BOOST_FIXTURE_TEST_CASE(TestName, Fixture) \
    { \
        CompareRefTestFunctionUsingTensorHandleFactory<FactoryType>(#TestName, &TestFunction, ##__VA_ARGS__); \
    }
