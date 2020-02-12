//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MockBackend.hpp"
#include "MockBackendId.hpp"
#include "Runtime.hpp"

#include <armnn/BackendId.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>

BOOST_AUTO_TEST_SUITE(BackendProfilingTestSuite)

BOOST_AUTO_TEST_CASE(BackendProfilingCounterRegisterMockBackendTest)
{
    // Reset the profiling service to the uninitialized state
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
    armnn::profiling::ProfilingService& profilingService = armnn::profiling::ProfilingService::Instance();
    profilingService.ConfigureProfilingService(options.m_ProfilingOptions, true);

    armnn::MockBackendInitialiser initialiser;
    // Create a runtime
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // Check if the MockBackends 3 dummy counters {0, 1, 2-5 (four cores)} are registered
    armnn::BackendId mockId = armnn::MockBackendId();
    const armnn::profiling::ICounterMappings& counterMap = profilingService.GetCounterMappings();
    BOOST_CHECK(counterMap.GetGlobalId(0, mockId) == 5);
    BOOST_CHECK(counterMap.GetGlobalId(1, mockId) == 6);
    BOOST_CHECK(counterMap.GetGlobalId(2, mockId) == 7);
    BOOST_CHECK(counterMap.GetGlobalId(3, mockId) == 8);
    BOOST_CHECK(counterMap.GetGlobalId(4, mockId) == 9);
    BOOST_CHECK(counterMap.GetGlobalId(5, mockId) == 10);
    options.m_ProfilingOptions.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options.m_ProfilingOptions, true);
}

BOOST_AUTO_TEST_SUITE_END()