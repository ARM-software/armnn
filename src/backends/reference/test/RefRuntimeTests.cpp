//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <test/RuntimeTests.hpp>

#include <LeakChecking.hpp>

#include <backendsCommon/test/RuntimeTestImpl.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(RefRuntime)

#ifdef ARMNN_LEAK_CHECKING_ENABLED
BOOST_AUTO_TEST_CASE(RuntimeMemoryLeaksCpuRef)
{
    BOOST_TEST(ARMNN_LEAK_CHECKER_IS_ACTIVE());

    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    {
        // Do a warmup of this so we make sure that all one-time
        // initialization happens before we do the leak checking.
        CreateAndDropDummyNetwork(backends, runtime);
    }

    {
        ARMNN_SCOPED_LEAK_CHECKER("LoadAndUnloadNetworkCpuRef");
        BOOST_TEST(ARMNN_NO_LEAKS_IN_SCOPE());
        // In the second run we check for all remaining memory
        // in use after the network was unloaded. If there is any
        // then it will be treated as a memory leak.
        CreateAndDropDummyNetwork(backends, runtime);
        BOOST_TEST(ARMNN_NO_LEAKS_IN_SCOPE());
        BOOST_TEST(ARMNN_BYTES_LEAKED_IN_SCOPE() == 0);
        BOOST_TEST(ARMNN_OBJECTS_LEAKED_IN_SCOPE() == 0);
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()