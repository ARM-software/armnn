//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <test/RuntimeTests.hpp>

#include <LeakChecking.hpp>

#include <backendsCommon/test/RuntimeTestImpl.hpp>

#include <doctest/doctest.h>


#ifdef ARMNN_LEAK_CHECKING_ENABLED
TEST_SUITE("RefRuntime")
{
TEST_CASE("RuntimeMemoryLeaksCpuRef")
{
    CHECK(ARMNN_LEAK_CHECKER_IS_ACTIVE());

    armnn::IRuntime::CreationOptions options;
    armnn::RuntimeImpl runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    {
        // Do a warmup of this so we make sure that all one-time
        // initialization happens before we do the leak checking.
        CreateAndDropDummyNetwork(backends, runtime);
    }

    {
        ARMNN_SCOPED_LEAK_CHECKER("LoadAndUnloadNetworkCpuRef");
        CHECK(ARMNN_NO_LEAKS_IN_SCOPE());
        // In the second run we check for all remaining memory
        // in use after the network was unloaded. If there is any
        // then it will be treated as a memory leak.
        CreateAndDropDummyNetwork(backends, runtime);
        CHECK(ARMNN_NO_LEAKS_IN_SCOPE());
        CHECK(ARMNN_BYTES_LEAKED_IN_SCOPE() == 0);
        CHECK(ARMNN_OBJECTS_LEAKED_IN_SCOPE() == 0);
    }
}
}
#endif

