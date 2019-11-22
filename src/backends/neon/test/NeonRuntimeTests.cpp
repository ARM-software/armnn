//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <test/RuntimeTests.hpp>

#include <LeakChecking.hpp>

#include <backendsCommon/test/RuntimeTestImpl.hpp>
#include <test/ProfilingTestUtils.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeonRuntime)

BOOST_AUTO_TEST_CASE(RuntimeValidateCpuAccDeviceSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == armnn::Status::Success);
}

#ifdef ARMNN_LEAK_CHECKING_ENABLED
BOOST_AUTO_TEST_CASE(RuntimeMemoryLeaksCpuAcc)
{
    BOOST_TEST(ARMNN_LEAK_CHECKER_IS_ACTIVE());
    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    {
        // Do a warmup of this so we make sure that all one-time
        // initialization happens before we do the leak checking.
        CreateAndDropDummyNetwork(backends, runtime);
    }

    {
        ARMNN_SCOPED_LEAK_CHECKER("LoadAndUnloadNetworkCpuAcc");
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

BOOST_AUTO_TEST_CASE(ProfilingPostOptimisationStructureCpuAcc)
{
    VerifyPostOptimisationStructureTestImpl(armnn::Compute::CpuAcc);
}

BOOST_AUTO_TEST_SUITE_END()