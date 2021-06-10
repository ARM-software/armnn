//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <test/RuntimeTests.hpp>

#include <LeakChecking.hpp>

#include <backendsCommon/test/RuntimeTestImpl.hpp>
#include <test/ProfilingTestUtils.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

#ifdef WITH_VALGRIND
#include <valgrind/memcheck.h>
#endif

TEST_SUITE("ClRuntime")
{
TEST_CASE("RuntimeValidateGpuDeviceSupportLayerNoFallback")
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    CHECK(optNet);

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    CHECK(runtime->LoadNetwork(netId, std::move(optNet)) == armnn::Status::Success);
}

#ifdef ARMNN_LEAK_CHECKING_ENABLED
TEST_CASE("RuntimeMemoryLeaksGpuAcc")
{
    CHECK(ARMNN_LEAK_CHECKER_IS_ACTIVE());
    armnn::IRuntime::CreationOptions options;
    armnn::RuntimeImpl runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    {
        // Do a warmup of this so we make sure that all one-time
        // initialization happens before we do the leak checking.
        CreateAndDropDummyNetwork(backends, runtime);
    }

    {
        ARMNN_SCOPED_LEAK_CHECKER("LoadAndUnloadNetworkGpuAcc");
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
#endif

// Note: this part of the code is due to be removed when we fully trust the gperftools based results.
#if defined(WITH_VALGRIND)
TEST_CASE("RuntimeMemoryUsage")
{
    // From documentation:

    // This means that no pointer to the block can be found. The block is classified as "lost",
    // because the programmer could not possibly have freed it at program exit, since no pointer to it exists.
    unsigned long leakedBefore = 0;
    unsigned long leakedAfter = 0;

    // A start-pointer or chain of start-pointers to the block is found. Since the block is still pointed at,
    // the programmer could, at least in principle, have freed it before program exit.
    // We want to test this in case memory is not freed as early as it could have been.
    unsigned long reachableBefore = 0;
    unsigned long reachableAfter = 0;

    // Needed as out params but we don't test them.
    unsigned long dubious = 0;
    unsigned long suppressed = 0;

    // Ensure that runtime is large enough before checking for memory leaks.
    // Otherwise, when loading the network, it will automatically reserve memory that won't be released
    // until destruction.
    armnn::NetworkId networkIdentifier;
    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    // Checks for leaks before we load the network and record them so that we can see the delta after unloading.
    VALGRIND_DO_QUICK_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedBefore, dubious, reachableBefore, suppressed);

    // build a mock-network and load it into the runtime
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    {
        armnn::TensorInfo inputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);
        armnn::TensorInfo outputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);

        armnn::INetworkPtr mockNetwork(armnn::INetwork::Create());

        armnn::IConnectableLayer* input = mockNetwork->AddInputLayer(0, "input");
        armnn::IConnectableLayer* layer = mockNetwork->AddActivationLayer(armnn::ActivationDescriptor(), "test");
        armnn::IConnectableLayer* output = mockNetwork->AddOutputLayer(0, "output");

        input->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
        layer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // Sets the tensors in the network.
        input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
        layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        // optimize the network
        armnn::IOptimizedNetworkPtr optNet = Optimize(*mockNetwork, backends, runtime.GetDeviceSpec());

        runtime.LoadNetwork(networkIdentifier, std::move(optNet));
    }

    runtime.UnloadNetwork(networkIdentifier);

    VALGRIND_DO_ADDED_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedAfter, dubious, reachableAfter, suppressed);

    // If we're not running under Valgrind, these vars will have been initialised to 0, so this will always pass.
    CHECK(leakedBefore == leakedAfter);

    // Add resonable threshold after and before running valgrind with the ACL clear cache function.
    // TODO Threshold set to 80k until the root cause of the memory leakage is found and fixed. Revert threshold
    // value to 1024 when fixed.
    CHECK(static_cast<long>(reachableAfter) - static_cast<long>(reachableBefore) < 81920);

    // These are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning.
    IgnoreUnused(dubious);
    IgnoreUnused(suppressed);
}
#endif

TEST_CASE("ProfilingPostOptimisationStructureGpuAcc")
{
    VerifyPostOptimisationStructureTestImpl(armnn::Compute::GpuAcc);
}

}
