//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include "armnn/TypesUtils.hpp"

#include "armnn/IRuntime.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Descriptors.hpp"
#include "Runtime.hpp"
#include "HeapProfiling.hpp"
#include "LeakChecking.hpp"

#ifdef WITH_VALGRIND
#include "valgrind/memcheck.h"
#endif

namespace armnn
{

void RuntimeLoadedNetworksReserve(armnn::Runtime* runtime)
{
    runtime->m_LoadedNetworks.reserve(1);
}

}

BOOST_AUTO_TEST_SUITE(Runtime)

BOOST_AUTO_TEST_CASE(RuntimeUnloadNetwork)
{
    // build 2 mock-networks and load them into the runtime
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // Mock network 1.
    armnn::NetworkId networkIdentifier1 = 1;
    armnn::INetworkPtr mockNetwork1(armnn::INetwork::Create());
    mockNetwork1->AddInputLayer(0, "test layer");
    std::vector<armnn::Compute> backends = {armnn::Compute::CpuRef};
    runtime->LoadNetwork(networkIdentifier1, Optimize(*mockNetwork1, backends, runtime->GetDeviceSpec()));

    // Mock network 2.
    armnn::NetworkId networkIdentifier2 = 2;
    armnn::INetworkPtr mockNetwork2(armnn::INetwork::Create());
    mockNetwork2->AddInputLayer(0, "test layer");
    runtime->LoadNetwork(networkIdentifier2, Optimize(*mockNetwork2, backends, runtime->GetDeviceSpec()));

    // Unloads one by its networkID.
    BOOST_TEST(runtime->UnloadNetwork(networkIdentifier1) == armnn::Status::Success);

    BOOST_TEST(runtime->UnloadNetwork(networkIdentifier1) == armnn::Status::Failure);
}

// Note: the current builds we don't do valgrind and gperftools based leak checking at the same
//       time, so in practice WITH_VALGRIND and ARMNN_LEAK_CHECKING_ENABLED are exclusive. The
//       valgrind tests can stay for x86 builds, but on hikey Valgrind is just way too slow
//       to be integrated into the CI system.

#ifdef ARMNN_LEAK_CHECKING_ENABLED

struct DisableGlobalLeakChecking
{
    DisableGlobalLeakChecking()
    {
        ARMNN_LOCAL_LEAK_CHECKING_ONLY();
    }
};

BOOST_GLOBAL_FIXTURE(DisableGlobalLeakChecking);

void CreateAndDropDummyNetwork(const std::vector<armnn::Compute>& backends, armnn::Runtime& runtime)
{
    armnn::NetworkId networkIdentifier;
    {
        armnn::TensorInfo inputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);
        armnn::TensorInfo outputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);

        armnn::INetworkPtr network(armnn::INetwork::Create());

        armnn::IConnectableLayer* input = network->AddInputLayer(0, "input");
        armnn::IConnectableLayer* layer = network->AddActivationLayer(armnn::ActivationDescriptor(), "test");
        armnn::IConnectableLayer* output = network->AddOutputLayer(0, "output");

        input->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
        layer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // Sets the tensors in the network.
        input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
        layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        // optimize the network
        armnn::IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime.GetDeviceSpec());

        runtime.LoadNetwork(networkIdentifier, std::move(optNet));
    }

    runtime.UnloadNetwork(networkIdentifier);
}

BOOST_AUTO_TEST_CASE(RuntimeHeapMemoryUsageSanityChecks)
{
    BOOST_TEST(ARMNN_LEAK_CHECKER_IS_ACTIVE());
    {
        ARMNN_SCOPED_LEAK_CHECKER("Sanity_Check_Outer");
        {
            ARMNN_SCOPED_LEAK_CHECKER("Sanity_Check_Inner");
            BOOST_TEST(ARMNN_NO_LEAKS_IN_SCOPE() == true);
            std::unique_ptr<char[]> dummyAllocation(new char[1000]);
            BOOST_CHECK_MESSAGE(ARMNN_NO_LEAKS_IN_SCOPE() == false,
                "A leak of 1000 bytes is expected here. "
                "Please make sure environment variable: HEAPCHECK=draconian is set!");
            BOOST_TEST(ARMNN_BYTES_LEAKED_IN_SCOPE() == 1000);
            BOOST_TEST(ARMNN_OBJECTS_LEAKED_IN_SCOPE() == 1);
        }
        BOOST_TEST(ARMNN_NO_LEAKS_IN_SCOPE());
        BOOST_TEST(ARMNN_BYTES_LEAKED_IN_SCOPE() == 0);
        BOOST_TEST(ARMNN_OBJECTS_LEAKED_IN_SCOPE() == 0);
    }
}

#ifdef ARMCOMPUTECL_ENABLED
BOOST_AUTO_TEST_CASE(RuntimeMemoryLeaksGpuAcc)
{
    BOOST_TEST(ARMNN_LEAK_CHECKER_IS_ACTIVE());
    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    std::vector<armnn::Compute> backends = {armnn::Compute::GpuAcc};
    {
        // Do a warmup of this so we make sure that all one-time
        // initialization happens before we do the leak checking.
        CreateAndDropDummyNetwork(backends, runtime);
    }

    {
        ARMNN_SCOPED_LEAK_CHECKER("LoadAndUnloadNetworkGpuAcc");
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
#endif // ARMCOMPUTECL_ENABLED

#ifdef ARMCOMPUTENEON_ENABLED
BOOST_AUTO_TEST_CASE(RuntimeMemoryLeaksCpuAcc)
{
    BOOST_TEST(ARMNN_LEAK_CHECKER_IS_ACTIVE());
    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    std::vector<armnn::Compute> backends = {armnn::Compute::CpuAcc};
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
#endif // ARMCOMPUTENEON_ENABLED

BOOST_AUTO_TEST_CASE(RuntimeMemoryLeaksCpuRef)
{
    BOOST_TEST(ARMNN_LEAK_CHECKER_IS_ACTIVE());

    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    std::vector<armnn::Compute> backends = {armnn::Compute::CpuRef};
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

#endif // ARMNN_LEAK_CHECKING_ENABLED

// Note: this part of the code is due to be removed when we fully trust the gperftools based results.
#if defined(ARMCOMPUTECL_ENABLED) && defined(WITH_VALGRIND)
BOOST_AUTO_TEST_CASE(RuntimeMemoryUsage)
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
    std::vector<armnn::Compute> backends = {armnn::Compute::GpuAcc};
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
    BOOST_TEST(leakedBefore == leakedAfter);

    // Add resonable threshold after and before running valgrind with the ACL clear cache function.
    // TODO Threshold set to 80k until the root cause of the memory leakage is found and fixed. Revert threshold
    // value to 1024 when fixed.
    BOOST_TEST(static_cast<long>(reachableAfter) - static_cast<long>(reachableBefore) < 81920);

    // These are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning.
    boost::ignore_unused(dubious);
    boost::ignore_unused(suppressed);
}
#endif

// Note: this part of the code is due to be removed when we fully trust the gperftools based results.
#ifdef WITH_VALGRIND
// Run with the following command to get all the amazing output (in the devenv/build folder) :)
// valgrind --leak-check=full --show-leak-kinds=all --log-file=Valgrind_Memcheck_Leak_Report.txt armnn/test/UnitTests
BOOST_AUTO_TEST_CASE(RuntimeMemoryLeak)
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

    armnn::NetworkId networkIdentifier1 = 1;

    // ensure that runtime is large enough before checking for memory leaks
    // otherwise when loading the network it will automatically reserve memory that won't be released until destruction
    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    // Checks for leaks before we load the network and record them so that we can see the delta after unloading.
    VALGRIND_DO_QUICK_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedBefore, dubious, reachableBefore, suppressed);

    // Builds a mock-network and load it into the runtime.
    {
        unsigned int inputShape[] = {1, 7, 1, 1};
        armnn::TensorInfo inputTensorInfo(4, inputShape, armnn::DataType::Float32);

        std::unique_ptr<armnn::Network> mockNetwork1 = std::make_unique<armnn::Network>();
        mockNetwork1->AddInputLayer(0, "test layer");


        std::vector<armnn::Compute> backends = {armnn::Compute::CpuRef};
        runtime.LoadNetwork(networkIdentifier1, Optimize(*mockNetwork1, backends, runtime.GetDeviceSpec()));
    }

    runtime.UnloadNetwork(networkIdentifier1);

    VALGRIND_DO_ADDED_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedAfter, dubious, reachableAfter, suppressed);

    // If we're not running under Valgrind, these vars will have been initialised to 0, so this will always pass.
    BOOST_TEST(leakedBefore == leakedAfter);

    #if defined(ARMCOMPUTECL_ENABLED)
        // reachableBefore == reachableAfter should hold, but on OpenCL with Android we are still
        // not entirely able to control the memory in the OpenCL driver. Testing is showing that
        // after this test (which clears all OpenCL memory) we are clearing a little bit more than
        // we expect, probably depending on the order in which other tests are run.
        BOOST_TEST(reachableBefore - reachableAfter <= 24);
    #else
        BOOST_TEST(reachableBefore == reachableAfter);
    #endif

    BOOST_TEST(reachableBefore >= reachableAfter);

    // These are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning.
    boost::ignore_unused(dubious);
    boost::ignore_unused(suppressed);
}
#endif

#if ARMCOMPUTENEON_ENABLED
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

    std::vector<armnn::Compute> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == armnn::Status::Success);
}
#endif // ARMCOMPUTENEON_ENABLED

#if ARMCOMPUTECL_ENABLED
BOOST_AUTO_TEST_CASE(RuntimeValidateGpuDeviceSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::Compute> backends = { armnn::Compute::GpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == armnn::Status::Success);
}
#endif // ARMCOMPUTECL_ENABLED

BOOST_AUTO_TEST_CASE(RuntimeCpuRef)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc, should be fall back to CpuRef.
    NormalizationDescriptor descriptor;
    IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<armnn::Compute> backends = { armnn::Compute::CpuRef };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

BOOST_AUTO_TEST_CASE(RuntimeFallbackToCpuRef)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc, should be fall back to CpuRef.
    NormalizationDescriptor descriptor;
    IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // Allow fallback to CpuRef.
    std::vector<armnn::Compute> backends = { armnn::Compute::CpuAcc, armnn::Compute::CpuRef };
    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Load it into the runtime. It should succeed.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

BOOST_AUTO_TEST_SUITE_END()
