//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include "armnn/TypesUtils.hpp"

#include "armnn/IRuntime.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Descriptors.hpp"
#include "Runtime.hpp"

#ifdef WITH_VALGRIND
#include "valgrind/memcheck.h"
#endif

#include <boost/core/ignore_unused.hpp>

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
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(armnn::Compute::CpuRef));

    // mock network 1
    armnn::NetworkId networkIdentifier1 = 1;
    armnn::INetworkPtr mockNetwork1(armnn::INetwork::Create());
    mockNetwork1->AddInputLayer(0, "test layer");
    runtime->LoadNetwork(networkIdentifier1, Optimize(*mockNetwork1, runtime->GetDeviceSpec()));

    // mock network 2
    armnn::NetworkId networkIdentifier2 = 2;
    armnn::INetworkPtr mockNetwork2(armnn::INetwork::Create());
    mockNetwork2->AddInputLayer(0, "test layer");
    runtime->LoadNetwork(networkIdentifier2, Optimize(*mockNetwork2, runtime->GetDeviceSpec()));

    // unload one by its networkID
    BOOST_TEST(runtime->UnloadNetwork(networkIdentifier1) == armnn::Status::Success);

    BOOST_TEST(runtime->UnloadNetwork(networkIdentifier1) == armnn::Status::Failure);
}

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
    // We want to test this in case memory is not freed as early as it could have been
    unsigned long reachableBefore = 0;
    unsigned long reachableAfter = 0;

    // needed as out params but we don't test them
    unsigned long dubious = 0;
    unsigned long suppressed = 0;

    // ensure that runtime is large enough before checking for memory leaks
    // otherwise when loading the network it will automatically reserve memory that won't be released until destruction
    armnn::NetworkId networkIdentifier;
    armnn::Runtime runtime(armnn::Compute::GpuAcc);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    // check for leaks before we load the network and record them so that we can see the delta after unloading
    VALGRIND_DO_QUICK_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedBefore, dubious, reachableBefore, suppressed);

    // build a mock-network and load it into the runtime
    {
        armnn::TensorInfo inputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);
        armnn::TensorInfo outputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);

        armnn::INetworkPtr mockNetwork(armnn::INetwork::Create());

        armnn::IConnectableLayer* input = mockNetwork->AddInputLayer(0, "input");
        armnn::IConnectableLayer* layer = mockNetwork->AddActivationLayer(armnn::ActivationDescriptor(), "test");
        armnn::IConnectableLayer* output = mockNetwork->AddOutputLayer(0, "output");

        input->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
        layer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // set the tensors in the network
        input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
        layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        // optimize the network
        armnn::IOptimizedNetworkPtr optNet = Optimize(*mockNetwork, runtime.GetDeviceSpec());

        runtime.LoadNetwork(networkIdentifier, std::move(optNet));
    }

    runtime.UnloadNetwork(networkIdentifier);

    VALGRIND_DO_ADDED_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedAfter, dubious, reachableAfter, suppressed);

    // if we're not running under Valgrind, these vars will have been initialised to 0, so this will always pass
    BOOST_TEST(leakedBefore == leakedAfter);

    // Add resonable threshold after and before running valgrind with the ACL clear cache function.
    BOOST_TEST(static_cast<long>(reachableAfter) - static_cast<long>(reachableBefore) < 1024);

    // these are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning
    boost::ignore_unused(dubious);
    boost::ignore_unused(suppressed);
}
#endif

#ifdef WITH_VALGRIND
// run with the following command to get all the amazing output (in the devenv/build folder) :)
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
    // We want to test this in case memory is not freed as early as it could have been
    unsigned long reachableBefore = 0;
    unsigned long reachableAfter = 0;

    // needed as out params but we don't test them
    unsigned long dubious = 0;
    unsigned long suppressed = 0;

    armnn::NetworkId networkIdentifier1 = 1;

    // ensure that runtime is large enough before checking for memory leaks
    // otherwise when loading the network it will automatically reserve memory that won't be released until destruction
    armnn::Runtime runtime(armnn::Compute::CpuRef);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    // check for leaks before we load the network and record them so that we can see the delta after unloading
    VALGRIND_DO_QUICK_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedBefore, dubious, reachableBefore, suppressed);

    // build a mock-network and load it into the runtime
    {
        unsigned int inputShape[] = {1, 7, 1, 1};
        armnn::TensorInfo inputTensorInfo(4, inputShape, armnn::DataType::Float32);

        std::unique_ptr<armnn::Network> mockNetwork1 = std::make_unique<armnn::Network>();
        mockNetwork1->AddInputLayer(0, "test layer");

        armnn::DeviceSpec device;
        device.DefaultComputeDevice = armnn::Compute::CpuRef;

        runtime.LoadNetwork(networkIdentifier1, Optimize(*mockNetwork1, device));
    }

    runtime.UnloadNetwork(networkIdentifier1);

    VALGRIND_DO_ADDED_LEAK_CHECK;
    VALGRIND_COUNT_LEAKS(leakedAfter, dubious, reachableAfter, suppressed);

    // if we're not running under Valgrind, these vars will have been initialised to 0, so this will always pass
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

    // these are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning
    boost::ignore_unused(dubious);
    boost::ignore_unused(suppressed);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
