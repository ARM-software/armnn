//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>
#include <Runtime.hpp>
#include <armnn/TypesUtils.hpp>

#include <LabelsAndEventClasses.hpp>
#include <test/ProfilingTestUtils.hpp>

#include <HeapProfiling.hpp>
#include <LeakChecking.hpp>

#ifdef WITH_VALGRIND
#include <valgrind/memcheck.h>
#endif

#include <boost/test/unit_test.hpp>
#include "RuntimeTests.hpp"

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
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
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

#endif // ARMNN_LEAK_CHECKING_ENABLED

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

    {
        std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};

        std::unique_ptr<armnn::Network> mockNetwork1 = std::make_unique<armnn::Network>();
        mockNetwork1->AddInputLayer(0, "test layer");

        // Warm-up load/unload pair to put the runtime in a stable state (memory-wise).
        runtime.LoadNetwork(networkIdentifier1, Optimize(*mockNetwork1, backends, runtime.GetDeviceSpec()));
        runtime.UnloadNetwork(networkIdentifier1);

        // Checks for leaks before we load the network and record them so that we can see the delta after unloading.
        VALGRIND_DO_QUICK_LEAK_CHECK;
        VALGRIND_COUNT_LEAKS(leakedBefore, dubious, reachableBefore, suppressed);

        // The actual test.
        runtime.LoadNetwork(networkIdentifier1, Optimize(*mockNetwork1, backends, runtime.GetDeviceSpec()));
        runtime.UnloadNetwork(networkIdentifier1);

        VALGRIND_DO_ADDED_LEAK_CHECK;
        VALGRIND_COUNT_LEAKS(leakedAfter, dubious, reachableAfter, suppressed);
    }

    // If we're not running under Valgrind, these vars will have been initialised to 0, so this will always pass.
    BOOST_TEST(leakedBefore    == leakedAfter);
    BOOST_TEST(reachableBefore == reachableAfter);

    // These are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning.
    boost::ignore_unused(dubious);
    boost::ignore_unused(suppressed);
}
#endif // WITH_VALGRIND

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
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
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
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc, armnn::Compute::CpuRef };
    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Load it into the runtime. It should succeed.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

BOOST_AUTO_TEST_CASE(IVGCVSW_1929_QuantizedSoftmaxIssue)
{
    // Test for issue reported by Chris Nix in https://jira.arm.com/browse/IVGCVSW-1929
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());
    armnn::IConnectableLayer* input = net->AddInputLayer(
            0,
            "input"
    );
    armnn::IConnectableLayer* softmax = net->AddSoftmaxLayer(
            armnn::SoftmaxDescriptor(),
            "softmax"
    );
    armnn::IConnectableLayer* output = net->AddOutputLayer(
            0,
            "output"
    );

    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(
            armnn::TensorShape({ 1, 5 }),
            armnn::DataType::QuantisedAsymm8,
            1.0f/255,
            0
    ));

    softmax->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(
            armnn::TensorShape({ 1, 5 }),
            armnn::DataType::QuantisedAsymm8
    ));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<std::string> errMessages;
    armnn::IOptimizedNetworkPtr optNet = Optimize(
            *net,
            backends,
            runtime->GetDeviceSpec(),
            OptimizerOptions(),
            errMessages
    );

    BOOST_TEST(errMessages.size() == 1);
    BOOST_TEST(errMessages[0] ==
        "ERROR: output 0 of layer Softmax (softmax) is of type "
        "Quantized 8 bit but its scale parameter has not been set");
    BOOST_TEST(!optNet);
}

BOOST_AUTO_TEST_CASE(ProfilingDisable)
{
    using namespace armnn;

    // Create runtime in which the test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc, should fall back to CpuRef.
    NormalizationDescriptor descriptor;
    IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Load it into the runtime. It should succeed.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper;
    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is not enabled, the post-optimisation structure should not be created
    BOOST_TEST(!readableBuffer);
}

BOOST_AUTO_TEST_CASE(ProfilingEnableCpuRef)
{
    using namespace armnn;
    using namespace armnn::profiling;

    // Create runtime in which the test will run
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "input");

    NormalizationDescriptor descriptor;
    IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor, "normalization");

    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    ProfilingGuid optNetGuid = optNet->GetGuid();

    // Load it into the runtime. It should succeed.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper;
    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is enabled, the post-optimisation structure should be created
    BOOST_CHECK(readableBuffer != nullptr);

    unsigned int size = readableBuffer->GetSize();
    BOOST_CHECK(size == 1356);

    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    unsigned int offset = 0;

    // Post-optimisation network
    // Network entity
    VerifyTimelineEntityBinaryPacket(optNetGuid, readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           optNetGuid,
                                           LabelsAndEventClasses::NETWORK_GUID,
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // Input layer
    // Input layer entity
    VerifyTimelineEntityBinaryPacket(input->GetGuid(), readableData, offset);

    // Name Entity
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), "input", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           input->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Name label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::NAME_GUID,
                                           readableData,
                                           offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           input->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // Network - Input layer relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           optNetGuid,
                                           input->GetGuid(),
                                           readableData,
                                           offset);

    // Normalization layer
    // Normalization layer entity
    VerifyTimelineEntityBinaryPacket(normalize->GetGuid(), readableData, offset);

    // Name entity
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), "normalization", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           normalize->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Name label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::NAME_GUID,
                                           readableData,
                                           offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           normalize->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // Network - Normalize layer relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           optNetGuid,
                                           normalize->GetGuid(),
                                           readableData,
                                           offset);

    // Input layer - Normalize layer relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           input->GetGuid(),
                                           normalize->GetGuid(),
                                           readableData,
                                           offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::CONNECTION_GUID,
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // Normalization workload
    // Normalization workload entity
    VerifyTimelineEntityBinaryPacket(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), "CpuRef", readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // BackendId label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::BACKENDID_GUID,
                                           readableData,
                                           offset);

    // Normalize layer - Normalize workload relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           normalize->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Output layer
    // Output layer entity
    VerifyTimelineEntityBinaryPacket(output->GetGuid(), readableData, offset);

    // Name entity
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), "output", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           output->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Name label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::NAME_GUID,
                                           readableData,
                                           offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           output->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // Network - Output layer relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           optNetGuid,
                                           output->GetGuid(),
                                           readableData,
                                           offset);

    // Normalize layer - Output layer relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           normalize->GetGuid(),
                                           output->GetGuid(),
                                           readableData,
                                           offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::CONNECTION_GUID,
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);
    
    bufferManager.MarkRead(readableBuffer);

    // Creates structures for input & output.
    std::vector<float> inputData(16);
    std::vector<float> outputData(16);

    InputTensors inputTensors
    {
        {0, ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0, Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Get readable buffer for output workload
    auto outputReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(outputReadableBuffer != nullptr);

    // Get readable buffer for input workload
    auto inputReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(inputReadableBuffer != nullptr);

    // Validate input workload data
    size = inputReadableBuffer->GetSize();
    BOOST_CHECK(size == 252);

    readableData = inputReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Input workload
    // Input workload entity
    VerifyTimelineEntityBinaryPacket(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), "CpuRef", readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // BackendId label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::BACKENDID_GUID,
                                           readableData,
                                           offset);

    // Input layer - Input workload relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           input->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    bufferManager.MarkRead(inputReadableBuffer);

    // Validate output workload data
    size = outputReadableBuffer->GetSize();
    BOOST_CHECK(size == 252);

    readableData = outputReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Output workload
    // Output workload entity
    VerifyTimelineEntityBinaryPacket(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::TYPE_GUID,
                                           readableData,
                                           offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacket(EmptyOptional(), "CpuRef", readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    // BackendId label relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                           EmptyOptional(),
                                           EmptyOptional(),
                                           LabelsAndEventClasses::BACKENDID_GUID,
                                           readableData,
                                           offset);

    // Output layer - Output workload relationship
    VerifyTimelineRelationshipBinaryPacket(ProfilingRelationshipType::RetentionLink,
                                           EmptyOptional(),
                                           output->GetGuid(),
                                           EmptyOptional(),
                                           readableData,
                                           offset);

    bufferManager.MarkRead(outputReadableBuffer);
}

BOOST_AUTO_TEST_CASE(ProfilingPostOptimisationStructureCpuRef)
{
    VerifyPostOptimisationStructureTestImpl(armnn::Compute::CpuRef);
}

BOOST_AUTO_TEST_SUITE_END()
