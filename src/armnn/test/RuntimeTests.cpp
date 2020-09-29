//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>
#include <Processes.hpp>
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
#include "TestUtils.hpp"

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
    armnn::IRuntimePtr               runtime(armnn::IRuntime::Create(options));

    // Mock network 1.
    armnn::NetworkId   networkIdentifier1 = 1;
    armnn::INetworkPtr mockNetwork1(armnn::INetwork::Create());
    mockNetwork1->AddInputLayer(0, "test layer");
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    runtime->LoadNetwork(networkIdentifier1, Optimize(*mockNetwork1, backends, runtime->GetDeviceSpec()));

    // Mock network 2.
    armnn::NetworkId   networkIdentifier2 = 2;
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
    unsigned long leakedAfter  = 0;

    // A start-pointer or chain of start-pointers to the block is found. Since the block is still pointed at,
    // the programmer could, at least in principle, have freed it before program exit.
    // We want to test this in case memory is not freed as early as it could have been.
    unsigned long reachableBefore = 0;
    unsigned long reachableAfter  = 0;

    // Needed as out params but we don't test them.
    unsigned long dubious    = 0;
    unsigned long suppressed = 0;

    armnn::NetworkId networkIdentifier1 = 1;

    // ensure that runtime is large enough before checking for memory leaks
    // otherwise when loading the network it will automatically reserve memory that won't be released until destruction
    armnn::IRuntime::CreationOptions options;
    armnn::Runtime                   runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };

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
    BOOST_TEST(leakedBefore == leakedAfter);
    BOOST_TEST(reachableBefore == reachableAfter);

    // These are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning.
    IgnoreUnused(dubious);
    IgnoreUnused(suppressed);
}
#endif // WITH_VALGRIND

BOOST_AUTO_TEST_CASE(RuntimeCpuRef)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr               runtime(armnn::IRuntime::Create(options));

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
    IOptimizedNetworkPtr          optNet   = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

BOOST_AUTO_TEST_CASE(RuntimeFallbackToCpuRef)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr               runtime(armnn::IRuntime::Create(options));

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
    IOptimizedNetworkPtr          optNet   = Optimize(*net, backends, runtime->GetDeviceSpec());

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
    armnn::IRuntimePtr               runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());
    armnn::IConnectableLayer* input   = net->AddInputLayer(0,"input");
    armnn::IConnectableLayer* softmax = net->AddSoftmaxLayer(armnn::SoftmaxDescriptor(), "softmax");
    armnn::IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(armnn::TensorShape({ 1, 5 }),
                                                            armnn::DataType::QAsymmU8,
                                                            1.0f / 255,
                                                            0));

    softmax->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(armnn::TensorShape({ 1, 5 }),
                                                              armnn::DataType::QAsymmU8));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    std::vector<std::string>      errMessages;

    try
    {
        armnn::IOptimizedNetworkPtr optNet = Optimize(*net,
                                                      backends,
                                                      runtime->GetDeviceSpec(),
                                                      OptimizerOptions(),
                                                      errMessages);
        BOOST_FAIL("An exception should have been thrown");
    }
    catch (const InvalidArgumentException& e)
    {
        // Different exceptions are thrown on different backends
    }
    BOOST_CHECK(errMessages.size() > 0);
}

BOOST_AUTO_TEST_CASE(RuntimeBackendOptions)
{
    using namespace armnn;

    IRuntime::CreationOptions creationOptions;
    auto& backendOptions = creationOptions.m_BackendOptions;


    // Define Options on explicit construction
    BackendOptions options1("FakeBackend1",
                            {
                                { "Option1", 1.3f },
                                { "Option2", true }
                            });

    // Add an option after construction
    options1.AddOption({ "Option3", "some_value" });

    // Add the options to CreationOptions struct
    backendOptions.push_back(options1);

    // Add more Options via inplace explicit construction
    backendOptions.emplace_back(BackendOptions{ "FakeBackend1",
                                                {{ "Option4", 42 }}
    });


    // First group
    BOOST_TEST(backendOptions[0].GetBackendId().Get() == "FakeBackend1");
    BOOST_TEST(backendOptions[0].GetOption(0).GetName() == "Option1");
    BOOST_TEST(backendOptions[0].GetOption(0).GetValue().IsFloat() == true);
    BOOST_TEST(backendOptions[0].GetOption(0).GetValue().AsFloat() == 1.3f);

    BOOST_TEST(backendOptions[0].GetOption(1).GetName() == "Option2");
    BOOST_TEST(backendOptions[0].GetOption(1).GetValue().IsBool() == true);
    BOOST_TEST(backendOptions[0].GetOption(1).GetValue().AsBool() == true);

    BOOST_TEST(backendOptions[0].GetOption(2).GetName() == "Option3");
    BOOST_TEST(backendOptions[0].GetOption(2).GetValue().IsString() == true);
    BOOST_TEST(backendOptions[0].GetOption(2).GetValue().AsString() == "some_value");

    // Second group
    BOOST_TEST(backendOptions[1].GetBackendId().Get() == "FakeBackend1");
    BOOST_TEST(backendOptions[1].GetOption(0).GetName() == "Option4");
    BOOST_TEST(backendOptions[1].GetOption(0).GetValue().IsInt() == true);
    BOOST_TEST(backendOptions[1].GetOption(0).GetValue().AsInt() == 42);
}

BOOST_AUTO_TEST_CASE(ProfilingDisable)
{
    using namespace armnn;

    // Create runtime in which the test will run
    armnn::IRuntime::CreationOptions options;
    armnn::Runtime runtime(options);

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
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime.GetDeviceSpec());

    // Load it into the runtime. It should succeed.
    armnn::NetworkId netId;
    BOOST_TEST(runtime.LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper(GetProfilingService(&runtime));
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
    options.m_ProfilingOptions.m_TimelineEnabled = true;

    armnn::Runtime runtime(options);
    GetProfilingService(&runtime).ResetExternalProfilingOptions(options.m_ProfilingOptions, false);

    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper(GetProfilingService(&runtime));
    profilingServiceHelper.ForceTransitionToState(ProfilingState::NotConnected);
    profilingServiceHelper.ForceTransitionToState(ProfilingState::WaitingForAck);
    profilingServiceHelper.ForceTransitionToState(ProfilingState::Active);

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
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime.GetDeviceSpec());

    ProfilingGuid optNetGuid = optNet->GetGuid();

    // Load it into the runtime. It should succeed.
    armnn::NetworkId netId;
    BOOST_TEST(runtime.LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is enabled, the post-optimisation structure should be created
    BOOST_CHECK(readableBuffer != nullptr);

    unsigned int size = readableBuffer->GetSize();

    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, size - 8);
    BOOST_TEST_MESSAGE("HEADER OK");

    // Post-optimisation network
    // Network entity
    VerifyTimelineEntityBinaryPacketData(optNetGuid, readableData, offset);
    BOOST_TEST_MESSAGE("NETWORK ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               LabelsAndEventClasses::NETWORK_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NETWORK TYPE RELATIONSHIP OK");

    // Network - START OF LIFE
    ProfilingGuid networkSolEventGuid = VerifyTimelineEventBinaryPacket(EmptyOptional(),
                                                                        EmptyOptional(),
                                                                        EmptyOptional(),
                                                                        readableData,
                                                                        offset);
    BOOST_TEST_MESSAGE("NETWORK START OF LIFE EVENT OK");

    // Network - START OF LIFE event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               networkSolEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NETWORK START OF LIFE RELATIONSHIP OK");

    // Process ID Label
    int processID = armnnUtils::Processes::GetCurrentId();
    std::stringstream ss;
    ss << processID;
    std::string processIdLabel = ss.str();
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), processIdLabel, readableData, offset);
    BOOST_TEST_MESSAGE("PROCESS ID LABEL OK");

    // Entity - Process ID relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               EmptyOptional(),
                                               LabelsAndEventClasses::PROCESS_ID_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NETWORK PROCESS ID RELATIONSHIP OK");

    // Input layer
    // Input layer entity
    VerifyTimelineEntityBinaryPacketData(input->GetGuid(), readableData, offset);
    BOOST_TEST_MESSAGE("INPUT ENTITY OK");

    // Name Entity
    ProfilingGuid inputLabelGuid = VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "input", readableData, offset);
    BOOST_TEST_MESSAGE("INPUT NAME LABEL OK");

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               inputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT NAME RELATIONSHIP OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT TYPE RELATIONSHIP OK");

    // Network - Input layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               input->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NETWORK - INPUT CHILD RELATIONSHIP OK");

    // Normalization layer
    // Normalization layer entity
    VerifyTimelineEntityBinaryPacketData(normalize->GetGuid(), readableData, offset);
    BOOST_TEST_MESSAGE("NORMALIZATION LAYER ENTITY OK");

    // Name entity
    ProfilingGuid normalizationLayerNameGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "normalization", readableData, offset);
    BOOST_TEST_MESSAGE("NORMALIZATION LAYER NAME LABEL OK");

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               normalizationLayerNameGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION LAYER NAME RELATIONSHIP OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION LAYER TYPE RELATIONSHIP OK");

    // Network - Normalize layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               normalize->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NETWORK - NORMALIZATION LAYER CHILD RELATIONSHIP OK");

    // Input layer - Normalize layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               normalize->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT - NORMALIZATION LAYER CONNECTION OK");

    // Normalization workload
    // Normalization workload entity
    ProfilingGuid normalizationWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalizationWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD TYPE RELATIONSHIP OK");

    // BackendId entity
    ProfilingGuid cpuRefLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "CpuRef", readableData, offset);
    BOOST_TEST_MESSAGE("CPUREF LABEL OK");

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalizationWorkloadGuid,
                                               cpuRefLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD BACKEND ID RELATIONSHIP OK");

    // Normalize layer - Normalize workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               normalizationWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION LAYER - WORKLOAD CHILD RELATIONSHIP OK");

    // Output layer
    // Output layer entity
    VerifyTimelineEntityBinaryPacketData(output->GetGuid(), readableData, offset);
    BOOST_TEST_MESSAGE("OUTPUT LAYER ENTITY OK");

    // Name entity
    ProfilingGuid outputLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "output", readableData, offset);
    BOOST_TEST_MESSAGE("OUTPUT LAYER NAME LABEL OK");

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               outputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT LAYER NAME RELATIONSHIP OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT LAYER TYPE RELATIONSHIP OK");

    // Network - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               output->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NETWORK - OUTPUT LAYER CHILD RELATIONSHIP OK");

    // Normalize layer - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               output->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZE LAYER - OUTPUT LAYER CONNECTION OK");

    bufferManager.MarkRead(readableBuffer);

    // Creates structures for input & output.
    std::vector<float> inputData(16);
    std::vector<float> outputData(16);

    InputTensors  inputTensors
    {
        {0, ConstTensor(runtime.GetInputTensorInfo(netId, 0), inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0, Tensor(runtime.GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime.EnqueueWorkload(netId, inputTensors, outputTensors);

    // Get readable buffer for input workload
    auto  inputReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(inputReadableBuffer != nullptr);

    // Get readable buffer for output workload
    auto outputReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(outputReadableBuffer != nullptr);

    // Get readable buffer for inference timeline
    auto inferenceReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(inferenceReadableBuffer != nullptr);

    // Validate input workload data
    size = inputReadableBuffer->GetSize();
    BOOST_CHECK(size == 164);

    readableData = inputReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 156);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD HEADER OK");

    // Input workload
    // Input workload entity
    ProfilingGuid inputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD TYPE RELATIONSHIP OK");

    // BackendId entity
    ProfilingGuid CpuRefLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "CpuRef", readableData, offset);
    BOOST_TEST_MESSAGE("CPUREF LABEL OK (INPUT WORKLOAD)");

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               CpuRefLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD BACKEND ID RELATIONSHIP OK");

    // Input layer - Input workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               inputWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT LAYER - INPUT WORKLOAD CHILD RELATIONSHIP OK");

    bufferManager.MarkRead(inputReadableBuffer);

    // Validate output workload data
    size = outputReadableBuffer->GetSize();
    BOOST_CHECK(size == 164);

    readableData = outputReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 156);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD HEADER OK");

    // Output workload
    // Output workload entity
    ProfilingGuid outputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD TYPE RELATIONSHIP OK");

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "CpuRef", readableData, offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD CPU REF LABEL OK");

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               CpuRefLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD BACKEND ID RELATIONSHIP OK");

    // Output layer - Output workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT LAYER - OUTPUT WORKLOAD CHILD RELATIONSHIP OK");

    bufferManager.MarkRead(outputReadableBuffer);

    // Validate inference data
    size = inferenceReadableBuffer->GetSize();
    BOOST_CHECK(size == 976 + 8 * ThreadIdSize);

    readableData = inferenceReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 968 + 8 * ThreadIdSize);
    BOOST_TEST_MESSAGE("INFERENCE HEADER OK");

    // Inference timeline trace
    // Inference entity
    ProfilingGuid inferenceGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("INFERENCE ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               LabelsAndEventClasses::INFERENCE_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INFERENCE TYPE RELATIONSHIP OK");

    // Network - Inference relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               inferenceGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NETWORK - INFERENCE EXECUTION_OF RELATIONSHIP OK");

    // Start Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("INFERENCE START OF LIFE EVENT OK");

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inferenceEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INFERENCE START OF LIFE RELATIONSHIP OK");

    // Execution
    // Input workload execution
    // Input workload execution entity
    ProfilingGuid inputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD EXECUTION ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD EXECUTION TYPE RELATIONSHIP OK");

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INFERENCE - INPUT WORKLOAD EXECUTION CHILD RELATIONSHIP OK");

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD - INPUT WORKLOAD EXECUTION RELATIONSHIP OK");

    // Start Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inputWorkloadExecutionSOLEventId = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               inputWorkloadExecutionSOLEventId,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD EXECUTION - START OF LIFE EVENT RELATIONSHIP OK");

    // End of Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inputWorkloadExecutionEOLEventId = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               inputWorkloadExecutionEOLEventId,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INPUT WORKLOAD EXECUTION - END OF LIFE EVENT RELATIONSHIP OK");

    // Normalize workload execution
    // Normalize workload execution entity
    ProfilingGuid normalizeWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("NORMALIZE WORKLOAD EXECUTION ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalizeWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZE WORKLOAD EXECUTION TYPE RELATIONSHIP OK");

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               normalizeWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INFERENCE - NORMALIZE WORKLOAD EXECUTION CHILD RELATIONSHIP OK");

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               normalizationWorkloadGuid,
                                               normalizeWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD - NORMALIZATION WORKLOAD EXECUTION RELATIONSHIP OK");

    // Start Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid normalizationWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD EXECUTION START OF LIFE EVENT OK");

    // Normalize workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               normalizeWorkloadExecutionGuid,
                                               normalizationWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD EXECUTION START OF LIFE RELATIONSHIP OK");

    // End of Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid normalizationWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD EXECUTION END OF LIFE EVENT OK");

    // Normalize workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               normalizeWorkloadExecutionGuid,
                                               normalizationWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("NORMALIZATION WORKLOAD EXECUTION END OF LIFE RELATIONSHIP OK");

    // Output workload execution
    // Output workload execution entity
    ProfilingGuid outputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD EXECUTION ENTITY OK");

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD EXECUTION TYPE RELATIONSHIP OK");

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INFERENCE - OUTPUT WORKLOAD EXECUTION CHILD RELATIONSHIP OK");

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);
     BOOST_TEST_MESSAGE("OUTPUT WORKLOAD - OUTPUT WORKLOAD EXECUTION EXECUTION_OF RELATIONSHIP OK");

    // Start Output workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD EXECUTION START OF LIFE EVENT OK");

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD EXECUTION - START OF LIFE EVENT RELATIONSHIP OK");

    // End of Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD EXECUTION END OF LIFE EVENT OK");

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("OUTPUT WORKLOAD EXECUTION - END OF LIFE EVENT RELATIONSHIP OK");

    // End of Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);
    BOOST_TEST_MESSAGE("INFERENCE END OF LIFE EVENT OK");

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inferenceEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);
    BOOST_TEST_MESSAGE("INFERENCE - END OF LIFE EVENT RELATIONSHIP OK");

    bufferManager.MarkRead(inferenceReadableBuffer);
}

BOOST_AUTO_TEST_CASE(ProfilingPostOptimisationStructureCpuRef)
{
    VerifyPostOptimisationStructureTestImpl(armnn::Compute::CpuRef);
}

BOOST_AUTO_TEST_SUITE_END()
