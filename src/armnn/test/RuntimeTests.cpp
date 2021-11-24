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

#include <common/include/LabelsAndEventClasses.hpp>
#include <test/ProfilingTestUtils.hpp>

#include <HeapProfiling.hpp>
#include <LeakChecking.hpp>

#ifdef WITH_VALGRIND
#include <valgrind/memcheck.h>
#endif

#include <doctest/doctest.h>
#include "RuntimeTests.hpp"
#include <TestUtils.hpp>

namespace armnn
{

void RuntimeLoadedNetworksReserve(armnn::RuntimeImpl* runtime)
{
    runtime->m_LoadedNetworks.reserve(1);
}

}

TEST_SUITE("Runtime")
{
TEST_CASE("RuntimeUnloadNetwork")
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
    CHECK(runtime->UnloadNetwork(networkIdentifier1) == armnn::Status::Success);

    CHECK(runtime->UnloadNetwork(networkIdentifier1) == armnn::Status::Failure);
}

TEST_CASE("RuntimePreImportInputs")
{
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::NetworkId networkId = 1;
    armnn::INetworkPtr testNetwork(armnn::INetwork::Create());

    auto inputLayer1 = testNetwork->AddInputLayer(0, "input 1 layer");
    auto inputLayer2 = testNetwork->AddInputLayer(1, "input 2 layer");
    auto addLayer = testNetwork->AddAdditionLayer("add layer");
    auto outputLayer = testNetwork->AddOutputLayer(2, "output layer");

    TensorInfo tensorInfo{{4}, armnn::DataType::Signed32};

    inputLayer1->GetOutputSlot(0).Connect(addLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    inputLayer2->GetOutputSlot(0).Connect(addLayer->GetInputSlot(1));
    inputLayer2->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    addLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    addLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};

    std::string er;
    armnn::INetworkProperties networkProperties(true, MemorySource::Malloc, MemorySource::Undefined);
    runtime->LoadNetwork(networkId,
                         Optimize(*testNetwork, backends, runtime->GetDeviceSpec()),
                         er,
                         networkProperties);

    std::vector<int> inputData1(4, 10);
    std::vector<int> inputData2(4, 20);
    std::vector<int> output(4);

    ConstTensor inputTensor1({{4}, armnn::DataType::Signed32, 0.0f, 0, true}, inputData1.data());
    ConstTensor inputTensor2({{4}, armnn::DataType::Signed32, 0.0f, 0, true}, inputData2.data());
    Tensor outputTensor({{4}, armnn::DataType::Signed32}, output.data());

    auto importedInputVec1 = runtime->ImportInputs(networkId, {{0, inputTensor1}});
    CHECK(importedInputVec1.size() == 1);
    CHECK(importedInputVec1[0] == 0);

    auto memHandle = runtime->CreateWorkingMemHandle(networkId);

    runtime->Execute(*memHandle.get(), {{1, inputTensor2}}, {{2, outputTensor}}, {0 /* pre-imported id */});
    for (auto val: output) {
        CHECK(val == 30);
    }

    auto importedInputVec2 = runtime->ImportInputs(networkId, {{1, inputTensor2}});
    CHECK(importedInputVec2.size() == 1);
    CHECK(importedInputVec2[0] == 1);

    runtime->Execute(*memHandle.get(), {{0, inputTensor1}}, {{2, outputTensor}}, {1 /* pre-imported id */});
    for (auto val: output) {
        CHECK(val == 30);
    }

    runtime->Execute(*memHandle.get(), {}, {{2, outputTensor}}, {0, 1});
    for (auto val: output) {
        CHECK(val == 30);
    }
    // Duplicate ImportedInputId and LayerBindingId
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {}, {{2, outputTensor}}, {0, 0});,
                    armnn::InvalidArgumentException);
    // Duplicate LayerBindingId
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {{1, inputTensor2}}, {{2, outputTensor}}, {1});,
                    armnn::InvalidArgumentException);
    // Incorrect ImportedInputId
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {{1, inputTensor2}}, {{2, outputTensor}}, {10});,
                    armnn::InvalidArgumentException);
    // Incorrect LayerBindingId
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {{-2, inputTensor2}}, {{2, outputTensor}}, {1});,
                    armnn::InvalidArgumentException);
    // Incorrect layer binding id and ImportedInputId
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {{-2, inputTensor2}}, {{2, outputTensor}}, {10});,
                    armnn::InvalidArgumentException);
    auto importedInputVec3 = runtime->ImportInputs(networkId, {{1, inputTensor2}});
    CHECK(importedInputVec3[0] == 2);
    // Too many ImportedInputIds
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {}, {{2, outputTensor}}, {0, 1, 2});,
                    armnn::InvalidArgumentException);
    // Too many InputTensors
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(),
                                     {{0, inputTensor2},
                                      {1, inputTensor2},
                                      {2, inputTensor2}},
                                      {{2, outputTensor}});, armnn::InvalidArgumentException);
    // Too few ImportedInputIds
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {}, {{2, outputTensor}}, {0});,
                    armnn::InvalidArgumentException);
    runtime->ClearImportedInputs(networkId, {1});
    runtime->Execute(*memHandle.get(), {{1, inputTensor2}}, {{2, outputTensor}}, {0}, {});
    for (auto val: output) {
        CHECK(val == 30);
    }
    // Using deleted pre-imported input
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {}, {{2, outputTensor}}, {0, 1}, {});,
                    armnn::InvalidArgumentException);

    // Trying to delete deleted pre-imported tensor
    CHECK_THROWS_AS(runtime->ClearImportedInputs(networkId, {1});, armnn::InvalidArgumentException);

    // Trying to delete unknown pre-imported tensor
    CHECK_THROWS_AS(runtime->ClearImportedInputs(networkId, {10});, armnn::InvalidArgumentException);
}
TEST_CASE("RuntimePreImportOutputs")
{
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr               runtime(armnn::IRuntime::Create(options));

    armnn::NetworkId   networkId = 1;

    armnn::INetworkPtr testNetwork(armnn::INetwork::Create());
    TensorInfo tensorInfo{{4}, armnn::DataType::Float32, 0.0f, 0, true};

    auto inputLayer1 = testNetwork->AddInputLayer(0, "input 1 layer");
    inputLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = 2.0f;
    activationDescriptor.m_B = 0.0f;
    auto activationLayer1 = testNetwork->AddActivationLayer(activationDescriptor, "add layer");
    auto outputLayer1 = testNetwork->AddOutputLayer(2, "output layer");

    inputLayer1->GetOutputSlot(0).Connect(activationLayer1->GetInputSlot(0));

    activationLayer1->GetOutputSlot(0).Connect(outputLayer1->GetInputSlot(0));
    activationLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    auto inputLayer2 = testNetwork->AddInputLayer(1, "input 1 layer");

    activationDescriptor.m_A = 4.0f;
    activationDescriptor.m_B = 2.0f;
    auto activationLayer2 = testNetwork->AddActivationLayer(activationDescriptor, "add layer");
    auto outputLayer2 = testNetwork->AddOutputLayer(3, "output layer");

    inputLayer2->GetOutputSlot(0).Connect(activationLayer2->GetInputSlot(0));
    inputLayer2->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    activationLayer2->GetOutputSlot(0).Connect(outputLayer2->GetInputSlot(0));
    activationLayer2->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };

    std::string er;
    armnn::INetworkProperties networkProperties(true, MemorySource::Malloc, MemorySource::Malloc);
    runtime->LoadNetwork(networkId,
                         Optimize(*testNetwork, backends, runtime->GetDeviceSpec()),
                         er,
                         networkProperties);

    std::vector<float> inputData1(4, 1.0f);
    std::vector<float> inputData2(4, 3.0f);

    std::vector<float> outputData1(4);
    std::vector<float> outputData2(4);

    ConstTensor inputTensor1(tensorInfo, inputData1.data());
    ConstTensor inputTensor2(tensorInfo, inputData2.data());

    Tensor outputTensor1{tensorInfo, outputData1.data()};
    Tensor outputTensor2{tensorInfo, outputData2.data()};

    InputTensors inputTensors = {{0, inputTensor1}, {1, inputTensor2}};

    std::pair<LayerBindingId, class Tensor> output1{2, outputTensor1};
    std::pair<LayerBindingId, class Tensor> output2{3, outputTensor2};

    auto testOutputs = [&]()
    {
        for (auto val : outputData1)
        {
                    CHECK(val == 1.0f);
        }

        for (auto val : outputData2)
        {
                    CHECK(val == 3.0f);
        }
    };

    auto memHandle = runtime->CreateWorkingMemHandle(networkId);

    runtime->Execute(*memHandle.get(),inputTensors, {output1, output2});
    testOutputs();

    auto importedOutputVec = runtime->ImportOutputs(networkId, {output1, output2 });
    CHECK(importedOutputVec.size() == 2);
    CHECK(importedOutputVec[0] == 0);
    CHECK(importedOutputVec[1] == 1);

    runtime->Execute(*memHandle.get(), inputTensors, {}, {}, importedOutputVec);
    testOutputs();

    runtime->Execute(*memHandle.get(), inputTensors, {output1}, {}, {1});
    testOutputs();

    runtime->Execute(*memHandle.get(), inputTensors, {output2}, {}, {0});
    testOutputs();

    auto importedInputVec = runtime->ImportInputs(networkId, inputTensors);
    CHECK(importedInputVec.size() == 2);
    CHECK(importedInputVec[0] == 0);
    CHECK(importedInputVec[1] == 1);

    runtime->Execute(*memHandle.get(), {}, {}, importedInputVec, importedOutputVec);
    testOutputs();

    runtime->Execute(*memHandle.get(), {{0, inputTensor1}}, {output2}, {1}, {0});
    testOutputs();

    // Too many ids
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {output1, output2}, {}, {0, 1});,
                    armnn::InvalidArgumentException);

    // Duplicate ids
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {output2}, {}, {1});,
                    armnn::InvalidArgumentException);

    // Duplicate ids
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {output1, output1}, {}, {});,
                    armnn::InvalidArgumentException);

    // Duplicate ids
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {}, {}, {0, 0}),
                    armnn::InvalidArgumentException);

    // Unknown id
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {output1}, {}, {3});,
                    armnn::InvalidArgumentException);

    // Unknown id
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {{4, outputTensor2}}, {}, {1});,
                    armnn::InvalidArgumentException);

    // Input id for output
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {{0, outputTensor2}}, {}, {1});,
                    armnn::InvalidArgumentException);

    // Input id for output
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {{0, outputTensor2}}, {}, {1});,
                    armnn::InvalidArgumentException);

    // Output id for input
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), {{2, inputTensor1}}, {{0, outputTensor2}}, {1}, {1, 0});,
                    armnn::InvalidArgumentException);

    runtime->ClearImportedOutputs(networkId, {1});

    runtime->Execute(*memHandle.get(), inputTensors, {output2}, {}, {0});
    testOutputs();

    // Trying to use deleted pre-imported tensor
    CHECK_THROWS_AS(runtime->Execute(*memHandle.get(), inputTensors, {}, {}, importedOutputVec),
                    armnn::InvalidArgumentException);

    // Trying to delete deleted pre-imported tensor
    CHECK_THROWS_AS(runtime->ClearImportedOutputs(networkId, {1});, armnn::InvalidArgumentException);

    // Trying to delete unknown pre-imported tensor
    CHECK_THROWS_AS(runtime->ClearImportedOutputs(networkId, {10});, armnn::InvalidArgumentException);
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

TEST_CASE_FIXTURE(DisableGlobalLeakChecking,  "RuntimeHeapMemoryUsageSanityChecks")
{
    CHECK(ARMNN_LEAK_CHECKER_IS_ACTIVE());
    {
        ARMNN_SCOPED_LEAK_CHECKER("Sanity_Check_Outer");
        {
            ARMNN_SCOPED_LEAK_CHECKER("Sanity_Check_Inner");
            CHECK(ARMNN_NO_LEAKS_IN_SCOPE() == true);
            std::unique_ptr<char[]> dummyAllocation(new char[1000]);
            // "A leak of 1000 bytes is expected here. "
            // "Please make sure environment variable: HEAPCHECK=draconian is set!"
            CHECK((ARMNN_NO_LEAKS_IN_SCOPE() == false));
            CHECK(ARMNN_BYTES_LEAKED_IN_SCOPE() == 1000);
            CHECK(ARMNN_OBJECTS_LEAKED_IN_SCOPE() == 1);
        }
        CHECK(ARMNN_NO_LEAKS_IN_SCOPE());
        CHECK(ARMNN_BYTES_LEAKED_IN_SCOPE() == 0);
        CHECK(ARMNN_OBJECTS_LEAKED_IN_SCOPE() == 0);
    }
}

#endif // ARMNN_LEAK_CHECKING_ENABLED

// Note: this part of the code is due to be removed when we fully trust the gperftools based results.
#ifdef WITH_VALGRIND
// Run with the following command to get all the amazing output (in the devenv/build folder) :)
// valgrind --leak-check=full --show-leak-kinds=all --log-file=Valgrind_Memcheck_Leak_Report.txt armnn/test/UnitTests
TEST_CASE("RuntimeMemoryLeak")
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
    armnn::RuntimeImpl                   runtime(options);
    armnn::RuntimeLoadedNetworksReserve(&runtime);

    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };

        armnn::INetworkPtr mockNetwork1(armnn::INetwork::Create());
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
    CHECK(leakedBefore == leakedAfter);
    CHECK(reachableBefore == reachableAfter);

    // These are needed because VALGRIND_COUNT_LEAKS is a macro that assigns to the parameters
    // so they are assigned to, but still considered unused, causing a warning.
    IgnoreUnused(dubious);
    IgnoreUnused(suppressed);
}
#endif // WITH_VALGRIND

TEST_CASE("RuntimeCpuRef")
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
    CHECK(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

TEST_CASE("RuntimeFallbackToCpuRef")
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
    CHECK(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

TEST_CASE("IVGCVSW_1929_QuantizedSoftmaxIssue")
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
        FAIL("An exception should have been thrown");
    }
    catch (const InvalidArgumentException&)
    {
        // Different exceptions are thrown on different backends
    }
    CHECK(errMessages.size() > 0);
}

TEST_CASE("RuntimeBackendOptions")
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
    CHECK(backendOptions[0].GetBackendId().Get() == "FakeBackend1");
    CHECK(backendOptions[0].GetOption(0).GetName() == "Option1");
    CHECK(backendOptions[0].GetOption(0).GetValue().IsFloat() == true);
    CHECK(backendOptions[0].GetOption(0).GetValue().AsFloat() == 1.3f);

    CHECK(backendOptions[0].GetOption(1).GetName() == "Option2");
    CHECK(backendOptions[0].GetOption(1).GetValue().IsBool() == true);
    CHECK(backendOptions[0].GetOption(1).GetValue().AsBool() == true);

    CHECK(backendOptions[0].GetOption(2).GetName() == "Option3");
    CHECK(backendOptions[0].GetOption(2).GetValue().IsString() == true);
    CHECK(backendOptions[0].GetOption(2).GetValue().AsString() == "some_value");

    // Second group
    CHECK(backendOptions[1].GetBackendId().Get() == "FakeBackend1");
    CHECK(backendOptions[1].GetOption(0).GetName() == "Option4");
    CHECK(backendOptions[1].GetOption(0).GetValue().IsInt() == true);
    CHECK(backendOptions[1].GetOption(0).GetValue().AsInt() == 42);
}

TEST_CASE("ProfilingDisable")
{
    using namespace armnn;

    // Create runtime in which the test will run
    armnn::IRuntime::CreationOptions options;
    armnn::RuntimeImpl runtime(options);

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
    CHECK(runtime.LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper(GetProfilingService(&runtime));
    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is not enabled, the post-optimisation structure should not be created
    CHECK(!readableBuffer);
}

TEST_CASE("ProfilingEnableCpuRef")
{
    using namespace armnn;
    using namespace armnn::profiling;

    // Create runtime in which the test will run
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
    options.m_ProfilingOptions.m_TimelineEnabled = true;

    armnn::RuntimeImpl runtime(options);
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
    CHECK(runtime.LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is enabled, the post-optimisation structure should be created
    CHECK(readableBuffer != nullptr);

    unsigned int size = readableBuffer->GetSize();

    const unsigned char* readableData = readableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, size - 8);

    // Post-optimisation network
    // Network entity
    VerifyTimelineEntityBinaryPacketData(optNetGuid, readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               LabelsAndEventClasses::NETWORK_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - START OF LIFE
    ProfilingGuid networkSolEventGuid = VerifyTimelineEventBinaryPacket(EmptyOptional(),
                                                                        EmptyOptional(),
                                                                        EmptyOptional(),
                                                                        readableData,
                                                                        offset);

    // Network - START OF LIFE event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               networkSolEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Process ID Label
    int processID = armnnUtils::Processes::GetCurrentId();
    std::stringstream ss;
    ss << processID;
    std::string processIdLabel = ss.str();
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), processIdLabel, readableData, offset);

    // Entity - Process ID relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               EmptyOptional(),
                                               LabelsAndEventClasses::PROCESS_ID_GUID,
                                               readableData,
                                               offset);

    // Input layer
    // Input layer entity
    VerifyTimelineEntityBinaryPacketData(input->GetGuid(), readableData, offset);

    // Name Entity
    ProfilingGuid inputLabelGuid = VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "input", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               inputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Input layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               input->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Normalization layer
    // Normalization layer entity
    VerifyTimelineEntityBinaryPacketData(normalize->GetGuid(), readableData, offset);

    // Name entity
    ProfilingGuid normalizationLayerNameGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "normalization", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               normalizationLayerNameGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Normalize layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               normalize->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Input layer - Normalize layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               normalize->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Normalization workload
    // Normalization workload entity
    ProfilingGuid normalizationWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalizationWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    ProfilingGuid cpuRefLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "CpuRef", readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalizationWorkloadGuid,
                                               cpuRefLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Normalize layer - Normalize workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               normalizationWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Output layer
    // Output layer entity
    VerifyTimelineEntityBinaryPacketData(output->GetGuid(), readableData, offset);

    // Name entity
    ProfilingGuid outputLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "output", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               outputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               output->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Normalize layer - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               normalize->GetGuid(),
                                               output->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(readableBuffer);

    // Creates structures for input & output.
    std::vector<float> inputData(16);
    std::vector<float> outputData(16);

    TensorInfo inputTensorInfo = runtime.GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors  inputTensors
    {
        {0, ConstTensor(inputTensorInfo, inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0, Tensor(runtime.GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime.EnqueueWorkload(netId, inputTensors, outputTensors);

    // Get readable buffer for input workload
    auto  inputReadableBuffer = bufferManager.GetReadableBuffer();
    CHECK(inputReadableBuffer != nullptr);

    // Get readable buffer for output workload
    auto outputReadableBuffer = bufferManager.GetReadableBuffer();
    CHECK(outputReadableBuffer != nullptr);

    // Get readable buffer for inference timeline
    auto inferenceReadableBuffer = bufferManager.GetReadableBuffer();
    CHECK(inferenceReadableBuffer != nullptr);

    // Validate input workload data
    size = inputReadableBuffer->GetSize();
    CHECK(size == 164);

    readableData = inputReadableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 156);

    // Input workload
    // Input workload entity
    ProfilingGuid inputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    ProfilingGuid CpuRefLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "CpuRef", readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               CpuRefLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Input layer - Input workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               inputWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(inputReadableBuffer);

    // Validate output workload data
    size = outputReadableBuffer->GetSize();
    CHECK(size == 164);

    readableData = outputReadableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 156);

    // Output workload
    // Output workload entity
    ProfilingGuid outputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "CpuRef", readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               CpuRefLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Output layer - Output workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(outputReadableBuffer);

    // Validate inference data
    size = inferenceReadableBuffer->GetSize();
    CHECK(size == 976 + 8 * ThreadIdSize);

    readableData = inferenceReadableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 968 + 8 * ThreadIdSize);

    // Inference timeline trace
    // Inference entity
    ProfilingGuid inferenceGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               LabelsAndEventClasses::INFERENCE_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Inference relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               inferenceGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inferenceEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Execution
    // Input workload execution
    // Input workload execution entity
    ProfilingGuid inputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

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

    // Normalize workload execution
    // Normalize workload execution entity
    ProfilingGuid normalizeWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               normalizeWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               normalizeWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               normalizationWorkloadGuid,
                                               normalizeWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid normalizationWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Normalize workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               normalizeWorkloadExecutionGuid,
                                               normalizationWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid normalizationWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Normalize workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               normalizeWorkloadExecutionGuid,
                                               normalizationWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Output workload execution
    // Output workload execution entity
    ProfilingGuid outputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Output workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inferenceEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(inferenceReadableBuffer);
}

TEST_CASE("ProfilingPostOptimisationStructureCpuRef")
{
    VerifyPostOptimisationStructureTestImpl(armnn::Compute::CpuRef);
}

}
