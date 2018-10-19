//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>

#include <backends/test/QuantizeHelper.hpp>
#include <boost/core/ignore_unused.hpp>

#include <set>

BOOST_AUTO_TEST_SUITE(EndToEnd)

namespace
{
template<typename T>
bool IsFloatIterFunc(T iter)
{
    boost::ignore_unused(iter);
    return IsFloatingPointIterator<T>::value;
}
} //namespace

BOOST_AUTO_TEST_CASE(QuantizedHelper)
{
    std::vector<float> fArray;
    BOOST_TEST(IsFloatIterFunc(fArray.begin()) == true);
    BOOST_TEST(IsFloatIterFunc(fArray.cbegin()) == true);

    std::vector<double> dArray;
    BOOST_TEST(IsFloatIterFunc(dArray.begin()) == true);

    std::vector<int> iArray;
    BOOST_TEST(IsFloatIterFunc(iArray.begin()) == false);

    float floats[5];
    BOOST_TEST(IsFloatIterFunc(&floats[0]) == true);

    int ints[5];
    BOOST_TEST(IsFloatIterFunc(&ints[0]) == false);
}

BOOST_AUTO_TEST_CASE(Unsigned8)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // Builds up the structure of the network.
    armnn::INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "input");
    IConnectableLayer* softmax = net->AddSoftmaxLayer(SoftmaxDescriptor(), "softmax");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    TensorInfo inputTensorInfo(TensorShape({1, 5}), DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationOffset(100);
    inputTensorInfo.SetQuantizationScale(10000.0f);
    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    TensorInfo outputTensorInfo(TensorShape({1, 5}), DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(1.0f/255.0f);
    softmax->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // optimize the network
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    auto error = runtime->LoadNetwork(netId, std::move(optNet));
    BOOST_TEST(error == Status::Success);

    // Creates structures for input & output.
    std::vector<uint8_t> inputData
    {
        1, 10, 3, 200, 5 // Some inputs - one of which is sufficiently larger than the others to saturate softmax.
    };
    std::vector<uint8_t> outputData(5);

    armnn::InputTensors inputTensors
    {
        {0, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())}
    };
    armnn::OutputTensors outputTensors
    {
        {0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    BOOST_TEST(outputData[0] == 0);
    BOOST_TEST(outputData[1] == 0);
    BOOST_TEST(outputData[2] == 0);
    BOOST_TEST(outputData[3] == 255); // softmax has been saturated.
    BOOST_TEST(outputData[4] == 0);
}

template <typename T>
void ConstantUsageTest(const std::vector<armnn::BackendId>& computeDevice,
    const armnn::TensorInfo& commonTensorInfo,
    const std::vector<T>& inputData,
    const std::vector<T>& constantData,
    const std::vector<T>& expectedOutputData)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);
    IConnectableLayer* constant = net->AddConstantLayer(ConstTensor(commonTensorInfo, constantData));
    IConnectableLayer* add = net->AddAdditionLayer();
    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    input->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    constant->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, computeDevice, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Creates structures for input & output.
    std::vector<T> outputData(inputData.size());

    InputTensors inputTensors
    {
        {0, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    BOOST_TEST(outputData == expectedOutputData);
}

static void ConstantUsageFloat32Test(const std::vector<armnn::BackendId>& computeDevice)
{
    const armnn::TensorInfo commonTensorInfo({ 2, 3 }, armnn::DataType::Float32);

    ConstantUsageTest(computeDevice,
        commonTensorInfo,
        std::vector<float>{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, // Input.
        std::vector<float>{ 6.f, 5.f, 4.f, 3.f, 2.f, 1.f }, // Const input.
        std::vector<float>{ 7.f, 7.f, 7.f, 7.f, 7.f, 7.f }  // Expected output.
    );
}

static void ConstantUsageUint8Test(const std::vector<armnn::BackendId>& computeDevice)
{
    armnn::TensorInfo commonTensorInfo({ 2, 3 }, armnn::DataType::QuantisedAsymm8);

    const float scale = 0.023529f;
    const int8_t offset = -43;

    commonTensorInfo.SetQuantizationScale(scale);
    commonTensorInfo.SetQuantizationOffset(offset);

    ConstantUsageTest(computeDevice,
        commonTensorInfo,
        QuantizedVector<uint8_t>(scale, offset, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }), // Input.
        QuantizedVector<uint8_t>(scale, offset, { 6.f, 5.f, 4.f, 3.f, 2.f, 1.f }), // Const input.
        QuantizedVector<uint8_t>(scale, offset, { 7.f, 7.f, 7.f, 7.f, 7.f, 7.f })  // Expected output.
    );
}

BOOST_AUTO_TEST_CASE(ConstantUsage_Ref_Float32)
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    ConstantUsageFloat32Test(backends);
}

#if ARMCOMPUTENEON_ENABLED
BOOST_AUTO_TEST_CASE(ConstantUsage_Neon_Float32)
{
    ConstantUsageFloat32Test({armnn::Compute::CpuAcc});
}
#endif

#if ARMCOMPUTECL_ENABLED
BOOST_AUTO_TEST_CASE(ConstantUsage_Cl_Float32)
{
    ConstantUsageFloat32Test({armnn::Compute::GpuAcc});
}
#endif

BOOST_AUTO_TEST_CASE(ConstantUsage_Ref_Uint8)
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    ConstantUsageUint8Test(backends);
}

BOOST_AUTO_TEST_CASE(TrivialAdd)
{
    // This test was designed to match "AddTwo" in android nn/runtime/test/TestTrivialModel.cpp.

    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // Builds up the structure of the network.
    armnn::INetworkPtr net(INetwork::Create());

    IConnectableLayer* input1 = net->AddInputLayer(0);
    IConnectableLayer* input2 = net->AddInputLayer(1);
    IConnectableLayer* add    = net->AddAdditionLayer();
    IConnectableLayer* output = net->AddOutputLayer(0);

    input1->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input2->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    TensorInfo tensorInfo(TensorShape({3, 4}), DataType::Float32);
    input1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    input2->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // optimize the network
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Creates structures for input & output - matching android nn test.
    std::vector<float> input1Data
    {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input2Data
    {
        100.f, 200.f, 300.f, 400.f, 500.f, 600.f, 700.f, 800.f, 900.f, 1000.f, 1100.f, 1200.f
    };
    std::vector<float> outputData(12);

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), input1Data.data())},
        {1,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), input2Data.data())}
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results
    BOOST_TEST(outputData[0] == 101);
    BOOST_TEST(outputData[1] == 202);
    BOOST_TEST(outputData[2] == 303);
    BOOST_TEST(outputData[3] == 404);
    BOOST_TEST(outputData[4] == 505);
    BOOST_TEST(outputData[5] == 606);
    BOOST_TEST(outputData[6] == 707);
    BOOST_TEST(outputData[7] == 808);
    BOOST_TEST(outputData[8] == 909);
    BOOST_TEST(outputData[9] == 1010);
    BOOST_TEST(outputData[10] == 1111);
    BOOST_TEST(outputData[11] == 1212);
}

BOOST_AUTO_TEST_CASE(MultipleOutputs)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr  runtime(armnn::IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // ReLu1
    ActivationDescriptor activation1Descriptor;
    activation1Descriptor.m_Function = ActivationFunction::BoundedReLu;
    activation1Descriptor.m_A = 1.f;
    activation1Descriptor.m_B = -1.f;
    IConnectableLayer* activation1 = net->AddActivationLayer(activation1Descriptor);

    // ReLu6
    ActivationDescriptor activation2Descriptor;
    activation2Descriptor.m_Function = ActivationFunction::BoundedReLu;
    activation2Descriptor.m_A = 6.0f;
    IConnectableLayer* activation2 = net->AddActivationLayer(activation2Descriptor);

    // BoundedReLu(min=2, max=5)
    ActivationDescriptor activation3Descriptor;
    activation3Descriptor.m_Function = ActivationFunction::BoundedReLu;
    activation3Descriptor.m_A = 5.0f;
    activation3Descriptor.m_B = 2.0f;
    IConnectableLayer* activation3 = net->AddActivationLayer(activation3Descriptor);

    IConnectableLayer* output1 = net->AddOutputLayer(0);
    IConnectableLayer* output2 = net->AddOutputLayer(1);
    IConnectableLayer* output3 = net->AddOutputLayer(2);

    input->GetOutputSlot(0).Connect(activation1->GetInputSlot(0));
    input->GetOutputSlot(0).Connect(activation2->GetInputSlot(0));
    input->GetOutputSlot(0).Connect(activation3->GetInputSlot(0));

    activation1->GetOutputSlot(0).Connect(output1->GetInputSlot(0));
    activation2->GetOutputSlot(0).Connect(output2->GetInputSlot(0));
    activation3->GetOutputSlot(0).Connect(output3->GetInputSlot(0));

    // Sets the tensors in the network.
    TensorInfo tensorInfo(TensorShape({ 10 }), DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    activation1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    activation2->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    activation3->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // optimize the network
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Creates structures for input & output.
    const std::vector<float> inputData{ 3.f, 5.f, 2.f, 3.f, 7.f, 0.f, -2.f, -1.f, 3.f, 3.f };

    std::vector<float> output1Data(inputData.size());
    std::vector<float> output2Data(inputData.size());
    std::vector<float> output3Data(inputData.size());

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), output1Data.data())},
        {1,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 1), output2Data.data())},
        {2,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 2), output3Data.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    BOOST_TEST(output1Data == std::vector<float>({ 1.f, 1.f, 1.f, 1.f, 1.f, 0.f, -1.f, -1.f, 1.f, 1.f })); // ReLu1
    BOOST_TEST(output2Data == std::vector<float>({ 3.f, 5.f, 2.f, 3.f, 6.f, 0.f, 0.f, 0.f, 3.f, 3.f })); // ReLu6
    BOOST_TEST(output3Data == std::vector<float>({ 3.f, 5.f, 2.f, 3.f, 5.f, 2.f, 2.f, 2.f, 3.f, 3.f })); // [2, 5]
}

#if ARMCOMPUTENEON_ENABLED
BOOST_AUTO_TEST_CASE(FallbackToCpuRef)
{
    using namespace armnn;

    // Create runtime in which test will run and allow fallback to CpuRef.
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc but we allow fallback to CpuRef so it shoud pass.
    NormalizationDescriptor descriptor;
    IConnectableLayer* pooling = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<BackendId> backends = {Compute::CpuAcc, Compute::CpuRef};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Load it into the runtime. It should pass.
    NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}
#endif // ARMCOMPUTENEON_ENABLED

BOOST_AUTO_TEST_CASE(ErrorOnLoadNetwork)
{
    using namespace armnn;

    // Create runtime in which test will run
    // Note we don't allow falling back to CpuRef if an operation (excluding inputs, outputs, etc.) isn't supported
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc and isn't allowed to fall back, so Optimize will return null.
    NormalizationDescriptor descriptor;
    IConnectableLayer* pooling = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<BackendId> backends = {Compute::CpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(!optNet);
}

BOOST_AUTO_TEST_SUITE_END()
