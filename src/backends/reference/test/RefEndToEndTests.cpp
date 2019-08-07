//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/test/EndToEndTestImpl.hpp>

#include <backendsCommon/test/ArithmeticTestImpl.hpp>
#include <backendsCommon/test/BatchToSpaceNdEndToEndTestImpl.hpp>
#include <backendsCommon/test/ConcatTestImpl.hpp>
#include <backendsCommon/test/DequantizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/DetectionPostProcessTestImpl.hpp>
#include <backendsCommon/test/GatherEndToEndTestImpl.hpp>
#include <backendsCommon/test/PreluEndToEndTestImpl.hpp>
#include <backendsCommon/test/ResizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/SpaceToDepthEndToEndTestImpl.hpp>
#include <backendsCommon/test/SplitterEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>

BOOST_AUTO_TEST_SUITE(RefEndToEnd)

std::vector<armnn::BackendId> defaultBackends = {armnn::Compute::CpuRef};

BOOST_AUTO_TEST_CASE(ConstantUsage_Ref_Float32)
{
    BOOST_TEST(ConstantUsageFloat32Test(defaultBackends));
}

BOOST_AUTO_TEST_CASE(ConstantUsage_Ref_Uint8)
{
    BOOST_TEST(ConstantUsageUint8Test(defaultBackends));
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
    IOptimizedNetworkPtr optNet = Optimize(*net, defaultBackends, runtime->GetDeviceSpec());

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
    IOptimizedNetworkPtr optNet = Optimize(*net, defaultBackends, runtime->GetDeviceSpec());

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
    IOptimizedNetworkPtr optNet = Optimize(*net, defaultBackends, runtime->GetDeviceSpec());

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

BOOST_AUTO_TEST_CASE(TrivialMin)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // Builds up the structure of the network.
    armnn::INetworkPtr net(INetwork::Create());

    IConnectableLayer* input1 = net->AddInputLayer(0);
    IConnectableLayer* input2 = net->AddInputLayer(1);
    IConnectableLayer* min    = net->AddMinimumLayer();
    IConnectableLayer* output = net->AddOutputLayer(0);

    input1->GetOutputSlot(0).Connect(min->GetInputSlot(0));
    input2->GetOutputSlot(0).Connect(min->GetInputSlot(1));
    min->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    TensorInfo tensorInfo(TensorShape({1, 1, 1, 4}), DataType::Float32);
    input1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    input2->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    min->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, defaultBackends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Creates structures for input & output - matching android nn test.
    std::vector<float> input1Data
        {
            1.0f, 2.0f, 3.0f, 4.0f
        };
    std::vector<float> input2Data
        {
            2.0f, 1.0f, 5.0f, 2.0f
        };
    std::vector<float> outputData(4);

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
    BOOST_TEST(outputData[0] == 1);
    BOOST_TEST(outputData[1] == 1);
    BOOST_TEST(outputData[2] == 3);
    BOOST_TEST(outputData[3] == 2);
}

BOOST_AUTO_TEST_CASE(RefNoCopyWorkloads)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    NormalizationDescriptor descriptor;
    IConnectableLayer* norm = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(norm->GetInputSlot(0));
    norm->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 1 }, DataType::Float32));
    norm->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 1 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, defaultBackends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    std::vector<float> outputData(4);

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // The result of the inference is not important, just the fact that there
    // should not be CopyMemGeneric workloads.
    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains RefNormalizationWorkload
    std::size_t found = dump.find("RefNormalizationWorkload");
    BOOST_TEST(found != std::string::npos);
    // Contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    BOOST_TEST(found != std::string::npos);
    // No contains CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    BOOST_TEST(found == std::string::npos);
}

BOOST_AUTO_TEST_CASE(RefEqualSimpleEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 1, 1, 1, 1,  0, 0, 0, 0,
                                                0, 0, 0, 0,  1, 1, 1, 1 });

    ArithmeticSimpleEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                 LayerType::Equal,
                                                                                 expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefGreaterSimpleEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ArithmeticSimpleEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                 LayerType::Greater,
                                                                                 expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefEqualSimpleEndToEndUint8Test)
{
    const std::vector<uint8_t> expectedOutput({ 1, 1, 1, 1,  0, 0, 0, 0,
                                                0, 0, 0, 0,  1, 1, 1, 1 });

    ArithmeticSimpleEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                         LayerType::Equal,
                                                                                         expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefGreaterSimpleEndToEndUint8Test)
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ArithmeticSimpleEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                         LayerType::Greater,
                                                                                         expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefEqualBroadcastEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 1, 0, 1, 1, 0, 0,
                                                0, 0, 0, 0, 0, 0 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                    LayerType::Equal,
                                                                                    expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefGreaterBroadcastEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                    LayerType::Greater,
                                                                                    expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefEqualBroadcastEndToEndUint8Test)
{
    const std::vector<uint8_t > expectedOutput({ 1, 0, 1, 1, 0, 0,
                                                 0, 0, 0, 0, 0, 0 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                            LayerType::Equal,
                                                                                            expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefGreaterBroadcastEndToEndUint8Test)
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                            LayerType::Greater,
                                                                                            expectedOutput);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndFloat32NHWCTest)
{
    BatchToSpaceNdEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndUint8NHWCTest)
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndQSymm16NHWCTest)
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndFloat32NCHWTest)
{
    BatchToSpaceNdEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndUint8NCHWTest)
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndQSymm16NCHWTest)
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndComplexFloat32NHWCTest)
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndComplexUint8NHWCTest)
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndComplexQSymm16NHWCTest)
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndComplexFloat32NCHWTest)
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndComplexUint8NCHWTest)
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefBatchToSpaceNdEndToEndComplexQSymm16NCHWTest)
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim0Test)
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim0Uint8Test)
{
    ConcatDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim1Test)
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim1Uint8Test)
{
    ConcatDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim2Test)
{
    ConcatDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim2Uint8Test)
{
    ConcatDim2EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim3Test)
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefConcatEndToEndDim3Uint8Test)
{
    ConcatDim3EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefGatherFloatTest)
{
    GatherEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefGatherUint8Test)
{
    GatherEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefGatherInt16Test)
{
    GatherEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefGatherMultiDimFloatTest)
{
    GatherMultiDimEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefGatherMultiDimUint8Test)
{
    GatherMultiDimEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefGatherMultiDimInt16Test)
{
    GatherMultiDimEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(DequantizeEndToEndSimpleTest)
{
    DequantizeEndToEndSimple<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(DequantizeEndToEndOffsetTest)
{
    DequantizeEndToEndOffset<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(DequantizeEndToEndSimpleInt16Test)
{
    DequantizeEndToEndSimple<armnn::DataType::QuantisedSymm16>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(DequantizeEndToEndOffsetInt16Test)
{
    DequantizeEndToEndOffset<armnn::DataType::QuantisedSymm16>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefDetectionPostProcessRegularNmsTest)
{
    std::vector<float> boxEncodings({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scores({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchors({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });
    DetectionPostProcessRegularNmsEndToEnd<armnn::DataType::Float32>(defaultBackends, boxEncodings, scores, anchors);
}

inline void QuantizeData(uint8_t* quant, const float* dequant, const TensorInfo& info)
{
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        quant[i] = armnn::Quantize<uint8_t>(dequant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

BOOST_AUTO_TEST_CASE(RefDetectionPostProcessRegularNmsUint8Test)
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo scoresInfo({ 1, 6, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);

    boxEncodingsInfo.SetQuantizationScale(1.0f);
    boxEncodingsInfo.SetQuantizationOffset(1);
    scoresInfo.SetQuantizationScale(0.01f);
    scoresInfo.SetQuantizationOffset(0);
    anchorsInfo.SetQuantizationScale(0.5f);
    anchorsInfo.SetQuantizationOffset(0);

    std::vector<float> boxEncodings({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scores({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchors({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });

    std::vector<uint8_t> qBoxEncodings(boxEncodings.size(), 0);
    std::vector<uint8_t> qScores(scores.size(), 0);
    std::vector<uint8_t> qAnchors(anchors.size(), 0);
    QuantizeData(qBoxEncodings.data(), boxEncodings.data(), boxEncodingsInfo);
    QuantizeData(qScores.data(), scores.data(), scoresInfo);
    QuantizeData(qAnchors.data(), anchors.data(), anchorsInfo);
    DetectionPostProcessRegularNmsEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, qBoxEncodings,
                                                                             qScores, qAnchors,
                                                                             1.0f, 1, 0.01f, 0, 0.5f, 0);
}

BOOST_AUTO_TEST_CASE(RefDetectionPostProcessFastNmsTest)
{
    std::vector<float> boxEncodings({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scores({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchors({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });
    DetectionPostProcessFastNmsEndToEnd<armnn::DataType::Float32>(defaultBackends, boxEncodings, scores, anchors);
}

BOOST_AUTO_TEST_CASE(RefDetectionPostProcessFastNmsUint8Test)
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo scoresInfo({ 1, 6, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);

    boxEncodingsInfo.SetQuantizationScale(1.0f);
    boxEncodingsInfo.SetQuantizationOffset(1);
    scoresInfo.SetQuantizationScale(0.01f);
    scoresInfo.SetQuantizationOffset(0);
    anchorsInfo.SetQuantizationScale(0.5f);
    anchorsInfo.SetQuantizationOffset(0);

    std::vector<float> boxEncodings({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scores({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchors({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });

    std::vector<uint8_t> qBoxEncodings(boxEncodings.size(), 0);
    std::vector<uint8_t> qScores(scores.size(), 0);
    std::vector<uint8_t> qAnchors(anchors.size(), 0);
    QuantizeData(qBoxEncodings.data(), boxEncodings.data(), boxEncodingsInfo);
    QuantizeData(qScores.data(), scores.data(), scoresInfo);
    QuantizeData(qAnchors.data(), anchors.data(), anchorsInfo);
    DetectionPostProcessFastNmsEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, qBoxEncodings,
                                                                          qScores, qAnchors,
                                                                          1.0f, 1, 0.01f, 0, 0.5f, 0);
}

BOOST_AUTO_TEST_CASE(RefPreluEndToEndTestFloat32)
{
    PreluEndToEndNegativeTest<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefPreluEndToEndTestUint8)
{
    PreluEndToEndPositiveTest<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefPreluEndToEndTestQSymm16)
{
    PreluEndToEndPositiveTest<armnn::DataType::QuantisedSymm16>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSpaceToDepthNHWCEndToEndTest1)
{
    SpaceToDepthNHWCEndToEndTest1(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSpaceToDepthNCHWEndToEndTest1)
{
    SpaceToDepthNCHWEndToEndTest1(defaultBackends);

}

BOOST_AUTO_TEST_CASE(RefSpaceToDepthNHWCEndToEndTest2)
{
    SpaceToDepthNHWCEndToEndTest2(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSpaceToDepthNCHWEndToEndTest2)
{
    SpaceToDepthNCHWEndToEndTest2(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter1dEndToEndTest)
{
    Splitter1dEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter1dEndToEndUint8Test)
{
    Splitter1dEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter2dDim0EndToEndTest)
{
    Splitter2dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter2dDim1EndToEndTest)
{
    Splitter2dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter2dDim0EndToEndUint8Test)
{
    Splitter2dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter2dDim1EndToEndUint8Test)
{
    Splitter2dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter3dDim0EndToEndTest)
{
    Splitter3dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter3dDim1EndToEndTest)
{
    Splitter3dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter3dDim2EndToEndTest)
{
    Splitter3dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter3dDim0EndToEndUint8Test)
{
    Splitter3dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter3dDim1EndToEndUint8Test)
{
    Splitter3dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter3dDim2EndToEndUint8Test)
{
    Splitter3dDim2EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim0EndToEndTest)
{
    Splitter4dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim1EndToEndTest)
{
    Splitter4dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim2EndToEndTest)
{
    Splitter4dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim3EndToEndTest)
{
    Splitter4dDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim0EndToEndUint8Test)
{
    Splitter4dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim1EndToEndUint8Test)
{
    Splitter4dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim2EndToEndUint8Test)
{
    Splitter4dDim2EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(RefSplitter4dDim3EndToEndUint8Test)
{
    Splitter4dDim3EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

// TransposeConvolution2d
BOOST_AUTO_TEST_CASE(RefTransposeConvolution2dEndToEndFloatNchwTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefTransposeConvolution2dEndToEndUint8NchwTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefTransposeConvolution2dEndToEndInt16NchwTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefTransposeConvolution2dEndToEndFloatNhwcTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefTransposeConvolution2dEndToEndUint8NhwcTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefTransposeConvolution2dEndToEndInt16NhwcTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedSymm16, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

// Resize Bilinear
BOOST_AUTO_TEST_CASE(RefResizeBilinearEndToEndFloatNchwTest)
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefResizeBilinearEndToEndUint8NchwTest)
{
    ResizeBilinearEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefResizeBilinearEndToEndInt16NchwTest)
{
    ResizeBilinearEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefResizeBilinearEndToEndFloatNhwcTest)
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefResizeBilinearEndToEndUint8NhwcTest)
{
    ResizeBilinearEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefResizeBilinearEndToEndInt16NhwcTest)
{
    ResizeBilinearEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NHWC);
}

// Resize NearestNeighbor
BOOST_AUTO_TEST_CASE(RefResizeNearestNeighborEndToEndFloatNchwTest)
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefResizeNearestNeighborEndToEndUint8NchwTest)
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefResizeNearestNeighborEndToEndInt16NchwTest)
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(RefResizeNearestNeighborEndToEndFloatNhwcTest)
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefResizeNearestNeighborEndToEndUint8NhwcTest)
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(RefResizeNearestNeighborEndToEndInt16NhwcTest)
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QuantisedSymm16>(defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_SUITE_END()