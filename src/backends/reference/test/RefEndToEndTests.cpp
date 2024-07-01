//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/test/EndToEndTestImpl.hpp>

#include <backendsCommon/test/ActivationEndToEndTestImpl.hpp>
#include <backendsCommon/test/AdditionEndToEndTestImpl.hpp>
#include <backendsCommon/test/ArgMinMaxEndToEndTestImpl.hpp>
#include <backendsCommon/test/BatchToSpaceNdEndToEndTestImpl.hpp>
#include <backendsCommon/test/BatchMatMulEndToEndTestImpl.hpp>
#include <backendsCommon/test/BroadcastToEndToEndTestImpl.hpp>
#include <backendsCommon/test/ChannelShuffleEndToEndTestImpl.hpp>
#include <backendsCommon/test/ComparisonEndToEndTestImpl.hpp>
#include <backendsCommon/test/ConcatEndToEndTestImpl.hpp>
#include <backendsCommon/test/Convolution2dEndToEndTestImpl.hpp>
#include <backendsCommon/test/Convolution3dEndToEndTestImpl.hpp>
#include <backendsCommon/test/DepthToSpaceEndToEndTestImpl.hpp>
#include <backendsCommon/test/DepthwiseConvolution2dEndToEndTests.hpp>
#include <backendsCommon/test/DequantizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/DetectionPostProcessEndToEndTestImpl.hpp>
#include <backendsCommon/test/ElementwiseBinaryEndToEndTestImpl.hpp>
#include <backendsCommon/test/ElementwiseUnaryEndToEndTestImpl.hpp>
#include <backendsCommon/test/FillEndToEndTestImpl.hpp>
#include <backendsCommon/test/FullyConnectedEndToEndTestImpl.hpp>
#include <backendsCommon/test/GatherEndToEndTestImpl.hpp>
#include <backendsCommon/test/GatherNdEndToEndTestImpl.hpp>
#include <backendsCommon/test/InstanceNormalizationEndToEndTestImpl.hpp>
#include <backendsCommon/test/LogSoftmaxEndToEndTestImpl.hpp>
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include <backendsCommon/test/PreluEndToEndTestImpl.hpp>
#include <backendsCommon/test/QLstmEndToEndTestImpl.hpp>
#include <backendsCommon/test/QuantizationEndToEndTestImpl.hpp>
#include <backendsCommon/test/RankEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReduceEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReshapeEndToEndTestImpl.hpp>
#include <backendsCommon/test/ResizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReverseV2EndToEndTestImpl.hpp>
#include <backendsCommon/test/ScatterNdEndToEndTestImpl.hpp>
#include <backendsCommon/test/SliceEndToEndTestImpl.hpp>
#include <backendsCommon/test/SoftmaxEndToEndTestImpl.hpp>
#include <backendsCommon/test/SpaceToDepthEndToEndTestImpl.hpp>
#include <backendsCommon/test/SplitterEndToEndTestImpl.hpp>
#include <backendsCommon/test/StridedSliceAsyncEndToEndTest.hpp>
#include <backendsCommon/test/SubgraphUtilsTest.hpp>
#include <backendsCommon/test/TileEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeEndToEndTestImpl.hpp>

#include <doctest/doctest.h>

TEST_SUITE("RefEndToEnd")
{
std::vector<armnn::BackendId> defaultBackends = {armnn::Compute::CpuRef};

// ElementwiseUnary
// Abs
TEST_CASE("RefAbsEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends,
                                                             UnaryOperation::Abs);
}

TEST_CASE("RefAbsEndToEndTestUint8")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends,
                                                              UnaryOperation::Abs);
}

TEST_CASE("RefAbsEndToEndTestInt16")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QSymmS16>(defaultBackends,
                                                              UnaryOperation::Abs);
}

// Rsqrt
TEST_CASE("RefRsqrtEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends,
                                                             UnaryOperation::Rsqrt);
}

TEST_CASE("RefRsqrtEndToEndTestUint8")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends,
                                                              UnaryOperation::Rsqrt);
}

TEST_CASE("RefRsqrtEndToEndTestInt16")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QSymmS16>(defaultBackends,
                                                              UnaryOperation::Rsqrt);
}

// Exp
TEST_CASE("RefExpEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends,
                                                             UnaryOperation::Exp);
}

TEST_CASE("RefExpEndToEndTestUint8")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends,
                                                              UnaryOperation::Exp);
}

TEST_CASE("RefExpEndToEndTestInt8")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends,
                                                              UnaryOperation::Exp);
}

TEST_CASE("RefExpEndToEndTestInt16")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QSymmS16>(defaultBackends,
                                                              UnaryOperation::Exp);
}

// Addition
TEST_CASE("RefAdditionEndtoEndFloat32")
{
    AdditionEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefAdditionEndtoEndUint8")
{
    AdditionEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

// Constant
TEST_CASE("ConstantUsage_Ref_Float32")
{
    CHECK(ConstantUsageFloat32Test(defaultBackends));
}

TEST_CASE("ConstantUsage_Ref_Uint8")
{
    CHECK(ConstantUsageUint8Test(defaultBackends));
}

TEST_CASE("Unsigned8")
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
    TensorInfo inputTensorInfo(TensorShape({1, 5}), DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(100);
    inputTensorInfo.SetQuantizationScale(10000.0f);
    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    TensorInfo outputTensorInfo(TensorShape({1, 5}), DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(1.0f/255.0f);
    softmax->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, defaultBackends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    auto error = runtime->LoadNetwork(netId, std::move(optNet));
    CHECK(error == Status::Success);

    // Creates structures for input & output.
    std::vector<uint8_t> inputData
    {
        1, 10, 3, 200, 5 // Some inputs - one of which is sufficiently larger than the others to saturate softmax.
    };
    std::vector<uint8_t> outputData(5);

    TensorInfo inputTensorInfo2 = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo2.SetConstant(true);
    armnn::InputTensors inputTensors
    {
        {0, armnn::ConstTensor(inputTensorInfo2, inputData.data())}
    };
    armnn::OutputTensors outputTensors
    {
        {0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    CHECK(outputData[0] == 0);
    CHECK(outputData[1] == 0);
    CHECK(outputData[2] == 0);
    CHECK(outputData[3] == 255); // softmax has been saturated.
    CHECK(outputData[4] == 0);
}

TEST_CASE("TrivialAdd")
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
    IConnectableLayer* add    = net->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Add));
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

    TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(inputTensorInfo, input1Data.data())},
        {1,armnn::ConstTensor(inputTensorInfo, input2Data.data())}
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results
    CHECK(outputData[0] == 101);
    CHECK(outputData[1] == 202);
    CHECK(outputData[2] == 303);
    CHECK(outputData[3] == 404);
    CHECK(outputData[4] == 505);
    CHECK(outputData[5] == 606);
    CHECK(outputData[6] == 707);
    CHECK(outputData[7] == 808);
    CHECK(outputData[8] == 909);
    CHECK(outputData[9] == 1010);
    CHECK(outputData[10] == 1111);
    CHECK(outputData[11] == 1212);
}

TEST_CASE("MultipleOutputs")
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

    TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(inputTensorInfo, inputData.data())}
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
    CHECK(output1Data == std::vector<float>({ 1.f, 1.f, 1.f, 1.f, 1.f, 0.f, -1.f, -1.f, 1.f, 1.f })); // ReLu1
    CHECK(output2Data == std::vector<float>({ 3.f, 5.f, 2.f, 3.f, 6.f, 0.f, 0.f, 0.f, 3.f, 3.f })); // ReLu6
    CHECK(output3Data == std::vector<float>({ 3.f, 5.f, 2.f, 3.f, 5.f, 2.f, 2.f, 2.f, 3.f, 3.f })); // [2, 5]
}

TEST_CASE("TrivialMin")
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // Builds up the structure of the network.
    armnn::INetworkPtr net(INetwork::Create());

    IConnectableLayer* input1 = net->AddInputLayer(0);
    IConnectableLayer* input2 = net->AddInputLayer(1);
    IConnectableLayer* min    = net->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Minimum));
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

    TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
        {
            {0,armnn::ConstTensor(inputTensorInfo, input1Data.data())},
            {1,armnn::ConstTensor(inputTensorInfo, input2Data.data())}
        };
    OutputTensors outputTensors
        {
            {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
        };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results
    CHECK(outputData[0] == 1);
    CHECK(outputData[1] == 1);
    CHECK(outputData[2] == 3);
    CHECK(outputData[3] == 2);
}

TEST_CASE("RefEqualSimpleEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 1, 1, 1, 1,  0, 0, 0, 0,
                                                0, 0, 0, 0,  1, 1, 1, 1 });

    ComparisonSimpleEndToEnd<armnn::DataType::Float32>(defaultBackends,
                                                       ComparisonOperation::Equal,
                                                       expectedOutput);
}

TEST_CASE("RefGreaterSimpleEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ComparisonSimpleEndToEnd<armnn::DataType::Float32>(defaultBackends,
                                                       ComparisonOperation::Greater,
                                                       expectedOutput);
}

TEST_CASE("RefEqualSimpleEndToEndUint8Test")
{
    const std::vector<uint8_t> expectedOutput({ 1, 1, 1, 1,  0, 0, 0, 0,
                                                0, 0, 0, 0,  1, 1, 1, 1 });

    ComparisonSimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends,
                                                               ComparisonOperation::Equal,
                                                               expectedOutput);
}

TEST_CASE("RefGreaterSimpleEndToEndUint8Test")
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ComparisonSimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends,
                                                               ComparisonOperation::Greater,
                                                               expectedOutput);
}

TEST_CASE("RefEqualBroadcastEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 1, 0, 1, 1, 0, 0,
                                                0, 0, 0, 0, 0, 0 });

    ComparisonBroadcastEndToEnd<armnn::DataType::Float32>(defaultBackends,
                                                          ComparisonOperation::Equal,
                                                          expectedOutput);
}

TEST_CASE("RefGreaterBroadcastEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ComparisonBroadcastEndToEnd<armnn::DataType::Float32>(defaultBackends,
                                                          ComparisonOperation::Greater,
                                                          expectedOutput);
}

TEST_CASE("RefEqualBroadcastEndToEndUint8Test")
{
    const std::vector<uint8_t > expectedOutput({ 1, 0, 1, 1, 0, 0,
                                                 0, 0, 0, 0, 0, 0 });

    ComparisonBroadcastEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends,
                                                                  ComparisonOperation::Equal,
                                                                  expectedOutput);
}

TEST_CASE("RefGreaterBroadcastEndToEndUint8Test")
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ComparisonBroadcastEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends,
                                                                  ComparisonOperation::Greater,
                                                                  expectedOutput);
}

TEST_CASE("RefBatchMatMulEndToEndFloat32Test")
{
    BatchMatMulEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefBatchMatMulEndToEndInt8Test")
{
    BatchMatMulEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefBatchMatMulNoTransposeEndToEndFloat32Test")
{
    BatchMatMulNoTransposeEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefBatchMatMulNoTransposeEndToEndInt8Test")
{
    BatchMatMulNoTransposeEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefBatchMatMulSimple4DEndToEndFloat32Test")
{
    BatchMatMulSimple4DEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefBatchMatMulSimple4DEndToEndInt8Test")
{
    BatchMatMulSimple4DEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefBatchMatMul4DEndToEndFloat32Test")
{
    BatchMatMul4DEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefBatchMatMul4DEndToEndInt8Test")
{
    BatchMatMul4DEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefBatchMatMulNotSquareEndToEndFloat32Test")
{
    BatchMatMulNotSquareEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefBatchMatMulNotSquareEndToEndInt8Test")
{
    BatchMatMulNotSquareEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefBatchToSpaceNdEndToEndFloat32NHWCTest")
{
    BatchToSpaceNdEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefBatchToSpaceNdEndToEndUint8NHWCTest")
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefBatchToSpaceNdEndToEndQSymm16NHWCTest")
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefBatchToSpaceNdEndToEndFloat32NCHWTest")
{
    BatchToSpaceNdEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefBatchToSpaceNdEndToEndUint8NCHWTest")
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefBatchToSpaceNdEndToEndQSymm16NCHWTest")
{
    BatchToSpaceNdEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefBatchToSpaceNdEndToEndComplexFloat32NHWCTest")
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefBatchToSpaceNdEndToEndComplexUint8NHWCTest")
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefBatchToSpaceNdEndToEndComplexQSymm16NHWCTest")
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefBatchToSpaceNdEndToEndComplexFloat32NCHWTest")
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefBatchToSpaceNdEndToEndComplexUint8NCHWTest")
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefBatchToSpaceNdEndToEndComplexQSymm16NCHWTest")
{
    BatchToSpaceNdComplexEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefChannelShuffleFloatTest")
{
    ChannelShuffleEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefChannelShuffleUint8Test")
{
    ChannelShuffleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim0Test")
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim0Uint8Test")
{
    ConcatDim0EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim1Test")
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim1Uint8Test")
{
    ConcatDim1EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim2Test")
{
    ConcatDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim2Uint8Test")
{
    ConcatDim2EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim3Test")
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefConcatEndToEndDim3Uint8Test")
{
    ConcatDim3EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefConvolution2dFloat32Test")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefConvolution2dNchwFloat32Test")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefConvolution2dFloat16Test")
{
    Convolution2dEndToEnd<armnn::DataType::Float16>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefConvolution2dInt8Test")
{
    Convolution2dEndToEnd<armnn::DataType::QSymmS8,
                          armnn::DataType::QSymmS8,
                          armnn::DataType::Signed32>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefConvolution3dFloat32Test")
{
    Convolution3dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(defaultBackends,
                                                                              armnn::DataLayout::NDHWC);
}

TEST_CASE("RefConvolution3dNcdhwFloat32Test")
{
    Convolution3dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(defaultBackends,
                                                                              armnn::DataLayout::NCDHW);
}

TEST_CASE("RefConvolution3dFloat16Test")
{
    Convolution3dEndToEnd<armnn::DataType::Float16, armnn::DataType::Float16>(defaultBackends,
                                                                              armnn::DataLayout::NDHWC);
}

TEST_CASE("RefConvolution3dUint8Test")
{
    Convolution3dEndToEnd<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(defaultBackends,
                                                                                armnn::DataLayout::NDHWC);
}

TEST_CASE("RefConvolution3dInt8Test")
{
    Convolution3dEndToEnd<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(defaultBackends,
                                                                                armnn::DataLayout::NDHWC);
}

TEST_CASE("RefDepthwiseConvolution2dEndtoEndFloat32Test")
{
    DepthwiseConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(defaultBackends,
                                                                                       armnn::DataLayout::NHWC);
}

TEST_CASE("RefDepthwiseConvolution2dEndtoEndFloat32TestBiasDisabled")
{
    DepthwiseConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(defaultBackends,
                                                                                       armnn::DataLayout::NHWC,
                                                                                       false);
}

TEST_CASE("RefFillEndToEndTest")
{
    FillEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefFillEndToEndTestFloat16")
{
    FillEndToEnd<armnn::DataType::Float16>(defaultBackends);
}

TEST_CASE("RefFillEndToEndTestInt32")
{
    FillEndToEnd<armnn::DataType::Signed32>(defaultBackends);
}

// Fully Connected
TEST_CASE("RefFullyConnectedEndToEndTestFloat32")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::Float32>(defaultBackends, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestNoBiasFloat32")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::Float32>(defaultBackends, false);
}

TEST_CASE("RefFullyConnectedEndToEndTestInt8")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QAsymmS8,
                                                 armnn::DataType::QAsymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QAsymmS8>(defaultBackends, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestNoBiasInt8")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QAsymmS8,
                                                 armnn::DataType::QAsymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QAsymmS8>(defaultBackends, false);
}

TEST_CASE("RefFullyConnectedEndToEndTestInt8Symm")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QSymmS8,
                                                 armnn::DataType::QSymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QSymmS8>(defaultBackends, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestNoBiasInt8Symm")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QSymmS8,
                                                 armnn::DataType::QSymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QSymmS8>(defaultBackends, false);
}

TEST_CASE("RefFullyConnectedEndToEndTestUint8")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QAsymmU8,
                                                 armnn::DataType::QAsymmU8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QAsymmU8>(defaultBackends, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestNoBiasUint8")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QAsymmU8,
                                                 armnn::DataType::QAsymmU8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QAsymmU8>(defaultBackends, false);
}

TEST_CASE("RefFullyConnectedEndToEndTestNoBiasOtherFloat32")
{
    FullyConnectedWithDynamicWeightsEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefFullyConnectedEndToEndTestNonConstantWeightsConstantBiasesFloat32")
{
    FullyConnectedWithDynamicOrConstantInputsEndToEnd<armnn::DataType::Float32>(defaultBackends, true, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestConstantWeightsNonConstantBiasesFloat32")
{
    FullyConnectedWithDynamicOrConstantInputsEndToEnd<armnn::DataType::Float32>(defaultBackends, true, false);
}

TEST_CASE("RefFullyConnectedEndToEndTestConstantWeightsTensorInfoNotSet")
{
    FullyConnectedErrorChecking<armnn::DataType::Float32>(defaultBackends, false, true, true, true, false);
}

TEST_CASE("RefFullyConnectedEndToEndTestWeightsNotConnectedExplicitCheck")
{
    FullyConnectedErrorChecking<armnn::DataType::Float32>(defaultBackends, true, true, false, true, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestBiasNotConnectedExplicitCheck")
{
    FullyConnectedErrorChecking<armnn::DataType::Float32>(defaultBackends, true, true, true, false, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestWeightsAndBiasNotConnected")
{
    FullyConnectedErrorChecking<armnn::DataType::Float32>(defaultBackends, false, true, false, false, true);
}

TEST_CASE("RefFullyConnectedEndToEndTestBiasDisabledConnectBias")
{
    FullyConnectedErrorChecking<armnn::DataType::Float32>(defaultBackends, true, false, false, true, true);
}

// Gather
TEST_CASE("RefGatherFloatTest")
{
    GatherEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefGatherUint8Test")
{
    GatherEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefGatherInt16Test")
{
    GatherEndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefGatherMultiDimFloatTest")
{
    GatherMultiDimEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefGatherMultiDimUint8Test")
{
    GatherMultiDimEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefGatherMultiDimInt16Test")
{
    GatherMultiDimEndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefGatherNdFloatTest")
{
    GatherNdEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefGatherNdUint8Test")
{
    GatherNdEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefGatherNdInt16Test")
{
    GatherNdEndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefGatherNdMultiDimFloatTest")
{
    GatherNdMultiDimEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefGatherNdMultiDimUint8Test")
{
    GatherNdMultiDimEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefGatherNdMultiDimInt16Test")
{
    GatherNdMultiDimEndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

// DepthToSpace
TEST_CASE("DepthToSpaceEndToEndNchwFloat32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DepthToSpaceEndToEndNchwFloat16")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float16>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DepthToSpaceEndToEndNchwUint8")
{
    DepthToSpaceEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DepthToSpaceEndToEndNchwInt16")
{
    DepthToSpaceEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DepthToSpaceEndToEndNhwcFloat32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("DepthToSpaceEndToEndNhwcFloat16")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float16>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("DepthToSpaceEndToEndNhwcUint8")
{
    DepthToSpaceEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("DepthToSpaceEndToEndNhwcInt16")
{
    DepthToSpaceEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NHWC);
}

// Quantization
TEST_CASE("RefQuantizeInt8")
{
    QuantizationEndToEndInt8(defaultBackends);
}

// Dequantize
TEST_CASE("DequantizeEndToEndSimpleTest")
{
    DequantizeEndToEndSimple<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("DequantizeEndToEndOffsetTest")
{
    DequantizeEndToEndOffset<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("DequantizeEndToEndSimpleInt16Test")
{
    DequantizeEndToEndSimple<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("DequantizeEndToEndOffsetInt16Test")
{
    DequantizeEndToEndOffset<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefDetectionPostProcessRegularNmsTest")
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

TEST_CASE("RefDetectionPostProcessRegularNmsUint8Test")
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
    DetectionPostProcessRegularNmsEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, qBoxEncodings,
                                                                             qScores, qAnchors,
                                                                             1.0f, 1, 0.01f, 0, 0.5f, 0);
}

TEST_CASE("RefDetectionPostProcessFastNmsTest")
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

TEST_CASE("RefDetectionPostProcessFastNmsUint8Test")
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
    DetectionPostProcessFastNmsEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, qBoxEncodings,
                                                                          qScores, qAnchors,
                                                                          1.0f, 1, 0.01f, 0, 0.5f, 0);
}

// Activations
// Linear
TEST_CASE("RefLinearEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(defaultBackends, ActivationFunction::Linear);
}

TEST_CASE("RefLinearEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(defaultBackends, ActivationFunction::Linear);
}

TEST_CASE("RefLinearEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(defaultBackends, ActivationFunction::Linear);
}

TEST_CASE("RefLinearEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmU8>(defaultBackends, ActivationFunction::Linear);
}

TEST_CASE("RefLinearEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<armnn::DataType::QSymmS16>(defaultBackends, ActivationFunction::Linear);
}

// Sigmoid
TEST_CASE("RefSigmoidEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(defaultBackends, ActivationFunction::Sigmoid);
}

TEST_CASE("RefSigmoidEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(defaultBackends, ActivationFunction::Sigmoid);
}

TEST_CASE("RefSigmoidEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(defaultBackends, ActivationFunction::Sigmoid);
}

TEST_CASE("RefSigmoidEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmU8>(defaultBackends, ActivationFunction::Sigmoid, 0.12f, 15);
}

TEST_CASE("RefSigmoidEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<armnn::DataType::QSymmS16>(defaultBackends, ActivationFunction::Sigmoid);
}

// ReLu
TEST_CASE("RefReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(defaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("RefReLuEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(defaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("RefReLuEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(defaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("RefReLuEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmU8>(defaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("RefReLuEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<armnn::DataType::QSymmS16>(defaultBackends, ActivationFunction::ReLu);
}

// BoundedReLu
TEST_CASE("RefBoundedReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(defaultBackends, ActivationFunction::BoundedReLu);
}

TEST_CASE("RefBoundedReLuEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(defaultBackends, ActivationFunction::BoundedReLu);
}

TEST_CASE("RefBoundedReLuEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(defaultBackends, ActivationFunction::BoundedReLu);
}

TEST_CASE("RefBoundedReLuEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmU8>(defaultBackends, ActivationFunction::BoundedReLu);
}

TEST_CASE("RefBoundedReLuEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<armnn::DataType::QSymmS16>(defaultBackends, ActivationFunction::BoundedReLu);
}

// SoftReLu
TEST_CASE("RefSoftReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(defaultBackends, ActivationFunction::SoftReLu);
}

TEST_CASE("RefSoftReLuEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(defaultBackends, ActivationFunction::SoftReLu);
}

TEST_CASE("RefSoftReLuEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(defaultBackends, ActivationFunction::SoftReLu);
}

TEST_CASE("RefSoftReLuEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmU8>(defaultBackends, ActivationFunction::SoftReLu, 0.12f, 15);
}

TEST_CASE("RefSoftReLuEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<armnn::DataType::QSymmS16>(defaultBackends, ActivationFunction::SoftReLu);
}

// LeakyRelu
TEST_CASE("RefLeakyReluActivationFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(defaultBackends, ActivationFunction::LeakyReLu, 1.f, 0, 0.01f);
}

TEST_CASE("RefLeakyReluActivationFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(defaultBackends, ActivationFunction::LeakyReLu, 0.3f, 5, 0.01f);
}

TEST_CASE("RefLeakyReluActivationInt8")
{
    ActivationEndToEndTest<DataType::QAsymmS8>(defaultBackends, ActivationFunction::LeakyReLu, 0.6f, 7, 0.01f);
}

TEST_CASE("RefLeakyReluActivationUInt8")
{
    ActivationEndToEndTest<DataType::QAsymmU8>(defaultBackends, ActivationFunction::LeakyReLu, 0.12f, 15, 0.01f);
}

TEST_CASE("RefLeakyReluActivationInt16")
{
    ActivationEndToEndTest<DataType::QSymmS16>(defaultBackends, ActivationFunction::LeakyReLu, 0.15f, 55, 0.01f);
}

// Elu
TEST_CASE("RefEluEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(defaultBackends, ActivationFunction::Elu);
}

TEST_CASE("RefEluEndToEndTestFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(defaultBackends, ActivationFunction::Elu);
}

TEST_CASE("RefEluEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<DataType::QAsymmS8>(defaultBackends, ActivationFunction::Elu);
}

TEST_CASE("RefEluEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<DataType::QAsymmU8>(defaultBackends, ActivationFunction::Elu);
}

TEST_CASE("RefEluEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<DataType::QSymmS16>(defaultBackends, ActivationFunction::Elu);
}

// HardSwish
TEST_CASE("RefHardSwishEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(defaultBackends, ActivationFunction::HardSwish);
}

TEST_CASE("RefHardSwishEndToEndTestFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(defaultBackends, ActivationFunction::HardSwish);
}

TEST_CASE("RefHardSwishEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<DataType::QAsymmS8>(defaultBackends, ActivationFunction::HardSwish);
}

TEST_CASE("RefHardSwishEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<DataType::QAsymmU8>(defaultBackends, ActivationFunction::HardSwish);
}

TEST_CASE("RefHardSwishEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<DataType::QSymmS16>(defaultBackends, ActivationFunction::HardSwish);
}

// TanH
TEST_CASE("RefTanHEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(defaultBackends, ActivationFunction::TanH, 1.f, 0, 2, 3);
}

TEST_CASE("RefTanHEndToEndTestFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(defaultBackends, ActivationFunction::TanH, 1.f, 0, 2, 3);
}

TEST_CASE("RefTanHEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<DataType::QAsymmS8>(defaultBackends, ActivationFunction::TanH, 1.f, 0, 3, 2);
}

TEST_CASE("RefTanHEndToEndTestQAsymmU8")
{
    ActivationEndToEndTest<DataType::QAsymmU8>(defaultBackends, ActivationFunction::TanH, 1.f, 0, 3, 2);
}

TEST_CASE("RefTanHEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<DataType::QSymmS16>(defaultBackends, ActivationFunction::TanH, 1.f, 0, 3, 2);
}

// LogSoftmax
TEST_CASE("RefLogSoftmaxEndToEndTest")
{
    LogSoftmaxEndToEndTest(defaultBackends);
}

// Softmax
TEST_CASE("RefSoftmaxEndToEndTestFloat32")
{
    SoftmaxEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

// Prelu
TEST_CASE("RefPreluEndToEndTestFloat32")
{
    PreluEndToEndNegativeTest<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefPreluEndToEndTestUint8")
{
    PreluEndToEndPositiveTest<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefPreluEndToEndTestQSymm16")
{
    PreluEndToEndPositiveTest<armnn::DataType::QSymmS16>(defaultBackends);
}

// Pooling 2D
// Average Pool 2D
TEST_CASE("RefAvgPool2DEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(defaultBackends);
}

TEST_CASE("RefAvgPool2DEndtoEndTestFloat16")
{
    AvgPool2dEndToEndFloat16<DataType::Float16>(defaultBackends);
}

TEST_CASE("RefAvgPool2DIgnoreValueEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(defaultBackends, PaddingMethod::IgnoreValue);
}

// Max Pool 2D
TEST_CASE("RefMaxPool2DEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(defaultBackends);
}

TEST_CASE("RefMaxPool2DEndtoEndTestFloat16")
{
    MaxPool2dEndToEndFloat16<DataType::Float16>(defaultBackends);
}

TEST_CASE("RefMaxPool2DIgnoreValueEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(defaultBackends, PaddingMethod::IgnoreValue);
}

TEST_CASE("RefMaxPool2DTwoLayerEndtoEndTestFloat32")
{
    MaxPool2dTwoLayerEndToEnd<DataType::Float32>(defaultBackends);
}

TEST_CASE("RefMaxPool2DThreeLayerEndtoEndTestFloat32")
{
    MaxPool2dThreeLayerEndToEnd<DataType::Float32>(defaultBackends);
}

// Quantization
TEST_CASE("QuantizationEndToEndFloat32_U8Test")
{
    QuantizationEndToEndFloat32<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("QuantizationEndToEndFloat32_I8Test")
{
    QuantizationEndToEndFloat32<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("QuantizationEndToEndFloat32_S16Test")
{
    QuantizationEndToEndFloat32<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("QuantizationEndToEndFloat16_U8Test")
{
    QuantizationEndToEndFloat16<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("QuantizationEndToEndFloat16_I8Test")
{
    QuantizationEndToEndFloat16<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("QuantizationEndToEndFloat16_S16Test")
{
    QuantizationEndToEndFloat16<armnn::DataType::QSymmS16>(defaultBackends);
}

// ScatterNd
TEST_CASE("RefScatterNd1DInputEndToEndFloat32Test")
{
    ScatterNd1DimUpdateWithInputEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefScatterNd1DInputEndToEndInt8Test")
{
    ScatterNd1DimUpdateWithInputEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefScatterNd1DNoInputEndToEndFloat32Test")
{
    ScatterNd1DimUpdateNoInputEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefScatterNd1DNoInputEndToEndInt8Test")
{
    ScatterNd1DimUpdateNoInputEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefScatterNd2DInputEndToEndFloat32Test")
{
    ScatterNd2DimUpdateWithInputEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefScatterNd2DInputEndToEndInt8Test")
{
    ScatterNd2DimUpdateWithInputEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefScatterNd2DNoInputEndToEndFloat32Test")
{
    ScatterNd2DimUpdateNoInputEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefScatterNd2DNoInputEndToEndInt8Test")
{
    ScatterNd2DimUpdateNoInputEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

// SpaceToDepth
TEST_CASE("RefSpaceToDepthNhwcEndToEndTest1")
{
    SpaceToDepthNhwcEndToEndTest1(defaultBackends);
}

TEST_CASE("RefSpaceToDepthNchwEndToEndTest1")
{
    SpaceToDepthNchwEndToEndTest1(defaultBackends);
}

TEST_CASE("RefSpaceToDepthNhwcEndToEndTest2")
{
    SpaceToDepthNhwcEndToEndTest2(defaultBackends);
}

TEST_CASE("RefSpaceToDepthNchwEndToEndTest2")
{
    SpaceToDepthNchwEndToEndTest2(defaultBackends);
}

// Split
TEST_CASE("RefSplit1dEndtoEndTestSigned16")
{
    Splitter1dEndToEnd<DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefSplit1dEndtoEndTestFloat16")
{
    Splitter1dEndToEndFloat16<DataType::Float16>(defaultBackends);
}

TEST_CASE("RefSplitter1dEndToEndTest")
{
    Splitter1dEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter1dEndToEndUint8Test")
{
    Splitter1dEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter2dDim0EndToEndTest")
{
    Splitter2dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter2dDim1EndToEndTest")
{
    Splitter2dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter2dDim0EndToEndUint8Test")
{
    Splitter2dDim0EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter2dDim1EndToEndUint8Test")
{
    Splitter2dDim1EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplit3dDim1EndtoEndTestSigned16")
{
    Splitter3dDim1EndToEnd<DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefSplit3dDim1EndtoEndTestFloat16")
{
    Splitter3dDim1EndToEndFloat16<DataType::Float16>(defaultBackends);
}

TEST_CASE("RefSplitter3dDim0EndToEndTest")
{
    Splitter3dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter3dDim1EndToEndTest")
{
    Splitter3dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter3dDim2EndToEndTest")
{
    Splitter3dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter3dDim0EndToEndUint8Test")
{
    Splitter3dDim0EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter3dDim1EndToEndUint8Test")
{
    Splitter3dDim1EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter3dDim2EndToEndUint8Test")
{
    Splitter3dDim2EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim0EndToEndTest")
{
    Splitter4dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim1EndToEndTest")
{
    Splitter4dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim2EndToEndTest")
{
    Splitter4dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplit4dDim2EndtoEndTestFloat16")
{
    Splitter4dDim2EndToEndFloat16<DataType::Float16>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim3EndToEndTest")
{
    Splitter4dDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim0EndToEndUint8Test")
{
    Splitter4dDim0EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim1EndToEndUint8Test")
{
    Splitter4dDim1EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim2EndToEndUint8Test")
{
    Splitter4dDim2EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefSplitter4dDim3EndToEndUint8Test")
{
    Splitter4dDim3EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

// Tile
TEST_CASE("RefTileEndToEndFloat32")
{
    TileEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefTileEndToEndFloat16")
{
    TileEndToEnd<armnn::DataType::Float16>(defaultBackends);
}

TEST_CASE("RefTileEndToEndQAsymmS8")
{
    TileEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefTileEndToEndQAsymmU8")
{
    TileEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefTileEndToEndQSymmS8")
{
    TileEndToEnd<armnn::DataType::QSymmS8>(defaultBackends);
}

TEST_CASE("RefTileEndToEndQSymmS16")
{
    TileEndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefTileEndToEndSigned32")
{
    TileEndToEnd<armnn::DataType::Signed32>(defaultBackends);
}

// TransposeConvolution2d
TEST_CASE("RefTransposeConvolution2dEndToEndFloatNchwTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefTransposeConvolution2dEndToEndUint8NchwTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefTransposeConvolution2dEndToEndInt16NchwTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefTransposeConvolution2dEndToEndFloatNhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefTransposeConvolution2dEndToEndUint8NhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefTransposeConvolution2dEndToEndInt16NhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

// Transpose
TEST_CASE("RefTransposeEndToEndTest")
{
    TransposeEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

// Resize Bilinear
TEST_CASE("RefResizeBilinearEndToEndFloatNchwTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeBilinearEndToEndUint8NchwTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeBilinearEndToEndInt8NchwTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::QSymmS8>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeBilinearEndToEndInt16NchwTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeBilinearEndToEndFloatNhwcTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefResizeBilinearEndToEndUint8NhwcTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefResizeBilinearEndToEndInt8NhwcTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::QSymmS8>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefResizeBilinearEndToEndInt16NhwcTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NHWC);
}

// Resize NearestNeighbor
TEST_CASE("RefResizeNearestNeighborEndToEndFloatNchwTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeNearestNeighborEndToEndUint8NchwTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeNearestNeighborEndToEndInt8NchwTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeNearestNeighborEndToEndInt16NchwTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("RefResizeNearestNeighborEndToEndFloatNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefResizeNearestNeighborEndToEndUint8NhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefResizeNearestNeighborEndToEndInt8NhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefResizeNearestNeighborEndToEndInt16NhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS16>(defaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("RefResizeNearestNeighborEndToEndFloatAlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("RefResizeNearestNeighborEndToEndFloatHalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(defaultBackends, armnn::DataLayout::NHWC, false, true);
}

TEST_CASE("RefResizeNearestNeighborEndToEndInt8AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("RefResizeNearestNeighborEndToEndInt8HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS8>(defaultBackends, armnn::DataLayout::NHWC, false, true);
}

// ReverseV2
TEST_CASE("RefReverseV2EndToEndFloat16Test")
{
    ReverseV2EndToEnd<armnn::DataType::Float16>(defaultBackends);
}

TEST_CASE("RefReverseV2EndToEndFloat32Test")
{
    ReverseV2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefReverseV2EndToEndQAsymmS8Test")
{
    ReverseV2EndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefReverseV2EndToEndQAsymmU8Test")
{
    ReverseV2EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefReverseV2EndToEndQSymmS8Test")
{
    ReverseV2EndToEnd<armnn::DataType::QSymmS8>(defaultBackends);
}

TEST_CASE("RefReverseV2EndToEndQSymmS16Test")
{
    ReverseV2EndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefReverseV2EndToEndSigned32Test")
{
    ReverseV2EndToEnd<armnn::DataType::Signed32>(defaultBackends);
}

// InstanceNormalization
TEST_CASE("RefInstanceNormalizationNhwcEndToEndTest1")
{
    InstanceNormalizationNhwcEndToEndTest1(defaultBackends);
}

TEST_CASE("RefInstanceNormalizationNchwEndToEndTest1")
{
    InstanceNormalizationNchwEndToEndTest1(defaultBackends);
}

TEST_CASE("RefInstanceNormalizationNhwcEndToEndTest2")
{
    InstanceNormalizationNhwcEndToEndTest2(defaultBackends);
}

TEST_CASE("RefInstanceNormalizationNchwEndToEndTest2")
{
    InstanceNormalizationNchwEndToEndTest2(defaultBackends);
}

// ArgMinMax
TEST_CASE("RefArgMaxSimpleTest")
{
    ArgMaxEndToEndSimple<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMaxSimpleUint8Test")
{
    ArgMaxEndToEndSimple<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMinSimpleTest")
{
    ArgMinEndToEndSimple<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMinSimpleUint8Test")
{
    ArgMinEndToEndSimple<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis0Test")
{
    ArgMaxAxis0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis0Uint8Test")
{
    ArgMaxAxis0EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMinAxis0Test")
{
    ArgMinAxis0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMinAxis0Uint8Test")
{

    ArgMinAxis0EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis1Test")
{
    ArgMaxAxis1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis1Uint8Test")
{
    ArgMaxAxis1EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMinAxis1Test")
{
    ArgMinAxis1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMinAxis1Uint8Test")
{

    ArgMinAxis1EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis2Test")
{
    ArgMaxAxis2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis2Uint8Test")
{
    ArgMaxAxis2EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMinAxis2Test")
{
    ArgMinAxis2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMinAxis2Uint8Test")
{

    ArgMinAxis2EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis3Test")
{
    ArgMaxAxis3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMaxAxis3Uint8Test")
{
    ArgMaxAxis3EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefArgMinAxis3Test")
{
    ArgMinAxis3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefArgMinAxis3Uint8Test")
{

    ArgMinAxis3EndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefQLstmEndToEndTest")
{
    QLstmEndToEnd(defaultBackends);
}

TEST_CASE("RefRankEndToEndTest")
{
    RankEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefRankEndToEndTestFloat16")
{
    RankEndToEnd<armnn::DataType::Float16>(defaultBackends);
}

TEST_CASE("RefRankEndToEndTestInt32")
{
    RankEndToEnd<armnn::DataType::Signed32>(defaultBackends);
}

TEST_CASE("RefRankEndToEndTestQAsymmS8")
{
    RankEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

TEST_CASE("RefRankEndToEndTestQSymmS16")
{
    RankEndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefRankEndToEndTestQSymmS8")
{
    RankEndToEnd<armnn::DataType::QSymmS8>(defaultBackends);
}

// Reduce
TEST_CASE("RefReduceEndToEndTest")
{
    ReduceEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefReduceEndToEndTestFloat16")
{
    ReduceEndToEnd<armnn::DataType::Float16>(defaultBackends);
}

// Reshape
TEST_CASE("RefReshapeEndToEndTest")
{
    ReshapeEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefReshapeEndToEndTestFloat16")
{
    ReshapeEndToEndFloat16<armnn::DataType::Float16>(defaultBackends);
}

TEST_CASE("RefReshapeEndToEndTestInt32")
{
    ReshapeEndToEnd<armnn::DataType::Signed32>(defaultBackends);
}

TEST_CASE("RefReshapeEndToEndTestInt16")
{
    ReshapeEndToEnd<armnn::DataType::QSymmS16>(defaultBackends);
}

TEST_CASE("RefReshapeEndToEndTestUInt8")
{
    ReshapeEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends);
}

TEST_CASE("RefReshapeEndToEndTestInt8")
{
    ReshapeEndToEnd<armnn::DataType::QAsymmS8>(defaultBackends);
}

// Force Import
TEST_CASE("RefForceImportWithAlignedBuffersEndToEndTest")
{
    ForceImportWithAlignedBuffersEndToEndTest(defaultBackends);
}

TEST_CASE("RefForceImportWithMisalignedInputBuffersEndToEndTest")
{
    ForceImportWithMisalignedInputBuffersEndToEndTest(defaultBackends);
}

TEST_CASE("RefForceImportWithMisalignedOutputBuffersEndToEndTest")
{
    ForceImportWithMisalignedOutputBuffersEndToEndTest(defaultBackends);
}

TEST_CASE("RefForceImportWithMisalignedInputAndOutputBuffersEndToEndTest")
{
    ForceImportWithMisalignedInputAndOutputBuffersEndToEndTest(defaultBackends);
}

TEST_CASE("RefForceImportRepeatedInferencesEndToEndTest")
{
    ForceImportRepeatedInferencesEndToEndTest(defaultBackends);
}

TEST_CASE("RefForceImportRepeatedInferencesInvertedEndToEndTest")
{
    ForceImportRepeatedInferencesInvertedEndToEndTest(defaultBackends);
}

#if !defined(__ANDROID__)
// Only run these tests on non Android platforms
TEST_CASE("RefImportNonAlignedPointerTest")
{
    ImportNonAlignedInputPointerTest(defaultBackends);
}

TEST_CASE("RefExportNonAlignedPointerTest")
{
    ExportNonAlignedOutputPointerTest(defaultBackends);
}

TEST_CASE("RefImportAlignedPointerTest")
{
    ImportAlignedPointerTest(defaultBackends);
}

TEST_CASE("RefImportOnlyWorkload")
{
    ImportOnlyWorkload(defaultBackends);
}

TEST_CASE("RefExportOnlyWorkload")
{
    ExportOnlyWorkload(defaultBackends);
}

TEST_CASE("RefImportAndExportWorkload")
{
    ImportAndExportWorkload(defaultBackends);
}

TEST_CASE("RefExportOutputWithSeveralOutputSlotConnectionsTest")
{
    ExportOutputWithSeveralOutputSlotConnectionsTest(defaultBackends);
}

// Slice
TEST_CASE("RefSliceEndtoEndTestFloat32")
{
    SliceEndToEnd<DataType::Float32>(defaultBackends);
}

TEST_CASE("RefSliceEndtoEndTestInt32")
{
    SliceEndToEnd<DataType::Signed32>(defaultBackends);
}

TEST_CASE("RefSliceEndtoEndTestFloat16")
{
    SliceEndToEndFloat16<DataType::Float16>(defaultBackends);
}

TEST_CASE("RefStridedSliceInvalidSliceEndToEndTest")
{
    StridedSliceInvalidSliceEndToEndTest(defaultBackends);
}

TEST_CASE("RefThreadSafeFP32StridedSlicedEndToEndTest")
{
    armnn::experimental::StridedSlicedEndToEndTest<armnn::DataType::Float32>(defaultBackends, 1);
}

TEST_CASE("RefAsyncFP32StridedSlicedMultiThreadedEndToEndTest")
{
    armnn::experimental::StridedSlicedMultiThreadedEndToEndTest<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefAsyncFP32StridedSlicedScheduledMultiThreadedEndToEndTest")
{
    armnn::experimental::StridedSlicedEndToEndTest<armnn::DataType::Float32>(defaultBackends, 3);
}

TEST_CASE("RefAddEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Add);
}
TEST_CASE("RefAddEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::Add);
}
TEST_CASE("RefAddEndToEndTestFloat32Simple3D")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Add);
}
TEST_CASE("RefAddEndToEndTestFloat16Simple3D")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(defaultBackends, BinaryOperation::Add);
}
TEST_CASE("RefDivEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Div);
}
TEST_CASE("RefDivEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::Div);
}
TEST_CASE("RefMulEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Mul);
}
TEST_CASE("RefMulEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::Mul);
}
TEST_CASE("RefSubEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Sub);
}
TEST_CASE("RefSubEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::Sub);
}
TEST_CASE("RefSubEndToEndTestFloat32Simple3D")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Sub);
}
TEST_CASE("RefSubEndToEndTestFloat16Simple3D")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(defaultBackends, BinaryOperation::Sub);
}
TEST_CASE("RefMaximumEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Maximum);
}
TEST_CASE("RefMaximumEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::Maximum);
}
TEST_CASE("RefMinimumEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Minimum);
}
TEST_CASE("RefMinimumEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::Minimum);
}
TEST_CASE("RefPowerEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::Power);
}
TEST_CASE("RefPowerEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::Power);
}
TEST_CASE("RefSquaredDifferenceEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(defaultBackends, BinaryOperation::SqDiff);
}
TEST_CASE("RefSquaredDifferenceEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(defaultBackends, BinaryOperation::SqDiff);
}

#endif

// Backend Optimization Tests
TEST_CASE("RefReshapeRemovalSimpleCaseEndToEnd")
{
    ReshapeRemovalEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefReshapeRemovalNCHWFirstEndToEnd")
{
    ReshapeRemovalNCHWEndToEnd<armnn::DataType::Float32>(defaultBackends, true, true);
}

TEST_CASE("RefReshapeRemovalNCHWSecondEndToEnd")
{
    ReshapeRemovalNCHWEndToEnd<armnn::DataType::Float32>(defaultBackends, true, false);
}

// BroadcastTo
TEST_CASE("RefBroadcastToEndToEndFloat32")
{
    BroadcastToEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

TEST_CASE("RefBroadcastToEndToEndWithMultiplicationFloat32")
{
    BroadcastToEndToEndElementWiseBinary<armnn::DataType::Float32>(defaultBackends,
                                                                   BinaryOperation::Mul);
}

TEST_CASE("RefBroadcastToEndToEndWithDivisionFloat32")
{
    BroadcastToEndToEndElementWiseBinary<armnn::DataType::Float32>(defaultBackends,
                                                                   BinaryOperation::Div);
}

TEST_CASE("RefBroadcastToEndToEndWithAdditionFloat32")
{
    BroadcastToEndToEndElementWiseBinary<armnn::DataType::Float32>(defaultBackends,
                                                                   BinaryOperation::Add);
}

TEST_CASE("RefBroadcastToEndToEndWithSubtractionFloat32")
{
    BroadcastToEndToEndElementWiseBinary<armnn::DataType::Float32>(defaultBackends,
                                                                   BinaryOperation::Sub);
}
}
