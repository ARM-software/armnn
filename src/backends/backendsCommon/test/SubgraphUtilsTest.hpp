//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <GraphUtils.hpp>
#include <CommonTestUtils.hpp>
#include <armnnTestUtils/DataLayoutUtils.hpp>

#include <doctest/doctest.h>

#include <vector>
#include "backendsCommon/SubgraphUtils.hpp"

namespace armnn
{

template<DataType ArmnnIType, DataType ArmnnOType,
        typename TInput = ResolveType<ArmnnIType>, typename TOutput = ResolveType<ArmnnOType>>
void EndToEndLayerTest(IRuntimePtr runtime,
                       IOptimizedNetworkPtr optNet,
                       const std::map<int, std::vector<TInput>>& inputTensorData,
                       const std::map<int, std::vector<TOutput>>& expectedOutputData,
                       float tolerance = 0.000001f)
{
    // Loads it into the runtime.
    NetworkId netId;
    std::string errorMessage;
    armnn::Status loadingStatus = runtime->LoadNetwork(netId, std::move(optNet), errorMessage);
    CHECK_MESSAGE(loadingStatus == Status::Success, errorMessage);

    InputTensors inputTensors;
    inputTensors.reserve(inputTensorData.size());
    for (auto&& it : inputTensorData)
    {
        inputTensors.push_back({it.first,
                                ConstTensor(runtime->GetInputTensorInfo(netId, it.first), it.second.data())});
    }
    OutputTensors outputTensors;
    outputTensors.reserve(expectedOutputData.size());
    std::map<int, std::vector<TOutput>> outputStorage;
    for (auto&& it : expectedOutputData)
    {
        std::vector<TOutput> out(it.second.size());
        outputStorage.emplace(it.first, out);
        outputTensors.push_back({it.first,
                                 Tensor(runtime->GetOutputTensorInfo(netId, it.first),
                                        outputStorage.at(it.first).data())});
    }

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    for (auto&& it : expectedOutputData)
    {
        std::vector<TOutput> out = outputStorage.at(it.first);
        for (unsigned int i = 0; i < out.size(); ++i)
        {
            CHECK_MESSAGE(Compare<ArmnnOType>(it.second[i], out[i], tolerance) == true,
                          "Actual output: " << out[i] << ". Expected output:" << it.second[i]);

        }
    }
}

template<armnn::DataType ArmnnType, typename T = ResolveType<ArmnnType>>
armnn::INetworkPtr CreateReshapeInOutNetwork(const armnn::TensorShape& inputShape,
                                             const armnn::TensorShape& outputShape,
                                             ReshapeDescriptor& descriptor,
                                             const float qScale = 1.0f,
                                             const int32_t qOffset = 0)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    IConnectableLayer* activation0 = network->AddActivationLayer(ActivationFunction::ReLu, "act0");
    IConnectableLayer* activation1 = network->AddActivationLayer(ActivationFunction::ReLu, "act1");
    IConnectableLayer* activation2 = network->AddActivationLayer(ActivationFunction::ReLu, "act2");
    IConnectableLayer* activation3 = network->AddActivationLayer(ActivationFunction::ReLu, "act3");
    IConnectableLayer* reshape = network->AddReshapeLayer(descriptor, "reshape");

    IConnectableLayer* input   = network->AddInputLayer(0, "input");
    IConnectableLayer* output1  = network->AddOutputLayer(0, "output1");
    IConnectableLayer* output2  = network->AddOutputLayer(1, "output2");
    IConnectableLayer* output3  = network->AddOutputLayer(2, "output3");

    Connect(input, activation0, inputTensorInfo, 0, 0);
    Connect(activation0, reshape, inputTensorInfo, 0, 0);

    Connect(reshape, activation1, outputTensorInfo, 0, 0);
    Connect(reshape, activation2, outputTensorInfo, 0, 0);
    Connect(reshape, activation3, outputTensorInfo, 0, 0);
    Connect(activation1, output1, outputTensorInfo, 0, 0);
    Connect(activation2, output2, outputTensorInfo, 0, 0);
    Connect(activation3, output3, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = ResolveType<ArmnnType>>
armnn::INetworkPtr CreateReshapeConv2dInOutNetwork(const armnn::TensorShape& inputShape,
                                                   const armnn::TensorShape& weightsShape,
                                                   const armnn::TensorShape& convOutputShape,
                                                   const armnn::TensorShape& outputShape,
                                                   std::vector<float>& weightsData,
                                                   ReshapeDescriptor& descriptor,
                                                   Convolution2dDescriptor& convolution2DDescriptor,
                                                   bool convFirst,
                                                   const float qScale = 1.0f,
                                                   const int32_t qOffset = 0)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());
    TensorInfo weightsTensorInfo(weightsShape, ArmnnType, qScale, qOffset, true);
    ConstTensor weights(weightsTensorInfo, weightsData);

    IConnectableLayer* convolution1 = network->AddConvolution2dLayer(convolution2DDescriptor, "conv2d");
    IConnectableLayer* weightsLayer = network->AddConstantLayer(weights, "weights");

    IConnectableLayer* activation1 = network->AddActivationLayer(ActivationFunction::ReLu, "act");
    IConnectableLayer* reshape = network->AddReshapeLayer(descriptor, "reshape");

    IConnectableLayer* input   = network->AddInputLayer(0, "input");
    IConnectableLayer* output  = network->AddOutputLayer(0, "output");

    TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset, true);
    TensorInfo convTensorInfo(convOutputShape, ArmnnType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);
    TensorInfo reshapeTensorInfo(descriptor.m_TargetShape, ArmnnType, qScale, qOffset, true);

    if (convFirst)
    {
        Connect(input, convolution1, inputTensorInfo, 0, 0);
        Connect(weightsLayer, convolution1, weightsTensorInfo, 0, 1);
        Connect(convolution1, reshape, convTensorInfo, 0, 0);
        Connect(reshape, activation1, reshapeTensorInfo, 0, 0);
        Connect(activation1, output, outputTensorInfo, 0, 0);
    }
    else
    {
        Connect(input, activation1, inputTensorInfo, 0, 0);
        Connect(activation1, reshape, inputTensorInfo, 0, 0);
        Connect(reshape, convolution1, reshapeTensorInfo, 0, 0);
        Connect(weightsLayer, convolution1, weightsTensorInfo, 0, 1);
        Connect(convolution1, output, outputTensorInfo, 0, 0);
    }
    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReshapeRemovalEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape = { 2, 3 };
    const TensorShape& outputShape = { 6 };

    ReshapeDescriptor descriptor;
    descriptor.m_TargetShape = outputShape;

    INetworkPtr network = CreateReshapeInOutNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<T> data{ 1, 2, 3,
                         4, 5, 6 };

    std::map<int, std::vector<float>> inputTensorData = { { 0, data } };
    std::map<int, std::vector<float>> expectedOutputData = { { 0, data }, { 1, data }, { 2, data } };

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec());

    Graph& graph = GetGraphForTesting(optNet.get());
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        LayerNameAndTypeCheck(LayerType::Input, "input"),
                        LayerNameAndTypeCheck(LayerType::Activation, "act0"),
                        LayerNameAndTypeCheck(LayerType::Activation, "act1"),
                        LayerNameAndTypeCheck(LayerType::Activation, "act2"),
                        LayerNameAndTypeCheck(LayerType::Activation, "act3"),
                        LayerNameAndTypeCheck(LayerType::Output, "output1"),
                        LayerNameAndTypeCheck(LayerType::Output, "output2"),
                        LayerNameAndTypeCheck(LayerType::Output, "output3")));

    EndToEndLayerTest<ArmnnType, ArmnnType>(std::move(runtime), std::move(optNet), inputTensorData, expectedOutputData);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReshapeRemovalNCHWEndToEnd(const std::vector<armnn::BackendId>& backends, bool shouldBeRemoved, bool convFirst)
{
    using namespace armnn;

    // shapes are different if convFirst or not
    //these are convFirst
    TensorShape inputShape;
    TensorShape convOutputShape;
    TensorShape weightsShape;
    TensorShape reshapeShape;
    TensorShape outputShape;

    if (convFirst)
    {
        inputShape = { 1, 1, 5, 5 };
        convOutputShape = { 1, 1, 3, 3 };
        weightsShape = { 1, 1, 3, 3 };
        reshapeShape = { 9 };
        outputShape = { 9 };
    }
    else
    {
        inputShape = { 5, 5 };
        reshapeShape = { 1, 1, 5, 5 };
        convOutputShape = { 1, 1, 3, 3 };
        weightsShape = { 1, 1, 3, 3 };
        outputShape = { 1, 1, 3, 3 };
    }

    ReshapeDescriptor descriptor;
    descriptor.m_TargetShape = reshapeShape;

    Convolution2dDescriptor convolution2DDescriptor;
    convolution2DDescriptor.m_PadLeft     = 0;
    convolution2DDescriptor.m_PadRight    = 0;
    convolution2DDescriptor.m_PadTop      = 0;
    convolution2DDescriptor.m_PadBottom   = 0;
    convolution2DDescriptor.m_StrideX     = 1;
    convolution2DDescriptor.m_StrideY     = 1;
    convolution2DDescriptor.m_DataLayout  = DataLayout::NCHW;
    convolution2DDescriptor.m_BiasEnabled = false;

    TensorInfo inputInfo(inputShape, DataType::Float32, true);
    TensorInfo outputInfo(convOutputShape, DataType::Float32);
    TensorInfo weightsInfo(weightsShape, DataType::Float32, true);

    std::vector<float> inputData =
    {
        1.0f, 8.0f, 3.0f, 4.0f, 6.0f,
        5.0f, 7.0f, 3.0f, 1.0f, 8.0f,
        2.0f, 3.0f, 9.0f, 8.0f, 1.0f,
        3.0f, 6.0f, 1.0f, 1.0f, 9.0f,
        5.0f, 3.0f, 9.0f, 3.0f, 2.0f
    };

    std::vector<float> weightsData =
    {
        4.0f, 0.0f, 3.0f,
        5.0f, 0.0f, 2.0f,
        6.0f, 0.0f, 1.0f
    };

    std::vector<float> outputData =
    {
        65.0f, 107.0f, 116.0f,
        76.0f,  99.0f,  98.0f,
        91.0f,  89.0f, 118.0f
    };

    INetworkPtr network = CreateReshapeConv2dInOutNetwork<DataType::Float32>(inputShape,
                                                                             weightsShape,
                                                                             convOutputShape,
                                                                             outputShape,
                                                                             weightsData,
                                                                             descriptor,
                                                                             convolution2DDescriptor,
                                                                             convFirst);
    CHECK(network);

    std::map<int, std::vector<float>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<float>> expectedOutputData = { { 0, outputData } };

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec());

    Graph& graph = GetGraphForTesting(optNet.get());

    if (shouldBeRemoved)
    {
        if (convFirst)
        {
            CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                                LayerNameAndTypeCheck(LayerType::Input, "input"),
                                LayerNameAndTypeCheck(LayerType::Constant, "weights"),
                                LayerNameAndTypeCheck(LayerType::Convolution2d, "conv2d"),
                                LayerNameAndTypeCheck(LayerType::Activation, "act"),
                                LayerNameAndTypeCheck(LayerType::Output, "output")));
        }
        else
        {
            CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                                LayerNameAndTypeCheck(LayerType::Input, "input"),
                                LayerNameAndTypeCheck(LayerType::Constant, "weights"),
                                LayerNameAndTypeCheck(LayerType::Activation, "act"),
                                LayerNameAndTypeCheck(LayerType::Convolution2d, "conv2d"),
                                LayerNameAndTypeCheck(LayerType::Output, "output")));
        }
    }
    else
    {
        if (convFirst)
        {
            CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                                LayerNameAndTypeCheck(LayerType::Input, "input"),
                                LayerNameAndTypeCheck(LayerType::Constant, "weights"),
                                LayerNameAndTypeCheck(LayerType::Convolution2d, "conv2d"),
                                LayerNameAndTypeCheck(LayerType::Reshape, "reshape"),
                                LayerNameAndTypeCheck(LayerType::Activation, "act"),
                                LayerNameAndTypeCheck(LayerType::Output, "output")));
        }
        else
        {
            CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                                LayerNameAndTypeCheck(LayerType::Input, "input"),
                                LayerNameAndTypeCheck(LayerType::Constant, "weights"),
                                LayerNameAndTypeCheck(LayerType::Activation, "act"),
                                LayerNameAndTypeCheck(LayerType::Reshape, "reshape"),
                                LayerNameAndTypeCheck(LayerType::Convolution2d, "conv2d"),
                                LayerNameAndTypeCheck(LayerType::Output, "output")));
        }
    }

    EndToEndLayerTest<ArmnnType, ArmnnType>(std::move(runtime), std::move(optNet), inputTensorData, expectedOutputData);
}

template<typename Descriptor, typename LayerType>
void CheckIsNCHW()
{
    Graph graph;
    Descriptor nchwDesc;
    nchwDesc.m_DataLayout = DataLayout::NCHW;
    auto nchwLayer = graph.AddLayer<LayerType>(nchwDesc, "");
    CHECK(IsNCHW(*nchwLayer));

    Descriptor nhwcDesc;
    nhwcDesc.m_DataLayout = DataLayout::NHWC;
    auto nhwcLayer = graph.AddLayer<LayerType>(nhwcDesc, "");
    CHECK_FALSE(IsNCHW(*nhwcLayer));
}

TEST_CASE("CheckIsNCHW")
{
    Graph graph;
    BatchMatMulDescriptor descriptor1;
    descriptor1.m_DataLayoutX = DataLayout::NHWC;
    descriptor1.m_DataLayoutY = DataLayout::NHWC;
    auto batchMatMulLayer1 = graph.AddLayer<BatchMatMulLayer>(descriptor1, "");
    CHECK_FALSE(IsNCHW(*batchMatMulLayer1));

    BatchMatMulDescriptor descriptor2;
    descriptor2.m_DataLayoutX = DataLayout::NCHW;
    descriptor2.m_DataLayoutY = DataLayout::NHWC;
    auto batchMatMulLayer2 = graph.AddLayer<BatchMatMulLayer>(descriptor2, "");
    CHECK(IsNCHW(*batchMatMulLayer2));

    BatchMatMulDescriptor descriptor3;
    descriptor3.m_DataLayoutX = DataLayout::NHWC;
    descriptor3.m_DataLayoutY = DataLayout::NCHW;
    auto batchMatMulLayer3 = graph.AddLayer<BatchMatMulLayer>(descriptor3, "");
    CHECK(IsNCHW(*batchMatMulLayer3));

    BatchMatMulDescriptor descriptor4;
    descriptor4.m_DataLayoutX = DataLayout::NCHW;
    descriptor4.m_DataLayoutY = DataLayout::NCHW;
    auto batchMatMulLayer4 = graph.AddLayer<BatchMatMulLayer>(descriptor4, "");
    CHECK(IsNCHW(*batchMatMulLayer4));

    CheckIsNCHW<BatchToSpaceNdDescriptor, BatchToSpaceNdLayer>();
    CheckIsNCHW<Convolution2dDescriptor, Convolution2dLayer>();
    CheckIsNCHW<Convolution3dDescriptor, Convolution3dLayer>();
    CheckIsNCHW<DepthwiseConvolution2dDescriptor, DepthwiseConvolution2dLayer>();
    CheckIsNCHW<InstanceNormalizationDescriptor, InstanceNormalizationLayer>();
    CheckIsNCHW<L2NormalizationDescriptor, L2NormalizationLayer>();
    CheckIsNCHW<NormalizationDescriptor, NormalizationLayer>();
    CheckIsNCHW<Pooling2dDescriptor, Pooling2dLayer>();
    CheckIsNCHW<Pooling3dDescriptor, Pooling3dLayer>();
    CheckIsNCHW<SpaceToBatchNdDescriptor, SpaceToBatchNdLayer>();
    CheckIsNCHW<SpaceToDepthDescriptor, SpaceToDepthLayer>();
    CheckIsNCHW<StridedSliceDescriptor, StridedSliceLayer>();

    // Check Default
    auto elementwiseLayer = graph.AddLayer<ElementwiseBinaryLayer>(BinaryOperation::Add, "");
    CHECK_FALSE(IsNCHW(*elementwiseLayer));
}


} // Namespace
