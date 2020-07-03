//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>

#include <armnn/Tensor.hpp>
#include <Graph.hpp>
#include <InternalTypes.hpp>
#include <layers/FullyConnectedLayer.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(ShapeInferenceTests)
using namespace armnn;
namespace
{

constexpr const bool maskPermutations[6][4] = {{false, false, false, false},
                                               {true,  false, false, false},
                                               {false, true,  false, false},
                                               {false, false, true,  false},
                                               {false, false, false,  true},
                                               {true,  true,  true,  true}};

template<typename LayerT, typename... Args>
LayerT* BuildGraph(Graph* graph, const std::vector<TensorShape>& inputShapes, Args &&... args)
{
    auto layer = graph->AddLayer<LayerT>(std::forward<Args>(args)...);

    uint32_t inputCount = 0;
    for (auto inputShape : inputShapes)
    {
        TensorInfo inputTensorInfo(inputShape, DataType::Float32);

        auto input = graph->AddLayer<InputLayer>(static_cast<int>(inputCount), "input");
        input->GetOutputSlot().SetTensorInfo(inputTensorInfo);
        input->GetOutputSlot().Connect(layer->GetInputSlot(inputCount));
        inputCount++;
    }

    return layer;
}

template<typename LayerT>
void RunShapeInferenceTest(LayerT* const layer,
                           const std::vector<std::initializer_list<unsigned int>> dimensionSizeLists)
{
    std::vector<unsigned int> numDimensions;
    std::vector<TensorShape> expectedOutputShapes;

    for (auto dimensionSizeList : dimensionSizeLists)
    {
        numDimensions.emplace_back(dimensionSizeList.size());
        expectedOutputShapes.emplace_back(TensorShape(dimensionSizeList));
    }

    const unsigned int outputSize = layer->GetNumOutputSlots();

    const auto runTestWithMask = [&](const bool maskPermutations[])
    {
        for (unsigned int i = 0; i < outputSize; ++i)
        {
            layer->GetOutputSlot(i).SetTensorInfo({{numDimensions[i], dimensionSizeLists[i].begin(), maskPermutations},
                                                  DataType::Float32});
        }

        layer->ValidateTensorShapesFromInputs();

        for (unsigned int i = 0; i < outputSize; ++i)
        {
            BOOST_CHECK(layer->GetOutputSlot(i).GetTensorInfo().GetShape() == expectedOutputShapes[i]);
        }
    };

    // Test inference with Dimensionality::NotSpecified
    for (unsigned int j = 0; j < outputSize; ++j)
    {
        layer->GetOutputSlot(j).SetTensorInfo({TensorShape(Dimensionality::NotSpecified), DataType::Float32});
    }

    layer->SetShapeInferenceMethod(ShapeInferenceMethod::ValidateOnly);

    BOOST_CHECK_THROW(layer->ValidateTensorShapesFromInputs(), LayerValidationException);

    layer->SetShapeInferenceMethod(ShapeInferenceMethod::InferAndValidate);
    layer->ValidateTensorShapesFromInputs();

    for (unsigned int i = 0; i < outputSize; ++i)
    {
        BOOST_CHECK(layer->GetOutputSlot(i).GetTensorInfo().GetShape() == expectedOutputShapes[i]);
    }

    // Test inference with Dimensionality::Specified and various combinations of dimensions of unknown size
    for (unsigned int i = 0; i < numDimensions[0]; ++i)
    {
        runTestWithMask(maskPermutations[i]);
    }

    // maskPermutations[5] equates to all dimensions being known
    runTestWithMask(maskPermutations[5]);
}

template<typename LayerT, typename... Args>
void CreateGraphAndRunTest(const std::vector<TensorShape>& inputShapes,
                           const std::vector<std::initializer_list<unsigned int>> dimensionSizeLists,
                           Args &&... args)
{
    Graph graph(true);

    auto layer = BuildGraph<LayerT>(&graph, inputShapes, std::forward<Args>(args)...);

    RunShapeInferenceTest<LayerT>(layer, dimensionSizeLists);
}

BOOST_AUTO_TEST_CASE(NetworkOptionsTest)
{
     BackendOptions ShapeInferenceMethodOption("ShapeInferenceMethod",
     {
        { "InferAndValidate", true }
     });

    INetworkPtr network = INetwork::Create({ShapeInferenceMethodOption});
    TensorInfo tensorInfo({ 5, 7, 6, 2 }, DataType::Float32);

    auto inputLayer = network->AddInputLayer(1, "inputLayer");
    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Abs;
    auto activationLayer = network->AddActivationLayer(descriptor, "activation");

    inputLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).SetTensorInfo({TensorShape{Dimensionality::NotSpecified}, DataType::Float32});

    BOOST_CHECK_NO_THROW(activationLayer->GetOutputSlot(0).IsTensorInfoSet());

    BOOST_CHECK(activationLayer->GetOutputSlot(0).GetTensorInfo() == tensorInfo);


    ShapeInferenceMethodOption = BackendOptions("ShapeInferenceMethod",
                                               {
                                                       { "InferAndValidate", false }
                                               });

    network = INetwork::Create({ShapeInferenceMethodOption});

    inputLayer = network->AddInputLayer(1, "inputLayer");
    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    activationLayer = network->AddActivationLayer(descriptor, "activation");

    inputLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).SetTensorInfo({TensorShape{Dimensionality::NotSpecified}, DataType::Float32});

    BOOST_CHECK_NO_THROW(activationLayer->GetOutputSlot(0).IsTensorInfoSet());

    network = INetwork::Create();

    inputLayer = network->AddInputLayer(1, "inputLayer");
    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    activationLayer = network->AddActivationLayer(descriptor, "activation");

    inputLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).SetTensorInfo({TensorShape{Dimensionality::NotSpecified}, DataType::Float32});

    BOOST_CHECK_NO_THROW(activationLayer->GetOutputSlot(0).IsTensorInfoSet());
}

BOOST_AUTO_TEST_CASE(AbsTest)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Abs;
    CreateGraphAndRunTest<ActivationLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, descriptor, "activation");
}

BOOST_AUTO_TEST_CASE(AdditionTest)
{
    CreateGraphAndRunTest<AdditionLayer>({{ 5, 7, 6, 2 }, { 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "add");
}

BOOST_AUTO_TEST_CASE(ArgMinMaxTest)
{
    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Function = ArgMinMaxFunction::Min;
    descriptor.m_Axis = 1;

    CreateGraphAndRunTest<ArgMinMaxLayer>({{ 1, 3, 2, 4 }}, {{ 1, 2, 4 }}, descriptor, "argMinMax");
}

BOOST_AUTO_TEST_CASE(BatchNormalizationTest)
{
    BatchNormalizationDescriptor descriptor;
    CreateGraphAndRunTest<BatchNormalizationLayer>({{ 1, 2, 3, 2 }}, {{ 1, 2, 3, 2 }}, descriptor, "batchNorm");
}

BOOST_AUTO_TEST_CASE(BatchToSpaceNdTest)
{
    BatchToSpaceNdDescriptor descriptor;

    std::vector<unsigned int> blockShape {2, 2};
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    descriptor.m_BlockShape = blockShape;
    descriptor.m_Crops = crops;
    descriptor.m_DataLayout = DataLayout::NHWC;

    CreateGraphAndRunTest<BatchToSpaceNdLayer>({{ 4, 2, 2, 1 }}, {{ 1, 4, 4, 1 }}, descriptor, "batchtospacend");
}

BOOST_AUTO_TEST_CASE(ComparisionTest)
{
    ComparisonDescriptor descriptor;
    descriptor.m_Operation = ComparisonOperation::Equal;
    CreateGraphAndRunTest<ComparisonLayer>({{ 5, 7, 6, 2 }, { 5, 7, 6, 2 }},
                                           {{ 5, 7, 6, 2 }},
                                           descriptor,
                                           "comparision");
}

BOOST_AUTO_TEST_CASE(ConcatTest)
{
    ConcatDescriptor descriptor(2, 3);

    descriptor.SetViewOriginCoord(0, 0, 0);
    descriptor.SetViewOriginCoord(1, 0, 1);

    CreateGraphAndRunTest<ConcatLayer>({{ 1, 2, 1 }, { 1, 2, 1 }}, {{ 2, 2, 1 }}, descriptor, "concat");
}

BOOST_AUTO_TEST_CASE(ConstantTesst)
{
    Graph graph;
    TensorShape outputShape{ 1, 1, 3, 3 };
    auto layer = BuildGraph<ConstantLayer>(&graph, {}, "constant");

    const float Datum = 0.0f;
    ConstTensor output0({outputShape, DataType::Float32}, &Datum);
    layer->m_LayerOutput = std::make_unique<ScopedCpuTensorHandle>(output0);

    layer->GetOutputSlot(0).SetTensorInfo({{1, 1, 3, 3}, DataType::Float32});

    layer->ValidateTensorShapesFromInputs();

    BOOST_CHECK(layer->GetOutputSlot(0).GetTensorInfo().GetShape() == outputShape);
}

BOOST_AUTO_TEST_CASE(ConvertBf16ToFp32Test)
{
    CreateGraphAndRunTest<ConvertBf16ToFp32Layer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "floor");
}

BOOST_AUTO_TEST_CASE(ConvertFp16ToBf16Test)
{
    const TensorShape tensorShape{5, 7, 6, 2};
    CreateGraphAndRunTest<ConvertFp32ToBf16Layer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "floor");
}

BOOST_AUTO_TEST_CASE(ConvertFp16ToFp32Test)
{
    CreateGraphAndRunTest<ConvertFp16ToFp32Layer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "floor");
}

BOOST_AUTO_TEST_CASE(ConvertFp32ToFp16Test)
{
    CreateGraphAndRunTest<ConvertFp32ToFp16Layer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "floor");
}

BOOST_AUTO_TEST_CASE(Convolution2dTest)
{
    const TensorShape inputShape{1, 1, 10, 10};

    Graph graph;

    Convolution2dDescriptor descriptor;

    descriptor.m_PadLeft = 0;
    descriptor.m_PadTop = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadBottom = 0;
    descriptor.m_StrideX = 1;
    descriptor.m_StrideY = 1;
    descriptor.m_DilationX = 3;
    descriptor.m_DilationY = 3;

    auto layer = BuildGraph<Convolution2dLayer>(&graph,
                                                 {inputShape},
                                                 descriptor,
                                                 "conv2d");

    const float Datum = 0.0f;
    ConstTensor weights({{1, 1, 3, 3}, DataType::Float32}, &Datum);
    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    RunShapeInferenceTest<Convolution2dLayer>(layer, {{ 1, 1, 4, 4 }});
}

BOOST_AUTO_TEST_CASE(DebugLayerTest)
{
    const TensorShape tensorShape;
    CreateGraphAndRunTest<DebugLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "debug");
}

BOOST_AUTO_TEST_CASE(DepthToSpaceTest)
{
    DepthToSpaceDescriptor descriptor;

    descriptor.m_BlockSize = 2;
    descriptor.m_DataLayout = DataLayout::NHWC;

    CreateGraphAndRunTest<DepthToSpaceLayer>({{ 1, 1, 1, 8}}, {{ 1, 2, 2, 2 }}, descriptor, "depthtospace");
}

BOOST_AUTO_TEST_CASE(DepthwiseConvolutionTest)
{
    DepthwiseConvolution2dDescriptor descriptor;

    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 0;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_DilationX = 0;
    descriptor.m_DilationY = 0;
    descriptor.m_DataLayout = DataLayout::NHWC;
    descriptor.m_BiasEnabled = false;

    Graph graph;

    auto layer = BuildGraph<DepthwiseConvolution2dLayer>(&graph,
                                                        {{ 8, 16, 2, 1 }},
                                                        descriptor,
                                                        "depthwiseconv2d");

    const float Datum = 0.0f;
    ConstTensor weights({{ 2, 5, 3, 2 }, DataType::Float32}, &Datum);
    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    RunShapeInferenceTest<DepthwiseConvolution2dLayer>(layer, {{ 8, 18, 1, 2 }});
}

BOOST_AUTO_TEST_CASE(DequantizeTest)
{
    const TensorShape tensorShape{5, 7, 6, 2};
    CreateGraphAndRunTest<DequantizeLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "dequantize");
}

BOOST_AUTO_TEST_CASE(DetectionPostProcessTest)
{
    const TensorShape detectionBoxesInfo{ 1, 3, 4 };
    const TensorShape detectionScoresInfo{ 1, 3, 4 };
    const TensorShape detectionClassesInfo{ 1, 3, 4 };

    armnn::DetectionPostProcessDescriptor descriptor;
    descriptor.m_UseRegularNms = true;
    descriptor.m_MaxDetections = 3;
    descriptor.m_MaxClassesPerDetection = 1;
    descriptor.m_DetectionsPerClass =1;
    descriptor.m_NmsScoreThreshold = 0.0;
    descriptor.m_NmsIouThreshold = 0.5;
    descriptor.m_NumClasses = 2;
    descriptor.m_ScaleY = 10.0;
    descriptor.m_ScaleX = 10.0;
    descriptor.m_ScaleH = 5.0;
    descriptor.m_ScaleW = 5.0;

    const float Datum = 0.0f;
    ConstTensor anchorsTensor({{1, 1, 3, 3}, DataType::Float32}, &Datum);

    Graph graph;

    auto layer = BuildGraph<DetectionPostProcessLayer>(&graph,
                                                       {detectionBoxesInfo, detectionScoresInfo},
                                                       descriptor,
                                                       "detectionpostprocess");

    layer->m_Anchors = std::make_unique<ScopedCpuTensorHandle>(anchorsTensor);

    RunShapeInferenceTest<DetectionPostProcessLayer>(layer, {{ 1, 3, 4 }, { 1, 3  }, { 1, 3 }, { 1 }});
}

BOOST_AUTO_TEST_CASE(FakeQuantizationTest)
{
    FakeQuantizationDescriptor descriptor;
    descriptor.m_Max = 1;
    descriptor.m_Min = 1;
    CreateGraphAndRunTest<FakeQuantizationLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, descriptor, "fakequantization");
}

BOOST_AUTO_TEST_CASE(FloorTest)
{
    const TensorShape tensorShape{5, 7, 6, 2};
    CreateGraphAndRunTest<FloorLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "floor");
}

BOOST_AUTO_TEST_CASE(FullyConnectedTest)
{
    Graph graph;

    const unsigned int inputWidth = 3u;
    const unsigned int inputHeight = 2u;
    const unsigned int inputChannels = 1u;
    const unsigned int outputChannels = 2u;

    auto layer = BuildGraph<FullyConnectedLayer>(&graph,
                                                 {{1, inputChannels, inputHeight, inputWidth}},
                                                 FullyConnectedDescriptor(),
                                                 "fc");


    const float Datum = 0.0f;
    ConstTensor weights({{inputChannels, outputChannels}, DataType::Float32}, &Datum);
    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    RunShapeInferenceTest<FullyConnectedLayer>(layer, {{ 1, outputChannels }});
}

BOOST_AUTO_TEST_CASE(GatherTest)
{
    CreateGraphAndRunTest<GatherLayer>({{ 7, 6, 2}, {2,3}}, {{ 2, 3, 6, 2 }}, GatherDescriptor(), "gather");
}

BOOST_AUTO_TEST_CASE(InstanceNormalizationTest)
{
    const TensorShape tensorShape{5, 7, 6, 2};

    CreateGraphAndRunTest<InstanceNormalizationLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }},
                                                      InstanceNormalizationDescriptor(),
                                                      "instancenorm");
}

BOOST_AUTO_TEST_CASE(L2NormalizationTest)
{
    const TensorShape tensorShape{5, 7, 6, 2};

    CreateGraphAndRunTest<L2NormalizationLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }},
                                                L2NormalizationDescriptor(),
                                                "l2norm");
}

BOOST_AUTO_TEST_CASE(LogSoftMaxTest)
{
    const TensorShape tensorShape{5, 7, 6, 2};

    CreateGraphAndRunTest<LogSoftmaxLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, LogSoftmaxDescriptor(), "logsoftmax");
}

BOOST_AUTO_TEST_CASE(LstmTest)
{
    const TensorShape inputShape{2, 5};
    const TensorShape inputCellState{2, 20};
    const TensorShape expectedOutputShape{2, 20};

    LstmDescriptor descriptor;

    descriptor.m_ActivationFunc = 4;
    descriptor.m_CifgEnabled = false;
    descriptor.m_PeepholeEnabled = false;
    descriptor.m_ProjectionEnabled = false;

    Graph graph;
    auto layer = BuildGraph<LstmLayer>(&graph, {inputShape, inputCellState, inputCellState}, descriptor, "lstm");

    float Datum = 0.0f;
    ConstTensor constTensor({{ 2, 5, 3, 2 }, DataType::Float32}, &Datum);

    layer->m_BasicParameters.m_InputToCellWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_InputToForgetWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_CellBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_ForgetGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_CifgParameters.m_InputGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_OutputGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_RecurrentToForgetWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_RecurrentToCellWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_InputToOutputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_RecurrentToOutputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_CifgParameters.m_RecurrentToInputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_CifgParameters.m_InputToInputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);

    RunShapeInferenceTest<LstmLayer>(layer, {{2, 80}, {2, 20}, {2, 20}, {2, 20}});
}

BOOST_AUTO_TEST_CASE(MeanLayerTest)
{
    MeanDescriptor descriptor;
    descriptor.m_Axis = {0};

    CreateGraphAndRunTest<MeanLayer>({{ 5, 7, 6, 2 }}, {{ 7, 6, 2 }}, descriptor, "mean");
}

BOOST_AUTO_TEST_CASE(MemCopyTest)
{
    CreateGraphAndRunTest<MemCopyLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "memcopy");
}

BOOST_AUTO_TEST_CASE(MemImportTest)
{
    CreateGraphAndRunTest<MemImportLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "memomport");
}

BOOST_AUTO_TEST_CASE(MergeTest)
{
    const TensorShape tensorShape{ 5, 7, 6, 2 };
    CreateGraphAndRunTest<MergeLayer>({ { 5, 7, 6, 2 }, { 5, 7, 6, 2 } }, {{ 5, 7, 6, 2 }}, "merge");
}

BOOST_AUTO_TEST_CASE(NormalizationTest)
{
    const TensorShape tensorShape{5, 7, 6, 2};

    CreateGraphAndRunTest<NormalizationLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, NormalizationDescriptor(), "l2norm");
}

BOOST_AUTO_TEST_CASE(PermuteTest)
{
    PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 2U, 3U, 1U};

    CreateGraphAndRunTest<PermuteLayer>({{ 1, 2, 2, 3 }}, {{ 1, 3, 2, 2 }}, descriptor, "permute");
}

BOOST_AUTO_TEST_CASE(Pooling2dTest)
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 4;
    descriptor.m_PadLeft = descriptor.m_PadRight = 3;
    descriptor.m_PadTop = descriptor.m_PadBottom = 0;
    descriptor.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    CreateGraphAndRunTest<Pooling2dLayer>({{ 1, 2, 8, 13 }}, {{ 1, 2, 2, 8 }}, descriptor, "pooling2d");
}

BOOST_AUTO_TEST_CASE(QLstmTest)
{
    const TensorShape inputShape{2, 5};
    const TensorShape inputCellState{2, 20};
    const TensorShape expectedOutputShape{2, 20};

    QLstmDescriptor descriptor;

    descriptor.m_CifgEnabled = false;
    descriptor.m_PeepholeEnabled = false;
    descriptor.m_ProjectionEnabled = false;

    Graph graph;
    auto layer = BuildGraph<QLstmLayer>(&graph, {inputShape, inputCellState, inputCellState}, descriptor, "qlstm");

    float Datum = 0.0f;
    ConstTensor constTensor({{ 2, 5, 3, 2 }, DataType::Float32}, &Datum);

    layer->m_BasicParameters.m_InputToCellWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_InputToForgetWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_CellBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_ForgetGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_CifgParameters.m_InputGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_OutputGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_RecurrentToForgetWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_RecurrentToCellWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_InputToOutputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_BasicParameters.m_RecurrentToOutputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_CifgParameters.m_RecurrentToInputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_CifgParameters.m_InputToInputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);

    RunShapeInferenceTest<QLstmLayer>(layer, {{2, 20}, {2, 20}, {2, 20}});
}

BOOST_AUTO_TEST_CASE(QuantizedLstmTest)
{
    const TensorShape inputShape{2, 5};
    const TensorShape inputCellState{2, 20};
    const TensorShape expectedOutputShape{2, 20};

    Graph graph;
    auto layer = BuildGraph<QuantizedLstmLayer>(&graph, {inputShape, inputCellState, inputCellState},  "quatizedlstm");

    float Datum = 0.0f;
    ConstTensor constTensor({{ 2, 5, 3, 2 }, DataType::Float32}, &Datum);

    layer->m_QuantizedLstmParameters.m_InputToCellWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_InputToForgetWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_CellBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_ForgetGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_InputGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_OutputGateBias = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_RecurrentToCellWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_InputToOutputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_RecurrentToInputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);
    layer->m_QuantizedLstmParameters.m_InputToInputWeights = std::make_unique<ScopedCpuTensorHandle>(constTensor);

    RunShapeInferenceTest<QuantizedLstmLayer>(layer, {{2, 20}, {2, 20}, {2, 20}});
}

BOOST_AUTO_TEST_CASE(QuantizeTest)
{
    const TensorShape tensorShape { 5, 4, 7, 6 };
    CreateGraphAndRunTest<QuantizeLayer>({{ 5, 7, 6, 2 }}, {{ 5, 7, 6, 2 }}, "mean");
}

BOOST_AUTO_TEST_CASE(RankTest)
{
   // due to rank having a scalar output we need a custom test
   const TensorShape expectedOutputs(Dimensionality::Scalar);

   Graph graph;
   auto layer = BuildGraph<RankLayer>(&graph, {{ 1, 1, 1, 1 }},  "rank");

   layer->GetOutputSlot(0).SetTensorInfo({TensorShape(Dimensionality::NotSpecified), DataType::Float32});

   BOOST_CHECK_THROW(
           layer->ValidateTensorShapesFromInputs(), LayerValidationException);

   layer->SetShapeInferenceMethod(ShapeInferenceMethod::InferAndValidate);

    layer->ValidateTensorShapesFromInputs();

   BOOST_CHECK(layer->GetOutputSlot(0).GetTensorInfo().GetShape() == expectedOutputs);

   layer->GetOutputSlot(0).SetTensorInfo({TensorShape(Dimensionality::Scalar), DataType::Float32});

    layer->ValidateTensorShapesFromInputs();

   BOOST_CHECK(layer->GetOutputSlot(0).GetTensorInfo().GetShape() == expectedOutputs);
}

BOOST_AUTO_TEST_CASE(ReshapeTest)
{
    ReshapeDescriptor descriptor;

    descriptor.m_TargetShape = { 1, 1, 1, 8 };

    CreateGraphAndRunTest<ReshapeLayer>({{ 2, 2, 2, 2 }}, {{ 1, 1, 1, 8 }}, descriptor, "reshape");
}

BOOST_AUTO_TEST_CASE(ResizeTest)
{
    ResizeDescriptor descriptor;

    descriptor.m_TargetHeight = 6;
    descriptor.m_TargetWidth = 2;

    CreateGraphAndRunTest<ResizeLayer>({{ 1, 7, 6, 2 }}, {{ 1, 7, 6, 2 }}, descriptor, "resize");
}

BOOST_AUTO_TEST_CASE(SliceTest)
{
    SliceDescriptor descriptor;
    descriptor.m_Begin  = { 1, 0, 1, 2 };
    descriptor.m_Size   = { 2, 1, 2, 3 };

    CreateGraphAndRunTest<SliceLayer>({{ 3, 2, 3, 5 }}, {{ 2, 1, 2, 3 }}, descriptor, "mean");
}

BOOST_AUTO_TEST_CASE(SpaceToBatchNdTest)
{
    SpaceToBatchNdDescriptor descriptor;

    std::vector<unsigned int> blockShape {2, 2};
    std::vector<std::pair<unsigned int, unsigned int>> padlist = {{0, 0}, {0, 0}};

    descriptor.m_BlockShape = blockShape;
    descriptor.m_PadList = padlist;
    descriptor.m_DataLayout = DataLayout::NHWC;

    CreateGraphAndRunTest<SpaceToBatchNdLayer>({{ 1, 4, 4, 1 }}, {{ 4, 2, 2, 1 }}, descriptor, "spacetobatchnd");
}

BOOST_AUTO_TEST_CASE(SpaceToDepth)
{
    SpaceToDepthDescriptor descriptor;

    descriptor.m_BlockSize = 2;
    descriptor.m_DataLayout = DataLayout::NHWC;

    CreateGraphAndRunTest<SpaceToDepthLayer>({{ 1, 2, 2, 2 }}, {{ 1, 1, 1, 8}}, descriptor, "spacetodepth");
}

BOOST_AUTO_TEST_CASE(SplitterTest)
{
    SplitterDescriptor descriptor(2, 3);

    descriptor.SetViewSize(0, 0, 1);
    descriptor.SetViewSize(0, 1, 2);
    descriptor.SetViewSize(0, 2, 2);

    descriptor.SetViewSize(1, 0, 1);
    descriptor.SetViewSize(1, 1, 2);
    descriptor.SetViewSize(1, 2, 2);

    CreateGraphAndRunTest<SplitterLayer>({{ 2, 2, 2 }}, {{ 1, 2, 2 }, { 1, 2, 2 }}, descriptor, "splitter");
}

BOOST_AUTO_TEST_CASE(StackTest)
{
    StackDescriptor descriptor;

    descriptor.m_Axis = 0;
    descriptor.m_NumInputs = 2;
    descriptor.m_InputShape = { 3, 2, 3 };

    CreateGraphAndRunTest<StackLayer>({{ 3, 2, 3 }, { 3, 2, 3 }}, {{ 2, 3, 2, 3 }}, descriptor, "stack");
}

BOOST_AUTO_TEST_CASE(StridedSliceTest)
{
    StridedSliceDescriptor descriptor;

    descriptor.m_Begin  = {0, 0, 0, 0};
    descriptor.m_End    = {3, 2, 3, 1};
    descriptor.m_Stride = {2, 2, 2, 1};

    CreateGraphAndRunTest<StridedSliceLayer>({{ 3, 2, 3, 1 }}, {{ 2, 1, 2, 1 }}, descriptor, "stridedslice");
}

BOOST_AUTO_TEST_CASE(Switchtest)
{
    CreateGraphAndRunTest<SwitchLayer>({{ 3, 2, 3, 1 }, { 3, 2, 3, 1 }}, {{ 3, 2, 3, 1 }, { 3, 2, 3, 1 }}, "switch");
}

BOOST_AUTO_TEST_CASE(TransposeConvolution2dTest)
{
    StridedSliceDescriptor descriptor;

    descriptor.m_Begin  = {0, 0, 0, 0};
    descriptor.m_End    = {3, 2, 3, 1};
    descriptor.m_Stride = {2, 2, 2, 1};

    CreateGraphAndRunTest<StridedSliceLayer>({{ 3, 2, 3, 1 }}, {{ 2, 1, 2, 1 }}, descriptor, "t");
}

BOOST_AUTO_TEST_CASE(TransposeTest)
{
    armnn::TransposeDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 3U, 1U, 2U};

    CreateGraphAndRunTest<TransposeLayer>({{ 1, 2, 2, 3 }}, {{ 1, 3, 2, 2 }}, descriptor, "stridedslice");
}

BOOST_AUTO_TEST_SUITE_END()
}