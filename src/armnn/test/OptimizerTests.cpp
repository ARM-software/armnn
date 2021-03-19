//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestUtils.hpp"

#include <BackendSettings.hpp>
#include <Graph.hpp>
#include <Network.hpp>
#include <Optimizer.hpp>

#include <armnn/BackendHelper.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/LayerVisitorBase.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnUtils/FloatingPointConverter.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/LayerSupportBase.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

namespace
{

void CreateLSTMLayerHelper(Graph &graph, bool CifgEnabled)
{
    LstmDescriptor layerDesc;
    layerDesc.m_ActivationFunc = 4;
    layerDesc.m_ClippingThresCell = 0.2f;
    layerDesc.m_ClippingThresProj = 0.4f;
    layerDesc.m_CifgEnabled = CifgEnabled;
    layerDesc.m_PeepholeEnabled = false;
    layerDesc.m_ProjectionEnabled = false;

    LstmLayer* const layer = graph.AddLayer<LstmLayer>(layerDesc, "layer");
    unsigned int batchSize = 3;
    unsigned int inputSize = 2;
    unsigned int numUnits = 4;
    unsigned int outputSize = 4;

    layer->m_BasicParameters.m_InputToForgetWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_InputToCellWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_InputToOutputWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToForgetWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToCellWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToOutputWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_ForgetGateBias = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));
    layer->m_BasicParameters.m_CellBias = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));
    layer->m_BasicParameters.m_OutputGateBias = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));

    layer->m_BasicParameters.m_InputToForgetWeights->Allocate();
    layer->m_BasicParameters.m_InputToCellWeights->Allocate();
    layer->m_BasicParameters.m_InputToOutputWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToForgetWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToCellWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToOutputWeights->Allocate();
    layer->m_BasicParameters.m_ForgetGateBias->Allocate();
    layer->m_BasicParameters.m_CellBias->Allocate();
    layer->m_BasicParameters.m_OutputGateBias->Allocate();

    if (!layerDesc.m_CifgEnabled)
    {
        layer->m_CifgParameters.m_InputToInputWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits, inputSize }, DataType::Float32));
        layer->m_CifgParameters.m_RecurrentToInputWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits, outputSize }, DataType::Float32));
        layer->m_CifgParameters.m_InputGateBias = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_CifgParameters.m_InputToInputWeights->Allocate();
        layer->m_CifgParameters.m_RecurrentToInputWeights->Allocate();
        layer->m_CifgParameters.m_InputGateBias->Allocate();
    }

    if (layerDesc.m_ProjectionEnabled)
    {
        layer->m_ProjectionParameters.m_ProjectionWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ outputSize, numUnits }, DataType::Float32));
        layer->m_ProjectionParameters.m_ProjectionBias = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ outputSize }, DataType::Float32));
        layer->m_ProjectionParameters.m_ProjectionWeights->Allocate();
        layer->m_ProjectionParameters.m_ProjectionBias->Allocate();
    }

    if (layerDesc.m_PeepholeEnabled)
    {
        if (!layerDesc.m_CifgEnabled)
        {
            layer->m_PeepholeParameters.m_CellToInputWeights = std::make_unique<ScopedCpuTensorHandle>
                    (TensorInfo({ numUnits }, DataType::Float32));
            layer->m_PeepholeParameters.m_CellToInputWeights->Allocate();
        }
        layer->m_PeepholeParameters.m_CellToForgetWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_PeepholeParameters.m_CellToOutputWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_PeepholeParameters.m_CellToForgetWeights->Allocate();
        layer->m_PeepholeParameters.m_CellToOutputWeights->Allocate();
    }

    // create input and output layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const outputStateIn = graph.AddLayer<InputLayer>(1, "outputStateIn");
    Layer* const cellStateIn = graph.AddLayer<InputLayer>(2, "cellStateIn");
    Layer* const scratchBuffer = graph.AddLayer<OutputLayer>(0, "scratchBuffer");
    Layer* const outputStateOut = graph.AddLayer<OutputLayer>(1, "outputStateOut");
    Layer* const cellStateOut = graph.AddLayer<OutputLayer>(2, "cellStateOut");
    Layer* const output = graph.AddLayer<OutputLayer>(3, "output");

    // connect up
    armnn::TensorInfo lstmTensorInfo1({ batchSize, inputSize }, DataType::Float32);
    armnn::TensorInfo lstmTensorInfo2({ batchSize, numUnits}, DataType::Float32);
    armnn::TensorInfo lstmTensorInfo3({ batchSize, outputSize }, DataType::Float32);
    armnn::TensorInfo lstmTensorInfoScratchBuff({ batchSize, numUnits * (layerDesc.m_CifgEnabled ? 3 : 4) },
                                                DataType::Float32);

    Connect(input, layer, lstmTensorInfo1, 0, 0);
    Connect(cellStateIn, layer, lstmTensorInfo2, 0, 1);
    Connect(outputStateIn, layer, lstmTensorInfo3, 0, 2);
    Connect(layer, scratchBuffer, lstmTensorInfoScratchBuff, 0, 0);
    Connect(layer, outputStateOut, lstmTensorInfo3, 1, 0);
    Connect(layer, cellStateOut, lstmTensorInfo2, 2, 0);
    Connect(layer, output, lstmTensorInfo3, 3, 0);
}

}    // namespace

BOOST_AUTO_TEST_SUITE(Optimizer)
using namespace armnn::optimizations;

BOOST_AUTO_TEST_CASE(LSTMValidateTensorShapesFromInputsCIFGDisabledTest)
{
    Graph graph;

    //Helper function creates graph containing LSTM layer with required input and output layers
    CreateLSTMLayerHelper(graph, false);

    //This function used to call ValidateShapesFromInputs();
    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(LSTMValidateTensorShapesFromInputsCIFGEnabledTest)
{
    Graph graph;

    //Helper function creates graph containing LSTM layer with required input and output layers
    CreateLSTMLayerHelper(graph, true);

    //This function used to call ValidateShapesFromInputs();
    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(InsertConvertersTest)
{
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float16);

    armnn::Graph graph;

    armnn::LayerBindingId inputId = 0;

    armnn::Layer* head = graph.AddLayer<armnn::OutputLayer>(0, "output");

    head = graph.InsertNewLayer<armnn::AdditionLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(1), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::FloorLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::MemCopyLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(0), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    // Check graph layer sequence before inserting convert layers
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::AdditionLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    // Check layers have Float16 DataType
    for (auto& layer : graph)
    {
        if(layer->GetType()==LayerType::Floor || layer->GetType() == LayerType::Addition)
        {
            ARMNN_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float16);
            ARMNN_ASSERT(layer->GetDataType() == DataType::Float16);
        }
    }

    // Insert convert layers either side of unsupported layer
    for (auto& layer : graph)
    {
        if(layer->GetType()==LayerType::Floor || layer->GetType() == LayerType::Addition)
        {
            InsertConvertFp16ToFp32LayersBefore(graph, *layer);
            InsertConvertFp32ToFp16LayersAfter(graph, *layer);
        }
    }

    // Check layers have correct DataType after inserting convert layers
    for (auto& layer : graph)
    {
        if (layer->GetType()==LayerType::Floor || layer->GetType() == LayerType::Addition)
        {
            ARMNN_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float32);
            ARMNN_ASSERT(layer->GetDataType() == DataType::Float32);
        }
        else if (layer->GetType() == LayerType::ConvertFp16ToFp32)
        {
            ARMNN_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float32);
            ARMNN_ASSERT(layer->GetDataType() == DataType::Float16);
        }
        else if (layer->GetType() == LayerType::ConvertFp32ToFp16)
        {
            ARMNN_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float16);
            ARMNN_ASSERT(layer->GetDataType() == DataType::Float32);
        }
    }

    // Check sequence of layers after inserting convert layers
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::AdditionLayer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

void CreateConvolution2dGraph(Graph &graph, const unsigned int* inputShape,
                              const unsigned int* weightsShape, const unsigned int* outputShape,
                              DataLayout dataLayout = DataLayout::NCHW)
{
    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    std::vector<float> weightsVector(90);
    armnn::ConstTensor weights(armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32), weightsVector);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX     = 1;
    desc.m_StrideY     = 1;
    desc.m_DataLayout  = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    Convolution2dLayer* layer = graph.AddLayer<Convolution2dLayer>(desc, "conv2d");
    layer->m_Weight           = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(Conv2dValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 3, 8, 16 };
    const unsigned int weightsShape[] = { 2, 3, 5, 3 };
    const unsigned int outputShape[] = { 1, 2, 4, 14 };
    CreateConvolution2dGraph(graph, inputShape, weightsShape, outputShape);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(Conv2dValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 8, 16, 3 };
    const unsigned int weightsShape[] = { 2, 5, 3, 3 };
    const unsigned int outputShape[] = { 1, 4, 14, 2 };
    CreateConvolution2dGraph(graph, inputShape, weightsShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void CreateDepthwiseConvolution2dGraph(Graph &graph, const unsigned int* inputShape,
                                       const unsigned int* weightsShape, const unsigned int* outputShape,
                                       DataLayout dataLayout = DataLayout::NCHW)
{
    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    std::vector<float> weightsVector(18);
    armnn::ConstTensor weights(armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32), weightsVector);

    DepthwiseConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX     = 1;
    desc.m_StrideY     = 1;
    desc.m_DataLayout  = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    DepthwiseConvolution2dLayer* layer = graph.AddLayer<DepthwiseConvolution2dLayer>(desc, "depthwiseConv2d");
    layer->m_Weight                    = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(DepthwiseConv2dValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 2, 3, 3 };
    const unsigned int weightsShape[] = { 1, 2, 3, 3 };
    const unsigned int outputShape[] = { 1, 2, 1, 1 };
    CreateDepthwiseConvolution2dGraph(graph, inputShape, weightsShape, outputShape);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(DepthwiseConv2dValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 3, 3, 2 };
    const unsigned int weightsShape[] = { 1, 2, 3, 3 };
    const unsigned int outputShape[] = { 1, 1, 1, 2 };
    CreateDepthwiseConvolution2dGraph(graph, inputShape, weightsShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void CreatePooling2dGraph(Graph& graph, const unsigned int* inputShape,  const unsigned int* outputShape,
                          DataLayout dataLayout = DataLayout::NCHW)
{
    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Pooling2dDescriptor desc;
    desc.m_PoolType  = armnn::PoolingAlgorithm::Average;
    desc.m_PoolWidth = desc.m_PoolHeight = 100;
    desc.m_StrideX = desc.m_StrideY = 5;
    desc.m_PadLeft                  = 50;
    desc.m_PadRight                 = 50;
    desc.m_PadTop                   = 50;
    desc.m_PadBottom                = 50;
    desc.m_PaddingMethod            = armnn::PaddingMethod::Exclude;
    desc.m_DataLayout               = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    Pooling2dLayer* layer = graph.AddLayer<Pooling2dLayer>(desc, "pooling2d");
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(Pooling2dValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[]  = { 5, 3, 52, 60 };
    const unsigned int outputShape[] = { 5, 3, 11, 13 };
    CreatePooling2dGraph(graph, inputShape, outputShape, DataLayout::NCHW);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(Pooling2dValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[]  = { 5, 52, 60, 3 };
    const unsigned int outputShape[] = { 5, 11, 13, 3 };
    CreatePooling2dGraph(graph, inputShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void CreateResizeBilinearGraph(Graph& graph,
                               const unsigned int* inputShape,
                               const unsigned int* outputShape,
                               DataLayout dataLayout = DataLayout::NCHW)
{
    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    ResizeDescriptor desc;
    desc.m_Method       = ResizeMethod::Bilinear;
    desc.m_TargetHeight = 3;
    desc.m_TargetWidth  = 4;
    desc.m_DataLayout   = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    ResizeLayer* layer = graph.AddLayer<ResizeLayer>(desc, "resizeBilinear");
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(ResizeBilinearValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[]  = { 1, 2, 4, 5 };
    const unsigned int outputShape[] = { 1, 2, 3, 4 };
    CreateResizeBilinearGraph(graph, inputShape, outputShape);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(ResizeBilinearValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[]  = { 1, 4, 5, 2 };
    const unsigned int outputShape[] = { 1, 3, 4, 2 };
    CreateResizeBilinearGraph(graph, inputShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void CreateGatherGraph(Graph& graph,
                       const armnn::TensorInfo& paramsInfo,
                       const armnn::TensorInfo& indicesInfo,
                       const armnn::TensorInfo& outputInfo)
{
    Layer* input0 = graph.AddLayer<InputLayer>(0, "params");
    input0->GetOutputSlot().SetTensorInfo(paramsInfo);

    Layer* input1 = graph.AddLayer<InputLayer>(1, "indices");
    input1->GetOutputSlot().SetTensorInfo(indicesInfo);

    GatherDescriptor descriptor;
    GatherLayer* layer = graph.AddLayer<GatherLayer>(descriptor, "gather");
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input0->GetOutputSlot().Connect(layer->GetInputSlot(0));
    input1->GetOutputSlot().Connect(layer->GetInputSlot(1));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(GatherValidateTensorShapesFromInputs)
{
    Graph graph;
    armnn::TensorInfo paramsInfo({10, 5}, DataType::Float32);
    armnn::TensorInfo indicesInfo({3}, DataType::Signed32);
    armnn::TensorInfo outputInfo({3, 5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(GatherValidateTensorShapesFromInputs1DParams)
{
    Graph graph;
    armnn::TensorInfo paramsInfo({8}, DataType::Float32);
    armnn::TensorInfo indicesInfo({5}, DataType::Signed32);
    armnn::TensorInfo outputInfo( {5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(GatherValidateTensorShapesFromInputsMultiDimIndices)
{
    Graph graph;
    armnn::TensorInfo paramsInfo({3, 2, 5}, DataType::Float32);
    armnn::TensorInfo indicesInfo({2, 2}, DataType::Signed32);
    armnn::TensorInfo outputInfo({2, 2, 2, 5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(DetectionPostProcessValidateTensorShapes)
{
    Graph graph;
    armnn::TensorInfo boxEncodingsInfo({1, 10, 4}, DataType::QAsymmU8);
    armnn::TensorInfo scoresInfo({1, 10, 4}, DataType::QAsymmU8);
    std::vector<uint8_t> anchorsVector(40);
    armnn::ConstTensor anchors(armnn::TensorInfo({10, 4}, armnn::DataType::QAsymmU8), anchorsVector);

    armnn::TensorInfo detectionBoxesInfo({1, 3, 4}, DataType::QAsymmU8);
    armnn::TensorInfo detectionScoresInfo({1, 3}, DataType::QAsymmU8);
    armnn::TensorInfo detectionClassesInfo({1, 3}, DataType::QAsymmU8);
    armnn::TensorInfo numDetectionInfo({1}, DataType::QAsymmU8);

    Layer* input0 = graph.AddLayer<InputLayer>(0, "boxEncodings");
    input0->GetOutputSlot().SetTensorInfo(boxEncodingsInfo);

    Layer* input1 = graph.AddLayer<InputLayer>(1, "score");
    input1->GetOutputSlot().SetTensorInfo(scoresInfo);

    DetectionPostProcessDescriptor descriptor;
    descriptor.m_MaxDetections = 3;

    DetectionPostProcessLayer* layer = graph.AddLayer<DetectionPostProcessLayer>(descriptor, "detectionPostProcess");
    layer->m_Anchors = std::make_unique<armnn::ScopedCpuTensorHandle>(anchors);
    layer->GetOutputSlot(0).SetTensorInfo(detectionBoxesInfo);
    layer->GetOutputSlot(1).SetTensorInfo(detectionScoresInfo);
    layer->GetOutputSlot(2).SetTensorInfo(detectionClassesInfo);
    layer->GetOutputSlot(3).SetTensorInfo(numDetectionInfo);

    input0->GetOutputSlot().Connect(layer->GetInputSlot(0));
    input1->GetOutputSlot().Connect(layer->GetInputSlot(1));

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(FoldPadLayerIntoConvolution2dLayer)
{
    Graph graph;
    const unsigned int inputShape[]   = { 1, 2, 2, 3 };
    const unsigned int paddedShape[]  = { 1, 6, 6, 3 };
    const unsigned int weightsShape[] = { 1, 2, 3, 3 };
    const unsigned int outputShape[]  = { 1, 2, 1, 1 };

    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({ { 0, 0 }, { 2, 2 }, { 2, 2 }, { 0, 0 } });

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_StrideX     = 1;
    convolution2dDescriptor.m_StrideY     = 1;
    convolution2dDescriptor.m_DataLayout  = DataLayout::NHWC;

    std::vector<float> weightsVector(18);
    armnn::ConstTensor weights(armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32), weightsVector);

    Convolution2dLayer* conv2dLayer = graph.AddLayer<Convolution2dLayer>(convolution2dDescriptor, "conv2d");
    conv2dLayer->m_Weight           = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    conv2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> conv2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(conv2dLayer->GetInputSlot(0));
    conv2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimpleConv2d = [](const armnn::Layer* const layer) -> bool {
        const auto conv2dLayer       = static_cast<const armnn::Convolution2dLayer*>(layer);
        const auto conv2dLayerParams = conv2dLayer->GetParameters();
        return IsLayerOfType<armnn::Convolution2dLayer>(layer) && (layer->GetNameStr() == "conv2d") &&
               (conv2dLayerParams.m_PadLeft == 0) && (conv2dLayerParams.m_PadRight == 0) &&
               (conv2dLayerParams.m_PadTop == 0) && (conv2dLayerParams.m_PadBottom == 0) &&
               (conv2dLayerParams.m_BiasEnabled == false) && (conv2dLayerParams.m_StrideX == 1) &&
               (conv2dLayerParams.m_StrideY == 1) && (conv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimpleConv2d,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FoldPadIntoConvolution2d()));

    auto checkPadFoldedIntoConv2d = [](const armnn::Layer* const layer) -> bool {
        const auto conv2dLayer       = static_cast<const armnn::Convolution2dLayer*>(layer);
        const auto conv2dLayerParams = conv2dLayer->GetParameters();
        return IsLayerOfType<armnn::Convolution2dLayer>(layer) && (layer->GetNameStr() == "folded-pad-into-conv2d") &&
               (conv2dLayerParams.m_PadLeft == 2) && (conv2dLayerParams.m_PadRight == 2) &&
               (conv2dLayerParams.m_PadTop == 2) && (conv2dLayerParams.m_PadBottom == 2) &&
               (conv2dLayerParams.m_BiasEnabled == false) && (conv2dLayerParams.m_StrideX == 1) &&
               (conv2dLayerParams.m_StrideY == 1) && (conv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             checkPadFoldedIntoConv2d,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(FoldPadLayerIntoPooling2dLayer)
{
    Graph graph;
    const unsigned int inputShape[]  = { 1, 2, 2, 3 };
    const unsigned int paddedShape[] = { 1, 4, 4, 3 };
    const unsigned int outputShape[] = { 1, 2, 2, 3 };

    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({ { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimplePool2d = [&](const armnn::Layer* const layer) {
        const auto pool2dLayer = static_cast<const armnn::Pooling2dLayer*>(layer);
        return IsLayerOfType<armnn::Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
               (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FoldPadIntoPooling2d()));

    auto checkPadFoldedIntoPool2d = [&](const armnn::Layer* const layer) {
        if (!IsLayerOfType<armnn::Pooling2dLayer>(layer) || (layer->GetNameStr() != "folded-pad-into-pool2d"))
        {
            return false;
        }

        const auto pool2dLayer                      = static_cast<const armnn::Pooling2dLayer*>(layer);
        const Pooling2dDescriptor pool2dLayerParams = pool2dLayer->GetParameters();

        Pooling2dDescriptor pool2dLayerParamsNoPad = pool2dLayerParams;
        pool2dLayerParamsNoPad.m_PadLeft           = 0;
        pool2dLayerParamsNoPad.m_PadRight          = 0;
        pool2dLayerParamsNoPad.m_PadTop            = 0;
        pool2dLayerParamsNoPad.m_PadBottom         = 0;
        // If we fold then PaddingMethod will be set to Ignore. The original will be Exclude.
        pool2dLayerParamsNoPad.m_PaddingMethod = PaddingMethod::Exclude;

        return (pool2dLayerParamsNoPad == pooling2dDescriptor) && (pool2dLayerParams.m_PadLeft == 1) &&
               (pool2dLayerParams.m_PadRight == 1) && (pool2dLayerParams.m_PadTop == 1) &&
               (pool2dLayerParams.m_PadBottom == 1) &&
               (pool2dLayerParams.m_PaddingMethod == PaddingMethod::IgnoreValue);
    };

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             checkPadFoldedIntoPool2d,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(FoldPadLayerIntoPooling2d_PadWithMultipleOutputsShouldNotBeOptimized)
{
    // In this test case we'll setup a pad layer with two outputs. One goes to a polling layers and the other
    // goes to an output layer. FoldPadLayerIntoPooling2d should not optimize this graph as it uses the
    // OptimizeForExclusiveConnection method.
    Graph graph;
    const unsigned int inputShape[]  = { 1, 2, 2, 3 };
    const unsigned int paddedShape[] = { 1, 4, 4, 3 };
    const unsigned int outputShape[] = { 1, 2, 2, 3 };

    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({ { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    // Add the alternative branch from the pas layer to an output layer.
    Layer* secondOutput = graph.AddLayer<OutputLayer>(1, "dummy output");
    padLayer->GetOutputSlot().Connect(secondOutput->GetInputSlot(0));

    auto checkSimplePool2d = [&](const armnn::Layer* const layer) {
        const auto pool2dLayer = static_cast<const armnn::Pooling2dLayer*>(layer);
        return IsLayerOfType<armnn::Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
               (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    // Initial sequence.
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FoldPadIntoPooling2d()));

    // The network should not change.
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(FoldPadLayerIntoPooling2dLayer_PoolingLayerWithExcludePaddingShouldNotTakeMorePadding)
{
    // In this test setup input, Pad layer, Pooling layer that includes padding, output layer. The optimization
    // should not work as the pooling layer already includes and existing pad and specifies PaddingMethod::Exclude.
    Graph graph;
    const unsigned int inputShape[]  = { 1, 2, 2, 3 };
    const unsigned int paddedShape[] = { 1, 4, 4, 3 };
    const unsigned int outputShape[] = { 1, 2, 2, 3 };

    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({ { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;
    // Include a pad with the pooling layer. This should prevent the optimization working.
    pooling2dDescriptor.m_PadLeft   = 1;
    pooling2dDescriptor.m_PadRight  = 1;
    pooling2dDescriptor.m_PadTop    = 1;
    pooling2dDescriptor.m_PadBottom = 1;
    pooling2dDescriptor.m_PaddingMethod = PaddingMethod::Exclude;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimplePool2d = [&](const armnn::Layer* const layer) {
        const auto pool2dLayer = static_cast<const armnn::Pooling2dLayer*>(layer);
        return IsLayerOfType<armnn::Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
               (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FoldPadIntoPooling2d()));

    // The optimization should not have modified the graph.
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(FoldPadLayerIntoPooling2dLayer_MaxPoolingLayerWithLargePadValueShouldNotBeFolded)
{
    // In this test setup input, Pad layer with a large pad value, Max Pooling layer, output layer. The optimization
    // should not work as the pad value will modify the result of the max pooling layer.
    Graph graph;
    const unsigned int inputShape[]  = { 1, 2, 2, 3 };
    const unsigned int paddedShape[] = { 1, 4, 4, 3 };
    const unsigned int outputShape[] = { 1, 2, 2, 3 };

    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({ { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });
    // For Max pooling of a float a pad value of 0 is more than enough to stop the fold happening.
    // Set this to -std::numeric_limits<float>::infinity() to make the fold happen.
    padDescriptor.m_PadValue = 0;

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Max;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimplePool2d = [&](const armnn::Layer* const layer) {
            const auto pool2dLayer = static_cast<const armnn::Pooling2dLayer*>(layer);
            return IsLayerOfType<armnn::Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
                   (pool2dLayer->GetParameters() == pooling2dDescriptor);
        };

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FoldPadIntoPooling2d()));

    // The optimization should not have modified the graph.
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<armnn::OutputLayer>));
}

#if defined(ARMNNREF_ENABLED)
BOOST_AUTO_TEST_CASE(FoldPadLayerIntoPooling2dLayer_ExecuteInferenceWithAndWithoutOptimization)
{
    // The idea of this test to run a simple pad+pool2d network twice. Once
    // with FoldPadLayerIntoPooling2dLayer enabled and a second time with it
    // avoided. The output tensors of each should match.
    const unsigned int inputShape[]  = { 1, 4, 4, 2 };
    const unsigned int paddedShape[] = { 1, 6, 6, 2 };
    const unsigned int outputShape[] = { 1, 4, 4, 2 };
    std::vector<float> inputData({
        2.0f,  2.0f,  6.0f,  6.0f,  4.0f,  4.0f,  8.0f,  8.0f,  10.0f, 12.0f, 14.0f, 16.0f, 10.0f, 12.0f, 16.0f, 14.0f,

        18.0f, 20.0f, 24.0f, 22.0f, 20.0f, 18.0f, 22.0f, 24.0f, 26.0f, 28.0f, 0.0f,  0.0f,  26.0f, 28.0f, 0.0f,  0.0f,
    });
    try
    {
        // Create a network of input, pad, pooling 2D, output.
        INetworkPtr network = INetwork::Create();

        IConnectableLayer* inputLayer = network->AddInputLayer(0);
        armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        PadDescriptor padDescriptor({ { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });
        IConnectableLayer* padLayer = network->AddPadLayer(padDescriptor, "Pad");
        armnn::TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
        padLayer->GetOutputSlot(0).SetTensorInfo(paddedInfo);

        Pooling2dDescriptor pooling2dDescriptor;
        pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
        pooling2dDescriptor.m_PoolWidth  = 3;
        pooling2dDescriptor.m_PoolHeight = 3;
        pooling2dDescriptor.m_StrideX    = 1;
        pooling2dDescriptor.m_StrideY    = 1;
        pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;
        IConnectableLayer* pool2dLayer   = network->AddPooling2dLayer(pooling2dDescriptor, "Pool2D");
        armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);
        pool2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        IConnectableLayer* outputLayer = network->AddOutputLayer(0);

        // Connect layers
        inputLayer->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
        padLayer->GetOutputSlot(0).Connect(pool2dLayer->GetInputSlot(0));
        pool2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Create ArmNN runtime
        IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions());    // default options
        // Optimise the network
        IOptimizedNetworkPtr optimizedNetwork = Optimize(*network, { Compute::CpuRef }, run->GetDeviceSpec());
        // Load network into runtime
        NetworkId networkIdentifier;
        BOOST_TEST(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);

        InputTensors inputTensors{ { 0,
                                     ConstTensor(run->GetInputTensorInfo(networkIdentifier, 0), inputData.data()) } };

        // Set the initial values of the data to different values to the golden data just in case the inference fails.
        std::vector<float> optimizedData(32, -std::numeric_limits<float>::infinity());
        armnn::OutputTensors outputTensors{ { 0, armnn::Tensor(outputInfo, optimizedData.data()) } };
        // Execute network
        run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        // Unload it.
        run->UnloadNetwork(networkIdentifier);

        // In this second case the pad will have two outputs, one connected to the pooling layer the second connected to
        // a second output layer. This will prevent the FoldPadLayerIntoPooling2dLayer optimization from working.
        // A previous test, FoldPadLayerIntoPooling2d_PadWithMultipleOutputsShouldNotBeOptimized, has proved that doing
        // this will avoid the optimization.
        IConnectableLayer* dummyOutputLayer = network->AddOutputLayer(1);
        padLayer->GetOutputSlot(0).Connect(dummyOutputLayer->GetInputSlot(0));

        // Optimize and load and execute it a second time.
        optimizedNetwork = Optimize(*network, { Compute::CpuRef }, run->GetDeviceSpec());
        BOOST_TEST(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);
        std::vector<float> goldenData(32, 0.0f);
        std::vector<float> padOutputData(72, 0.0f);
        armnn::OutputTensors goldenTensors{ { 0, armnn::Tensor(outputInfo, goldenData.data()) },
                                            { 1, armnn::Tensor(paddedInfo, padOutputData.data()) } };
        run->EnqueueWorkload(networkIdentifier, inputTensors, goldenTensors);

        // Now we can compare goldenData against optimizedData. They should be the same.
        BOOST_TEST(std::equal(goldenData.begin(), goldenData.end(), optimizedData.begin()));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        ARMNN_ASSERT_MSG(false, e.what());
    }
}
#endif

class MockLayerSupport : public LayerSupportBase
{
public:
    bool IsInputSupported(const TensorInfo& /*input*/,
                          Optional<std::string&> /*reasonIfUnsupported = EmptyOptional()*/) const override
    {
        return true;
    }

    bool IsOutputSupported(const TensorInfo& /*input*/,
                           Optional<std::string&> /*reasonIfUnsupported = EmptyOptional()*/) const override
    {
        return true;
    }

    bool IsActivationSupported(const TensorInfo& /*input0*/,
                               const TensorInfo& /*output*/,
                               const ActivationDescriptor& /*descriptor*/,
                               Optional<std::string&> /*reasonIfUnsupported = EmptyOptional()*/) const override
    {
        return true;
    }
};

template <typename NamePolicy>
class MockBackend : public IBackendInternal
{
public:
    MockBackend()  = default;
    ~MockBackend() = default;

    static const BackendId& GetIdStatic()
    {
        return NamePolicy::GetIdStatic();
    }
    const BackendId& GetId() const override
    {
        return GetIdStatic();
    }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override
    {
        return nullptr;
    };

    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr&) const override
    {
        return nullptr;
    }

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override
    {
        return nullptr;
    }

    IBackendInternal::Optimizations GetOptimizations() const override
    {
        return {};
    }
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return std::make_shared<MockLayerSupport>();
    }

    OptimizationViews OptimizeSubgraphView(const SubgraphView&) const override
    {
        return {};
    };
};

BOOST_AUTO_TEST_CASE(BackendCapabilityTest)
{
    BackendId backendId = "MockBackend";
    // MockBackend does not support the NonConstWeights capability
    BOOST_CHECK(!armnn::IsCapabilitySupported(backendId, armnn::BackendCapability::NonConstWeights));

    // MockBackend does not support the AsyncExecution capability
    BOOST_CHECK(!armnn::IsCapabilitySupported(backendId, armnn::BackendCapability::AsyncExecution));
}

BOOST_AUTO_TEST_CASE(BackendHintTest)
{
    class TestBackendAssignment : public LayerVisitorBase<VisitorNoThrowPolicy>
    {
    public:
        void VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name = nullptr) override
        {
            IgnoreUnused(id, name);
            auto inputLayer = PolymorphicDowncast<const InputLayer*>(layer);
            BOOST_TEST((inputLayer->GetBackendId() == "MockBackend"));
        }

        void VisitOutputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name = nullptr) override
        {
            IgnoreUnused(id, name);
            auto outputLayer = PolymorphicDowncast<const OutputLayer*>(layer);
            BOOST_TEST((outputLayer->GetBackendId() == "MockBackend"));
        }

        void VisitActivationLayer(const IConnectableLayer* layer,
                                  const ActivationDescriptor& activationDescriptor,
                                  const char* name = nullptr) override
        {
            IgnoreUnused(activationDescriptor, name);
            auto activation = PolymorphicDowncast<const ActivationLayer*>(layer);
            BOOST_TEST((activation->GetBackendId() == "CustomBackend"));
        }
    };

    struct CustomPolicy
    {
        static const BackendId& GetIdStatic()
        {
            static BackendId id = "CustomBackend";
            return id;
        }
    };

    struct MockPolicy
    {
        static const BackendId& GetIdStatic()
        {
            static BackendId id = "MockBackend";
            return id;
        }
    };

    auto& backendRegistry = BackendRegistryInstance();

    backendRegistry.Register("MockBackend", []() { return std::make_unique<MockBackend<MockPolicy>>(); });

    backendRegistry.Register("CustomBackend", []() { return std::make_unique<MockBackend<CustomPolicy>>(); });

    // Define the network
    auto network = INetwork::Create();
    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Linear;

    std::unique_ptr<Graph> graph = std::make_unique<Graph>();
    auto input                   = graph->AddLayer<InputLayer>(0, "input");
    auto act                     = graph->AddLayer<ActivationLayer>(desc, "activation");
    auto output                  = graph->AddLayer<OutputLayer>(0, "output");

    BackendId customBackendId("CustomBackend");
    act->BackendSelectionHint(customBackendId);

    input->GetOutputSlot(0).Connect(act->GetInputSlot(0));
    act->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    OptimizedNetworkImpl optNet(std::move(graph));

    // Get the optimized graph
    Graph& optGraph = optNet.GetGraph();

    std::vector<BackendId> prefs{ "MockBackend", "CustomBackend" };

    BackendIdSet availableBackends = { "CustomBackend", "MockBackend" };
    DeviceSpec spec(availableBackends);

    BackendSettings backendSettings(prefs, spec);

    // Assign an available backend to each layer
    Graph::Iterator firstLayer = optGraph.begin();
    Graph::Iterator lastLayer  = optGraph.end();

    OptimizedNetworkImpl* optNetObjPtr = &optNet;
    OptimizationResult res = AssignBackends(optNetObjPtr,
                                            backendSettings,
                                            firstLayer,
                                            lastLayer,
                                            EmptyOptional());

    BOOST_TEST(res.IsOk());

    TestBackendAssignment visitor;
    for (auto it = firstLayer; it != lastLayer; ++it)
    {
        (*it)->Accept(visitor);
    }
}

// Tests that OptimizeForExclusiveConnections works, fusing when needed, using BatchNorm fusing as example
BOOST_AUTO_TEST_CASE(OptimizeForExclusiveConnectionsFuseTest)
{
    using namespace armnn;
    // Define layers information
    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_DataLayout  = DataLayout::NHWC;
    BatchNormalizationDescriptor batchNormDescriptor;
    batchNormDescriptor.m_DataLayout = DataLayout::NHWC;

    const unsigned int inputDimensionSizes[]   = { 1, 4, 4, 3 };                 // NHWCin
    const unsigned int weightsDimensionSizes[] = { 1, 2, 2, 3 };                 // CoutHWCin
    const unsigned int outputDimensionSizes[]  = { 1, 3, 3, 1 };                 // NHWCout
    const unsigned int outputChannelSize[]     = { outputDimensionSizes[3] };    // Cout

    TensorInfo inputInfo(4, inputDimensionSizes, DataType::Float32);
    TensorInfo outputInfo(4, outputDimensionSizes, DataType::Float32);

    std::vector<float> weightsVector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    ConstTensor weights(TensorInfo(4, weightsDimensionSizes, DataType::Float32), weightsVector);

    std::vector<float> betaVector     = { 0.1f };
    std::vector<float> gammaVector    = { 0.5f };
    std::vector<float> meanVector     = { 0 };
    std::vector<float> varianceVector = { 1 };
    ConstTensor beta(TensorInfo(1, outputChannelSize, DataType::Float32), betaVector);
    ConstTensor gamma(TensorInfo(1, outputChannelSize, DataType::Float32), gammaVector);
    ConstTensor mean(TensorInfo(1, outputChannelSize, DataType::Float32), meanVector);
    ConstTensor variance(TensorInfo(1, outputChannelSize, DataType::Float32), varianceVector);

    // Define the network
    Graph graph;
    auto input     = graph.AddLayer<InputLayer>(0, "input");
    auto conv      = graph.AddLayer<Convolution2dLayer>(convolution2dDescriptor, "convolution");
    auto batchNorm = graph.AddLayer<BatchNormalizationLayer>(batchNormDescriptor, "batchNorm");
    auto output    = graph.AddLayer<OutputLayer>(0, "output");

    // Set layer information
    input->GetOutputSlot().SetTensorInfo(inputInfo);
    conv->GetOutputSlot().SetTensorInfo(outputInfo);
    batchNorm->GetOutputSlot().SetTensorInfo(outputInfo);
    conv->m_Weight        = std::make_unique<ScopedCpuTensorHandle>(weights);
    batchNorm->m_Beta     = std::make_unique<ScopedCpuTensorHandle>(beta);
    batchNorm->m_Gamma    = std::make_unique<ScopedCpuTensorHandle>(gamma);
    batchNorm->m_Mean     = std::make_unique<ScopedCpuTensorHandle>(mean);
    batchNorm->m_Variance = std::make_unique<ScopedCpuTensorHandle>(variance);
    if (convolution2dDescriptor.m_BiasEnabled)
    {
        std::vector<float> biasVector = { 11 };
        ConstTensor bias(TensorInfo(1, outputChannelSize, DataType::Float32), biasVector);
        conv->m_Bias = std::make_unique<ScopedCpuTensorHandle>(bias);
    }

    // Connect layers
    input->GetOutputSlot(0).Connect(conv->GetInputSlot(0));
    conv->GetOutputSlot(0).Connect(batchNorm->GetInputSlot(0));
    batchNorm->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    BOOST_CHECK(4 == graph.GetNumLayers());
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<Convolution2dLayer>,
                             &IsLayerOfType<BatchNormalizationLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Optimize graph
    armnn::Optimizer::Pass(graph, MakeOptimizations(FuseBatchNormIntoConvolution2DFloat32()));

    auto checkFusedConv2d = [](const armnn::Layer* const layer) -> bool {
        return IsLayerOfType<armnn::Convolution2dLayer>(layer) &&
               (layer->GetNameStr() == "fused-batchNorm-into-convolution");
    };

    BOOST_CHECK(3 == graph.GetNumLayers());
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             checkFusedConv2d,
                             &IsLayerOfType<OutputLayer>));
}

// Tests that OptimizeForExclusiveConnections works, not fusing when not needed, using BatchNorm fusing as example
BOOST_AUTO_TEST_CASE(OptimizeForExclusiveConnectionsWithoutFuseTest)
{
    // Define the network
    Graph graph;
    Convolution2dDescriptor convolution2dDescriptor;
    BatchNormalizationDescriptor batchNormDescriptor;

    auto input     = graph.AddLayer<InputLayer>(0, "input");
    auto conv      = graph.AddLayer<Convolution2dLayer>(convolution2dDescriptor, "convolution");
    auto batchNorm = graph.AddLayer<BatchNormalizationLayer>(batchNormDescriptor, "batchNorm");
    auto output    = graph.AddLayer<OutputLayer>(0, "output");
    auto output2   = graph.AddLayer<OutputLayer>(1, "output2");

    // Connect layers
    input->GetOutputSlot(0).Connect(conv->GetInputSlot(0));
    conv->GetOutputSlot(0).Connect(batchNorm->GetInputSlot(0));
    batchNorm->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    conv->GetOutputSlot(0).Connect(output2->GetInputSlot(0));

    BOOST_CHECK(5 == graph.GetNumLayers());
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::Convolution2dLayer>,
                             &IsLayerOfType<armnn::BatchNormalizationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
    // Optimize graph
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FuseBatchNormIntoConvolution2DFloat32()));

    BOOST_CHECK(5 == graph.GetNumLayers());
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::Convolution2dLayer>,
                             &IsLayerOfType<armnn::BatchNormalizationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}
BOOST_AUTO_TEST_SUITE_END()
