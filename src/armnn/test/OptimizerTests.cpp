//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <BackendSettings.hpp>
#include <Graph.hpp>
#include <Network.hpp>
#include <Optimizer.hpp>

#include <armnn/BackendHelper.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/StrategyBase.hpp>

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/IBackendInternal.hpp>

#include <backendsCommon/LayerSupportBase.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <doctest/doctest.h>

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

    layer->m_BasicParameters.m_InputToForgetWeights = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_InputToCellWeights = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_InputToOutputWeights = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToForgetWeights = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToCellWeights = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToOutputWeights = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_ForgetGateBias = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));
    layer->m_BasicParameters.m_CellBias = std::make_unique<ScopedTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));
    layer->m_BasicParameters.m_OutputGateBias = std::make_unique<ScopedTensorHandle>
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
        layer->m_CifgParameters.m_InputToInputWeights = std::make_unique<ScopedTensorHandle>
                (TensorInfo({ numUnits, inputSize }, DataType::Float32));
        layer->m_CifgParameters.m_RecurrentToInputWeights = std::make_unique<ScopedTensorHandle>
                (TensorInfo({ numUnits, outputSize }, DataType::Float32));
        layer->m_CifgParameters.m_InputGateBias = std::make_unique<ScopedTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_CifgParameters.m_InputToInputWeights->Allocate();
        layer->m_CifgParameters.m_RecurrentToInputWeights->Allocate();
        layer->m_CifgParameters.m_InputGateBias->Allocate();
    }

    if (layerDesc.m_ProjectionEnabled)
    {
        layer->m_ProjectionParameters.m_ProjectionWeights = std::make_unique<ScopedTensorHandle>
                (TensorInfo({ outputSize, numUnits }, DataType::Float32));
        layer->m_ProjectionParameters.m_ProjectionBias = std::make_unique<ScopedTensorHandle>
                (TensorInfo({ outputSize }, DataType::Float32));
        layer->m_ProjectionParameters.m_ProjectionWeights->Allocate();
        layer->m_ProjectionParameters.m_ProjectionBias->Allocate();
    }

    if (layerDesc.m_PeepholeEnabled)
    {
        if (!layerDesc.m_CifgEnabled)
        {
            layer->m_PeepholeParameters.m_CellToInputWeights = std::make_unique<ScopedTensorHandle>
                    (TensorInfo({ numUnits }, DataType::Float32));
            layer->m_PeepholeParameters.m_CellToInputWeights->Allocate();
        }
        layer->m_PeepholeParameters.m_CellToForgetWeights = std::make_unique<ScopedTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_PeepholeParameters.m_CellToOutputWeights = std::make_unique<ScopedTensorHandle>
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


class MockLayerSupport : public LayerSupportBase
{
public:
    bool IsLayerSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& descriptor,
                          const Optional<LstmInputParamsInfo>& /*lstmParamsInfo*/,
                          const Optional<QuantizedLstmInputParamsInfo>& /*quantizedLstmParamsInfo*/,
                          Optional<std::string&> reasonIfUnsupported) const override
    {
        switch (type)
        {
            case LayerType::Input:
                return IsInputSupported(infos[0], reasonIfUnsupported);
            case LayerType::Output:
                return IsOutputSupported(infos[0], reasonIfUnsupported);
            case LayerType::Activation:
                return IsActivationSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const ActivationDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
            default:
                return false;
        }
    }

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
class CustomAllocatorBackend : public IBackendInternal
{
public:
    CustomAllocatorBackend() :
            m_BackendCapabilities(NamePolicy::GetIdStatic(), {{"NullCapability", false}}),
            m_CustomAllocator(false) {};
    CustomAllocatorBackend(const BackendCapabilities& capabilities) :
            m_BackendCapabilities(capabilities),
            m_CustomAllocator(false) {};
    ~CustomAllocatorBackend() = default;

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

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return std::make_shared<MockLayerSupport>();
    }

    OptimizationViews OptimizeSubgraphView(const SubgraphView&) const override
    {
        return {};
    };

    BackendCapabilities GetCapabilities() const override
    {
        return m_BackendCapabilities;
    };

    virtual bool UseCustomMemoryAllocator(std::shared_ptr<ICustomAllocator> allocator,
                                          armnn::Optional<std::string&> errMsg) override
    {
        IgnoreUnused(errMsg, allocator);
        m_CustomAllocator = true;
        return m_CustomAllocator;
    }

    BackendCapabilities m_BackendCapabilities;
    bool m_CustomAllocator;
};

template <typename NamePolicy>
class NoProtectedModeMockBackend : public IBackendInternal
{
public:
    NoProtectedModeMockBackend() : m_BackendCapabilities(NamePolicy::GetIdStatic(), {{"NullCapability", false}}) {};
    NoProtectedModeMockBackend(const BackendCapabilities& capabilities) : m_BackendCapabilities(capabilities) {};
    ~NoProtectedModeMockBackend() = default;

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

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return std::make_shared<MockLayerSupport>();
    }

    OptimizationViews OptimizeSubgraphView(const SubgraphView&) const override
    {
        return {};
    };

    BackendCapabilities GetCapabilities() const override
    {
        return m_BackendCapabilities;
    };

    BackendCapabilities m_BackendCapabilities;
};

}    // namespace

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("LSTMValidateTensorShapesFromInputsCIFGDisabledTest")
{
    Graph graph;

    //Helper function creates graph containing LSTM layer with required input and output layers
    CreateLSTMLayerHelper(graph, false);

    //This function used to call ValidateShapesFromInputs();
    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("LSTMValidateTensorShapesFromInputsCIFGEnabledTest")
{
    Graph graph;

    //Helper function creates graph containing LSTM layer with required input and output layers
    CreateLSTMLayerHelper(graph, true);

    //This function used to call ValidateShapesFromInputs();
    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("InsertConvertersTest")
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
    CHECK(CheckSequence(graph.cbegin(),
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
    CHECK(CheckSequence(graph.cbegin(),
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
    armnn::ConstTensor weights(
            armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32, 0.0f, 0, true),
            weightsVector);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX     = 1;
    desc.m_StrideY     = 1;
    desc.m_DataLayout  = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    Convolution2dLayer* layer = graph.AddLayer<Convolution2dLayer>(desc, "conv2d");
    layer->m_Weight           = std::make_unique<armnn::ScopedTensorHandle>(weights);
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

TEST_CASE("Conv2dValidateTensorShapesFromInputs")
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 3, 8, 16 };
    const unsigned int weightsShape[] = { 2, 3, 5, 3 };
    const unsigned int outputShape[] = { 1, 2, 4, 14 };
    CreateConvolution2dGraph(graph, inputShape, weightsShape, outputShape);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("Conv2dValidateTensorShapesFromInputsNhwc")
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 8, 16, 3 };
    const unsigned int weightsShape[] = { 2, 5, 3, 3 };
    const unsigned int outputShape[] = { 1, 4, 14, 2 };
    CreateConvolution2dGraph(graph, inputShape, weightsShape, outputShape, DataLayout::NHWC);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

void CreateDepthwiseConvolution2dGraph(Graph &graph, const unsigned int* inputShape,
                                       const unsigned int* weightsShape, const unsigned int* outputShape,
                                       DataLayout dataLayout = DataLayout::NCHW)
{
    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    std::vector<float> weightsVector(18);
    armnn::ConstTensor weights(
            armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32, 0.0f, 0, true),
            weightsVector);

    DepthwiseConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX     = 1;
    desc.m_StrideY     = 1;
    desc.m_DataLayout  = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    DepthwiseConvolution2dLayer* layer = graph.AddLayer<DepthwiseConvolution2dLayer>(desc, "depthwiseConv2d");
    layer->m_Weight                    = std::make_unique<armnn::ScopedTensorHandle>(weights);
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

TEST_CASE("DepthwiseConv2dValidateTensorShapesFromInputs")
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 2, 3, 3 };
    const unsigned int weightsShape[] = { 1, 3, 3, 2 };
    const unsigned int outputShape[] = { 1, 2, 1, 1 };
    CreateDepthwiseConvolution2dGraph(graph, inputShape, weightsShape, outputShape);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("DepthwiseConv2dValidateTensorShapesFromInputsNhwc")
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 3, 3, 2 };
    const unsigned int weightsShape[] = { 1, 3, 3, 2 };
    const unsigned int outputShape[] = { 1, 1, 1, 2 };
    CreateDepthwiseConvolution2dGraph(graph, inputShape, weightsShape, outputShape, DataLayout::NHWC);

    CHECK_NOTHROW(graph.InferTensorInfos());
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

TEST_CASE("Pooling2dValidateTensorShapesFromInputs")
{
    Graph graph;
    const unsigned int inputShape[]  = { 5, 3, 52, 60 };
    const unsigned int outputShape[] = { 5, 3, 11, 13 };
    CreatePooling2dGraph(graph, inputShape, outputShape, DataLayout::NCHW);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("Pooling2dValidateTensorShapesFromInputsNhwc")
{
    Graph graph;
    const unsigned int inputShape[]  = { 5, 52, 60, 3 };
    const unsigned int outputShape[] = { 5, 11, 13, 3 };
    CreatePooling2dGraph(graph, inputShape, outputShape, DataLayout::NHWC);

    CHECK_NOTHROW(graph.InferTensorInfos());
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

TEST_CASE("ResizeBilinearValidateTensorShapesFromInputs")
{
    Graph graph;
    const unsigned int inputShape[]  = { 1, 2, 4, 5 };
    const unsigned int outputShape[] = { 1, 2, 3, 4 };
    CreateResizeBilinearGraph(graph, inputShape, outputShape);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("ResizeBilinearValidateTensorShapesFromInputsNhwc")
{
    Graph graph;
    const unsigned int inputShape[]  = { 1, 4, 5, 2 };
    const unsigned int outputShape[] = { 1, 3, 4, 2 };
    CreateResizeBilinearGraph(graph, inputShape, outputShape, DataLayout::NHWC);

    CHECK_NOTHROW(graph.InferTensorInfos());
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

TEST_CASE("GatherValidateTensorShapesFromInputs")
{
    Graph graph;
    armnn::TensorInfo paramsInfo({10, 5}, DataType::Float32);
    armnn::TensorInfo indicesInfo({3}, DataType::Signed32);
    armnn::TensorInfo outputInfo({3, 5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("GatherValidateTensorShapesFromInputs1DParams")
{
    Graph graph;
    armnn::TensorInfo paramsInfo({8}, DataType::Float32);
    armnn::TensorInfo indicesInfo({5}, DataType::Signed32);
    armnn::TensorInfo outputInfo( {5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("GatherValidateTensorShapesFromInputsMultiDimIndices")
{
    Graph graph;
    armnn::TensorInfo paramsInfo({3, 2, 5}, DataType::Float32);
    armnn::TensorInfo indicesInfo({2, 2}, DataType::Signed32);
    armnn::TensorInfo outputInfo({2, 2, 2, 5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("DetectionPostProcessValidateTensorShapes")
{
    Graph graph;
    armnn::TensorInfo boxEncodingsInfo({1, 10, 4}, DataType::QAsymmU8);
    armnn::TensorInfo scoresInfo({1, 10, 4}, DataType::QAsymmU8);
    std::vector<uint8_t> anchorsVector(40);
    armnn::ConstTensor anchors(armnn::TensorInfo({10, 4}, armnn::DataType::QAsymmU8, 0.0f, 0, true), anchorsVector);

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
    layer->m_Anchors = std::make_unique<armnn::ScopedTensorHandle>(anchors);
    layer->GetOutputSlot(0).SetTensorInfo(detectionBoxesInfo);
    layer->GetOutputSlot(1).SetTensorInfo(detectionScoresInfo);
    layer->GetOutputSlot(2).SetTensorInfo(detectionClassesInfo);
    layer->GetOutputSlot(3).SetTensorInfo(numDetectionInfo);

    input0->GetOutputSlot().Connect(layer->GetInputSlot(0));
    input1->GetOutputSlot().Connect(layer->GetInputSlot(1));

    CHECK_NOTHROW(graph.InferTensorInfos());
}

TEST_CASE("BackendCapabilityTest")
{
    BackendId backendId = "MockBackend";

    armnn::BackendOptions::BackendOption nonConstWeights{"NonConstWeights", true};

    // MockBackend does not support the NonConstWeights capability
    CHECK(!armnn::HasCapability(nonConstWeights, backendId));
    CHECK(!armnn::HasCapability("NonConstWeights", backendId));

    // MockBackend does not support the AsyncExecution capability
    CHECK(!armnn::GetCapability("AsyncExecution", backendId).has_value());
}

TEST_CASE("BackendHintTest")
{
    class TestBackendAssignment : public StrategyBase<NoThrowStrategy>
    {
    public:

        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name,
                             const armnn::LayerBindingId id = 0) override
        {
            armnn::IgnoreUnused(descriptor, constants, id, name);
            switch (layer->GetType())
            {
                case armnn::LayerType::Input:
                {
                    auto inputLayer = PolymorphicDowncast<const InputLayer*>(layer);
                    const auto connectedLayerBackendId = inputLayer->GetOutputSlot(0).GetOwningLayer().GetBackendId();
                    CHECK((inputLayer->GetBackendId() == connectedLayerBackendId));
                    break;
                }
                case armnn::LayerType::Output:
                {
                    auto outputLayer = PolymorphicDowncast<const OutputLayer*>(layer);
                    CHECK((outputLayer->GetBackendId() == "MockBackend"));
                    break;
                }
                case armnn::LayerType::Activation:
                {
                    auto activation = PolymorphicDowncast<const ActivationLayer*>(layer);
                    CHECK((activation->GetBackendId() == "CustomBackend"));
                    break;
                }
                default:
                {
                    m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
                }
            }
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

    backendRegistry.Register("MockBackend", []() { return std::make_unique<CustomAllocatorBackend<MockPolicy>>(); });

    backendRegistry.Register("CustomBackend",
                             []() { return std::make_unique<CustomAllocatorBackend<CustomPolicy>>(); });

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

    CHECK(res.IsOk());

    TestBackendAssignment visitor;
    for (auto it = firstLayer; it != lastLayer; ++it)
    {
        (*it)->ExecuteStrategy(visitor);
    }
    // Clean up the registry for the next test.
    backendRegistry.Deregister("MockBackend");
    backendRegistry.Deregister("CustomBackend");
}

// Tests that OptimizeForExclusiveConnections works, fusing when needed, using BatchNorm fusing as example
TEST_CASE("OptimizeForExclusiveConnectionsFuseTest")
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
    ConstTensor weights(TensorInfo(4, weightsDimensionSizes, DataType::Float32, 0.0f, 0, true), weightsVector);

    std::vector<float> betaVector     = { 0.1f };
    std::vector<float> gammaVector    = { 0.5f };
    std::vector<float> meanVector     = { 0 };
    std::vector<float> varianceVector = { 1 };
    ConstTensor beta(TensorInfo(1, outputChannelSize, DataType::Float32, 0.0f, 0, true), betaVector);
    ConstTensor gamma(TensorInfo(1, outputChannelSize, DataType::Float32, 0.0f, 0, true), gammaVector);
    ConstTensor mean(TensorInfo(1, outputChannelSize, DataType::Float32, 0.0f, 0, true), meanVector);
    ConstTensor variance(TensorInfo(1, outputChannelSize, DataType::Float32, 0.0f, 0, true), varianceVector);

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
    conv->m_Weight        = std::make_unique<ScopedTensorHandle>(weights);
    batchNorm->m_Beta     = std::make_unique<ScopedTensorHandle>(beta);
    batchNorm->m_Gamma    = std::make_unique<ScopedTensorHandle>(gamma);
    batchNorm->m_Mean     = std::make_unique<ScopedTensorHandle>(mean);
    batchNorm->m_Variance = std::make_unique<ScopedTensorHandle>(variance);
    if (convolution2dDescriptor.m_BiasEnabled)
    {
        std::vector<float> biasVector = { 11 };
        ConstTensor bias(TensorInfo(1, outputChannelSize, DataType::Float32, 0.0f, 0, true), biasVector);
        conv->m_Bias = std::make_unique<ScopedTensorHandle>(bias);
    }

    // Connect layers
    input->GetOutputSlot(0).Connect(conv->GetInputSlot(0));
    conv->GetOutputSlot(0).Connect(batchNorm->GetInputSlot(0));
    batchNorm->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    CHECK(4 == graph.GetNumLayers());
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
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

    CHECK(3 == graph.GetNumLayers());
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             checkFusedConv2d,
                             &IsLayerOfType<OutputLayer>));
}

// Tests that OptimizeForExclusiveConnections works, not fusing when not needed, using BatchNorm fusing as example
TEST_CASE("OptimizeForExclusiveConnectionsWithoutFuseTest")
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

    CHECK(5 == graph.GetNumLayers());
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::Convolution2dLayer>,
                             &IsLayerOfType<armnn::BatchNormalizationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
    // Optimize graph
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FuseBatchNormIntoConvolution2DFloat32()));

    CHECK(5 == graph.GetNumLayers());
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::Convolution2dLayer>,
                             &IsLayerOfType<armnn::BatchNormalizationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}
} // Optimizer TestSuite
