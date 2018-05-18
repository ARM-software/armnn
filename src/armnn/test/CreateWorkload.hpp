//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <boost/test/unit_test.hpp>

#include <boost/cast.hpp>

#include "backends/WorkloadData.hpp"
#include "Graph.hpp"

#include <utility>

#include "backends/CpuTensorHandle.hpp"

using namespace armnn;

namespace
{

using namespace std;

// Calls CreateWorkload for a layer, and checks the returned pointer is of the correct type
template<typename Workload>
std::unique_ptr<Workload> MakeAndCheckWorkload(Layer& layer, Graph& graph, const IWorkloadFactory& factory)
{
    std::unique_ptr<IWorkload> workload = layer.CreateWorkload(graph, factory);
    BOOST_TEST(workload.get() == boost::polymorphic_downcast<Workload*>(workload.get()),
               "Cannot convert to derived class");
    std::string reasonIfUnsupported;
    BOOST_TEST(factory.IsLayerSupported(layer, layer.GetDataType(), reasonIfUnsupported));
    return std::unique_ptr<Workload>(static_cast<Workload*>(workload.release()));
}

// connects two layers
void Connect(Layer* from, Layer* to, const TensorInfo& tensorInfo, unsigned int fromIndex = 0, unsigned int toIndex = 0)
{
    from->GetOutputSlot(fromIndex).Connect(to->GetInputSlot(toIndex));
    from->GetOutputHandler(fromIndex).SetTensorInfo(tensorInfo);
}

// helper function to create tensor handlers for workloads, assuming they all use the same factory
void CreateTensorHandles(armnn::Graph& graph, armnn::IWorkloadFactory& factory)
{
    for (auto&& layer : graph.TopologicalSort())
    {
        layer->CreateTensorHandles(graph, factory);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// The following functions are called by backends/test/CreateWorkload*.cpp
// They build very simple graphs, and then create a workload.
// Some checks are performed on the workload to ensure parameters have been passed correctly.
// They return the created workloads so that backend-specific checks can be performed.
/////////////////////////////////////////////////////////////////////////////////////////////

template <typename ActivationWorkload>
std::unique_ptr<ActivationWorkload> CreateActivationWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                 armnn::Graph&            graph)
{
    // create the layer we're testing
    ActivationDescriptor layerDesc;
    layerDesc.m_Function = ActivationFunction::Abs;
    layerDesc.m_A        = 3.5f;
    layerDesc.m_B        = -10.0f;

    ActivationLayer* const layer = graph.AddLayer<ActivationLayer>(layerDesc, "layer");

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo tensorInfo({1, 1}, ActivationWorkload::ms_DataType);

    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);

    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<ActivationWorkload>(*layer, graph, factory);

    ActivationQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_A == 3.5f);
    BOOST_TEST(queueDescriptor.m_Parameters.m_B == -10.0f);
    BOOST_TEST((queueDescriptor.m_Parameters.m_Function == ActivationFunction::Abs));

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename AdditionWorkload>
std::unique_ptr<AdditionWorkload> CreateAdditionWorkloadTest(armnn::IWorkloadFactory& factory,
                                                             armnn::Graph&            graph)
{
    // create the layer we're testing
    Layer* const layer = graph.AddLayer<AdditionLayer>("layer");

    // create extra layers
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo tensorInfo({2, 3}, AdditionWorkload::ms_DataType);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<AdditionWorkload>(*layer, graph, factory);

    AdditionQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 2);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename BatchNormalizationFloat32Workload>
std::unique_ptr<BatchNormalizationFloat32Workload> CreateBatchNormalizationWorkloadTest(
    armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // create the layer we're testing
    BatchNormalizationDescriptor layerDesc;
    layerDesc.m_Eps = 0.05f;

    BatchNormalizationLayer* const layer = graph.AddLayer<BatchNormalizationLayer>(layerDesc, "layer");

    armnn::TensorInfo weightInfo({3}, armnn::DataType::Float32);
    layer->m_Mean     = std::make_unique<ScopedCpuTensorHandle>(weightInfo);
    layer->m_Variance = std::make_unique<ScopedCpuTensorHandle>(weightInfo);
    layer->m_Beta     = std::make_unique<ScopedCpuTensorHandle>(weightInfo);
    layer->m_Gamma    = std::make_unique<ScopedCpuTensorHandle>(weightInfo);
    layer->m_Mean->Allocate();
    layer->m_Variance->Allocate();
    layer->m_Beta->Allocate();
    layer->m_Gamma->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo tensorInfo({2, 3, 1, 1}, armnn::DataType::Float32);
    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<BatchNormalizationFloat32Workload>(*layer, graph, factory);

    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Parameters.m_Eps == 0.05f);
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);
    BOOST_TEST((queueDescriptor.m_Mean->GetTensorInfo() == TensorInfo({3}, DataType::Float32)));
    BOOST_TEST((queueDescriptor.m_Variance->GetTensorInfo() == TensorInfo({3}, DataType::Float32)));
    BOOST_TEST((queueDescriptor.m_Gamma->GetTensorInfo() == TensorInfo({3}, DataType::Float32)));
    BOOST_TEST((queueDescriptor.m_Beta->GetTensorInfo() == TensorInfo({3}, DataType::Float32)));

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename Convolution2dWorkload>
std::unique_ptr<Convolution2dWorkload> CreateConvolution2dWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                              armnn::Graph&            graph)
{
    // create the layer we're testing
    Convolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft = 3;
    layerDesc.m_PadRight = 3;
    layerDesc.m_PadTop = 1;
    layerDesc.m_PadBottom = 1;
    layerDesc.m_StrideX = 2;
    layerDesc.m_StrideY = 4;
    layerDesc.m_BiasEnabled = true;

    Convolution2dLayer* const layer = graph.AddLayer<Convolution2dLayer>(layerDesc, "layer");

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(TensorInfo({2, 3, 5, 3},
                                                                         Convolution2dWorkload::ms_DataType));
    layer->m_Bias   = std::make_unique<ScopedCpuTensorHandle>
        (TensorInfo({2}, GetBiasDataType(Convolution2dWorkload::ms_DataType)));

    layer->m_Weight->Allocate();
    layer->m_Bias->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({2, 3, 8, 16}, Convolution2dWorkload::ms_DataType));
    Connect(layer, output, TensorInfo({2, 2, 2, 10}, Convolution2dWorkload::ms_DataType));
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<Convolution2dWorkload>(*layer, graph, factory);

    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideX == 2);
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideY == 4);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadLeft == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadRight == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadTop == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadBottom == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_BiasEnabled == true);

    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);
    BOOST_TEST((queueDescriptor.m_Weight->GetTensorInfo() == TensorInfo({2, 3, 5, 3},
                                                                        Convolution2dWorkload::ms_DataType)));
    BOOST_TEST((queueDescriptor.m_Bias->GetTensorInfo() ==
        TensorInfo({2}, GetBiasDataType(Convolution2dWorkload::ms_DataType))));

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename Convolution2dWorkload>
std::unique_ptr<Convolution2dWorkload> CreateDirectConvolution2dWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                       armnn::Graph&            graph)
{
    // create the layer we're testing
    Convolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft = 1;
    layerDesc.m_PadRight = 1;
    layerDesc.m_PadTop = 1;
    layerDesc.m_PadBottom = 1;
    layerDesc.m_StrideX = 1;
    layerDesc.m_StrideY = 1;
    layerDesc.m_BiasEnabled = true;

    Convolution2dLayer* const layer = graph.AddLayer<Convolution2dLayer>(layerDesc, "layer");

    float inputsQScale = Convolution2dWorkload::ms_DataType == DataType::QuantisedAsymm8 ? 1.0f : 0.0;
    float outputQScale = Convolution2dWorkload::ms_DataType == DataType::QuantisedAsymm8 ? 2.0f : 0.0;

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(TensorInfo({ 2, 3, 3, 3 },
        Convolution2dWorkload::ms_DataType, inputsQScale));
    layer->m_Bias   = std::make_unique<ScopedCpuTensorHandle>
        (TensorInfo({2},  GetBiasDataType(Convolution2dWorkload::ms_DataType), inputsQScale));
    layer->m_Weight->Allocate();
    layer->m_Bias->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({2, 3, 6, 6}, Convolution2dWorkload::ms_DataType, inputsQScale));
    Connect(layer, output, TensorInfo({2, 2, 6, 6}, Convolution2dWorkload::ms_DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<Convolution2dWorkload>(*layer, graph, factory);

    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideX == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideY == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadLeft == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadRight == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadTop == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadBottom == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_BiasEnabled == true);

    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);
    BOOST_TEST((queueDescriptor.m_Weight->GetTensorInfo() == TensorInfo({2, 3, 3, 3},
        Convolution2dWorkload::ms_DataType, inputsQScale)));
    BOOST_TEST((queueDescriptor.m_Bias->GetTensorInfo()
                == TensorInfo({2},  GetBiasDataType(Convolution2dWorkload::ms_DataType), inputsQScale)));

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename DepthwiseConvolution2dFloat32Workload>
std::unique_ptr<DepthwiseConvolution2dFloat32Workload> CreateDepthwiseConvolution2dWorkloadTest(
    armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // create the layer we're testing
    DepthwiseConvolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft         = 3;
    layerDesc.m_PadRight        = 3;
    layerDesc.m_PadTop          = 1;
    layerDesc.m_PadBottom       = 1;
    layerDesc.m_StrideX         = 2;
    layerDesc.m_StrideY         = 4;
    layerDesc.m_BiasEnabled     = true;

    DepthwiseConvolution2dLayer* const layer = graph.AddLayer<DepthwiseConvolution2dLayer>(layerDesc, "layer");

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(TensorInfo({3, 3, 5, 3}, DataType::Float32));
    layer->m_Bias   = std::make_unique<ScopedCpuTensorHandle>(TensorInfo({9}, DataType::Float32));
    layer->m_Weight->Allocate();
    layer->m_Bias->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({2, 3, 8, 16}, armnn::DataType::Float32));
    Connect(layer, output, TensorInfo({2, 9, 2, 10}, armnn::DataType::Float32));
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<DepthwiseConvolution2dFloat32Workload>(*layer, graph, factory);

    DepthwiseConvolution2dQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideX == 2);
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideY == 4);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadLeft == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadRight == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadTop == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadBottom == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_BiasEnabled == true);

    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);
    BOOST_TEST((queueDescriptor.m_Weight->GetTensorInfo() == TensorInfo({3, 3, 5, 3}, DataType::Float32)));
    BOOST_TEST((queueDescriptor.m_Bias->GetTensorInfo() == TensorInfo({9}, DataType::Float32)));

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename FullyConnectedWorkload>
std::unique_ptr<FullyConnectedWorkload> CreateFullyConnectedWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                         armnn::Graph&            graph)
{
    // create the layer we're testing
    FullyConnectedDescriptor layerDesc;
    layerDesc.m_BiasEnabled = true;
    layerDesc.m_TransposeWeightMatrix = true;

    FullyConnectedLayer* const layer = graph.AddLayer<FullyConnectedLayer>(layerDesc, "layer");

    float inputsQScale = FullyConnectedWorkload::ms_DataType == DataType::QuantisedAsymm8 ? 1.0f : 0.0;
    float outputQScale = FullyConnectedWorkload::ms_DataType == DataType::QuantisedAsymm8 ? 2.0f : 0.0;

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(TensorInfo({7, 20},
        FullyConnectedWorkload::ms_DataType, inputsQScale, 0));
    layer->m_Bias   = std::make_unique<ScopedCpuTensorHandle>(TensorInfo({7},
        GetBiasDataType(FullyConnectedWorkload::ms_DataType), inputsQScale));
    layer->m_Weight->Allocate();
    layer->m_Bias->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({3, 1, 4, 5}, FullyConnectedWorkload::ms_DataType, inputsQScale));
    Connect(layer, output, TensorInfo({3, 7}, FullyConnectedWorkload::ms_DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<FullyConnectedWorkload>(*layer, graph, factory);

    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Parameters.m_BiasEnabled == true);
    BOOST_TEST(queueDescriptor.m_Parameters.m_TransposeWeightMatrix == true);

    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);
    BOOST_TEST((queueDescriptor.m_Weight->GetTensorInfo() ==
        TensorInfo({7, 20}, FullyConnectedWorkload::ms_DataType, inputsQScale)));
    BOOST_TEST((queueDescriptor.m_Bias->GetTensorInfo() ==
        TensorInfo({7}, GetBiasDataType(FullyConnectedWorkload::ms_DataType), inputsQScale)));

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename MultiplicationWorkload>
std::unique_ptr<MultiplicationWorkload> CreateMultiplicationWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                         armnn::Graph&            graph)
{
    // create the layer we're testing
    Layer* const layer = graph.AddLayer<MultiplicationLayer>("layer");

    // create extra layers
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo tensorInfo({2, 3}, MultiplicationWorkload::ms_DataType);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<MultiplicationWorkload>(*layer, graph, factory);

    MultiplicationQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 2);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename NormalizationFloat32Workload>
std::unique_ptr<NormalizationFloat32Workload> CreateNormalizationWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                              armnn::Graph&            graph)
{
    // create the layer we're testing
    NormalizationDescriptor layerDesc;
    layerDesc.m_NormChannelType = NormalizationAlgorithmChannel::Across;
    layerDesc.m_NormMethodType = NormalizationAlgorithmMethod::LocalBrightness;
    layerDesc.m_NormSize = 3;
    layerDesc.m_Alpha = 0.5f;
    layerDesc.m_Beta = -1.0f;
    layerDesc.m_K = 0.2f;

    NormalizationLayer* layer = graph.AddLayer<NormalizationLayer>(layerDesc, "layer");

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({3, 5, 5, 1}, armnn::DataType::Float32));
    Connect(layer, output, TensorInfo({3, 5, 5, 1}, armnn::DataType::Float32));
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<NormalizationFloat32Workload>(*layer, graph, factory);

    NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST((queueDescriptor.m_Parameters.m_NormChannelType == NormalizationAlgorithmChannel::Across));
    BOOST_TEST((queueDescriptor.m_Parameters.m_NormMethodType == NormalizationAlgorithmMethod::LocalBrightness));
    BOOST_TEST(queueDescriptor.m_Parameters.m_NormSize == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_Alpha == 0.5f);
    BOOST_TEST(queueDescriptor.m_Parameters.m_Beta == -1.0f);
    BOOST_TEST(queueDescriptor.m_Parameters.m_K == 0.2f);

    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename Pooling2dWorkload>
std::unique_ptr<Pooling2dWorkload> CreatePooling2dWorkloadTest(armnn::IWorkloadFactory& factory,
                                                               armnn::Graph&            graph)
{
    // create the layer we're testing
    Pooling2dDescriptor layerDesc;
    layerDesc.m_PoolType = PoolingAlgorithm::Average;
    layerDesc.m_PoolWidth = 3;
    layerDesc.m_PoolHeight = 3;
    layerDesc.m_PadLeft = 2;
    layerDesc.m_PadRight = 2;
    layerDesc.m_PadTop = 1;
    layerDesc.m_PadBottom = 1;
    layerDesc.m_StrideX = 2;
    layerDesc.m_StrideY = 3;
    layerDesc.m_OutputShapeRounding = OutputShapeRounding::Floor;

    Pooling2dLayer* const layer = graph.AddLayer<Pooling2dLayer>(layerDesc, "layer");

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({3, 2, 5, 5}, Pooling2dWorkload::ms_DataType));
    Connect(layer, output, TensorInfo({3, 2, 2, 4}, Pooling2dWorkload::ms_DataType));
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<Pooling2dWorkload>(*layer, graph, factory);

    Pooling2dQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST((queueDescriptor.m_Parameters.m_PoolType == PoolingAlgorithm::Average));
    BOOST_TEST((queueDescriptor.m_Parameters.m_OutputShapeRounding == OutputShapeRounding::Floor));
    BOOST_TEST(queueDescriptor.m_Parameters.m_PoolWidth == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PoolHeight == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideX == 2);
    BOOST_TEST(queueDescriptor.m_Parameters.m_StrideY == 3);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadLeft == 2);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadRight == 2);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadTop == 1);
    BOOST_TEST(queueDescriptor.m_Parameters.m_PadBottom == 1);

    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename SoftmaxWorkload>
std::unique_ptr<SoftmaxWorkload> CreateSoftmaxWorkloadTest(armnn::IWorkloadFactory& factory,
                                                           armnn::Graph&            graph)
{
    // create the layer we're testing
    SoftmaxDescriptor softmaxDescriptor;
    Layer* const layer = graph.AddLayer<SoftmaxLayer>(softmaxDescriptor, "layer");

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo tensorInfo({4, 1}, SoftmaxWorkload::ms_DataType);
    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<SoftmaxWorkload>(*layer, graph, factory);

    SoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

template<typename SplitterWorkload>
std::unique_ptr<SplitterWorkload>
    CreateSplitterWorkloadTest(armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // create the layer we're testing
    // NOTE: need three dimensions channels, height/y, width/x because the Compute
    //       library restricts subtensors to have the same x and y dimensions as
    //       their parent tensors, and therefore the origin on the x and y dimension
    //       has to be zero for any view. So we need a third dimension to split...
    // NOTE: arguments are: number of views, number of dimensions
    ViewsDescriptor layerDesc(3, 3);
    // NOTE: arguments are: view, dimension, value
    layerDesc.SetViewOriginCoord(0, 0, 0);
    layerDesc.SetViewOriginCoord(1, 0, 1);
    layerDesc.SetViewOriginCoord(2, 0, 3);

    Layer* const layer = graph.AddLayer<SplitterLayer>(layerDesc, "layer");

    // add extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output0 = graph.AddLayer<OutputLayer>(0, "output0");
    Layer* const output1 = graph.AddLayer<OutputLayer>(1, "output1");
    Layer* const output2 = graph.AddLayer<OutputLayer>(2, "output2");

    // connect up
    armnn::TensorInfo tensorInfo({5, 7, 7}, SplitterWorkload::ms_DataType);
    Connect(input, layer, tensorInfo);

    armnn::TensorInfo output0Info({1, 7, 7}, SplitterWorkload::ms_DataType);
    armnn::TensorInfo output1Info({2, 7, 7}, SplitterWorkload::ms_DataType);
    armnn::TensorInfo output2Info({2, 7, 7}, SplitterWorkload::ms_DataType);

    Connect(layer, output0, output0Info, 0, 0);
    Connect(layer, output1, output1Info, 1, 0);
    Connect(layer, output2, output2Info, 2, 0);

    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<SplitterWorkload>(*layer, graph, factory);

    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 3);
    BOOST_TEST(queueDescriptor.m_ViewOrigins.size() == 3);

    BOOST_TEST(queueDescriptor.m_ViewOrigins[0].m_Origin[0] == 0);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[1].m_Origin[0] == 1);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[2].m_Origin[0] == 3);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[0].m_Origin[1] == 0);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[1].m_Origin[1] == 0);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[2].m_Origin[1] == 0);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[0].m_Origin[2] == 0);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[1].m_Origin[2] == 0);
    BOOST_TEST(queueDescriptor.m_ViewOrigins[2].m_Origin[2] == 0);

    // return so we can do extra, backend-specific tests
    return workload;
}

/// This function constructs a graph with both a splitter and a merger, and returns a pair of the workloads
template<typename SplitterWorkload, typename MergerWorkload>
std::pair<std::unique_ptr<SplitterWorkload>, std::unique_ptr<MergerWorkload>>
    CreateSplitterMergerWorkloadTest(armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    static_assert(SplitterWorkload::ms_DataType == MergerWorkload::ms_DataType,
        "Splitter and merger workloads must have the same data type");

    armnn::TensorInfo inputTensorInfo({ 1, 2, 100, 10 }, SplitterWorkload::ms_DataType);

    armnn::TensorInfo splitTensorInfo1({ 1, 1, 100, 10 }, SplitterWorkload::ms_DataType);
    armnn::TensorInfo splitTensorInfo2({ 1, 1, 100, 10 }, SplitterWorkload::ms_DataType);

    //construct the  graph
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");

    armnn::ViewsDescriptor splitterViews(2);
    splitterViews.SetViewOriginCoord(0, 0, 0);
    splitterViews.SetViewOriginCoord(0, 1, 0);
    splitterViews.SetViewOriginCoord(0, 2, 0);
    splitterViews.SetViewOriginCoord(0, 3, 0);

    splitterViews.SetViewOriginCoord(1, 0, 0);
    splitterViews.SetViewOriginCoord(1, 1, 1);
    splitterViews.SetViewOriginCoord(1, 2, 0);
    splitterViews.SetViewOriginCoord(1, 3, 0);

    Layer* const splitter = graph.AddLayer<SplitterLayer>(splitterViews, "splitter");
    BOOST_TEST_CHECKPOINT("created splitter layer");

    armnn::OriginsDescriptor mergerViews(2);
    mergerViews.SetViewOriginCoord(0, 0, 0);
    mergerViews.SetViewOriginCoord(0, 1, 1);
    mergerViews.SetViewOriginCoord(0, 2, 0);
    mergerViews.SetViewOriginCoord(0, 3, 0);

    mergerViews.SetViewOriginCoord(1, 0, 0);
    mergerViews.SetViewOriginCoord(1, 1, 0);
    mergerViews.SetViewOriginCoord(1, 2, 0);
    mergerViews.SetViewOriginCoord(1, 3, 0);

    Layer* const merger = graph.AddLayer<MergerLayer>(mergerViews, "merger");
    BOOST_TEST_CHECKPOINT("created merger layer");

    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // add connections
    Connect(input, splitter, inputTensorInfo, 0, 0);
    BOOST_TEST_CHECKPOINT("connect input to splitter");
    Connect(splitter, merger, splitTensorInfo1, 0, 1); // The splitter & merger are connected up
    BOOST_TEST_CHECKPOINT("connect splitter[0] to merger[1]");
    Connect(splitter, merger, splitTensorInfo2, 1, 0); // so that the outputs are flipped round
    BOOST_TEST_CHECKPOINT("connect splitter[1] to merger[0]");
    Connect(merger, output, inputTensorInfo, 0, 0);
    BOOST_TEST_CHECKPOINT("connect merger to output");

    CreateTensorHandles(graph, factory);
    BOOST_TEST_CHECKPOINT("created tensor handles");

    auto workloadSplitter = MakeAndCheckWorkload<SplitterWorkload>(*splitter, graph, factory);
    BOOST_TEST_CHECKPOINT("created splitter workload");
    auto workloadMerger = MakeAndCheckWorkload<MergerWorkload>(*merger, graph, factory);
    BOOST_TEST_CHECKPOINT("created merger workload");

    return {std::move(workloadSplitter), std::move(workloadMerger)};
}


/// This function constructs a graph with a splitter with two outputs. Each of the outputs is then
/// connected to two different activation layers
template<typename SplitterWorkload, typename ActivationWorkload>
void CreateSplitterMultipleInputsOneOutputWorkloadTest(armnn::IWorkloadFactory& factory, armnn::Graph& graph,
                                 std::unique_ptr<SplitterWorkload>& wlSplitter,
                                 std::unique_ptr<ActivationWorkload>& wlActiv0_0,
                                 std::unique_ptr<ActivationWorkload>& wlActiv0_1,
                                 std::unique_ptr<ActivationWorkload>& wlActiv1_0,
                                 std::unique_ptr<ActivationWorkload>& wlActiv1_1)
{
    static_assert(SplitterWorkload::ms_DataType == ActivationWorkload::ms_DataType,
        "Splitter and activation workloads must have the same data type");

    armnn::TensorInfo inputTensorInfo ({ 1, 3, 100, 50 }, SplitterWorkload::ms_DataType);
    armnn::TensorInfo splitTensorInfo1({ 1, 1, 100, 50 }, SplitterWorkload::ms_DataType);
    armnn::TensorInfo splitTensorInfo2({ 1, 2, 100, 50 }, SplitterWorkload::ms_DataType);

    //construct the  graph
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");

    armnn::ViewsDescriptor splitterViews(2);

    splitterViews.SetViewOriginCoord(0, 0, 0);
    splitterViews.SetViewOriginCoord(0, 1, 0);
    splitterViews.SetViewOriginCoord(0, 2, 0);
    splitterViews.SetViewOriginCoord(0, 3, 0);

    splitterViews.SetViewOriginCoord(1, 0, 0);
    splitterViews.SetViewOriginCoord(1, 1, 1);
    splitterViews.SetViewOriginCoord(1, 2, 0);
    splitterViews.SetViewOriginCoord(1, 3, 0);

    Layer* const splitter = graph.AddLayer<SplitterLayer>(splitterViews, "splitter");

    armnn::ActivationDescriptor activationDesc;

    Layer* const activ0_0 = graph.AddLayer<ActivationLayer>(activationDesc, "activ0_0");
    Layer* const activ0_1 = graph.AddLayer<ActivationLayer>(activationDesc, "activ0_1");
    Layer* const activ1_0 = graph.AddLayer<ActivationLayer>(activationDesc, "activ1_0");
    Layer* const activ1_1 = graph.AddLayer<ActivationLayer>(activationDesc, "activ1_1");

    Layer* const output1 = graph.AddLayer<OutputLayer>(1, "output1");
    Layer* const output2 = graph.AddLayer<OutputLayer>(2, "output2");
    Layer* const output3 = graph.AddLayer<OutputLayer>(3, "output3");
    Layer* const output4 = graph.AddLayer<OutputLayer>(4, "output4");

    // add connections
    Connect(input, splitter, inputTensorInfo, 0, 0);
    Connect(splitter, activ0_0, splitTensorInfo1, 0, 0);
    Connect(splitter, activ0_1, splitTensorInfo1, 0, 0);

    Connect(splitter, activ1_0, splitTensorInfo2, 1, 0);
    Connect(splitter, activ1_1, splitTensorInfo2, 1, 0);

    Connect(activ0_0, output1, splitTensorInfo1, 0, 0);
    Connect(activ0_1, output2, splitTensorInfo1, 0, 0);
    Connect(activ1_0, output3, splitTensorInfo2, 0, 0);
    Connect(activ1_1, output4, splitTensorInfo2, 0, 0);

    CreateTensorHandles(graph, factory);

    auto workloadSplitter = MakeAndCheckWorkload<SplitterWorkload>(*splitter, graph, factory);
    auto workloadActiv0_0 = MakeAndCheckWorkload<ActivationWorkload>(*activ0_0, graph, factory);
    auto workloadActiv0_1 = MakeAndCheckWorkload<ActivationWorkload>(*activ0_1, graph, factory);
    auto workloadActiv1_0 = MakeAndCheckWorkload<ActivationWorkload>(*activ1_0, graph, factory);
    auto workloadActiv1_1 = MakeAndCheckWorkload<ActivationWorkload>(*activ1_1, graph, factory);

    wlSplitter = std::move(workloadSplitter);
    wlActiv0_0 = std::move(workloadActiv0_0);
    wlActiv0_1 = std::move(workloadActiv0_1);
    wlActiv1_0 = std::move(workloadActiv1_0);
    wlActiv1_1 = std::move(workloadActiv1_1);
}

template <typename ResizeBilinearWorkload>
std::unique_ptr<ResizeBilinearWorkload> CreateResizeBilinearWorkloadTest(armnn::IWorkloadFactory& factory,
    armnn::Graph& graph)
{
    // create the layer we're testing
    TensorShape outputShape({ 2, 3, 2, 2 });
    ResizeBilinearDescriptor resizeDesc;
    resizeDesc.m_TargetWidth = outputShape[3];
    resizeDesc.m_TargetHeight = outputShape[2];
    Layer* const layer = graph.AddLayer<ResizeBilinearLayer>(resizeDesc, "layer");

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo inputTensorInfo({ 2, 3, 4, 4 }, ResizeBilinearWorkload::ms_DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, ResizeBilinearWorkload::ms_DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<ResizeBilinearWorkload>(*layer, graph, factory);

    ResizeBilinearQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename L2NormalizationWorkload>
std::unique_ptr<L2NormalizationWorkload> CreateL2NormalizationWorkloadTest(armnn::IWorkloadFactory& factory,
    armnn::Graph& graph)
{
    // create the layer we're testing
    Layer* const layer = graph.AddLayer<L2NormalizationLayer>("l2norm");

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo inputTensorInfo({ 5, 20, 50, 67 }, L2NormalizationWorkload::ms_DataType);
    armnn::TensorInfo outputTensorInfo({ 5, 20, 50, 67 }, L2NormalizationWorkload::ms_DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<L2NormalizationWorkload>(*layer, graph, factory);

    L2NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

template <typename ReshapeWorkload>
std::unique_ptr<ReshapeWorkload> CreateReshapeWorkloadTest(armnn::IWorkloadFactory& factory,
    armnn::Graph& graph)
{
    // create the layer we're testing
    TensorShape outputShape({ 1, 4 });
    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputShape;
    Layer* const layer = graph.AddLayer<ReshapeLayer>(reshapeDesc, "layer");

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo inputTensorInfo({ 4, 1 }, ReshapeWorkload::ms_DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, ReshapeWorkload::ms_DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<ReshapeWorkload>(*layer, graph, factory);

    ReshapeQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // return so we can do extra, backend-specific tests
    return workload;
}

}
