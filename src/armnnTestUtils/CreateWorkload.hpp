//
// Copyright © 2017,2021-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TestUtils.hpp"

#include <Graph.hpp>
#include <Network.hpp>
#include <ResolveType.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <doctest/doctest.h>

#include <utility>

using namespace armnn;

namespace
{

using namespace std;

// Calls CreateWorkload for a layer, and checks the returned pointer is of the correct type.
template<typename Workload>
std::unique_ptr<Workload> MakeAndCheckWorkload(Layer& layer,
                                               const IWorkloadFactory& factory,
                                               const ModelOptions& modelOptions = {})
{
    std::unique_ptr<IWorkload> workload = layer.CreateWorkload(factory);
    CHECK_MESSAGE(workload.get() == PolymorphicDowncast<Workload*>(workload.get()),
               "Cannot convert to derived class");
    std::string reasonIfUnsupported;
    layer.SetBackendId(factory.GetBackendId());
    CHECK(factory.IsLayerSupported(layer, layer.GetDataType(), reasonIfUnsupported, modelOptions));
    return std::unique_ptr<Workload>(static_cast<Workload*>(workload.release()));
}

// Helper function to create tensor handlers for workloads, assuming they all use the same factory.
void CreateTensorHandles(armnn::Graph& graph,
                         armnn::IWorkloadFactory& factory)
{
    TensorHandleFactoryRegistry tmpRegistry;
    for (auto&& layer : graph.TopologicalSort())
    {
        layer->CreateTensorHandles(tmpRegistry, factory);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// The following functions are called by backendsCommon/test/CreateWorkload*.cpp
// They build very simple graphs, and then create a workload.
// Some checks are performed on the workload to ensure parameters have been passed correctly.
// They return the created workloads so that backend-specific checks can be performed.
/////////////////////////////////////////////////////////////////////////////////////////////

template <typename ActivationWorkload, armnn::DataType DataType>
std::unique_ptr<ActivationWorkload> CreateActivationWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                 armnn::Graph&            graph)
{
    // Creates the layer we're testing.
    ActivationDescriptor layerDesc;
    layerDesc.m_Function = ActivationFunction::ReLu;
    layerDesc.m_A        = 3.5f;
    layerDesc.m_B        = -10.0f;

    ActivationLayer* const layer = graph.AddLayer<ActivationLayer>(layerDesc, "layer");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo({1, 1}, DataType);

    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);

    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<ActivationWorkload>(*layer, factory);

    ActivationQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);
    CHECK(queueDescriptor.m_Parameters.m_A == 3.5f);
    CHECK(queueDescriptor.m_Parameters.m_B == -10.0f);
    CHECK((queueDescriptor.m_Parameters.m_Function == ActivationFunction::ReLu));

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename WorkloadType,
          typename DescriptorType,
          typename LayerType,
          armnn::DataType DataType>
std::unique_ptr<WorkloadType> CreateElementwiseWorkloadTest(armnn::IWorkloadFactory & factory,
                                                            armnn::Graph & graph)
{
    // Creates the layer we're testing.
    Layer* const layer = graph.AddLayer<LayerType>("layer");

    // Creates extra layers.
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo({2, 3}, DataType);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<WorkloadType>(*layer, factory);

    auto queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename WorkloadType, armnn::DataType DataType>
std::unique_ptr<WorkloadType> CreateElementwiseBinaryWorkloadTest(armnn::IWorkloadFactory & factory,
                                                                  armnn::Graph & graph,
                                                                  armnn::BinaryOperation binaryOperation)
{
    // Creates the layer we're testing.
    ElementwiseBinaryDescriptor descriptor(binaryOperation);
    //ElementwiseBinaryDescriptor descriptor = ElementwiseBinaryDescriptor(binaryOperation);

    Layer* const layer = graph.AddLayer<ElementwiseBinaryLayer>(descriptor, "layer");

    // Creates extra layers.
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo({2, 3}, DataType);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<WorkloadType>(*layer, factory);

    auto queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template<typename WorkloadType,
         typename DescriptorType,
         armnn::DataType DataType>
std::unique_ptr<WorkloadType> CreateSubtractionWithBlobWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                    armnn::Graph& graph)
{
    // Creates the layer we're testing.
    SubtractionLayer* const layer = graph.AddLayer<SubtractionLayer>("layer");

    auto activationDesc = std::make_shared<ActivationDescriptor>();
    activationDesc->m_A        = 10.0f;
    activationDesc->m_B        = 5.0f;
    activationDesc->m_Function = armnn::ActivationFunction::BoundedReLu;

    layer->SetAdditionalInfoForObject(activationDesc);

    // Creates extra layers.
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo({2, 3}, DataType);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Check that the additional information can be queried from the layer
    std::shared_ptr<ActivationDescriptor>
        activationDescPtr = layer->GetAdditionalInformation<ActivationDescriptor>();

    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(activationDescPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<WorkloadType>(*layer, factory);

    DescriptorType queueDescriptor = workload->GetData();

    const ActivationDescriptor* queueDescBlobPtr =
        queueDescriptor.template GetAdditionalInformation<ActivationDescriptor>();
    IgnoreUnused(queueDescBlobPtr);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(queueDescBlobPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    return workload;
}


template<typename WorkloadType,
         typename DescriptorType,
         armnn::DataType DataType>
std::unique_ptr<WorkloadType> CreateMultiplicationWithBlobWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                       armnn::Graph& graph)
{
    // Creates the layer we're testing.
    MultiplicationLayer* const layer = graph.AddLayer<MultiplicationLayer>("layer");

    auto activationDesc = std::make_shared<ActivationDescriptor>();
    activationDesc->m_A        = 10.0f;
    activationDesc->m_B        = 5.0f;
    activationDesc->m_Function = armnn::ActivationFunction::BoundedReLu;

    layer->SetAdditionalInfoForObject(activationDesc);

    // Creates extra layers.
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo({2, 3}, DataType);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Check that the additional information can be queried from the layer
    std::shared_ptr<ActivationDescriptor>
        activationDescPtr = layer->GetAdditionalInformation<ActivationDescriptor>();

    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(activationDescPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<WorkloadType>(*layer, factory);

    DescriptorType queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);
    const ActivationDescriptor* queueDescBlobPtr =
        queueDescriptor.template GetAdditionalInformation<ActivationDescriptor>();
    IgnoreUnused(queueDescBlobPtr);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(queueDescBlobPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    return workload;// Returns so we can do extra, backend-specific tests.
}

template<typename WorkloadType,
         typename DescriptorType,
         armnn::DataType DataType>
std::unique_ptr<WorkloadType> CreateAdditionWithBlobWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                 armnn::Graph& graph)
{
    // Creates the layer we're testing.
    AdditionLayer* const layer = graph.AddLayer<AdditionLayer>("layer");

    auto activationDesc = std::make_shared<ActivationDescriptor>();
    activationDesc->m_A        = 10.0f;
    activationDesc->m_B        = 5.0f;
    activationDesc->m_Function = armnn::ActivationFunction::BoundedReLu;

    layer->SetAdditionalInfoForObject(activationDesc);

    // Creates extra layers.
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo({2, 3}, DataType);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Check that the additional information can be queried from the layer
    std::shared_ptr<ActivationDescriptor>
        activationDescPtr = layer->template GetAdditionalInformation<ActivationDescriptor>();

    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(activationDescPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<WorkloadType>(*layer, factory);

    DescriptorType queueDescriptor = workload->GetData();
    const ActivationDescriptor* queueDescBlobPtr =
        queueDescriptor.template GetAdditionalInformation<ActivationDescriptor>();
    IgnoreUnused(queueDescBlobPtr);
    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(queueDescBlobPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    return workload;
}

template <typename WorkloadType,
          typename DescriptorType,
          armnn::DataType DataType>
std::unique_ptr<WorkloadType> CreateElementwiseUnaryWorkloadTest(armnn::IWorkloadFactory & factory,
                                                                 armnn::Graph & graph,
                                                                 armnn::UnaryOperation op)
{
    ElementwiseUnaryDescriptor desc = ElementwiseUnaryDescriptor(op);
    Layer* const layer = graph.AddLayer<armnn::ElementwiseUnaryLayer>(desc, "layer");

    Layer* const input  = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    armnn::TensorInfo tensorInfo({ 2, 3 }, DataType);
    Connect(input, layer, tensorInfo, 0, 0);
    Connect(layer, output, tensorInfo, 0, 0);
    CreateTensorHandles(graph, factory);

    auto workload = MakeAndCheckWorkload<WorkloadType>(*layer, factory);
    DescriptorType queueDescriptor = workload->GetData();

    CHECK(queueDescriptor.m_Inputs.size()  == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    return workload;
}

template <typename BatchNormalizationWorkloadType, armnn::DataType DataType>
std::unique_ptr<BatchNormalizationWorkloadType> CreateBatchNormalizationWorkloadTest(
    armnn::IWorkloadFactory& factory, armnn::Graph& graph, DataLayout dataLayout = DataLayout::NCHW)
{
    TensorShape tensorShape;
    switch (dataLayout)
    {
        case DataLayout::NHWC:
            tensorShape = { 2, 4, 4, 3 };
            break;
        case DataLayout::NCHW:
        default:
            tensorShape = { 2, 3, 4, 4 };
    }

    // Creates the layer we're testing.
    BatchNormalizationDescriptor layerDesc;
    layerDesc.m_Eps = 0.05f;
    layerDesc.m_DataLayout = dataLayout;

    BatchNormalizationLayer* const layer = graph.AddLayer<BatchNormalizationLayer>(layerDesc, "layer");

    armnn::TensorInfo weightInfo({3}, DataType);
    layer->m_Mean     = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Variance = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Beta     = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Gamma    = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Mean->Allocate();
    layer->m_Variance->Allocate();
    layer->m_Beta->Allocate();
    layer->m_Gamma->Allocate();

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo(tensorShape, DataType);
    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<BatchNormalizationWorkloadType>(*layer, factory);
    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_Eps == 0.05f);
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);
    CHECK((queueDescriptor.m_Mean->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Variance->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Gamma->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Beta->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename BatchNormalizationWorkloadType, armnn::DataType DataType>
std::unique_ptr<BatchNormalizationWorkloadType> CreateBatchNormalizationWithBlobWorkloadTest(
    armnn::IWorkloadFactory& factory, armnn::Graph& graph, DataLayout dataLayout = DataLayout::NCHW)
{
    TensorShape tensorShape;
    switch (dataLayout)
    {
        case DataLayout::NHWC:
            tensorShape = { 2, 4, 4, 3 };
            break;
        case DataLayout::NCHW:
        default:
            tensorShape = { 2, 3, 4, 4 };
    }

    // Creates the layer we're testing.
    BatchNormalizationDescriptor layerDesc;
    layerDesc.m_Eps = 0.05f;
    layerDesc.m_DataLayout = dataLayout;

    BatchNormalizationLayer* const layer = graph.AddLayer<BatchNormalizationLayer>(layerDesc, "layer");

    armnn::TensorInfo weightInfo({3}, DataType);
    layer->m_Mean     = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Variance = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Beta     = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Gamma    = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Mean->Allocate();
    layer->m_Variance->Allocate();
    layer->m_Beta->Allocate();
    layer->m_Gamma->Allocate();

    auto activationDesc = std::make_shared<ActivationDescriptor>();
    activationDesc->m_A        = 10.0f;
    activationDesc->m_B        = 5.0f;
    activationDesc->m_Function = armnn::ActivationFunction::BoundedReLu;

    layer->SetAdditionalInfoForObject(activationDesc);

    // Check that the additional information can be queried from the layer
    std::shared_ptr<ActivationDescriptor> activationDescPtr = layer->GetAdditionalInformation<ActivationDescriptor>();
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(activationDescPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo(tensorShape, DataType);
    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<BatchNormalizationWorkloadType>(*layer, factory);
    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    const ActivationDescriptor* queueDescBlobPtr = queueDescriptor.GetAdditionalInformation<ActivationDescriptor>();
    IgnoreUnused(queueDescBlobPtr);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(queueDescBlobPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    CHECK(queueDescriptor.m_Parameters.m_Eps == 0.05f);
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);
    CHECK((queueDescriptor.m_Mean->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Variance->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Gamma->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Beta->GetTensorInfo() == TensorInfo({3}, DataType)));
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename Convolution2dWorkload, armnn::DataType DataType>
std::unique_ptr<Convolution2dWorkload> CreateConvolution2dWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                       armnn::Graph&            graph,
                                                                       DataLayout dataLayout = DataLayout::NCHW,
                                                                       const ModelOptions& modelOptions = {})
{
    // Creates the layer we're testing.
    Convolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft = 3;
    layerDesc.m_PadRight = 3;
    layerDesc.m_PadTop = 1;
    layerDesc.m_PadBottom = 1;
    layerDesc.m_StrideX = 2;
    layerDesc.m_StrideY = 4;
    layerDesc.m_BiasEnabled = false;
    layerDesc.m_DataLayout = dataLayout;

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    Convolution2dLayer* const layer = graph.AddLayer<Convolution2dLayer>(layerDesc, "layer");

    TensorShape weightShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 5, 3} : TensorShape{2, 5, 3, 3};
    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 8, 16} : TensorShape{2, 8, 16, 3};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 2, 2, 10} : TensorShape{2, 2, 10, 2};

    armnn::TensorInfo weightsTensorInfo(weightShape, DataType, inputsQScale);
    weightsTensorInfo.SetConstant();

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    auto const weights = graph.AddLayer<ConstantLayer>("weights");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightsTensorInfo);
    weights->m_LayerOutput->Allocate();

    // Connects up.
    Connect(input, layer, TensorInfo(inputShape, DataType, inputsQScale));
    Connect(weights, layer, weightsTensorInfo, 0, 1);
    Connect(layer, output, TensorInfo(outputShape, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<Convolution2dWorkload>(*layer, factory, modelOptions);

    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_StrideX == 2);
    CHECK(queueDescriptor.m_Parameters.m_StrideY == 4);
    CHECK(queueDescriptor.m_Parameters.m_PadLeft == 3);
    CHECK(queueDescriptor.m_Parameters.m_PadRight == 3);
    CHECK(queueDescriptor.m_Parameters.m_PadTop == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadBottom == 1);
    CHECK(!queueDescriptor.m_Parameters.m_BiasEnabled);
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template<typename Convolution2dWorkload, armnn::DataType DataType>
std::unique_ptr<Convolution2dWorkload> CreateConvolution2dFusedActivationWithBlobWorkloadTest(
    armnn::IWorkloadFactory& factory,
    armnn::Graph& graph,
    DataLayout dataLayout = DataLayout::NCHW,
    const ModelOptions& modelOptions = {})
{
    // Creates the layer we're testing.
    Convolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft = 3;
    layerDesc.m_PadRight = 3;
    layerDesc.m_PadTop = 1;
    layerDesc.m_PadBottom = 1;
    layerDesc.m_StrideX = 2;
    layerDesc.m_StrideY = 4;
    layerDesc.m_BiasEnabled = true;
    layerDesc.m_DataLayout = dataLayout;

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    Convolution2dLayer* const layer = graph.AddLayer<Convolution2dLayer>(layerDesc, "layer");

    TensorShape weightShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 5, 3} : TensorShape{2, 5, 3, 3};
    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 8, 16} : TensorShape{2, 8, 16, 3};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 2, 2, 10} : TensorShape{2, 2, 10, 2};

    armnn::TensorInfo weightsTensorInfo(weightShape, DataType, inputsQScale);
    weightsTensorInfo.SetConstant();
    armnn::TensorInfo biasTensorInfo({2}, DataType, inputsQScale);
    biasTensorInfo.SetConstant();

    auto activationDesc = std::make_shared<ActivationDescriptor>();
    activationDesc->m_A        = 10.0f;
    activationDesc->m_B        = 5.0f;
    activationDesc->m_Function = armnn::ActivationFunction::BoundedReLu;

    layer->SetAdditionalInfoForObject(activationDesc);

    // Check that the additional information can be queried from the layer
    std::shared_ptr<ActivationDescriptor> activationDescPtr = layer->GetAdditionalInformation<ActivationDescriptor>();

    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(activationDescPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    auto const weights = graph.AddLayer<ConstantLayer>("weights");
    auto const bias = graph.AddLayer<ConstantLayer>("bias");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightsTensorInfo);
    weights->m_LayerOutput->Allocate();
    bias->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasTensorInfo);
    bias->m_LayerOutput->Allocate();

    // Connects up.
    Connect(input, layer, TensorInfo(inputShape, DataType, inputsQScale));
    Connect(weights, layer, weightsTensorInfo, 0, 1);
    Connect(bias, layer, biasTensorInfo, 0, 2);
    Connect(layer, output, TensorInfo(outputShape, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<Convolution2dWorkload>(*layer, factory, modelOptions);

    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    const ActivationDescriptor* queueDescBlobPtr = queueDescriptor.GetAdditionalInformation<ActivationDescriptor>();
    IgnoreUnused(queueDescBlobPtr);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(queueDescBlobPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    CHECK(queueDescriptor.m_Parameters.m_StrideX == 2);
    CHECK(queueDescriptor.m_Parameters.m_StrideY == 4);
    CHECK(queueDescriptor.m_Parameters.m_PadLeft == 3);
    CHECK(queueDescriptor.m_Parameters.m_PadRight == 3);
    CHECK(queueDescriptor.m_Parameters.m_PadTop == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadBottom == 1);
    CHECK(queueDescriptor.m_Parameters.m_BiasEnabled);
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    CHECK(queueDescriptor.m_Outputs.size() == 1);
    CHECK(queueDescriptor.m_Inputs.size() == 3);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename Convolution2dWorkload, armnn::DataType DataType>
std::unique_ptr<Convolution2dWorkload> CreateConvolution2dWorkloadFastMathTest(armnn::IWorkloadFactory& factory,
                                                                               armnn::Graph&            graph,
                                                                               DataLayout dataLayout = DataLayout::NCHW,
                                                                               const ModelOptions& modelOptions = {})
{
    // Creates the layer we're testing.
    Convolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft = 0;
    layerDesc.m_PadRight = 0;
    layerDesc.m_PadTop = 0;
    layerDesc.m_PadBottom = 0;
    layerDesc.m_StrideX = 1;
    layerDesc.m_StrideY = 1;
    layerDesc.m_BiasEnabled = true;
    layerDesc.m_DataLayout = dataLayout;

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    Convolution2dLayer* const layer = graph.AddLayer<Convolution2dLayer>(layerDesc, "layer");

    TensorShape weightShape = TensorShape{ 32, 32, 3, 3 };
    TensorShape biasShape = TensorShape{ 32 };
    TensorShape inputShape = TensorShape{ 1, 32, 149, 149 };
    TensorShape outputShape = TensorShape{ 1, 32, 147, 147 };

    armnn::TensorInfo weightsTensorInfo(weightShape, DataType, inputsQScale);
    weightsTensorInfo.SetConstant();
    armnn::TensorInfo biasTensorInfo(biasShape, DataType, inputsQScale);
    biasTensorInfo.SetConstant();

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    auto const weights = graph.AddLayer<ConstantLayer>("weights");
    auto const bias = graph.AddLayer<ConstantLayer>("bias");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    Connect(input, layer, TensorInfo(inputShape, DataType));
    Connect(weights, layer, weightsTensorInfo, 0, 1);
    Connect(bias, layer, biasTensorInfo, 0, 2);
    Connect(layer, output, TensorInfo(outputShape, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<Convolution2dWorkload>(*layer, factory, modelOptions);

    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_StrideX == 1);
    CHECK(queueDescriptor.m_Parameters.m_StrideY == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadLeft == 0);
    CHECK(queueDescriptor.m_Parameters.m_PadRight == 0);
    CHECK(queueDescriptor.m_Parameters.m_PadTop == 0);
    CHECK(queueDescriptor.m_Parameters.m_PadBottom == 0);
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    CHECK(queueDescriptor.m_Inputs.size() == 3);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename LstmWorkload>
std::unique_ptr<LstmWorkload> CreateLstmWorkloadTest(armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // This parameter setting is for withCifgWithPeepholeNoProjection
    LstmDescriptor layerDesc;
    layerDesc.m_ActivationFunc = 4;
    layerDesc.m_ClippingThresCell = 0.0f;
    layerDesc.m_ClippingThresProj = 0.0f;
    layerDesc.m_CifgEnabled = true;
    layerDesc.m_PeepholeEnabled = true;
    layerDesc.m_ProjectionEnabled = false;

    LstmLayer* const layer = graph.AddLayer<LstmLayer>(layerDesc, "layer");
    unsigned int batchSize = 2;
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


    if (layerDesc.m_PeepholeEnabled)
    {
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

    CreateTensorHandles(graph, factory);

    // make the workload and check it
    auto workload = MakeAndCheckWorkload<LstmWorkload>(*layer, factory);
    LstmQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_ActivationFunc == 4);
    CHECK(queueDescriptor.m_Parameters.m_ClippingThresCell == 0.0f);
    CHECK(queueDescriptor.m_Parameters.m_ClippingThresProj == 0.0f);
    CHECK(queueDescriptor.m_Inputs.size() == 3);
    CHECK(queueDescriptor.m_Outputs.size() == 4);

    CHECK((queueDescriptor.m_InputToForgetWeights->GetTensorInfo() == TensorInfo({ numUnits, inputSize },
                                                                                     DataType::Float32)));
    CHECK((queueDescriptor.m_OutputGateBias->GetTensorInfo() == TensorInfo({ numUnits },
                                                                                     DataType::Float32)));
    CHECK((queueDescriptor.m_CellBias->GetTensorInfo() == TensorInfo({ numUnits }, DataType::Float32)));
    return workload;
}

template <typename QuantizedLstmWorkload>
std::unique_ptr<QuantizedLstmWorkload> CreateQuantizedLstmWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                       armnn::Graph& graph)
{
    auto layer = graph.AddLayer<QuantizedLstmLayer>("quantizedLstmlayer");
    unsigned int numBatches = 2;
    unsigned int inputSize = 2;
    unsigned int outputSize = 4;

    // Scale/Offset for input/output, cellState In/Out, weights, bias
    float inputOutputScale = 0.0078125f;
    int32_t inputOutputOffset = 128;

    float cellStateScale = 0.00048828125f;
    int32_t cellStateOffset = 0;

    float weightsScale = 0.00408021f;
    int32_t weightsOffset = 100;

    float biasScale = 3.1876640625e-05f;
    int32_t biasOffset = 0;

    // Weights and bias tensor and quantization info
    armnn::TensorInfo inputWeightsInfo({outputSize, inputSize},
                                       armnn::DataType::QAsymmU8,
                                       weightsScale,
                                       weightsOffset);

    armnn::TensorInfo recurrentWeightsInfo({outputSize, outputSize},
                                           armnn::DataType::QAsymmU8,
                                           weightsScale,
                                           weightsOffset);

    armnn::TensorInfo biasInfo({outputSize},
                               armnn::DataType::Signed32,
                               biasScale,
                               biasOffset);

    // Weights and bias
    layer->m_QuantizedLstmParameters.m_InputToInputWeights =
            std::make_unique<ScopedTensorHandle>(inputWeightsInfo);
    layer->m_QuantizedLstmParameters.m_InputToForgetWeights =
            std::make_unique<ScopedTensorHandle>(inputWeightsInfo);
    layer->m_QuantizedLstmParameters.m_InputToCellWeights =
            std::make_unique<ScopedTensorHandle>(inputWeightsInfo);
    layer->m_QuantizedLstmParameters.m_InputToOutputWeights =
            std::make_unique<ScopedTensorHandle>(inputWeightsInfo);

    layer->m_QuantizedLstmParameters.m_RecurrentToInputWeights =
            std::make_unique<ScopedTensorHandle>(recurrentWeightsInfo);
    layer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights =
            std::make_unique<ScopedTensorHandle>(recurrentWeightsInfo);
    layer->m_QuantizedLstmParameters.m_RecurrentToCellWeights =
            std::make_unique<ScopedTensorHandle>(recurrentWeightsInfo);
    layer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights =
            std::make_unique<ScopedTensorHandle>(recurrentWeightsInfo);

    layer->m_QuantizedLstmParameters.m_InputGateBias = std::make_unique<ScopedTensorHandle>(biasInfo);
    layer->m_QuantizedLstmParameters.m_ForgetGateBias = std::make_unique<ScopedTensorHandle>(biasInfo);
    layer->m_QuantizedLstmParameters.m_CellBias = std::make_unique<ScopedTensorHandle>(biasInfo);
    layer->m_QuantizedLstmParameters.m_OutputGateBias = std::make_unique<ScopedTensorHandle>(biasInfo);

    // Allocate weights and bias
    layer->m_QuantizedLstmParameters.m_InputToInputWeights->Allocate();
    layer->m_QuantizedLstmParameters.m_InputToForgetWeights->Allocate();
    layer->m_QuantizedLstmParameters.m_InputToCellWeights->Allocate();
    layer->m_QuantizedLstmParameters.m_InputToOutputWeights->Allocate();

    layer->m_QuantizedLstmParameters.m_RecurrentToInputWeights->Allocate();
    layer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights->Allocate();
    layer->m_QuantizedLstmParameters.m_RecurrentToCellWeights->Allocate();
    layer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights->Allocate();

    layer->m_QuantizedLstmParameters.m_InputGateBias->Allocate();
    layer->m_QuantizedLstmParameters.m_ForgetGateBias->Allocate();
    layer->m_QuantizedLstmParameters.m_CellBias->Allocate();
    layer->m_QuantizedLstmParameters.m_OutputGateBias->Allocate();

    // Create input and output layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const cellStateIn = graph.AddLayer<InputLayer>(1, "cellStateIn");
    Layer* const outputStateIn = graph.AddLayer<InputLayer>(2, "outputStateIn");

    Layer* const cellStateOut = graph.AddLayer<OutputLayer>(0, "cellStateOut");
    Layer* const outputStateOut = graph.AddLayer<OutputLayer>(1, "outputStateOut");

    // Input/output tensor info and quantization info
    armnn::TensorInfo inputInfo({numBatches , inputSize},
                                armnn::DataType::QAsymmU8,
                                inputOutputScale,
                                inputOutputOffset);

    armnn::TensorInfo cellStateInfo({numBatches , outputSize},
                                    armnn::DataType::QSymmS16,
                                    cellStateScale,
                                    cellStateOffset);

    armnn::TensorInfo outputStateInfo({numBatches , outputSize},
                                      armnn::DataType::QAsymmU8,
                                      inputOutputScale,
                                      inputOutputOffset);

    // Connect input/output slots
    Connect(input, layer, inputInfo, 0, 0);
    Connect(cellStateIn, layer, cellStateInfo, 0, 1);
    Connect(outputStateIn, layer, outputStateInfo, 0, 2);

    Connect(layer, cellStateOut, cellStateInfo, 0, 0);
    Connect(layer, outputStateOut, outputStateInfo, 1, 0);

    CreateTensorHandles(graph, factory);

    // Create workload and check layer support
    auto workload = MakeAndCheckWorkload<QuantizedLstmWorkload>(*layer, factory);
    QuantizedLstmQueueDescriptor queueDescriptor = workload->GetData();

    // Validate input/output sizes
    CHECK(queueDescriptor.m_Inputs.size() == 3);
    CHECK(queueDescriptor.m_Outputs.size() == 2);

    // Validate weight tensor info
    CHECK((queueDescriptor.m_InputToInputWeights->GetTensorInfo() == inputWeightsInfo));
    CHECK((queueDescriptor.m_InputToForgetWeights->GetTensorInfo() == inputWeightsInfo));
    CHECK((queueDescriptor.m_InputToCellWeights->GetTensorInfo() == inputWeightsInfo));
    CHECK((queueDescriptor.m_InputToOutputWeights->GetTensorInfo() == inputWeightsInfo));

    CHECK((queueDescriptor.m_RecurrentToInputWeights->GetTensorInfo() == recurrentWeightsInfo));
    CHECK((queueDescriptor.m_RecurrentToForgetWeights->GetTensorInfo() == recurrentWeightsInfo));
    CHECK((queueDescriptor.m_RecurrentToCellWeights->GetTensorInfo() == recurrentWeightsInfo));
    CHECK((queueDescriptor.m_RecurrentToOutputWeights->GetTensorInfo() == recurrentWeightsInfo));

    CHECK((queueDescriptor.m_InputGateBias->GetTensorInfo() == biasInfo));
    CHECK((queueDescriptor.m_ForgetGateBias->GetTensorInfo() == biasInfo));
    CHECK((queueDescriptor.m_CellBias->GetTensorInfo() == biasInfo));
    CHECK((queueDescriptor.m_OutputGateBias->GetTensorInfo() == biasInfo));

    return workload;
}

template <typename QLstmWorkload>
std::unique_ptr<QLstmWorkload> CreateQLstmWorkloadTest(armnn::IWorkloadFactory& factory,
                                                       armnn::Graph& graph)
{
    QLstmDescriptor layerDesc;
    layerDesc.m_CifgEnabled       = true;
    layerDesc.m_PeepholeEnabled   = false;
    layerDesc.m_ProjectionEnabled = false;
    layerDesc.m_LayerNormEnabled  = true;

    layerDesc.m_CellClip       = 0.0f;
    layerDesc.m_ProjectionClip = 0.0f;

    layerDesc.m_HiddenStateZeroPoint = 0;
    layerDesc.m_HiddenStateScale     = 0.007f;

    layerDesc.m_InputIntermediateScale  = 0.007059f;
    layerDesc.m_ForgetIntermediateScale = 0.007812f;
    layerDesc.m_CellIntermediateScale   = 0.007059f;
    layerDesc.m_OutputIntermediateScale = 0.007812f;

    QLstmLayer* const layer = graph.AddLayer<QLstmLayer>(layerDesc, "qLstm");

    unsigned int numBatches = 2;
    unsigned int inputSize  = 4;
    unsigned int numUnits   = 4;
    unsigned int outputSize = 4;

    // Scale/Offset quantization info
    float inputScale    = 0.0078125f;
    int32_t inputOffset = 0;

    // if (!projectionEnabled) outputScale == hiddenStateScale
    float outputScale    = layerDesc.m_HiddenStateScale;
    int32_t outputOffset = layerDesc.m_HiddenStateZeroPoint;

    float cellStateScale    = 3.05176e-05f;
    int32_t cellStateOffset = 0;

    float weightsScale    = 0.00784314f;
    int32_t weightsOffset = 0;

    float layerNormScale    = 3.05182e-05f;
    int32_t layerNormOffset = 0;

    float biasScale    = layerNormScale / 1024;
    int32_t biasOffset = 0;

    // Weights and bias tensor and quantization info
    armnn::TensorInfo inputWeightsInfo({outputSize, inputSize},
                                       armnn::DataType::QSymmS8,
                                       weightsScale,
                                       weightsOffset);

    armnn::TensorInfo recurrentWeightsInfo({outputSize, outputSize},
                                           armnn::DataType::QSymmS8,
                                           weightsScale,
                                           weightsOffset);

    armnn::TensorInfo biasInfo({outputSize}, armnn::DataType::Signed32, biasScale, biasOffset);

    armnn::TensorInfo layerNormWeightsInfo({numUnits}, armnn::DataType::QSymmS16, layerNormScale, layerNormOffset);

    // Create and allocate tensors
    layer->m_BasicParameters.m_InputToForgetWeights = std::make_unique<ScopedTensorHandle>(inputWeightsInfo);
    layer->m_BasicParameters.m_InputToCellWeights = std::make_unique<ScopedTensorHandle>(inputWeightsInfo);
    layer->m_BasicParameters.m_InputToOutputWeights = std::make_unique<ScopedTensorHandle>(inputWeightsInfo);

    layer->m_BasicParameters.m_RecurrentToForgetWeights =
            std::make_unique<ScopedTensorHandle>(recurrentWeightsInfo);
    layer->m_BasicParameters.m_RecurrentToCellWeights =
            std::make_unique<ScopedTensorHandle>(recurrentWeightsInfo);
    layer->m_BasicParameters.m_RecurrentToOutputWeights =
            std::make_unique<ScopedTensorHandle>(recurrentWeightsInfo);

    layer->m_BasicParameters.m_ForgetGateBias = std::make_unique<ScopedTensorHandle>(biasInfo);
    layer->m_BasicParameters.m_CellBias = std::make_unique<ScopedTensorHandle>(biasInfo);
    layer->m_BasicParameters.m_OutputGateBias = std::make_unique<ScopedTensorHandle>(biasInfo);

    layer->m_LayerNormParameters.m_ForgetLayerNormWeights =
            std::make_unique<ScopedTensorHandle>(layerNormWeightsInfo);
    layer->m_LayerNormParameters.m_CellLayerNormWeights =
            std::make_unique<ScopedTensorHandle>(layerNormWeightsInfo);
    layer->m_LayerNormParameters.m_OutputLayerNormWeights =
            std::make_unique<ScopedTensorHandle>(layerNormWeightsInfo);

    layer->m_BasicParameters.m_InputToForgetWeights->Allocate();
    layer->m_BasicParameters.m_InputToCellWeights->Allocate();
    layer->m_BasicParameters.m_InputToOutputWeights->Allocate();

    layer->m_BasicParameters.m_RecurrentToForgetWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToCellWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToOutputWeights->Allocate();

    layer->m_BasicParameters.m_ForgetGateBias->Allocate();
    layer->m_BasicParameters.m_CellBias->Allocate();
    layer->m_BasicParameters.m_OutputGateBias->Allocate();

    layer->m_LayerNormParameters.m_ForgetLayerNormWeights->Allocate();
    layer->m_LayerNormParameters.m_CellLayerNormWeights->Allocate();
    layer->m_LayerNormParameters.m_OutputLayerNormWeights->Allocate();

    // Input and output layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const outputStateIn = graph.AddLayer<InputLayer>(1, "outputStateIn");
    Layer* const cellStateIn = graph.AddLayer<InputLayer>(2, "cellStateIn");

    Layer* const outputStateOut = graph.AddLayer<OutputLayer>(0, "outputStateOut");
    Layer* const cellStateOut = graph.AddLayer<OutputLayer>(1, "cellStateOut");
    Layer* const output = graph.AddLayer<OutputLayer>(2, "output");

    // Input/Output tensor info
    armnn::TensorInfo inputInfo({numBatches , inputSize},
                                armnn::DataType::QAsymmS8,
                                inputScale,
                                inputOffset);

    armnn::TensorInfo cellStateInfo({numBatches , numUnits},
                                    armnn::DataType::QSymmS16,
                                    cellStateScale,
                                    cellStateOffset);

    armnn::TensorInfo outputStateInfo({numBatches , outputSize},
                                      armnn::DataType::QAsymmS8,
                                      outputScale,
                                      outputOffset);

    // Connect layers to slots
    Connect(input, layer, inputInfo, 0, 0);
    Connect(outputStateIn, layer, outputStateInfo, 0, 1);
    Connect(cellStateIn, layer, cellStateInfo, 0, 2);

    Connect(layer, outputStateOut, outputStateInfo, 0, 0);
    Connect(layer, cellStateOut, cellStateInfo, 1, 0);
    Connect(layer, output, outputStateInfo, 2, 0);

    CreateTensorHandles(graph, factory);

    // Create and check workload
    auto workload = MakeAndCheckWorkload<QLstmWorkload>(*layer, factory);
    QLstmQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_CellClip == 0.0f);
    CHECK(queueDescriptor.m_Parameters.m_ProjectionClip == 0.0f);
    CHECK(queueDescriptor.m_Inputs.size() == 3);
    CHECK(queueDescriptor.m_Outputs.size() == 3);

    CHECK((queueDescriptor.m_InputToForgetWeights->GetTensorInfo() == inputWeightsInfo));
    CHECK((queueDescriptor.m_InputToCellWeights->GetTensorInfo() == inputWeightsInfo));
    CHECK((queueDescriptor.m_InputToOutputWeights->GetTensorInfo() == inputWeightsInfo));

    CHECK((queueDescriptor.m_RecurrentToForgetWeights->GetTensorInfo() == recurrentWeightsInfo));
    CHECK((queueDescriptor.m_RecurrentToCellWeights->GetTensorInfo() == recurrentWeightsInfo));
    CHECK((queueDescriptor.m_RecurrentToOutputWeights->GetTensorInfo() == recurrentWeightsInfo));

    CHECK((queueDescriptor.m_ForgetGateBias->GetTensorInfo() == biasInfo));
    CHECK((queueDescriptor.m_CellBias->GetTensorInfo() == biasInfo));
    CHECK((queueDescriptor.m_OutputGateBias->GetTensorInfo() == biasInfo));

    return workload;
}

template<typename Convolution2dWorkload, armnn::DataType DataType>
std::unique_ptr<Convolution2dWorkload> CreateDirectConvolution2dWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                             armnn::Graph& graph)
{
    // Creates the layer we're testing.
    Convolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft = 1;
    layerDesc.m_PadRight = 1;
    layerDesc.m_PadTop = 1;
    layerDesc.m_PadBottom = 1;
    layerDesc.m_StrideX = 1;
    layerDesc.m_StrideY = 1;
    layerDesc.m_BiasEnabled = true;

    Convolution2dLayer* const layer = graph.AddLayer<Convolution2dLayer>(layerDesc, "layer");

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    TensorShape biasShape = TensorShape{ 2 };
    TensorShape weightShape = TensorShape{ 2, 3, 3, 3 };
    armnn::TensorInfo weightsTensorInfo(weightShape, DataType, inputsQScale);
    weightsTensorInfo.SetConstant();
    armnn::TensorInfo biasTensorInfo(biasShape, GetBiasDataType(DataType), inputsQScale);
    biasTensorInfo.SetConstant();

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    auto const weights = graph.AddLayer<ConstantLayer>("weights");
    auto const bias = graph.AddLayer<ConstantLayer>("bias");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightsTensorInfo);
    weights->m_LayerOutput->Allocate();
    bias->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasTensorInfo);
    bias->m_LayerOutput->Allocate();

    // Connects up.
    Connect(input, layer, TensorInfo({2, 3, 6, 6}, DataType, inputsQScale));
    Connect(weights, layer, weightsTensorInfo, 0, 1);
    Connect(bias, layer, biasTensorInfo, 0, 2);
    Connect(layer, output, TensorInfo({2, 2, 6, 6}, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<Convolution2dWorkload>(*layer, factory);

    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_StrideX == 1);
    CHECK(queueDescriptor.m_Parameters.m_StrideY == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadLeft == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadRight == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadTop == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadBottom == 1);
    CHECK(queueDescriptor.m_Parameters.m_BiasEnabled == true);

    CHECK(queueDescriptor.m_Inputs.size() == 3);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename DepthwiseConvolution2dFloat32Workload, armnn::DataType DataType>
std::unique_ptr<DepthwiseConvolution2dFloat32Workload> CreateDepthwiseConvolution2dWorkloadTest(
    armnn::IWorkloadFactory& factory, armnn::Graph& graph, DataLayout dataLayout = DataLayout::NCHW)
{
    // Creates the layer we're testing.
    DepthwiseConvolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft     = 1;
    layerDesc.m_PadRight    = 2;
    layerDesc.m_PadTop      = 1;
    layerDesc.m_PadBottom   = 2;
    layerDesc.m_StrideX     = 1;
    layerDesc.m_StrideY     = 1;
    layerDesc.m_BiasEnabled = false;
    layerDesc.m_DataLayout  = dataLayout;

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    TensorShape weightShape({1, 4, 4, 2});
    TensorShape inputShape = (dataLayout == DataLayout::NCHW) ?
                             TensorShape{ 2, 2, 5, 5 } : TensorShape{ 2, 5, 5, 2 };
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ?
                              TensorShape{ 2, 2, 5, 5 } : TensorShape{ 2, 5, 5, 2 };

    DepthwiseConvolution2dLayer* const layer = graph.AddLayer<DepthwiseConvolution2dLayer>(layerDesc, "layer");


    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const weights = graph.AddLayer<ConstantLayer>("weights");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    Connect(input, layer, TensorInfo(inputShape, DataType, inputsQScale));
    Connect(weights, layer, TensorInfo(weightShape, DataType, inputsQScale, 0.0f, true), 0, 1);
    Connect(layer, output, TensorInfo(outputShape, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<DepthwiseConvolution2dFloat32Workload>(*layer, factory);

    DepthwiseConvolution2dQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_StrideX == 1);
    CHECK(queueDescriptor.m_Parameters.m_StrideY == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadLeft == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadRight == 2);
    CHECK(queueDescriptor.m_Parameters.m_PadTop == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadBottom == 2);
    CHECK(queueDescriptor.m_Parameters.m_BiasEnabled == false);
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename FullyConnectedWorkload, armnn::DataType DataType>
std::unique_ptr<FullyConnectedWorkload> CreateFullyConnectedWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                         armnn::Graph&            graph)
{
    // Creates the layer we're testing.
    FullyConnectedDescriptor layerDesc;
    layerDesc.m_BiasEnabled = false;
    layerDesc.m_TransposeWeightMatrix = true;

    FullyConnectedLayer* const layer = graph.AddLayer<FullyConnectedLayer>(layerDesc, "layer");

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    armnn::TensorInfo weightsTensorInfo({7, 20}, DataType, inputsQScale);
    weightsTensorInfo.SetConstant();

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    auto const weights = graph.AddLayer<ConstantLayer>("weights");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightsTensorInfo);
    weights->m_LayerOutput->Allocate();

    // Connects up.
    Connect(input, layer, TensorInfo({3, 1, 4, 5}, DataType, inputsQScale), 0, 0);
    Connect(weights, layer, weightsTensorInfo, 0, 1);
    Connect(layer, output, TensorInfo({3, 7}, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<FullyConnectedWorkload>(*layer, factory);

    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_TransposeWeightMatrix == true);

    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename FullyConnectedWorkload, armnn::DataType DataType>
std::unique_ptr<FullyConnectedWorkload> CreateFullyConnectedWithBlobWorkloadTest
    (armnn::IWorkloadFactory& factory,
     armnn::Graph& graph)
{
    // Creates the layer we're testing.
    FullyConnectedDescriptor layerDesc;
    layerDesc.m_BiasEnabled = true;
    layerDesc.m_TransposeWeightMatrix = true;

    FullyConnectedLayer* const layer = graph.AddLayer<FullyConnectedLayer>(layerDesc, "layer");

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    armnn::TensorInfo weightsTensorInfo({7, 20}, DataType, inputsQScale);
    armnn::TensorInfo biasesTensorInfo({7}, GetBiasDataType(DataType), inputsQScale);
    weightsTensorInfo.SetConstant();
    biasesTensorInfo.SetConstant();

    auto activationDesc = std::make_shared<ActivationDescriptor>();
    activationDesc->m_A        = 10.0f;
    activationDesc->m_B        = 5.0f;
    activationDesc->m_Function = armnn::ActivationFunction::BoundedReLu;

    layer->SetAdditionalInfoForObject(activationDesc);

    // Check that the additional information can be queried from the layer
    std::shared_ptr<ActivationDescriptor> activationDescPtr = layer->GetAdditionalInformation<ActivationDescriptor>();
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(activationDescPtr->m_B) == 5.0f);
    ARMNN_ASSERT(static_cast<ActivationFunction>(activationDescPtr->m_Function) ==
        armnn::ActivationFunction::BoundedReLu);

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    auto const weights = graph.AddLayer<ConstantLayer>("weights");
    auto const biases = graph.AddLayer<ConstantLayer>("biases");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightsTensorInfo);
    weights->m_LayerOutput->Allocate();
    biases->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasesTensorInfo);
    biases->m_LayerOutput->Allocate();

    // Connects up.
    Connect(input, layer, TensorInfo({3, 1, 4, 5}, DataType, inputsQScale), 0, 0);
    Connect(weights, layer, weightsTensorInfo, 0, 1);
    Connect(biases, layer, biasesTensorInfo, 0, 2);
    Connect(layer, output, TensorInfo({3, 7}, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<FullyConnectedWorkload>(*layer, factory);

    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();

    const ActivationDescriptor* queueDescBlobPtr = queueDescriptor.GetAdditionalInformation<ActivationDescriptor>();
    IgnoreUnused(queueDescBlobPtr);

    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_A) == 10.0f);
    ARMNN_ASSERT(static_cast<float>(queueDescBlobPtr->m_B) == 5.0f);
    ARMNN_ASSERT(
        static_cast<ActivationFunction>(queueDescBlobPtr->m_Function) == armnn::ActivationFunction::BoundedReLu
    );

    CHECK(queueDescriptor.m_Parameters.m_BiasEnabled == true);
    CHECK(queueDescriptor.m_Parameters.m_TransposeWeightMatrix == true);
    CHECK(queueDescriptor.m_Inputs.size() == 3);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename FullyConnectedWorkload, armnn::DataType DataType>
std::unique_ptr<FullyConnectedWorkload> CreateFullyConnectedWorkloadWeightsBiasesAsInputsTest
    (armnn::IWorkloadFactory& factory,
     armnn::Graph&            graph)
{
    // Creates the layer we're testing.
    FullyConnectedDescriptor layerDesc;
    layerDesc.m_BiasEnabled = true;
    layerDesc.m_TransposeWeightMatrix = true;
    layerDesc.m_ConstantWeights = false;

    FullyConnectedLayer* const layer = graph.AddLayer<FullyConnectedLayer>(layerDesc, "layer");

    float inputsQScale = 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0;

    // Creates extra layers with weights and biases as input layers.
    Layer* const input   = graph.AddLayer<InputLayer>(1, "input");
    Layer* const weights = graph.AddLayer<InputLayer>(2, "weights");
    Layer* const biases  = graph.AddLayer<InputLayer>(3, "biases");
    Layer* const output  = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    Connect(input, layer, TensorInfo({3, 1, 4, 5}, DataType, inputsQScale), 0, 0);
    Connect(weights, layer, TensorInfo({7, 20}, DataType, inputsQScale), 0, 1);
    Connect(biases, layer, TensorInfo({7}, GetBiasDataType(DataType), inputsQScale), 0, 2);
    Connect(layer, output, TensorInfo({3, 7}, DataType, outputQScale));
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<FullyConnectedWorkload>(*layer, factory);

    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();

    CHECK(queueDescriptor.m_Parameters.m_BiasEnabled == true);
    CHECK(queueDescriptor.m_Parameters.m_TransposeWeightMatrix == true);
    CHECK(queueDescriptor.m_Parameters.m_ConstantWeights == false);
    CHECK(queueDescriptor.m_Inputs.size() == 3);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}


template <typename NormalizationWorkload, armnn::DataType DataType>
std::unique_ptr<NormalizationWorkload> CreateNormalizationWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                       armnn::Graph& graph,
                                                                       DataLayout dataLayout = DataLayout::NCHW)
{
    // Creates the layer we're testing.
    NormalizationDescriptor layerDesc;
    layerDesc.m_NormChannelType = NormalizationAlgorithmChannel::Across;
    layerDesc.m_NormMethodType = NormalizationAlgorithmMethod::LocalBrightness;
    layerDesc.m_NormSize = 3;
    layerDesc.m_Alpha = 0.5f;
    layerDesc.m_Beta = -1.0f;
    layerDesc.m_K = 0.2f;
    layerDesc.m_DataLayout = dataLayout;

    NormalizationLayer* layer = graph.AddLayer<NormalizationLayer>(layerDesc, "layer");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    TensorShape inputShape = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 3, 5, 5, 1 } : TensorShape{ 3, 1, 5, 5 };
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 3, 5, 5, 1 } : TensorShape{ 3, 1, 5, 5 };

    // Connects up.
    armnn::TensorInfo inputTensorInfo(inputShape, DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<NormalizationWorkload>(*layer, factory);

    NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    CHECK((queueDescriptor.m_Parameters.m_NormChannelType == NormalizationAlgorithmChannel::Across));
    CHECK((queueDescriptor.m_Parameters.m_NormMethodType == NormalizationAlgorithmMethod::LocalBrightness));
    CHECK(queueDescriptor.m_Parameters.m_NormSize == 3);
    CHECK(queueDescriptor.m_Parameters.m_Alpha == 0.5f);
    CHECK(queueDescriptor.m_Parameters.m_Beta == -1.0f);
    CHECK(queueDescriptor.m_Parameters.m_K == 0.2f);
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename Pooling2dWorkload, armnn::DataType DataType>
std::unique_ptr<Pooling2dWorkload> CreatePooling2dWorkloadTest(armnn::IWorkloadFactory& factory,
                                                               armnn::Graph&            graph,
                                                               DataLayout dataLayout = DataLayout::NCHW)
{
    // Creates the layer we're testing.
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
    layerDesc.m_DataLayout = dataLayout;

    Pooling2dLayer* const layer = graph.AddLayer<Pooling2dLayer>(layerDesc, "layer");

    // Create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 2, 5, 5} : TensorShape{3, 5, 5, 2};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 2, 2, 4} : TensorShape{3, 2, 4, 2};

    // Connect up
    Connect(input, layer, TensorInfo(inputShape, DataType));
    Connect(layer, output, TensorInfo(outputShape, DataType));
    CreateTensorHandles(graph, factory);

    // Make the workload and checks it
    auto workload = MakeAndCheckWorkload<Pooling2dWorkload>(*layer, factory);

    Pooling2dQueueDescriptor queueDescriptor = workload->GetData();
    CHECK((queueDescriptor.m_Parameters.m_PoolType == PoolingAlgorithm::Average));
    CHECK((queueDescriptor.m_Parameters.m_OutputShapeRounding == OutputShapeRounding::Floor));
    CHECK(queueDescriptor.m_Parameters.m_PoolWidth == 3);
    CHECK(queueDescriptor.m_Parameters.m_PoolHeight == 3);
    CHECK(queueDescriptor.m_Parameters.m_StrideX == 2);
    CHECK(queueDescriptor.m_Parameters.m_StrideY == 3);
    CHECK(queueDescriptor.m_Parameters.m_PadLeft == 2);
    CHECK(queueDescriptor.m_Parameters.m_PadRight == 2);
    CHECK(queueDescriptor.m_Parameters.m_PadTop == 1);
    CHECK(queueDescriptor.m_Parameters.m_PadBottom == 1);
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));

    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Return so we can do extra, backend-specific tests
    return workload;
}

template <typename SoftmaxWorkload, armnn::DataType DataType>
std::unique_ptr<SoftmaxWorkload> CreateSoftmaxWorkloadTest(armnn::IWorkloadFactory& factory,
                                                           armnn::Graph&            graph)
{
    // Create the layer we're testing.
    SoftmaxDescriptor softmaxDescriptor;
    // Set Axis to -1 if CL or Neon until further Axes are supported.
    if (factory.GetBackendId() == armnn::Compute::CpuAcc || factory.GetBackendId() == armnn::Compute::GpuAcc)
    {
        softmaxDescriptor.m_Axis = -1;
    }

    Layer* const layer = graph.AddLayer<SoftmaxLayer>(softmaxDescriptor, "layer");
    // Create extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up
    armnn::TensorInfo tensorInfo({4, 1}, DataType);
    if (DataType == armnn::DataType::QAsymmU8)
    {
        tensorInfo.SetQuantizationOffset(0);
        tensorInfo.SetQuantizationScale(1.f / 256);
    }
    else if (DataType == armnn::DataType::QAsymmS8)
    {
        tensorInfo.SetQuantizationOffset(-128);
        tensorInfo.SetQuantizationScale(1.f / 256);
    }

    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Make the workload and checks it.
    auto workload = MakeAndCheckWorkload<SoftmaxWorkload>(*layer, factory);

    SoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Return so we can do extra, backend-specific tests.
    return workload;
}

template<typename SplitterWorkload, armnn::DataType DataType>
std::unique_ptr<SplitterWorkload>
    CreateSplitterWorkloadTest(armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // Create the layer we're testing.
    // NOTE: need three dimensions channels, height/y, width/x because the Compute
    //       library restricts subtensors to have the same x and y dimensions as
    //       their parent tensors, and therefore the origin on the x and y dimension
    //       has to be zero for any view. So we need a third dimension to split...
    // NOTE: arguments are: number of views, number of dimensions.
    ViewsDescriptor layerDesc(3, 3);
    // NOTE: arguments are: view, dimension, value.
    layerDesc.SetViewOriginCoord(0, 0, 0);
    layerDesc.SetViewOriginCoord(1, 0, 1);
    layerDesc.SetViewOriginCoord(2, 0, 3);

    Layer* const layer = graph.AddLayer<SplitterLayer>(layerDesc, "layer");

    // Adds extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output0 = graph.AddLayer<OutputLayer>(0, "output0");
    Layer* const output1 = graph.AddLayer<OutputLayer>(1, "output1");
    Layer* const output2 = graph.AddLayer<OutputLayer>(2, "output2");

    // Connects up.
    armnn::TensorInfo tensorInfo({5, 7, 7}, DataType);
    Connect(input, layer, tensorInfo);

    armnn::TensorInfo output0Info({1, 7, 7}, DataType);
    armnn::TensorInfo output1Info({2, 7, 7}, DataType);
    armnn::TensorInfo output2Info({2, 7, 7}, DataType);

    Connect(layer, output0, output0Info, 0, 0);
    Connect(layer, output1, output1Info, 1, 0);
    Connect(layer, output2, output2Info, 2, 0);

    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<SplitterWorkload>(*layer, factory);

    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 3);
    CHECK(queueDescriptor.m_ViewOrigins.size() == 3);

    CHECK(queueDescriptor.m_ViewOrigins[0].m_Origin[0] == 0);
    CHECK(queueDescriptor.m_ViewOrigins[1].m_Origin[0] == 1);
    CHECK(queueDescriptor.m_ViewOrigins[2].m_Origin[0] == 3);
    CHECK(queueDescriptor.m_ViewOrigins[0].m_Origin[1] == 0);
    CHECK(queueDescriptor.m_ViewOrigins[1].m_Origin[1] == 0);
    CHECK(queueDescriptor.m_ViewOrigins[2].m_Origin[1] == 0);
    CHECK(queueDescriptor.m_ViewOrigins[0].m_Origin[2] == 0);
    CHECK(queueDescriptor.m_ViewOrigins[1].m_Origin[2] == 0);
    CHECK(queueDescriptor.m_ViewOrigins[2].m_Origin[2] == 0);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

/// This function constructs a graph with both a splitter and a concat, and returns a pair of the workloads.
template<typename SplitterWorkload, typename ConcatWorkload, armnn::DataType DataType>
std::pair<std::unique_ptr<SplitterWorkload>, std::unique_ptr<ConcatWorkload>>
    CreateSplitterConcatWorkloadTest(armnn::IWorkloadFactory &factory, armnn::Graph &graph)
{
    armnn::TensorInfo inputTensorInfo({ 1, 2, 100, 10 }, DataType);

    armnn::TensorInfo splitTensorInfo1({ 1, 1, 100, 10 }, DataType);
    armnn::TensorInfo splitTensorInfo2({ 1, 1, 100, 10 }, DataType);

    //Constructs the graph.
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

    // create splitter layer
    Layer* const splitter = graph.AddLayer<SplitterLayer>(splitterViews, "splitter");
    CHECK(splitter);

    armnn::OriginsDescriptor concatViews(2);
    concatViews.SetViewOriginCoord(0, 0, 0);
    concatViews.SetViewOriginCoord(0, 1, 1);
    concatViews.SetViewOriginCoord(0, 2, 0);
    concatViews.SetViewOriginCoord(0, 3, 0);

    concatViews.SetViewOriginCoord(1, 0, 0);
    concatViews.SetViewOriginCoord(1, 1, 0);
    concatViews.SetViewOriginCoord(1, 2, 0);
    concatViews.SetViewOriginCoord(1, 3, 0);

    // create concat layer
    Layer* const concat = graph.AddLayer<ConcatLayer>(concatViews, "concat");
    CHECK(concat);

    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Adds connections.
    // connect input to splitter
    Connect(input, splitter, inputTensorInfo, 0, 0);
    // connect splitter[0] to concat[1]
    Connect(splitter, concat, splitTensorInfo1, 0, 1); // The splitter & concat are connected up.
    // connect splitter[1] to concat[0]
    Connect(splitter, concat, splitTensorInfo2, 1, 0); // So that the outputs are flipped round.
    // connect concat to output
    Connect(concat, output, inputTensorInfo, 0, 0);

    // created tensor handles
    CreateTensorHandles(graph, factory);

    // created splitter workload
    auto workloadSplitter = MakeAndCheckWorkload<SplitterWorkload>(*splitter, factory);
    CHECK(workloadSplitter);
    // created concat workload
    auto workloadConcat = MakeAndCheckWorkload<ConcatWorkload>(*concat, factory);
    CHECK(workloadConcat);

    return {std::move(workloadSplitter), std::move(workloadConcat)};
}


/// This function constructs a graph with a splitter with two outputs. Each of the outputs is then
/// connected to two different activation layers
template<typename SplitterWorkload, typename ActivationWorkload, armnn::DataType DataType>
void CreateSplitterMultipleInputsOneOutputWorkloadTest(armnn::IWorkloadFactory& factory, armnn::Graph& graph,
                                 std::unique_ptr<SplitterWorkload>& wlSplitter,
                                 std::unique_ptr<ActivationWorkload>& wlActiv0_0,
                                 std::unique_ptr<ActivationWorkload>& wlActiv0_1,
                                 std::unique_ptr<ActivationWorkload>& wlActiv1_0,
                                 std::unique_ptr<ActivationWorkload>& wlActiv1_1)
{
    armnn::TensorInfo inputTensorInfo ({ 1, 3, 100, 50 }, DataType);
    armnn::TensorInfo splitTensorInfo1({ 1, 1, 100, 50 }, DataType);
    armnn::TensorInfo splitTensorInfo2({ 1, 2, 100, 50 }, DataType);

    //Constructs the graph.
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

    // Adds connections.
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

    auto workloadSplitter = MakeAndCheckWorkload<SplitterWorkload>(*splitter, factory);
    auto workloadActiv0_0 = MakeAndCheckWorkload<ActivationWorkload>(*activ0_0, factory);
    auto workloadActiv0_1 = MakeAndCheckWorkload<ActivationWorkload>(*activ0_1, factory);
    auto workloadActiv1_0 = MakeAndCheckWorkload<ActivationWorkload>(*activ1_0, factory);
    auto workloadActiv1_1 = MakeAndCheckWorkload<ActivationWorkload>(*activ1_1, factory);

    wlSplitter = std::move(workloadSplitter);
    wlActiv0_0 = std::move(workloadActiv0_0);
    wlActiv0_1 = std::move(workloadActiv0_1);
    wlActiv1_0 = std::move(workloadActiv1_0);
    wlActiv1_1 = std::move(workloadActiv1_1);
}

template <typename ResizeWorkload, armnn::DataType DataType>
std::unique_ptr<ResizeWorkload> CreateResizeBilinearWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                 armnn::Graph& graph,
                                                                 DataLayout dataLayout = DataLayout::NCHW)
{
    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout) {
        case DataLayout::NHWC:
            inputShape =  { 2, 4, 4, 3 };
            outputShape = { 2, 2, 2, 3 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape =  { 2, 3, 4, 4 };
            outputShape = { 2, 3, 2, 2 };
    }

    // Creates the layer we're testing.
    ResizeDescriptor resizeDesc;
    armnnUtils::DataLayoutIndexed dimensionIndices = dataLayout;
    resizeDesc.m_Method       = ResizeMethod::Bilinear;
    resizeDesc.m_TargetWidth  = outputShape[dimensionIndices.GetWidthIndex()];
    resizeDesc.m_TargetHeight = outputShape[dimensionIndices.GetHeightIndex()];
    resizeDesc.m_DataLayout   = dataLayout;
    Layer* const layer = graph.AddLayer<ResizeLayer>(resizeDesc, "resize");

    // Creates extra layers.
    Layer* const input  = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo inputTensorInfo(inputShape, DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<ResizeWorkload>(*layer, factory);

    auto queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size()  == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);
    CHECK(queueDescriptor.m_Parameters.m_DataLayout == dataLayout);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename BatchToSpaceNdWorkload, armnn::DataType DataType>
std::unique_ptr<BatchToSpaceNdWorkload> CreateBatchToSpaceNdWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                         armnn::Graph&  graph)
{
    BatchToSpaceNdDescriptor desc;
    Layer* const layer = graph.AddLayer<BatchToSpaceNdLayer>(desc, "batchToSpace");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo tensorInfo({1, 1, 1, 1}, DataType);

    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);

    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<BatchToSpaceNdWorkload>(*layer, factory);

    BatchToSpaceNdQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    return workload;
}

template <typename LogSoftmaxWorkload, armnn::DataType DataType>
std::unique_ptr<LogSoftmaxWorkload> CreateLogSoftmaxWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                 armnn::Graph& graph)
{
    // Create the layer we're testing.
    LogSoftmaxDescriptor logSoftmaxDescriptor;
    // Set Axis to -1 if CL or Neon until further Axes are supported.
    if (factory.GetBackendId() == armnn::Compute::CpuAcc || factory.GetBackendId() == armnn::Compute::GpuAcc)
    {
        logSoftmaxDescriptor.m_Axis = -1;
    }

    Layer* const layer = graph.AddLayer<LogSoftmaxLayer>(logSoftmaxDescriptor, "layer");
    // Create extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up
    armnn::TensorInfo tensorInfo({4, 1}, DataType);

    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Make the workload and checks it.
    auto workload = MakeAndCheckWorkload<LogSoftmaxWorkload>(*layer, factory);

    LogSoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Return so we can do extra, backend-specific tests.
    return workload;
}

template <typename L2NormalizationWorkload, armnn::DataType DataType>
std::unique_ptr<L2NormalizationWorkload> CreateL2NormalizationWorkloadTest(armnn::IWorkloadFactory& factory,
    armnn::Graph& graph, DataLayout dataLayout = DataLayout::NCHW)
{
    // Creates the layer we're testing.
    L2NormalizationDescriptor layerDesc;
    layerDesc.m_DataLayout = dataLayout;

    Layer* const layer = graph.AddLayer<L2NormalizationLayer>(layerDesc, "l2norm");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    TensorShape inputShape = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 5, 20, 50, 67 } : TensorShape{ 5, 50, 67, 20 };
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 5, 20, 50, 67 } : TensorShape{ 5, 50, 67, 20 };

    // Connects up.
    armnn::TensorInfo inputTensorInfo(inputShape, DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<L2NormalizationWorkload>(*layer, factory);

    L2NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    CHECK((queueDescriptor.m_Parameters.m_DataLayout == dataLayout));
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename ReshapeWorkload, armnn::DataType DataType>
std::unique_ptr<ReshapeWorkload> CreateReshapeWorkloadTest(armnn::IWorkloadFactory& factory,
    armnn::Graph& graph)
{
    // Creates the layer we're testing.
    TensorShape outputShape({ 1, 4 });
    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputShape;
    Layer* const layer = graph.AddLayer<ReshapeLayer>(reshapeDesc, "layer");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo inputTensorInfo({ 4, 1 }, DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<ReshapeWorkload>(*layer, factory);

    ReshapeQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename ConvertFp16ToFp32Float32Workload>
std::unique_ptr<ConvertFp16ToFp32Float32Workload> CreateConvertFp16ToFp32WorkloadTest(
    armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // Creates the layer we're testing.
    ConvertFp16ToFp32Layer* const layer = graph.AddLayer<ConvertFp16ToFp32Layer>("Fp16ToFp32Converter");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float16);
    armnn::TensorInfo outputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float32);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<ConvertFp16ToFp32Float32Workload>(*layer, factory);

    ConvertFp16ToFp32QueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename ConvertFp32ToFp16Float16Workload>
std::unique_ptr<ConvertFp32ToFp16Float16Workload> CreateConvertFp32ToFp16WorkloadTest(
    armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // Creates the layer we're testing.
    ConvertFp32ToFp16Layer* const layer = graph.AddLayer<ConvertFp32ToFp16Layer>("Fp32ToFp16Converter");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float16);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<ConvertFp32ToFp16Float16Workload>(*layer, factory);

    ConvertFp32ToFp16QueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename MeanWorkload, armnn::DataType DataType>
std::unique_ptr<MeanWorkload> CreateMeanWorkloadTest(armnn::IWorkloadFactory& factory, armnn::Graph& graph)
{
    // Reduce along the first and second dimensions, and do not keep the reduced dimensions.
    MeanDescriptor descriptor({ 1, 2 }, false);

    // Creates the layer we're testing.
    Layer* const layer = graph.AddLayer<MeanLayer>(descriptor, "mean");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo inputTensorInfo({ 1, 3, 7, 4 }, DataType);
    armnn::TensorInfo outputTensorInfo({ 1, 4 }, DataType);
    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<MeanWorkload>(*layer, factory);

    MeanQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Parameters.m_Axis == descriptor.m_Axis);
    CHECK(queueDescriptor.m_Parameters.m_KeepDims == descriptor.m_KeepDims);
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template<typename ConcatWorkload, armnn::DataType DataType>
std::unique_ptr<ConcatWorkload> CreateConcatWorkloadTest(armnn::IWorkloadFactory &factory,
                                                         armnn::Graph &graph,
                                                         const armnn::TensorShape &outputShape,
                                                         unsigned int concatAxis)
{
    armnn::TensorInfo inputTensorInfo({ 2, 3, 2, 5 }, DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, DataType);

    // Constructs the graph.
    Layer* const input0 = graph.AddLayer<InputLayer>(0, "input0");
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    armnn::OriginsDescriptor descriptor;

    std::vector<armnn::TensorShape> inputShapes{{ 2, 3, 2, 5 }, { 2, 3, 2, 5 }};

    descriptor = CreateDescriptorForConcatenation(inputShapes.begin(),
                                                  inputShapes.end(),
                                                  concatAxis);

    // create concat layer
    Layer* const concat = graph.AddLayer<ConcatLayer>(descriptor, "concat");
    CHECK(concat);

    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Adds connections.
    // connect input0 to concat
    Connect(input0, concat, inputTensorInfo, 0, 0);
    // connect input1 to concat
    Connect(input1, concat, inputTensorInfo, 0, 1);
    // connect concat to output
    Connect(concat, output, outputTensorInfo, 0, 0);

    // create tensor handles
    CreateTensorHandles(graph, factory);

    // create concat workload
    auto workloadConcat = MakeAndCheckWorkload<ConcatWorkload>(*concat, factory);
    CHECK(workloadConcat);

    return workloadConcat;
}

template <typename PreCompiledWorkload, armnn::DataType dataType>
std::pair<armnn::IOptimizedNetworkPtr, std::unique_ptr<PreCompiledWorkload>> CreatePreCompiledWorkloadTest(
    armnn::IWorkloadFactory& factory,
    armnn::Graph& graph,
    bool biasEnabled = false)
{
    IgnoreUnused(graph);

    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    // Add an input layer
    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    // ArmNN weights tensor shape is OIHW (out channels, in channels, height, width) for NCHW
    // ArmNN weights tensor shape is OHWI (out channels, height, width, in channels) for NHWC
    // this test is using NHWC, so the weights shape is OHWI
    TensorInfo weightsTensorInfo(TensorShape({16, 1, 1, 16}), dataType, 0.9f, 0, true);
    unsigned int weightsLength = weightsTensorInfo.GetNumElements();

    using WeightType = armnn::ResolveType<dataType>;
    std::vector<WeightType> convWeightsData(weightsLength);
    for (unsigned int i = 0; i < weightsLength; ++i)
    {
        convWeightsData[i] = static_cast<WeightType>(i);
    }

    armnn::ConstTensor weights(weightsTensorInfo, convWeightsData);

    // Add a layer that can be used in the PreCompiled layer
    armnn::Convolution2dDescriptor convDesc2d;
    convDesc2d.m_StrideX = 1;
    convDesc2d.m_StrideY = 1;
    convDesc2d.m_BiasEnabled = biasEnabled;
    convDesc2d.m_DataLayout = armnn::DataLayout::NHWC;


    const std::string convLayerName("conv layer");

    armnn::IConnectableLayer* convLayer = net->AddConvolution2dLayer(convDesc2d, convLayerName.c_str());

    IConnectableLayer* weightsLayer = net->AddConstantLayer(weights);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weights.GetInfo());
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1u));

    if (biasEnabled)
    {
        constexpr armnn::DataType biasDataType = ( dataType == armnn::DataType::QAsymmU8) ?
            armnn::DataType::Signed32 : armnn::DataType::Float32;

        TensorInfo biasTensorInfo(TensorShape({16}), biasDataType, 0.9f * 0.9f, 0, true);
        unsigned int biasLength = biasTensorInfo.GetNumElements();

        using BiasType = armnn::ResolveType<biasDataType>;
        std::vector<BiasType> biasData(biasLength);
        std::fill(biasData.begin(), biasData.end(), static_cast<BiasType>(0));

        armnn::ConstTensor biases(biasTensorInfo, biasData);

        IConnectableLayer* biasLayer = net->AddConstantLayer(biases);

        biasLayer->GetOutputSlot(0).SetTensorInfo(biases.GetInfo());
        biasLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2u));
    }

    CHECK(convLayer);

    // Add an output layer
    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

    // set the tensors in the network (NHWC format)
    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), dataType);
    if (dataType == armnn::DataType::QAsymmU8)
    {
        inputTensorInfo.SetQuantizationOffset(0);
        inputTensorInfo.SetQuantizationScale(0.9f);
    }

    TensorInfo outputTensorInfo(TensorShape({1, 16, 16, 16}), dataType);
    if (dataType == armnn::DataType::QAsymmU8)
    {
        outputTensorInfo.SetQuantizationOffset(0);
        outputTensorInfo.SetQuantizationScale(0.9f);
    }

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimize the network for the backend supported by the factory
    std::vector<armnn::BackendId> backends = {factory.GetBackendId()};
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::OptimizerOptionsOpaque optimizerOptions;
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec(),
                                                               optimizerOptions);
    CHECK(optimizedNet != nullptr);

    // Find the PreCompiled layer in the optimised graph
    armnn::Graph& optimisedGraph = GetGraphForTesting(optimizedNet.get());
    Layer* preCompiledLayer = nullptr;
    for (auto& layer : optimisedGraph)
    {
        if (layer->GetType() == LayerType::PreCompiled)
        {
            preCompiledLayer = layer;
        }
    }
    CHECK(preCompiledLayer != nullptr);

    // Create the TensorHandles.
    CreateTensorHandles(optimisedGraph, factory);

    // Make the workload and check it.
    auto workload = MakeAndCheckWorkload<PreCompiledWorkload>(*preCompiledLayer, factory);

    PreCompiledQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size()  == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns the workload so we can do extra, backend-specific tests.
    // NOTE: We need to return the optimised network as well, otherwise it gets
    // out of scope and the tensor handles get destructed
    return std::make_pair(std::move(optimizedNet), std::move(workload));
}

template<typename ConstantWorkload, armnn::DataType DataType>
std::unique_ptr<ConstantWorkload> CreateConstantWorkloadTest(armnn::IWorkloadFactory& factory,
                                                             armnn::Graph& graph,
                                                             const armnn::TensorShape& outputShape)
{
    armnn::TensorInfo outputTensorInfo(outputShape, DataType);

    // create constant layer
    auto constant = graph.AddLayer<ConstantLayer>("constant");
    CHECK(constant);
    constant->m_LayerOutput = std::make_unique<ScopedTensorHandle>(outputTensorInfo);

    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Adds connections.
    // connect constant to output
    Connect(constant, output, outputTensorInfo, 0, 0);

    // create tensor handles
    CreateTensorHandles(graph, factory);

    // create Constant workload"
    auto workloadConstant = MakeAndCheckWorkload<ConstantWorkload>(*constant, factory);
    CHECK(workloadConstant);

    return workloadConstant;
}

template <typename PreluWorkload>
std::unique_ptr<PreluWorkload> CreatePreluWorkloadTest(armnn::IWorkloadFactory& factory,
                                                       armnn::Graph& graph,
                                                       const armnn::TensorShape& inputShape,
                                                       const armnn::TensorShape& alphaShape,
                                                       const armnn::TensorShape& outputShape,
                                                       armnn::DataType dataType)
{
    // Creates the PReLU layer
    Layer* const layer = graph.AddLayer<PreluLayer>("prelu");
    CHECK(layer != nullptr);

    // Creates extra layers
    Layer* const input  = graph.AddLayer<InputLayer> (0, "input");
    Layer* const alpha  = graph.AddLayer<InputLayer> (1, "alpha");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");
    CHECK(input  != nullptr);
    CHECK(alpha  != nullptr);
    CHECK(output != nullptr);

    // Connects up
    armnn::TensorInfo inputTensorInfo (inputShape,  dataType);
    armnn::TensorInfo alphaTensorInfo (alphaShape,  dataType);
    armnn::TensorInfo outputTensorInfo(outputShape, dataType);
    Connect(input, layer,  inputTensorInfo,  0, 0);
    Connect(alpha, layer,  alphaTensorInfo,  0, 1);
    Connect(layer, output, outputTensorInfo, 0, 0);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it
    auto workload = MakeAndCheckWorkload<PreluWorkload>(*layer, factory);

    PreluQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 2);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    // Returns so we can do extra, backend-specific tests.
    return workload;
}

template <typename SpaceToDepthWorkload, armnn::DataType DataType>
std::unique_ptr<SpaceToDepthWorkload> CreateSpaceToDepthWorkloadTest(armnn::IWorkloadFactory& factory,
                                                                     armnn::Graph&  graph)
{
    SpaceToDepthDescriptor desc;
    desc.m_BlockSize = 2;
    Layer* const layer = graph.AddLayer<SpaceToDepthLayer>(desc, "spaceToDepth");

    // Creates extra layers.
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connects up.
    armnn::TensorInfo inputTensorInfo({ 1, 2, 2, 1 }, DataType);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 1, 4 }, DataType);

    Connect(input, layer, inputTensorInfo);
    Connect(layer, output, outputTensorInfo);

    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it.
    auto workload = MakeAndCheckWorkload<SpaceToDepthWorkload>(*layer, factory);

    SpaceToDepthQueueDescriptor queueDescriptor = workload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == 1);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    return workload;
}

template <typename StackWorkload, armnn::DataType DataType>
std::unique_ptr<StackWorkload> CreateStackWorkloadTest(armnn::IWorkloadFactory& factory,
                                                       armnn::Graph& graph,
                                                       const armnn::TensorShape& inputShape,
                                                       const armnn::TensorShape& outputShape,
                                                       unsigned int axis,
                                                       unsigned int numInputs)
{
    armnn::TensorInfo inputTensorInfo(inputShape, DataType);
    armnn::TensorInfo outputTensorInfo(outputShape, DataType);

    // Constructs the Stack layer.
    armnn::StackDescriptor descriptor(axis, numInputs, inputShape);
    Layer* const stackLayer = graph.AddLayer<StackLayer>(descriptor, "stack");
    CHECK(stackLayer != nullptr);

    // Constructs layer inputs and output.
    std::vector<Layer*> inputs;
    for (unsigned int i=0; i<numInputs; ++i)
    {
        inputs.push_back(graph.AddLayer<InputLayer>(
            static_cast<int>(i),
            ("input" + std::to_string(i)).c_str()
        ));
        CHECK(inputs[i] != nullptr);
    }
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");
    CHECK(output != nullptr);

    // Adds connections.
    for (unsigned int i=0; i<numInputs; ++i)
    {
        Connect(inputs[i], stackLayer, inputTensorInfo, 0, i);
    }
    Connect(stackLayer, output, outputTensorInfo, 0, 0);

    CreateTensorHandles(graph, factory);

    auto stackWorkload = MakeAndCheckWorkload<StackWorkload>(*stackLayer, factory);
    StackQueueDescriptor queueDescriptor = stackWorkload->GetData();
    CHECK(queueDescriptor.m_Inputs.size() == numInputs);
    CHECK(queueDescriptor.m_Outputs.size() == 1);

    return stackWorkload;
}

} // Anonymous namespace
