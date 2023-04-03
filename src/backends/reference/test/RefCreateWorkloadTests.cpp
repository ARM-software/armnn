//
// Copyright © 2017, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <CreateWorkload.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <reference/RefTensorHandle.hpp>
#include <reference/RefTensorHandleFactory.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <reference/workloads/RefWorkloads.hpp>

#include <doctest/doctest.h>

namespace
{

template<typename Workload>
void CheckInputOutput(std::unique_ptr<Workload> workload, const TensorInfo& inputInfo, const TensorInfo& outputInfo)
{
    auto queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((inputHandle->GetTensorInfo() == inputInfo));
    CHECK((outputHandle->GetTensorInfo() == outputInfo));
}

template <typename Workload>
void CheckInputsOutput(std::unique_ptr<Workload> workload,
                       const TensorInfo&         inputInfo0,
                       const TensorInfo&         inputInfo1,
                       const TensorInfo&         outputInfo)
{
    auto queueDescriptor = workload->GetData();
    auto inputHandle0     = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle1     = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle    = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((inputHandle0->GetTensorInfo() == inputInfo0));
    CHECK((inputHandle1->GetTensorInfo() == inputInfo1));
    CHECK((outputHandle->GetTensorInfo() == outputInfo));
}

armnn::RefWorkloadFactory GetFactory()
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();
    return RefWorkloadFactory(memoryManager);
}

}

TEST_SUITE("CreateWorkloadRef")
{
template <typename ActivationWorkloadType, armnn::DataType DataType>
static void RefCreateActivationWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateActivationWorkloadTest<ActivationWorkloadType, DataType>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateActivationWorkloadTest).
    CheckInputOutput(std::move(workload),
        TensorInfo({ 1, 1 }, DataType),
        TensorInfo({ 1, 1 }, DataType));
}

TEST_CASE("CreateActivationFloat32Workload")
{
    RefCreateActivationWorkloadTest<RefActivationWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateActivationUint8Workload")
{
    RefCreateActivationWorkloadTest<RefActivationWorkload, armnn::DataType::QAsymmU8>();
}

template <typename WorkloadType,
          typename DescriptorType,
          typename LayerType,
          armnn::DataType DataType>
static void RefCreateElementwiseWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateElementwiseWorkloadTest<WorkloadType, DescriptorType, LayerType, DataType>(
        factory, graph);

    CheckInputsOutput(std::move(workload),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType));
}

TEST_CASE("CreateSubtractionWorkloadWithBlobTest")
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    armnn::DataType DataType = armnn::DataType::Float32;

    auto workload = CreateSubtractionWithBlobWorkloadTest<RefSubtractionWorkload<>,
                                                          SubtractionQueueDescriptor,
                                                          armnn::DataType::Float32>
                                                          (factory, graph);

    CheckInputsOutput(std::move(workload),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType));
}

TEST_CASE("CreateAdditionWorkloadWithBlobTest")
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    armnn::DataType DataType = armnn::DataType::Float32;

    auto workload = CreateAdditionWithBlobWorkloadTest<RefAdditionWorkload<>,
                                                       AdditionQueueDescriptor,
                                                       armnn::DataType::Float32>(factory, graph);

    CheckInputsOutput(std::move(workload),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType));
}

TEST_CASE("CreateMultiplicationWorkloadWithBlobTest")
{
    Graph              graph;
    RefWorkloadFactory factory  = GetFactory();
    armnn::DataType    DataType = armnn::DataType::Float32;

    auto workload = CreateMultiplicationWithBlobWorkloadTest<RefMultiplicationWorkload<>,
                                                             MultiplicationQueueDescriptor,
                                                             armnn::DataType::Float32>(factory, graph);

    CheckInputsOutput(std::move(workload),
                      TensorInfo({2, 3}, DataType),
                      TensorInfo({2, 3}, DataType),
                      TensorInfo({2, 3}, DataType));
}

TEST_CASE("CreateAdditionFloatWorkload")
{
    RefCreateElementwiseWorkloadTest<RefAdditionWorkload<>,
        AdditionQueueDescriptor,
        AdditionLayer,
        armnn::DataType::Float32>();
}

TEST_CASE("CreateAdditionUint8Workload")
{
    RefCreateElementwiseWorkloadTest<RefAdditionWorkload<>,
        AdditionQueueDescriptor,
        AdditionLayer,
        armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateAdditionInt16Workload")
{
    RefCreateElementwiseWorkloadTest<RefAdditionWorkload<>,
        AdditionQueueDescriptor,
        AdditionLayer,
        armnn::DataType::QSymmS16>();
}

TEST_CASE("CreateAdditionInt32Workload")
{
    RefCreateElementwiseWorkloadTest<RefAdditionWorkload<int32_t>,
            AdditionQueueDescriptor,
            AdditionLayer,
            armnn::DataType::Signed32>();
}

TEST_CASE("CreateSubtractionFloat32Workload")
{
    RefCreateElementwiseWorkloadTest<RefSubtractionWorkload<>,
        SubtractionQueueDescriptor,
        SubtractionLayer,
        armnn::DataType::Float32>();
}

TEST_CASE("CreateSubtractionFloat16Workload")
{
    RefCreateElementwiseWorkloadTest<RefSubtractionWorkload<>,
        SubtractionQueueDescriptor,
        SubtractionLayer,
        armnn::DataType::Float16>();
}

TEST_CASE("CreateSubtractionUint8Workload")
{
    RefCreateElementwiseWorkloadTest<RefSubtractionWorkload<>,
        SubtractionQueueDescriptor,
        SubtractionLayer,
        armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateSubtractionInt16Workload")
{
    RefCreateElementwiseWorkloadTest<RefSubtractionWorkload<>,
        SubtractionQueueDescriptor,
        SubtractionLayer,
        armnn::DataType::QSymmS16>();
}

TEST_CASE("CreateSubtractionInt32Workload")
{
    RefCreateElementwiseWorkloadTest<RefSubtractionWorkload<int32_t>,
            SubtractionQueueDescriptor,
            SubtractionLayer,
            armnn::DataType::Signed32>();
}

TEST_CASE("CreateMultiplicationFloatWorkload")
{
    RefCreateElementwiseWorkloadTest<RefMultiplicationWorkload<>,
        MultiplicationQueueDescriptor,
        MultiplicationLayer,
        armnn::DataType::Float32>();
}

TEST_CASE("CreateMultiplicationUint8Workload")
{
    RefCreateElementwiseWorkloadTest<RefMultiplicationWorkload<>,
        MultiplicationQueueDescriptor,
        MultiplicationLayer,
        armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateMultiplicationInt16Workload")
{
    RefCreateElementwiseWorkloadTest<RefMultiplicationWorkload<>,
        MultiplicationQueueDescriptor,
        MultiplicationLayer,
        armnn::DataType::QSymmS16>();
}

TEST_CASE("CreateMultiplicationInt32Workload")
{
    RefCreateElementwiseWorkloadTest<RefMultiplicationWorkload<int32_t>,
            MultiplicationQueueDescriptor,
            MultiplicationLayer,
            armnn::DataType::Signed32>();
}

TEST_CASE("CreateDivisionFloat32Workload")
{
    RefCreateElementwiseWorkloadTest<RefDivisionWorkload<>,
        DivisionQueueDescriptor,
        DivisionLayer,
        armnn::DataType::Float32>();
}

TEST_CASE("CreateDivisionFloat16Workload")
{
    RefCreateElementwiseWorkloadTest<RefDivisionWorkload<>,
        DivisionQueueDescriptor,
        DivisionLayer,
        armnn::DataType::Float16>();
}

TEST_CASE("CreateDivisionUint8Workload")
{
    RefCreateElementwiseWorkloadTest<RefDivisionWorkload<>,
        DivisionQueueDescriptor,
        DivisionLayer,
        armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateDivisionInt16Workload")
{
    RefCreateElementwiseWorkloadTest<RefDivisionWorkload<>,
        DivisionQueueDescriptor,
        DivisionLayer,
        armnn::DataType::QSymmS16>();
}

TEST_CASE("CreateDivisionInt32Workload")
{
    RefCreateElementwiseWorkloadTest<RefDivisionWorkload<int32_t>,
            DivisionQueueDescriptor,
            DivisionLayer,
            armnn::DataType::Signed32>();
}

template <typename BatchNormalizationWorkloadType, armnn::DataType DataType>
static void RefCreateBatchNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>(factory,
                                                                                                   graph,
                                                                                                   dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 2, 4, 4, 3 };
            outputShape = { 2, 4, 4, 3 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape  = { 2, 3, 4, 4 };
            outputShape = { 2, 3, 4, 4 };
            break;
    }

    // Checks that outputs and inputs are as we expect them (see definition of CreateBatchNormalizationWorkloadTest).
    CheckInputOutput(std::move(workload), TensorInfo(inputShape, DataType), TensorInfo(outputShape, DataType));
}

TEST_CASE("CreateBatchNormalizationWithBlobFloat32Workload")
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto dataType = armnn::DataType::Float32;
    auto workload = CreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload,
                                                         armnn::DataType::Float32>(factory, graph, DataLayout::NHWC);

    TensorShape inputShape;
    TensorShape outputShape;

    inputShape  = { 2, 4, 4, 3 };
    outputShape = { 2, 4, 4, 3 };

    // Checks that outputs and inputs are as we expect them (see definition of CreateBatchNormalizationWorkloadTest).
    CheckInputOutput(std::move(workload), TensorInfo(inputShape, dataType), TensorInfo(outputShape, dataType));
}

TEST_CASE("CreateBatchNormalizationFloat32Workload")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload,armnn::DataType::Float32>
            (DataLayout::NCHW);
}

TEST_CASE("CreateBatchNormalizationFloat32WorkloadNhwc")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload, armnn::DataType::Float32>
            (DataLayout::NHWC);
}

TEST_CASE("CreateBatchNormalizationFloat16Workload")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload,armnn::DataType::Float16>
            (DataLayout::NCHW);
}

TEST_CASE("CreateBatchNormalizationFloat16WorkloadNhwc")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload, armnn::DataType::Float16>
            (DataLayout::NHWC);
}

TEST_CASE("CreateBatchNormalizationUint8Workload")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload, armnn::DataType::QAsymmU8>
            (DataLayout::NCHW);
}

TEST_CASE("CreateBatchNormalizationUint8WorkloadNhwc")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload, armnn::DataType::QAsymmU8>
            (DataLayout::NHWC);
}

TEST_CASE("CreateBatchNormalizationInt16Workload")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload, armnn::DataType::QSymmS16>
            (DataLayout::NCHW);
}

TEST_CASE("CreateBatchNormalizationInt16WorkloadNhwc")
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload, armnn::DataType::QSymmS16>
            (DataLayout::NHWC);
}

TEST_CASE("CreateConvertFp16ToFp32Float32Workload")
{
    Graph                graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateConvertFp16ToFp32WorkloadTest<RefConvertFp16ToFp32Workload>(factory, graph);

    // Checks that outputs and inputs are as we expect them
    CheckInputOutput(
        std::move(workload), TensorInfo({1, 3, 2, 3}, DataType::Float16), TensorInfo({1, 3, 2, 3}, DataType::Float32));
}

TEST_CASE("CreateConvertFp32ToFp16Float16Workload")
{
    Graph                graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateConvertFp32ToFp16WorkloadTest<RefConvertFp32ToFp16Workload>(factory, graph);

    // Checks that outputs and inputs are as we expect them
    CheckInputOutput(
        std::move(workload), TensorInfo({1, 3, 2, 3}, DataType::Float32), TensorInfo({1, 3, 2, 3}, DataType::Float16));
}

static void RefCreateConvolution2dWorkloadTest(DataLayout dataLayout = DataLayout::NCHW)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateConvolution2dWorkloadTest<RefConvolution2dWorkload, DataType::Float32>
                    (factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({2, 3, 8, 16})
                                                               : std::initializer_list<unsigned int>({2, 8, 16, 3});
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({2, 2, 2, 10})
                                                               : std::initializer_list<unsigned int>({2, 2, 10, 2});

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType::Float32),
                     TensorInfo(outputShape, DataType::Float32));
}

TEST_CASE("CreateConvolution2dFloatNchwWorkload")
{
    RefCreateConvolution2dWorkloadTest(DataLayout::NCHW);
}

TEST_CASE("CreateConvolution2dFloatNhwcWorkload")
{
    RefCreateConvolution2dWorkloadTest(DataLayout::NHWC);
}

TEST_CASE("CreateConvolution2dWithBlobWorkload")
{
    DataLayout dataLayout = DataLayout::NHWC;
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateConvolution2dFusedActivationWithBlobWorkloadTest<RefConvolution2dWorkload, DataType::Float32>
                    (factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({2, 3, 8, 16})
                                                               : std::initializer_list<unsigned int>({2, 8, 16, 3});
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({2, 2, 2, 10})
                                                               : std::initializer_list<unsigned int>({2, 2, 10, 2});

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType::Float32),
                     TensorInfo(outputShape, DataType::Float32));
}

static void RefCreateDepthwiseConvolutionWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateDepthwiseConvolution2dWorkloadTest<RefDepthwiseConvolution2dWorkload, DataType::Float32>
            (factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });

    // Checks that inputs/outputs are as we expect them (see definition of CreateDepthwiseConvolution2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType::Float32),
                     TensorInfo(outputShape, DataType::Float32));
}

TEST_CASE("CreateDepthwiseConvolutionFloat32NhwcWorkload")
{
    RefCreateDepthwiseConvolutionWorkloadTest(DataLayout::NHWC);
}

TEST_CASE("RefCreateFullyConnectedWithBlobWorkloadTest")
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateFullyConnectedWithBlobWorkloadTest<RefFullyConnectedWorkload,
                                                         armnn::DataType::Float32>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    float inputsQScale = 1.0f;
    float outputQScale = 1.0f;
    CheckInputOutput(std::move(workload),
        TensorInfo({ 3, 1, 4, 5 }, armnn::DataType::Float32, inputsQScale),
        TensorInfo({ 3, 7 }, armnn::DataType::Float32, outputQScale));
}

TEST_CASE("CreateFullyConnectedWorkloadWeightsBiasesAsInputsFloat32")
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();

    auto workload =
            CreateFullyConnectedWorkloadWeightsBiasesAsInputsTest<RefFullyConnectedWorkload,
                                                                  armnn::DataType::Float32>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    float inputsQScale = 1.0f;
    float outputQScale = 1.0f;
    CheckInputsOutput(std::move(workload),
                      TensorInfo({ 3, 1, 4, 5 }, armnn::DataType::Float32, inputsQScale),
                      TensorInfo({ 7, 20 }, armnn::DataType::Float32, inputsQScale),
                      TensorInfo({ 3, 7 }, armnn::DataType::Float32, outputQScale));
}

template <typename FullyConnectedWorkloadType, armnn::DataType DataType>
static void RefCreateFullyConnectedWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateFullyConnectedWorkloadTest<FullyConnectedWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    float inputsQScale = DataType == armnn::DataType::QAsymmU8 ? 1.0f : 1.0f;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 1.0f;
    CheckInputOutput(std::move(workload),
        TensorInfo({ 3, 1, 4, 5 }, DataType, inputsQScale),
        TensorInfo({ 3, 7 }, DataType, outputQScale));
}

TEST_CASE("CreateFullyConnectedWorkloadFloat32")
{
    RefCreateFullyConnectedWorkloadTest<RefFullyConnectedWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateFullyConnectedWorkloadQuantisedAsymm8")
{
    RefCreateFullyConnectedWorkloadTest<RefFullyConnectedWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateFullyConnectedWorkloadQuantisedSymm16")
{
    RefCreateFullyConnectedWorkloadTest<RefFullyConnectedWorkload, armnn::DataType::QSymmS16>();
}

template <typename NormalizationWorkloadType, armnn::DataType DataType>
static void RefCreateNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateNormalizationWorkloadTest<NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 3, 1, 5, 5 };
            outputShape = { 3, 1, 5, 5 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape  = { 3, 5, 5, 1 };
            outputShape = { 3, 5, 5, 1 };
            break;
    }

    // Checks that outputs and inputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    CheckInputOutput(std::move(workload), TensorInfo(inputShape, DataType), TensorInfo(outputShape, DataType));
}

TEST_CASE("CreateRefNormalizationFloat32NchwWorkload")
{
    RefCreateNormalizationWorkloadTest<RefNormalizationWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreateRefNormalizationFloat32NhwcWorkload")
{
    RefCreateNormalizationWorkloadTest<RefNormalizationWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE("CreateRefNormalizationUint8NchwWorkload")
{
    RefCreateNormalizationWorkloadTest<RefNormalizationWorkload, armnn::DataType::QAsymmU8>(DataLayout::NCHW);
}

TEST_CASE("CreateRefNormalizationUint8NhwcWorkload")
{
    RefCreateNormalizationWorkloadTest<RefNormalizationWorkload, armnn::DataType::QAsymmU8>(DataLayout::NHWC);
}

TEST_CASE("CreateRefNormalizationInt16NchwWorkload")
{
    RefCreateNormalizationWorkloadTest<RefNormalizationWorkload, armnn::DataType::QSymmS16>(DataLayout::NCHW);
}

TEST_CASE("CreateRefNormalizationInt16NhwcWorkload")
{
    RefCreateNormalizationWorkloadTest<RefNormalizationWorkload, armnn::DataType::QSymmS16>(DataLayout::NHWC);
}

template <typename Pooling2dWorkloadType, armnn::DataType DataType>
static void RefCreatePooling2dWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreatePooling2dWorkloadTest<Pooling2dWorkloadType, DataType>(factory, graph, dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 3, 5, 5, 2 };
            outputShape = { 3, 2, 4, 2 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape =  { 3, 2, 5, 5 };
            outputShape = { 3, 2, 2, 4 };
    }

    // Checks that outputs and inputs are as we expect them (see definition of CreatePooling2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType),
                     TensorInfo(outputShape, DataType));
}

TEST_CASE("CreatePooling2dFloat32Workload")
{
    RefCreatePooling2dWorkloadTest<RefPooling2dWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreatePooling2dFloat32NhwcWorkload")
{
    RefCreatePooling2dWorkloadTest<RefPooling2dWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE("CreatePooling2dUint8Workload")
{
    RefCreatePooling2dWorkloadTest<RefPooling2dWorkload, armnn::DataType::QAsymmU8>(DataLayout::NCHW);
}

TEST_CASE("CreatePooling2dUint8NhwcWorkload")
{
    RefCreatePooling2dWorkloadTest<RefPooling2dWorkload, armnn::DataType::QAsymmU8>(DataLayout::NHWC);
}

TEST_CASE("CreatePooling2dInt16Workload")
{
    RefCreatePooling2dWorkloadTest<RefPooling2dWorkload, armnn::DataType::QSymmS16>(DataLayout::NCHW);
}

TEST_CASE("CreatePooling2dInt16NhwcWorkload")
{
    RefCreatePooling2dWorkloadTest<RefPooling2dWorkload, armnn::DataType::QSymmS16>(DataLayout::NHWC);
}

template <typename SoftmaxWorkloadType, armnn::DataType DataType>
static void RefCreateSoftmaxWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateSoftmaxWorkloadTest<SoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateSoftmaxWorkloadTest).

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
    CheckInputOutput(
        std::move(workload),
        tensorInfo,
        tensorInfo);
}

TEST_CASE("CreateSoftmaxFloat32Workload")
{
    RefCreateSoftmaxWorkloadTest<RefSoftmaxWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateSoftmaxFloat16Workload")
{
    RefCreateSoftmaxWorkloadTest<RefSoftmaxWorkload, armnn::DataType::Float16>();
}

TEST_CASE("CreateSoftmaxQuantisedAsymm8Workload")
{
    RefCreateSoftmaxWorkloadTest<RefSoftmaxWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateSoftmaxQuantisedSymm16Workload")
{
    RefCreateSoftmaxWorkloadTest<RefSoftmaxWorkload, armnn::DataType::QSymmS16>();
}

template <typename SplitterWorkloadType, armnn::DataType DataType>
static void RefCreateSplitterWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateSplitterWorkloadTest<SplitterWorkloadType, DataType>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateSplitterWorkloadTest).
    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Inputs[0]);
    CHECK((inputHandle->GetTensorInfo() == TensorInfo({ 5, 7, 7 }, DataType)));

    auto outputHandle0 = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((outputHandle0->GetTensorInfo() == TensorInfo({ 1, 7, 7 }, DataType)));

    auto outputHandle1 = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[1]);
    CHECK((outputHandle1->GetTensorInfo() == TensorInfo({ 2, 7, 7 }, DataType)));

    auto outputHandle2 = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[2]);
    CHECK((outputHandle2->GetTensorInfo() == TensorInfo({ 2, 7, 7 }, DataType)));
}

TEST_CASE("CreateSplitterFloat32Workload")
{
    RefCreateSplitterWorkloadTest<RefSplitterWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateSplitterFloat16Workload")
{
    RefCreateSplitterWorkloadTest<RefSplitterWorkload, armnn::DataType::Float16>();
}

TEST_CASE("CreateSplitterUint8Workload")
{
    RefCreateSplitterWorkloadTest<RefSplitterWorkload, armnn::DataType::QAsymmU8>();
}

template <typename SplitterWorkloadType, typename ConcatWorkloadType, armnn::DataType DataType>
static void RefCreateSplitterConcatWorkloadTest()
{
    // Tests that it is possible to decide which output of the splitter layer
    // should be lined to which input of the concat layer.
    // We tested that is is possible to specify 0th output
    // of the splitter to be the 1st input to the concat and the 1st output of the splitter to be 0th input
    // of the concat.

    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workloads = CreateSplitterConcatWorkloadTest<SplitterWorkloadType, ConcatWorkloadType, DataType>
            (factory, graph);

    auto wlSplitter = std::move(workloads.first);
    auto wlConcat = std::move(workloads.second);

    //Checks that the index of inputs/outputs matches what we declared on InputDescriptor construction.
    armnn::RefTensorHandle* sOut0 = dynamic_cast<armnn::RefTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::RefTensorHandle* sOut1 = dynamic_cast<armnn::RefTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::RefTensorHandle* mIn0 = dynamic_cast<armnn::RefTensorHandle*>(wlConcat->GetData().m_Inputs[0]);
    armnn::RefTensorHandle* mIn1 = dynamic_cast<armnn::RefTensorHandle*>(wlConcat->GetData().m_Inputs[1]);

    CHECK(sOut0);
    CHECK(sOut1);
    CHECK(mIn0);
    CHECK(mIn1);

    bool validDataPointers = (sOut0 == mIn1) && (sOut1 == mIn0);

    CHECK(validDataPointers);
}

TEST_CASE("CreateSplitterConcatFloat32")
{
    RefCreateSplitterConcatWorkloadTest<RefSplitterWorkload, RefConcatWorkload, DataType::Float32>();
}

TEST_CASE("CreateSplitterConcatFloat16")
{
    RefCreateSplitterConcatWorkloadTest<RefSplitterWorkload, RefConcatWorkload, DataType::Float16>();
}

TEST_CASE("CreateSplitterConcatUint8")
{
    RefCreateSplitterConcatWorkloadTest<RefSplitterWorkload, RefConcatWorkload, DataType::QAsymmU8>();
}

template <typename SplitterWorkloadType, typename ActivationWorkloadType, armnn::DataType DataType>
static void RefCreateSingleOutputMultipleInputsTest()
{
    // Tests that it is possible to assign multiple (two) different layers to each of the outputs of a splitter layer.
    // We created a splitter with two outputs. That each of those outputs is used by two different activation layers.

    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    std::unique_ptr<SplitterWorkloadType> wlSplitter;
    std::unique_ptr<ActivationWorkloadType> wlActiv0_0;
    std::unique_ptr<ActivationWorkloadType> wlActiv0_1;
    std::unique_ptr<ActivationWorkloadType> wlActiv1_0;
    std::unique_ptr<ActivationWorkloadType> wlActiv1_1;

    CreateSplitterMultipleInputsOneOutputWorkloadTest<SplitterWorkloadType,
        ActivationWorkloadType, DataType>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1, wlActiv1_0, wlActiv1_1);

    armnn::RefTensorHandle* sOut0 = dynamic_cast<armnn::RefTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::RefTensorHandle* sOut1 = dynamic_cast<armnn::RefTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::RefTensorHandle* activ0_0Im = dynamic_cast<armnn::RefTensorHandle*>(wlActiv0_0->GetData().m_Inputs[0]);
    armnn::RefTensorHandle* activ0_1Im = dynamic_cast<armnn::RefTensorHandle*>(wlActiv0_1->GetData().m_Inputs[0]);
    armnn::RefTensorHandle* activ1_0Im = dynamic_cast<armnn::RefTensorHandle*>(wlActiv1_0->GetData().m_Inputs[0]);
    armnn::RefTensorHandle* activ1_1Im = dynamic_cast<armnn::RefTensorHandle*>(wlActiv1_1->GetData().m_Inputs[0]);


    CHECK(sOut0);
    CHECK(sOut1);
    CHECK(activ0_0Im);
    CHECK(activ0_1Im);
    CHECK(activ1_0Im);
    CHECK(activ1_1Im);

    bool validDataPointers = (sOut0 == activ0_0Im) && (sOut0 == activ0_1Im) &&
                             (sOut1 == activ1_0Im) && (sOut1 == activ1_1Im);

    CHECK(validDataPointers);
}

TEST_CASE("CreateSingleOutputMultipleInputsFloat32")
{
    RefCreateSingleOutputMultipleInputsTest<RefSplitterWorkload, RefActivationWorkload,
        armnn::DataType::Float32>();
}

TEST_CASE("CreateSingleOutputMultipleInputsUint8")
{
    RefCreateSingleOutputMultipleInputsTest<RefSplitterWorkload, RefActivationWorkload,
        armnn::DataType::QAsymmU8>();
}

template <typename ResizeBilinearWorkloadType, armnn::DataType DataType>
static void RefCreateResizeBilinearTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateResizeBilinearWorkloadTest<ResizeBilinearWorkloadType, DataType>(factory, graph, dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 2, 4, 4, 3 };
            outputShape = { 2, 2, 2, 3 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape  = { 2, 3, 4, 4 };
            outputShape = { 2, 3, 2, 2 };
    }

    // Checks that outputs and inputs are as we expect them (see definition of CreateResizeBilinearWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType),
                     TensorInfo(outputShape, DataType));
}

TEST_CASE("CreateResizeBilinearFloat32")
{
    RefCreateResizeBilinearTest<RefResizeWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreateResizeBilinearFloat16")
{
    RefCreateResizeBilinearTest<RefResizeWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE("CreateResizeBilinearUint8")
{
    RefCreateResizeBilinearTest<RefResizeWorkload, armnn::DataType::QAsymmU8>(DataLayout::NCHW);
}

TEST_CASE("CreateResizeBilinearQuantisedAsymm16")
{
    RefCreateResizeBilinearTest<RefResizeWorkload, armnn::DataType::QSymmS16>(DataLayout::NCHW);
}

TEST_CASE("CreateResizeBilinearFloat32Nhwc")
{
    RefCreateResizeBilinearTest<RefResizeWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

template <typename BatchToSpaceNdWorkloadType, armnn::DataType DataType>
static void RefCreateBatchToSpaceNdTest()
{
    Graph graph;
    RefWorkloadFactory factory;

    auto workload = CreateBatchToSpaceNdWorkloadTest<BatchToSpaceNdWorkloadType, DataType>(factory, graph);

    CheckInputOutput(std::move(workload),
                     TensorInfo({ 1, 1, 1, 1 }, DataType),
                     TensorInfo({ 1, 1, 1, 1 }, DataType));
}

TEST_CASE("CreateBatchToSpaceNdFloat32")
{
    RefCreateBatchToSpaceNdTest<RefBatchToSpaceNdWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateBatchToSpaceNdFloat16")
{
    RefCreateBatchToSpaceNdTest<RefBatchToSpaceNdWorkload, armnn::DataType::Float16>();
}

TEST_CASE("CreateBatchToSpaceNdUint8")
{
    RefCreateBatchToSpaceNdTest<RefBatchToSpaceNdWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateBatchToSpaceNdQSymm16")
{
    RefCreateBatchToSpaceNdTest<RefBatchToSpaceNdWorkload, armnn::DataType::QSymmS16>();
}

template <typename L2NormalizationWorkloadType, armnn::DataType DataType>
static void RefCreateL2NormalizationTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload =
            CreateL2NormalizationWorkloadTest<L2NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 5, 50, 67, 20 };
            outputShape = { 5, 50, 67, 20 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape  = { 5, 20, 50, 67 };
            outputShape = { 5, 20, 50, 67 };
            break;
    }

    // Checks that outputs and inputs are as we expect them (see definition of CreateL2NormalizationWorkloadTest).
    CheckInputOutput(std::move(workload), TensorInfo(inputShape, DataType), TensorInfo(outputShape, DataType));
}

TEST_CASE("CreateL2NormalizationFloat32")
{
    RefCreateL2NormalizationTest<RefL2NormalizationWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreateL2NormalizationFloat32Nhwc")
{
    RefCreateL2NormalizationTest<RefL2NormalizationWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE("CreateL2NormalizationInt16")
{
    RefCreateL2NormalizationTest<RefL2NormalizationWorkload, armnn::DataType::QSymmS16>(DataLayout::NCHW);
}

TEST_CASE("CreateL2NormalizationInt16Nhwc")
{
    RefCreateL2NormalizationTest<RefL2NormalizationWorkload, armnn::DataType::QSymmS16>(DataLayout::NHWC);
}

TEST_CASE("CreateL2NormalizationUint8")
{
    RefCreateL2NormalizationTest<RefL2NormalizationWorkload, armnn::DataType::QAsymmU8>(DataLayout::NCHW);
}

TEST_CASE("CreateL2NormalizationUint8Nhwc")
{
    RefCreateL2NormalizationTest<RefL2NormalizationWorkload, armnn::DataType::QAsymmU8>(DataLayout::NHWC);
}

template <typename ReshapeWorkloadType, armnn::DataType DataType>
static void RefCreateReshapeWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateReshapeWorkloadTest<ReshapeWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateReshapeWorkloadTest).
    CheckInputOutput(
        std::move(workload),
        TensorInfo({ 4, 1 }, DataType),
        TensorInfo({ 1, 4 }, DataType));
}

TEST_CASE("CreateReshapeWorkloadFloat32")
{
    RefCreateReshapeWorkloadTest<RefReshapeWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateReshapeWorkloadQuantisedAsymm8")
{
    RefCreateReshapeWorkloadTest<RefReshapeWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateReshapeWorkloadQuantisedSymm16")
{
    RefCreateReshapeWorkloadTest<RefReshapeWorkload, armnn::DataType::QSymmS16>();
}

template <typename ConcatWorkloadType, armnn::DataType DataType>
static void RefCreateConcatWorkloadTest(const armnn::TensorShape& outputShape,
                                        unsigned int concatAxis)
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateConcatWorkloadTest<ConcatWorkloadType, DataType>(factory, graph, outputShape, concatAxis);

    CheckInputsOutput(std::move(workload),
                      TensorInfo({ 2, 3, 2, 5 }, DataType),
                      TensorInfo({ 2, 3, 2, 5 }, DataType),
                      TensorInfo(outputShape, DataType));
}

TEST_CASE("CreateConcatDim0Float32Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::Float32>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE("CreateConcatDim0Float16Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::Float16>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE("CreateConcatDim0Uint8Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::QAsymmU8>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE("CreateConcatDim0Uint16Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::QSymmS16>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE("CreateConcatDim1Float32Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::Float32>({ 2, 6, 2, 5 }, 1);
}

TEST_CASE("CreateConcatDim1Uint8Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 6, 2, 5 }, 1);
}

TEST_CASE("CreateConcatDim2Float32Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::Float32>({ 2, 3, 4, 5 }, 2);
}

TEST_CASE("CreateConcatDim2Uint8Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 3, 4, 5 }, 2);
}

TEST_CASE("CreateConcatDim3Float32Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::Float32>({ 2, 3, 2, 10 }, 3);
}

TEST_CASE("CreateConcatDim3Uint8Workload")
{
    RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 3, 2, 10 }, 3);
}

template <typename ConstantWorkloadType, armnn::DataType DataType>
static void RefCreateConstantWorkloadTest(const armnn::TensorShape& outputShape)
{
    armnn::Graph graph;
    RefWorkloadFactory factory = GetFactory();
    auto workload = CreateConstantWorkloadTest<ConstantWorkloadType, DataType>(factory, graph, outputShape);

    // Check output is as expected
    auto queueDescriptor = workload->GetData();
    auto outputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((outputHandle->GetTensorInfo() == TensorInfo(outputShape, DataType)));
}

TEST_CASE("CreateConstantUint8Workload")
{
    RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::QAsymmU8>({ 2, 3, 2, 10 });
}

TEST_CASE("CreateConstantInt16Workload")
{
    RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::QSymmS16>({ 2, 3, 2, 10 });
}

TEST_CASE("CreateConstantFloat32Workload")
{
    RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::Float32>({ 2, 3, 2, 10 });
}

TEST_CASE("CreateConstantSigned32Workload")
{
    RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::Signed32>({ 2, 3, 2, 10 });
}

static void RefCreatePreluWorkloadTest(const armnn::TensorShape& inputShape,
                                       const armnn::TensorShape& alphaShape,
                                       const armnn::TensorShape& outputShape,
                                       armnn::DataType dataType)
{
    armnn::Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreatePreluWorkloadTest<RefPreluWorkload>(factory,
                                                              graph,
                                                              inputShape,
                                                              alphaShape,
                                                              outputShape,
                                                              dataType);

    // Check output is as expected
    auto queueDescriptor = workload->GetData();
    auto outputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((outputHandle->GetTensorInfo() == TensorInfo(outputShape, dataType)));
}

TEST_CASE("CreatePreluFloat32Workload")
{
    RefCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, armnn::DataType::Float32);
}

TEST_CASE("CreatePreluFloat16Workload")
{
    RefCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, armnn::DataType::Float16);
}

TEST_CASE("CreatePreluUint8Workload")
{
    RefCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, armnn::DataType::QAsymmU8);
}

TEST_CASE("CreatePreluInt16Workload")
{
    RefCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, armnn::DataType::QSymmS16);
}

TEST_CASE("CreatePreluFloat32NoBroadcastWorkload")
{
    CHECK_THROWS_AS(RefCreatePreluWorkloadTest({ 1, 4, 7, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 },
                                                 armnn::DataType::Float32),
                      armnn::InvalidArgumentException);
}

TEST_CASE("CreatePreluFloat16NoBroadcastWorkload")
{
    CHECK_THROWS_AS(RefCreatePreluWorkloadTest({ 1, 4, 7, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 },
                                                 armnn::DataType::Float16),
                      armnn::InvalidArgumentException);
}

TEST_CASE("CreatePreluUint8NoBroadcastWorkload")
{
    CHECK_THROWS_AS(RefCreatePreluWorkloadTest({ 1, 4, 7, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 },
                                                 armnn::DataType::QAsymmU8),
                      armnn::InvalidArgumentException);
}

TEST_CASE("CreatePreluInt16NoBroadcastWorkload")
{
    CHECK_THROWS_AS(RefCreatePreluWorkloadTest({ 1, 4, 7, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 },
                                                 armnn::DataType::QSymmS16),
                      armnn::InvalidArgumentException);
}

template <typename SpaceToDepthWorkloadType, armnn::DataType DataType>
static void RefCreateSpaceToDepthWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;

    auto workload = CreateSpaceToDepthWorkloadTest<SpaceToDepthWorkloadType, DataType>(factory, graph);

    CheckInputOutput(std::move(workload),
                     TensorInfo({ 1, 2, 2, 1 }, DataType),
                     TensorInfo({ 1, 1, 1, 4 }, DataType));
}

TEST_CASE("CreateSpaceToDepthWorkloadFloat32")
{
    RefCreateSpaceToDepthWorkloadTest<RefSpaceToDepthWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateSpaceToDepthWorkloadFloat16")
{
    RefCreateSpaceToDepthWorkloadTest<RefSpaceToDepthWorkload, armnn::DataType::Float16>();
}

TEST_CASE("CreateSpaceToDepthWorkloadQASymm8")
{
    RefCreateSpaceToDepthWorkloadTest<RefSpaceToDepthWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateSpaceToDepthWorkloadQSymm16")
{
    RefCreateSpaceToDepthWorkloadTest<RefSpaceToDepthWorkload, armnn::DataType::QSymmS16>();
}

template <armnn::DataType DataType>
static void RefCreateStackWorkloadTest(const armnn::TensorShape& inputShape,
                                       const armnn::TensorShape& outputShape,
                                       unsigned int axis,
                                       unsigned int numInputs)
{
    armnn::Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateStackWorkloadTest<RefStackWorkload, DataType>(factory,
                                                                        graph,
                                                                        inputShape,
                                                                        outputShape,
                                                                        axis,
                                                                        numInputs);

    // Check inputs and output are as expected
    StackQueueDescriptor queueDescriptor = workload->GetData();
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        auto inputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Inputs[i]);
        CHECK((inputHandle->GetTensorInfo() == TensorInfo(inputShape, DataType)));
    }
    auto outputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((outputHandle->GetTensorInfo() == TensorInfo(outputShape, DataType)));
}

TEST_CASE("CreateStackFloat32Workload")
{
    RefCreateStackWorkloadTest<armnn::DataType::Float32>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

TEST_CASE("CreateStackUint8Workload")
{
    RefCreateStackWorkloadTest<armnn::DataType::QAsymmU8>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

TEST_CASE("CreateStackUint16Workload")
{
    RefCreateStackWorkloadTest<armnn::DataType::QSymmS16>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

template <typename QLstmWorkloadType>
static void RefCreateQLstmWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;

    auto workload = CreateQLstmWorkloadTest<QLstmWorkloadType>(factory, graph);

    armnn::TensorInfo inputInfo({2 , 4}, armnn::DataType::QAsymmS8, 0.0078125f, 0);

    armnn::TensorInfo cellStateInfo({2 , 4}, armnn::DataType::QSymmS16, 3.05176e-05f, 0);

    armnn::TensorInfo outputInfo({2 , 4}, armnn::DataType::QAsymmS8, 0.007f, 0);

    QLstmQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto cellStateOutHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[1]);
    auto outputHandle = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[2]);

    CHECK((inputHandle->GetTensorInfo() == inputInfo));
    CHECK((cellStateOutHandle->GetTensorInfo() == cellStateInfo));
    CHECK((outputHandle->GetTensorInfo() == outputInfo));
}

TEST_CASE("CreateQLstmWorkload")
{
    RefCreateQLstmWorkloadTest<RefQLstmWorkload>();
}

template <armnn::DataType DataType>
static void RefCreateActivationWorkloadReplaceFunctionsTest()
{
    Graph graph;
    RefWorkloadFactory factory = GetFactory();
    // input and output are created as armnn::TensorInfo tensorInfo({1, 1}, DataType)
    auto workloadPtr = CreateActivationWorkloadTest<RefActivationWorkload, DataType>(factory, graph);

    // new input and output tensor handlers are created and then replace in the workload
    shared_ptr<RefMemoryManager> memoryManager = make_shared<RefMemoryManager>();
    const RefTensorHandleFactory tensorHandleFactory(memoryManager);
    TensorInfo inputInfo({2 , 2}, armnn::DataType::Float16);
    TensorInfo outputInfo({2 , 2}, armnn::DataType::Float16);
    unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);
    unsigned int slot = 0;
    workloadPtr->ReplaceInputTensorHandle(inputHandle.get(), slot);
    workloadPtr->ReplaceOutputTensorHandle(outputHandle.get(), slot);

    // Check if the tensor handlers inside the workload are the same as ones we replace with
    auto queueDescriptor = workloadPtr->GetData();
    auto inputHandleTest  = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandleTest = PolymorphicDowncast<RefTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((inputHandleTest->GetTensorInfo() == inputInfo));
    CHECK((outputHandleTest->GetTensorInfo() == outputInfo));
    CHECK(inputHandle.get() == inputHandleTest);
    CHECK(outputHandle.get() == outputHandleTest);
    inputHandle->Allocate();
    CHECK(inputHandle->Map() == inputHandleTest->Map());
    outputHandle->Allocate();
    CHECK(outputHandle->Map() == outputHandleTest->Map());
}

TEST_CASE("ReplaceFunctionsfromFloat32toFloat16ActivationWorkload")
{
    RefCreateActivationWorkloadReplaceFunctionsTest<armnn::DataType::Float32>();
}

TEST_CASE("ReplaceFunctionsfromUint8toFloat16ActivationWorkload")
{
    RefCreateActivationWorkloadReplaceFunctionsTest<armnn::DataType::QAsymmU8>();
}

}
