//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <test/CreateWorkload.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <reference/workloads/RefWorkloads.hpp>

namespace
{

template<typename Workload>
void CheckInputOutput(std::unique_ptr<Workload> workload, const TensorInfo& inputInfo, const TensorInfo& outputInfo)
{
    auto queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<ConstCpuTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((inputHandle->GetTensorInfo() == inputInfo));
    BOOST_TEST((outputHandle->GetTensorInfo() == outputInfo));
}

template <typename Workload>
void CheckInputsOutput(std::unique_ptr<Workload> workload,
                       const TensorInfo&         inputInfo0,
                       const TensorInfo&         inputInfo1,
                       const TensorInfo&         outputInfo)
{
    auto queueDescriptor = workload->GetData();
    auto inputHandle0     = boost::polymorphic_downcast<ConstCpuTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle1     = boost::polymorphic_downcast<ConstCpuTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle    = boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((inputHandle0->GetTensorInfo() == inputInfo0));
    BOOST_TEST((inputHandle1->GetTensorInfo() == inputInfo1));
    BOOST_TEST((outputHandle->GetTensorInfo() == outputInfo));
}
}

BOOST_AUTO_TEST_SUITE(CreateWorkloadRef)

template <typename ActivationWorkloadType, armnn::DataType DataType>
static void RefCreateActivationWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateActivationWorkloadTest<ActivationWorkloadType, DataType>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateActivationWorkloadTest).
    CheckInputOutput(std::move(workload),
        TensorInfo({ 1, 1 }, DataType),
        TensorInfo({ 1, 1 }, DataType));
}

BOOST_AUTO_TEST_CASE(CreateActivationFloat32Workload)
{
    RefCreateActivationWorkloadTest<RefActivationFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateActivationUint8Workload)
{
    RefCreateActivationWorkloadTest<RefActivationUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename WorkloadType,
          typename DescriptorType,
          typename LayerType,
          armnn::DataType DataType>
static void RefCreateElementwiseWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateElementwiseWorkloadTest<WorkloadType, DescriptorType, LayerType, DataType>(
        factory, graph);

    CheckInputsOutput(std::move(workload),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType),
        TensorInfo({ 2, 3 }, DataType));
}

BOOST_AUTO_TEST_CASE(CreateAdditionFloatWorkload)
{
    RefCreateElementwiseWorkloadTest<RefAdditionFloat32Workload,
        AdditionQueueDescriptor,
        AdditionLayer,
        armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateAdditionUint8Workload)
{
    RefCreateElementwiseWorkloadTest<RefAdditionUint8Workload,
        AdditionQueueDescriptor,
        AdditionLayer,
        armnn::DataType::QuantisedAsymm8>();
}

BOOST_AUTO_TEST_CASE(CreateSubtractionFloatWorkload)
{
    RefCreateElementwiseWorkloadTest<RefSubtractionFloat32Workload,
        SubtractionQueueDescriptor,
        SubtractionLayer,
        armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSubtractionUint8Workload)
{
    RefCreateElementwiseWorkloadTest<RefSubtractionUint8Workload,
        SubtractionQueueDescriptor,
        SubtractionLayer,
        armnn::DataType::QuantisedAsymm8>();
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloatWorkload)
{
    RefCreateElementwiseWorkloadTest<RefMultiplicationFloat32Workload,
        MultiplicationQueueDescriptor,
        MultiplicationLayer,
        armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationUint8Workload)
{
    RefCreateElementwiseWorkloadTest<RefMultiplicationUint8Workload,
        MultiplicationQueueDescriptor,
        MultiplicationLayer,
        armnn::DataType::QuantisedAsymm8>();
}

BOOST_AUTO_TEST_CASE(CreateDivisionFloatWorkload)
{
    RefCreateElementwiseWorkloadTest<RefDivisionFloat32Workload,
        DivisionQueueDescriptor,
        DivisionLayer,
        armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateDivisionUint8Workload)
{
    RefCreateElementwiseWorkloadTest<RefDivisionUint8Workload,
        DivisionQueueDescriptor,
        DivisionLayer,
        armnn::DataType::QuantisedAsymm8>();
}

template <typename BatchNormalizationWorkloadType, armnn::DataType DataType>
static void RefCreateBatchNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload =
            CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>(factory, graph, dataLayout);

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

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat32Workload)
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationFloat32Workload,armnn::DataType::Float32>
            (DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat32WorkloadNhwc)
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationFloat32Workload, armnn::DataType::Float32>
            (DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationUint8Workload)
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationUint8Workload, armnn::DataType::QuantisedAsymm8>
            (DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationUint8WorkloadNhwc)
{
    RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationUint8Workload, armnn::DataType::QuantisedAsymm8>
            (DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateConvertFp16ToFp32Float32Workload)
{
    Graph                graph;
    RefWorkloadFactory factory;
    auto workload = CreateConvertFp16ToFp32WorkloadTest<RefConvertFp16ToFp32Workload>(factory, graph);

    // Checks that outputs and inputs are as we expect them
    CheckInputOutput(
        std::move(workload), TensorInfo({1, 3, 2, 3}, DataType::Float16), TensorInfo({1, 3, 2, 3}, DataType::Float32));
}

BOOST_AUTO_TEST_CASE(CreateConvertFp32ToFp16Float16Workload)
{
    Graph                graph;
    RefWorkloadFactory factory;
    auto workload = CreateConvertFp32ToFp16WorkloadTest<RefConvertFp32ToFp16Workload>(factory, graph);

    // Checks that outputs and inputs are as we expect them
    CheckInputOutput(
        std::move(workload), TensorInfo({1, 3, 2, 3}, DataType::Float32), TensorInfo({1, 3, 2, 3}, DataType::Float16));
}

static void RefCreateConvolution2dWorkloadTest(DataLayout dataLayout = DataLayout::NCHW)
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateConvolution2dWorkloadTest<RefConvolution2dFloat32Workload, DataType::Float32>
                    (factory, graph, dataLayout);

    std::initializer_list<unsigned int> inputShape  = (dataLayout == DataLayout::NCHW) ?
        std::initializer_list<unsigned int>({2, 3, 8, 16}) : std::initializer_list<unsigned int>({2, 8, 16, 3});
    std::initializer_list<unsigned int> outputShape = (dataLayout == DataLayout::NCHW) ?
        std::initializer_list<unsigned int>({2, 2, 2, 10}) : std::initializer_list<unsigned int>({2, 2, 10, 2});

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType::Float32),
                     TensorInfo(outputShape, DataType::Float32));
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNchwWorkload)
{
    RefCreateConvolution2dWorkloadTest(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNhwcWorkload)
{
    RefCreateConvolution2dWorkloadTest(DataLayout::NHWC);
}

template <typename FullyConnectedWorkloadType, armnn::DataType DataType>
static void RefCreateFullyConnectedWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateFullyConnectedWorkloadTest<FullyConnectedWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    float inputsQScale = DataType == armnn::DataType::QuantisedAsymm8 ? 1.0f : 0.0;
    float outputQScale = DataType == armnn::DataType::QuantisedAsymm8 ? 2.0f : 0.0;
    CheckInputOutput(std::move(workload),
        TensorInfo({ 3, 1, 4, 5 }, DataType, inputsQScale),
        TensorInfo({ 3, 7 }, DataType, outputQScale));
}

BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloat32Workload)
{
    RefCreateFullyConnectedWorkloadTest<RefFullyConnectedFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateFullyConnectedUint8Workload)
{
    RefCreateFullyConnectedWorkloadTest<RefFullyConnectedUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename NormalizationWorkloadType, armnn::DataType DataType>
static void RefCreateNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory;
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

BOOST_AUTO_TEST_CASE(CreateRefNormalizationNchwWorkload)
{
    RefCreateNormalizationWorkloadTest<RefNormalizationFloat32Workload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateRefNormalizationNhwcWorkload)
{
    RefCreateNormalizationWorkloadTest<RefNormalizationFloat32Workload, armnn::DataType::Float32>(DataLayout::NHWC);
}

template <typename Pooling2dWorkloadType, armnn::DataType DataType>
static void RefCreatePooling2dWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory;
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

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat32Workload)
{
    RefCreatePooling2dWorkloadTest<RefPooling2dFloat32Workload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat32NhwcWorkload)
{
    RefCreatePooling2dWorkloadTest<RefPooling2dFloat32Workload, armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dUint8Workload)
{
    RefCreatePooling2dWorkloadTest<RefPooling2dUint8Workload, armnn::DataType::QuantisedAsymm8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dUint8NhwcWorkload)
{
    RefCreatePooling2dWorkloadTest<RefPooling2dUint8Workload, armnn::DataType::QuantisedAsymm8>(DataLayout::NHWC);
}

template <typename SoftmaxWorkloadType, armnn::DataType DataType>
static void RefCreateSoftmaxWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateSoftmaxWorkloadTest<SoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateSoftmaxWorkloadTest).
    CheckInputOutput(
        std::move(workload),
        TensorInfo({4, 1}, DataType),
        TensorInfo({4, 1}, DataType));
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxFloat32Workload)
{
    RefCreateSoftmaxWorkloadTest<RefSoftmaxFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxUint8Workload)
{
    RefCreateSoftmaxWorkloadTest<RefSoftmaxUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename SplitterWorkloadType, armnn::DataType DataType>
static void RefCreateSplitterWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateSplitterWorkloadTest<SplitterWorkloadType, DataType>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateSplitterWorkloadTest).
    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<ConstCpuTensorHandle*>(queueDescriptor.m_Inputs[0]);
    BOOST_TEST((inputHandle->GetTensorInfo() == TensorInfo({ 5, 7, 7 }, DataType)));

    auto outputHandle0 = boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((outputHandle0->GetTensorInfo() == TensorInfo({ 1, 7, 7 }, DataType)));

    auto outputHandle1 = boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST((outputHandle1->GetTensorInfo() == TensorInfo({ 2, 7, 7 }, DataType)));

    auto outputHandle2 = boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[2]);
    BOOST_TEST((outputHandle2->GetTensorInfo() == TensorInfo({ 2, 7, 7 }, DataType)));
}

BOOST_AUTO_TEST_CASE(CreateSplitterFloat32Workload)
{
    RefCreateSplitterWorkloadTest<RefSplitterFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterUint8Workload)
{
    RefCreateSplitterWorkloadTest<RefSplitterUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename SplitterWorkloadType, typename MergerWorkloadType, armnn::DataType DataType>
static void RefCreateSplitterMergerWorkloadTest()
{
    // Tests that it is possible to decide which output of the splitter layer
    // should be lined to which input of the merger layer.
    // We tested that is is possible to specify 0th output
    // of the splitter to be the 1st input to the merger and the 1st output of the splitter to be 0th input
    // of the merger.

    Graph graph;
    RefWorkloadFactory factory;
    auto workloads = CreateSplitterMergerWorkloadTest<SplitterWorkloadType, MergerWorkloadType, DataType>
        (factory, graph);

    auto wlSplitter = std::move(workloads.first);
    auto wlMerger = std::move(workloads.second);

    //Checks that the index of inputs/outputs matches what we declared on InputDescriptor construction.
    armnn::CpuTensorHandle* sOut0 = dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::CpuTensorHandle* sOut1 = dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::CpuTensorHandle* mIn0 = dynamic_cast<armnn::CpuTensorHandle*>(wlMerger->GetData().m_Inputs[0]);
    armnn::CpuTensorHandle* mIn1 = dynamic_cast<armnn::CpuTensorHandle*>(wlMerger->GetData().m_Inputs[1]);

    BOOST_TEST(sOut0);
    BOOST_TEST(sOut1);
    BOOST_TEST(mIn0);
    BOOST_TEST(mIn1);

    bool validDataPointers = (sOut0 == mIn1) && (sOut1 == mIn0);

    BOOST_TEST(validDataPointers);
}

BOOST_AUTO_TEST_CASE(CreateSplitterMergerFloat32)
{
    RefCreateSplitterMergerWorkloadTest<RefSplitterFloat32Workload, RefMergerFloat32Workload, DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterMergerUint8)
{
    RefCreateSplitterMergerWorkloadTest<RefSplitterUint8Workload, RefMergerUint8Workload, DataType::QuantisedAsymm8>();
}

template <typename SplitterWorkloadType, typename ActivationWorkloadType, armnn::DataType DataType>
static void RefCreateSingleOutputMultipleInputsTest()
{
    // Tests that it is possible to assign multiple (two) different layers to each of the outputs of a splitter layer.
    // We created a splitter with two outputs. That each of those outputs is used by two different activation layers.

    Graph graph;
    RefWorkloadFactory factory;
    std::unique_ptr<SplitterWorkloadType> wlSplitter;
    std::unique_ptr<ActivationWorkloadType> wlActiv0_0;
    std::unique_ptr<ActivationWorkloadType> wlActiv0_1;
    std::unique_ptr<ActivationWorkloadType> wlActiv1_0;
    std::unique_ptr<ActivationWorkloadType> wlActiv1_1;

    CreateSplitterMultipleInputsOneOutputWorkloadTest<SplitterWorkloadType,
        ActivationWorkloadType, DataType>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1, wlActiv1_0, wlActiv1_1);

    armnn::CpuTensorHandle* sOut0 = dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::CpuTensorHandle* sOut1 = dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::CpuTensorHandle* activ0_0Im = dynamic_cast<armnn::CpuTensorHandle*>(wlActiv0_0->GetData().m_Inputs[0]);
    armnn::CpuTensorHandle* activ0_1Im = dynamic_cast<armnn::CpuTensorHandle*>(wlActiv0_1->GetData().m_Inputs[0]);
    armnn::CpuTensorHandle* activ1_0Im = dynamic_cast<armnn::CpuTensorHandle*>(wlActiv1_0->GetData().m_Inputs[0]);
    armnn::CpuTensorHandle* activ1_1Im = dynamic_cast<armnn::CpuTensorHandle*>(wlActiv1_1->GetData().m_Inputs[0]);


    BOOST_TEST(sOut0);
    BOOST_TEST(sOut1);
    BOOST_TEST(activ0_0Im);
    BOOST_TEST(activ0_1Im);
    BOOST_TEST(activ1_0Im);
    BOOST_TEST(activ1_1Im);

    bool validDataPointers = (sOut0 == activ0_0Im) && (sOut0 == activ0_1Im) &&
                             (sOut1 == activ1_0Im) && (sOut1 == activ1_1Im);

    BOOST_TEST(validDataPointers);
}

BOOST_AUTO_TEST_CASE(CreateSingleOutputMultipleInputsFloat32)
{
    RefCreateSingleOutputMultipleInputsTest<RefSplitterFloat32Workload, RefActivationFloat32Workload,
        armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSingleOutputMultipleInputsUint8)
{
    RefCreateSingleOutputMultipleInputsTest<RefSplitterUint8Workload, RefActivationUint8Workload,
        armnn::DataType::QuantisedAsymm8>();
}

template <typename ResizeBilinearWorkloadType, armnn::DataType DataType>
static void RefCreateResizeBilinearTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory;
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

BOOST_AUTO_TEST_CASE(CreateResizeBilinearFloat32)
{
    RefCreateResizeBilinearTest<RefResizeBilinearFloat32Workload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateResizeBilinearUint8)
{
    RefCreateResizeBilinearTest<RefResizeBilinearUint8Workload, armnn::DataType::QuantisedAsymm8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateResizeBilinearFloat32Nhwc)
{
    RefCreateResizeBilinearTest<RefResizeBilinearFloat32Workload, armnn::DataType::Float32>(DataLayout::NHWC);
}

template <typename L2NormalizationWorkloadType, armnn::DataType DataType>
static void RefCreateL2NormalizationTest(DataLayout dataLayout)
{
    Graph graph;
    RefWorkloadFactory factory;
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

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat32)
{
    RefCreateL2NormalizationTest<RefL2NormalizationFloat32Workload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat32Nhwc)
{
    RefCreateL2NormalizationTest<RefL2NormalizationFloat32Workload, armnn::DataType::Float32>(DataLayout::NHWC);
}

template <typename ReshapeWorkloadType, armnn::DataType DataType>
static void RefCreateReshapeWorkloadTest()
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateReshapeWorkloadTest<ReshapeWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateReshapeWorkloadTest).
    CheckInputOutput(
        std::move(workload),
        TensorInfo({ 4, 1 }, DataType),
        TensorInfo({ 1, 4 }, DataType));
}

BOOST_AUTO_TEST_CASE(CreateReshapeFloat32Workload)
{
    RefCreateReshapeWorkloadTest<RefReshapeFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeUint8Workload)
{
    RefCreateReshapeWorkloadTest<RefReshapeUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename MergerWorkloadType, armnn::DataType DataType>
static void RefCreateMergerWorkloadTest(const armnn::TensorShape& outputShape,
                                        unsigned int concatAxis)
{
    Graph graph;
    RefWorkloadFactory factory;
    auto workload = CreateMergerWorkloadTest<MergerWorkloadType, DataType>(factory, graph, outputShape, concatAxis);

    CheckInputsOutput(std::move(workload),
                      TensorInfo({ 2, 3, 2, 5 }, DataType),
                      TensorInfo({ 2, 3, 2, 5 }, DataType),
                      TensorInfo(outputShape, DataType));
}

BOOST_AUTO_TEST_CASE(CreateMergerDim0Float32Workload)
{
    RefCreateMergerWorkloadTest<RefMergerFloat32Workload, armnn::DataType::Float32>({ 4, 3, 2, 5 }, 0);
}

BOOST_AUTO_TEST_CASE(CreateMergerDim0Uint8Workload)
{
    RefCreateMergerWorkloadTest<RefMergerUint8Workload, armnn::DataType::QuantisedAsymm8>({ 4, 3, 2, 5 }, 0);
}

BOOST_AUTO_TEST_CASE(CreateMergerDim1Float32Workload)
{
    RefCreateMergerWorkloadTest<RefMergerFloat32Workload, armnn::DataType::Float32>({ 2, 6, 2, 5 }, 1);
}

BOOST_AUTO_TEST_CASE(CreateMergerDim1Uint8Workload)
{
    RefCreateMergerWorkloadTest<RefMergerUint8Workload, armnn::DataType::QuantisedAsymm8>({ 2, 6, 2, 5 }, 1);
}

BOOST_AUTO_TEST_CASE(CreateMergerDim2Float32Workload)
{
    RefCreateMergerWorkloadTest<RefMergerFloat32Workload, armnn::DataType::Float32>({ 2, 3, 4, 5 }, 2);
}

BOOST_AUTO_TEST_CASE(CreateMergerDim2Uint8Workload)
{
    RefCreateMergerWorkloadTest<RefMergerUint8Workload, armnn::DataType::QuantisedAsymm8>({ 2, 3, 4, 5 }, 2);
}

BOOST_AUTO_TEST_CASE(CreateMergerDim3Float32Workload)
{
    RefCreateMergerWorkloadTest<RefMergerFloat32Workload, armnn::DataType::Float32>({ 2, 3, 2, 10 }, 3);
}

BOOST_AUTO_TEST_CASE(CreateMergerDim3Uint8Workload)
{
    RefCreateMergerWorkloadTest<RefMergerUint8Workload, armnn::DataType::QuantisedAsymm8>({ 2, 3, 2, 10 }, 3);
}

BOOST_AUTO_TEST_SUITE_END()
