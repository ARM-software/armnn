//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "backends/ClWorkloadFactory.hpp"
#include "backends/RefWorkloadFactory.hpp"
#include "backends/MemCopyWorkload.hpp"
#include "backends/ClWorkloadUtils.hpp"
#include "backends/ClWorkloads.hpp"
#include "backends/ClTensorHandle.hpp"
#include "ClContextControlFixture.hpp"

#include "test/CreateWorkloadClNeon.hpp"

boost::test_tools::predicate_result CompareIClTensorHandleShape(IClTensorHandle*                    tensorHandle,
                                                                std::initializer_list<unsigned int> expectedDimensions)
{
    return CompareTensorHandleShape<IClTensorHandle>(tensorHandle, expectedDimensions);
}

BOOST_FIXTURE_TEST_SUITE(CreateWorkloadCl, ClContextControlFixture)

template <typename ActivationWorkloadType, armnn::DataType DataType>
static void ClCreateActivationWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreateActivationWorkloadTest<ActivationWorkloadType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateActivationWorkloadTest).
    ActivationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {1}));
}

BOOST_AUTO_TEST_CASE(CreateActivationFloatWorkload)
{
    ClCreateActivationWorkloadTest<ClActivationFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateActivationFloat16Workload)
{
    ClCreateActivationWorkloadTest<ClActivationFloatWorkload, armnn::DataType::Float16>();
}

template <typename AdditionWorkloadType, armnn::DataType DataType>
static void ClCreateAdditionWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;
    auto workload = CreateAdditionWorkloadTest<AdditionWorkloadType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateAdditionWorkloadTest).
    AdditionQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle1 = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle2 = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle1, {2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle2, {2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 3}));
}

BOOST_AUTO_TEST_CASE(CreateAdditionFloatWorkload)
{
    ClCreateAdditionWorkloadTest<ClAdditionFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateAdditionFloat16Workload)
{
    ClCreateAdditionWorkloadTest<ClAdditionFloatWorkload, armnn::DataType::Float16>();
}

template <typename BatchNormalizationWorkloadType, armnn::DataType DataType>
static void ClCreateBatchNormalizationWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>
                    (factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateBatchNormalizationWorkloadTest).
    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {2, 3, 1, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 3, 1, 1}));
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloatWorkload)
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat16Workload)
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload, armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateConvertFp16ToFp32Workload)
{
    Graph graph;
    ClWorkloadFactory factory;
    auto workload = CreateConvertFp16ToFp32WorkloadTest<ClConvertFp16ToFp32Workload>(factory, graph);

    ConvertFp16ToFp32QueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {3, 2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {3, 2, 3}));
    BOOST_TEST((inputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F16));
    BOOST_TEST((outputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F32));
}

BOOST_AUTO_TEST_CASE(CreateConvertFp32ToFp16Workload)
{
    Graph graph;
    ClWorkloadFactory factory;
    auto workload = CreateConvertFp32ToFp16WorkloadTest<ClConvertFp32ToFp16Workload>(factory, graph);

    ConvertFp32ToFp16QueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {3, 2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {3, 2, 3}));
    BOOST_TEST((inputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F32));
    BOOST_TEST((outputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F16));
}

template <typename Convolution2dWorkloadType, typename armnn::DataType DataType>
static void ClConvolution2dWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;
    auto                workload = CreateConvolution2dWorkloadTest<Convolution2dWorkloadType, DataType>
                                   (factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {2, 3, 8, 16}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 2, 2, 10}));
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatWorkload)
{
    ClConvolution2dWorkloadTest<ClConvolution2dFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloat16Workload)
{
    ClConvolution2dWorkloadTest<ClConvolution2dFloatWorkload, armnn::DataType::Float16>();
}


template <typename Convolution2dWorkloadType, typename armnn::DataType DataType>
static void ClDirectConvolution2dWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;
    auto workload = CreateDirectConvolution2dWorkloadTest<Convolution2dWorkloadType, DataType>(
            factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateDirectConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {2, 3, 6, 6}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 2, 6, 6}));
}

BOOST_AUTO_TEST_CASE(CreateDirectConvolution2dFloatWorkload)
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateDirectConvolution2dFloat16Workload)
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dFloatWorkload, armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateDirectConvolution2dUint8Workload)
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename FullyConnectedWorkloadType, typename armnn::DataType DataType>
static void ClCreateFullyConnectedWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;
    auto workload =
        CreateFullyConnectedWorkloadTest<FullyConnectedWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {3, 1, 4, 5}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {3, 7}));
}


BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloatWorkloadTest)
{
    ClCreateFullyConnectedWorkloadTest<ClFullyConnectedFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloat16WorkloadTest)
{
    ClCreateFullyConnectedWorkloadTest<ClFullyConnectedFloatWorkload, armnn::DataType::Float16>();
}


template <typename MultiplicationWorkloadType, typename armnn::DataType DataType>
static void ClCreateMultiplicationWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload =
        CreateMultiplicationWorkloadTest<MultiplicationWorkloadType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateMultiplicationWorkloadTest).
    MultiplicationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle1 = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle2 = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle1, {2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle2, {2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 3}));
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloatWorkloadTest)
{
    ClCreateMultiplicationWorkloadTest<ClMultiplicationFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloat16WorkloadTest)
{
    ClCreateMultiplicationWorkloadTest<ClMultiplicationFloatWorkload, armnn::DataType::Float16>();
}

template <typename NormalizationWorkloadType, typename armnn::DataType DataType>
static void ClNormalizationWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreateNormalizationWorkloadTest<NormalizationWorkloadType, DataType>
                    (factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {3, 5, 5, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {3, 5, 5, 1}));
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloatWorkload)
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloat16Workload)
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float16>();
}

template <typename Pooling2dWorkloadType, typename armnn::DataType DataType>
static void ClPooling2dWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreatePooling2dWorkloadTest<Pooling2dWorkloadType, DataType>(factory, graph);

    // Check that inputs/outputs are as we expect them (see definition of CreatePooling2dWorkloadTest).
    Pooling2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {3, 2, 5, 5}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {3, 2, 2, 4}));
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloatWorkload)
{
    ClPooling2dWorkloadTest<ClPooling2dFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat16Workload)
{
    ClPooling2dWorkloadTest<ClPooling2dFloatWorkload, armnn::DataType::Float16>();
}

template <typename ReshapeWorkloadType, typename armnn::DataType DataType>
static void ClCreateReshapeWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreateReshapeWorkloadTest<ReshapeWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateReshapeWorkloadTest).
    ReshapeQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {4, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {4})); // Leading size 1 dimensions are collapsed by ACL.
}

BOOST_AUTO_TEST_CASE(CreateReshapeFloatWorkload)
{
    ClCreateReshapeWorkloadTest<ClReshapeFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeFloat16Workload)
{
    ClCreateReshapeWorkloadTest<ClReshapeFloatWorkload, armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeUint8Workload)
{
    ClCreateReshapeWorkloadTest<ClReshapeUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename SoftmaxWorkloadType, typename armnn::DataType DataType>
static void ClSoftmaxWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreateSoftmaxWorkloadTest<SoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of ClSoftmaxFloatWorkload).
    SoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {4, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {4, 1}));
}


BOOST_AUTO_TEST_CASE(CreateSoftmaxFloatWorkloadTest)
{
    ClSoftmaxWorkloadTest<ClSoftmaxFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxFloat16WorkloadTest)
{
    ClSoftmaxWorkloadTest<ClSoftmaxFloatWorkload, armnn::DataType::Float16>();
}

template <typename SplitterWorkloadType, typename armnn::DataType DataType>
static void ClSplitterWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreateSplitterWorkloadTest<SplitterWorkloadType, DataType>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateSplitterWorkloadTest).
    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {5, 7, 7}));

    auto outputHandle1 = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle1, {2, 7, 7}));

    auto outputHandle2 = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[2]);
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle2, {2, 7, 7}));

    auto outputHandle0 = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    // NOTE: At the moment the CL collapses the tensor to a 2 dim when dimension zero = 1
    //       we are raising this difference between the NEON and CL libs as an issue with the compute library team.
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle0, {7, 7}));
}

BOOST_AUTO_TEST_CASE(CreateSplitterFloatWorkload)
{
    ClSplitterWorkloadTest<ClSplitterFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterFloat16Workload)
{
    ClSplitterWorkloadTest<ClSplitterFloatWorkload, armnn::DataType::Float16>();
}

template <typename SplitterWorkloadType, typename MergerWorkloadType, typename armnn::DataType DataType>
static void ClSplitterMergerTest()
{
    // Tests that it is possible to decide which output of the splitter layer
    // should be lined to which input of the merger layer.
    // We test that is is possible to specify 0th output
    // of the splitter to be the 1st input to the merger and the 1st output of the splitter  to be 0th input
    // of the merger.

    Graph graph;
    ClWorkloadFactory factory;

    auto workloads =
        CreateSplitterMergerWorkloadTest<SplitterWorkloadType, MergerWorkloadType, DataType>
            (factory, graph);

    auto wlSplitter = std::move(workloads.first);
    auto wlMerger = std::move(workloads.second);

    //Checks that the index of inputs/outputs matches what we declared on InputDescriptor construction.
    armnn::ClSubTensorHandle* sOut0 = dynamic_cast<armnn::ClSubTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::ClSubTensorHandle* sOut1 = dynamic_cast<armnn::ClSubTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::ClSubTensorHandle* mIn0 = dynamic_cast<armnn::ClSubTensorHandle*>(wlMerger->GetData().m_Inputs[0]);
    armnn::ClSubTensorHandle* mIn1 = dynamic_cast<armnn::ClSubTensorHandle*>(wlMerger->GetData().m_Inputs[1]);

    BOOST_TEST(sOut0);
    BOOST_TEST(sOut1);
    BOOST_TEST(mIn0);
    BOOST_TEST(mIn1);

    //Fliped order of inputs/outputs.
    bool validDataPointers = (sOut0 == mIn1) && (sOut1 == mIn0);
    BOOST_TEST(validDataPointers);


    //Also make sure that the inputs are subtensors of one tensor and outputs are sub tensors of another tensor.
    bool validSubTensorParents = (mIn0->GetTensor().parent() == mIn1->GetTensor().parent())
                                    && (sOut0->GetTensor().parent() == sOut1->GetTensor().parent());

    BOOST_TEST(validSubTensorParents);
}

BOOST_AUTO_TEST_CASE(CreateSplitterMergerFloatWorkload)
{
    ClSplitterMergerTest<ClSplitterFloatWorkload, ClMergerFloatWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterMergerFloat16Workload)
{
    ClSplitterMergerTest<ClSplitterFloatWorkload, ClMergerFloatWorkload, armnn::DataType::Float16>();
}


BOOST_AUTO_TEST_CASE(CreateSingleOutputMultipleInputs)
{
    // Test that it is possible to assign multiple (two) different layers to each of the outputs of a splitter layer.
    // We create a splitter with two outputs. That each of those outputs is used by two different activation layers.

    Graph graph;
    ClWorkloadFactory factory;
    std::unique_ptr<ClSplitterFloatWorkload> wlSplitter;
    std::unique_ptr<ClActivationFloatWorkload> wlActiv0_0;
    std::unique_ptr<ClActivationFloatWorkload> wlActiv0_1;
    std::unique_ptr<ClActivationFloatWorkload> wlActiv1_0;
    std::unique_ptr<ClActivationFloatWorkload> wlActiv1_1;

    CreateSplitterMultipleInputsOneOutputWorkloadTest<ClSplitterFloatWorkload,
        ClActivationFloatWorkload, armnn::DataType::Float32>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1,
                                                               wlActiv1_0, wlActiv1_1);

    //Checks that the index of inputs/outputs matches what we declared on InputDescriptor construction.
    armnn::ClSubTensorHandle* sOut0 = dynamic_cast<armnn::ClSubTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::ClSubTensorHandle* sOut1 = dynamic_cast<armnn::ClSubTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::ClSubTensorHandle* activ0_0Im = dynamic_cast<armnn::ClSubTensorHandle*>(wlActiv0_0->GetData().m_Inputs[0]);
    armnn::ClSubTensorHandle* activ0_1Im = dynamic_cast<armnn::ClSubTensorHandle*>(wlActiv0_1->GetData().m_Inputs[0]);
    armnn::ClSubTensorHandle* activ1_0Im = dynamic_cast<armnn::ClSubTensorHandle*>(wlActiv1_0->GetData().m_Inputs[0]);
    armnn::ClSubTensorHandle* activ1_1Im = dynamic_cast<armnn::ClSubTensorHandle*>(wlActiv1_1->GetData().m_Inputs[0]);


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

BOOST_AUTO_TEST_CASE(CreateMemCopyWorkloadsCl)
{
    ClWorkloadFactory    factory;
    CreateMemCopyWorkloads<IClTensorHandle>(factory);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationWorkload)
{
    Graph graph;
    ClWorkloadFactory factory;

    auto workload = CreateL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float32>
        (factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    L2NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 5, 20, 50, 67 }));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 5, 20, 50, 67 }));
}

template <typename LstmWorkloadType>
static void ClCreateLstmWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory;
    auto workload = CreateLstmWorkloadTest<LstmWorkloadType>(factory, graph);

    LstmQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<IClTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 2, 2 }));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 2, 4 }));
}

BOOST_AUTO_TEST_CASE(CreateLSTMWorkloadFloatWorkload)
{
    ClCreateLstmWorkloadTest<ClLstmFloatWorkload>();
}


BOOST_AUTO_TEST_SUITE_END()
