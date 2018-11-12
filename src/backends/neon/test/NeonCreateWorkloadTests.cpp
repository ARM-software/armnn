//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <backendsCommon/MemCopyWorkload.hpp>

#include <aclCommon/test/CreateWorkloadClNeon.hpp>

#include <neon/NeonWorkloadFactory.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>
#include <neon/workloads/NeonWorkloads.hpp>

BOOST_AUTO_TEST_SUITE(CreateWorkloadNeon)

namespace
{

bool TestNeonTensorHandleInfo(armnn::INeonTensorHandle* handle, const armnn::TensorInfo& expectedInfo)
{
    using namespace armnn::armcomputetensorutils;

    const arm_compute::ITensorInfo* handleInfo = handle->GetTensor().info();
    const arm_compute::TensorInfo expectedAclInfo = BuildArmComputeTensorInfo(expectedInfo);

    if (handleInfo->data_type() != expectedAclInfo.data_type())
    {
        return false;
    }

    if (handleInfo->num_dimensions() != expectedAclInfo.num_dimensions())
    {
        return false;
    }

    if (handleInfo->quantization_info() != expectedAclInfo.quantization_info())
    {
        return false;
    }

    for (std::size_t d = 0; d < expectedAclInfo.num_dimensions(); ++d)
    {
        if (handleInfo->dimension(d) != expectedAclInfo.dimension(d))
        {
            return false;
        }
    }

    return true;
}

} // namespace

template <typename armnn::DataType DataType>
static void NeonCreateActivationWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto workload = CreateActivationWorkloadTest<NeonActivationWorkload, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateActivationWorkloadTest).
    ActivationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({1, 1}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({1, 1}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateActivationFloat16Workload)
{
    NeonCreateActivationWorkloadTest<DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateActivationFloatWorkload)
{
    NeonCreateActivationWorkloadTest<DataType::Float32>();
}

template <typename WorkloadType,
          typename DescriptorType,
          typename LayerType,
          armnn::DataType DataType>
static void NeonCreateArithmethicWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto workload = CreateArithmeticWorkloadTest<WorkloadType, DescriptorType, LayerType, DataType>(factory, graph);

    DescriptorType queueDescriptor = workload->GetData();
    auto inputHandle1 = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle2 = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle1, TensorInfo({2, 3}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle2, TensorInfo({2, 3}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({2, 3}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateAdditionFloat16Workload)
{
    NeonCreateArithmethicWorkloadTest<NeonAdditionFloatWorkload,
                                      AdditionQueueDescriptor,
                                      AdditionLayer,
                                      DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateAdditionFloatWorkload)
{
    NeonCreateArithmethicWorkloadTest<NeonAdditionFloatWorkload,
                                      AdditionQueueDescriptor,
                                      AdditionLayer,
                                      DataType::Float32>();
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateSubtractionFloat16Workload)
{
    NeonCreateArithmethicWorkloadTest<NeonSubtractionFloatWorkload,
                                      SubtractionQueueDescriptor,
                                      SubtractionLayer,
                                      DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateSubtractionFloatWorkload)
{
    NeonCreateArithmethicWorkloadTest<NeonSubtractionFloatWorkload,
                                      SubtractionQueueDescriptor,
                                      SubtractionLayer,
                                      DataType::Float32>();
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateMultiplicationFloat16Workload)
{
    NeonCreateArithmethicWorkloadTest<NeonMultiplicationFloatWorkload,
                                      MultiplicationQueueDescriptor,
                                      MultiplicationLayer,
                                      DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloatWorkload)
{
    NeonCreateArithmethicWorkloadTest<NeonMultiplicationFloatWorkload,
                                      MultiplicationQueueDescriptor,
                                      MultiplicationLayer,
                                      DataType::Float32>();
}

template <typename BatchNormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateBatchNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph                graph;
    NeonWorkloadFactory  factory = NeonWorkloadFactoryHelper::GetFactory();
    auto workload = CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>
                    (factory, graph, dataLayout);

    // Checks that outputs and inputs are as we expect them (see definition of CreateBatchNormalizationWorkloadTest).
    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 4, 4} : TensorShape{2, 4, 4, 3};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 4, 4} : TensorShape{2, 4, 4, 3};

    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat16NchwWorkload)
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationFloatWorkload, DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat16NhwcWorkload)
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationFloatWorkload, DataType::Float16>(DataLayout::NHWC);
}
#endif

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloatNchwWorkload)
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationFloatWorkload, DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloatNhwcWorkload)
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationFloatWorkload, DataType::Float32>(DataLayout::NHWC);
}

template <typename armnn::DataType DataType>
static void NeonCreateConvolution2dWorkloadTest(DataLayout dataLayout = DataLayout::NCHW)
{
    Graph                graph;
    NeonWorkloadFactory  factory = NeonWorkloadFactoryHelper::GetFactory();
    auto                 workload = CreateConvolution2dWorkloadTest<NeonConvolution2dWorkload,
                                    DataType>(factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 8, 16} : TensorShape{2, 8, 16, 3};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 2, 2, 10} : TensorShape{2, 2, 10, 2};

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle,  TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateConvolution2dFloat16NchwWorkload)
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloat16NhwcWorkload)
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float16>(DataLayout::NHWC);
}

#endif
BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNchwWorkload)
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNhwcWorkload)
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float32>(DataLayout::NHWC);
}

template <typename armnn::DataType DataType>
static void NeonCreateDepthWiseConvolutionWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();

    auto workload = CreateDepthwiseConvolution2dWorkloadTest<NeonDepthwiseConvolutionWorkload,
                                                             DataType>(factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    DepthwiseConvolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);

    std::initializer_list<unsigned int> inputShape  = (dataLayout == DataLayout::NCHW)
            ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
            : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });
    std::initializer_list<unsigned int> outputShape = (dataLayout == DataLayout::NCHW)
            ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
            : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });

    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

BOOST_AUTO_TEST_CASE(CreateDepthWiseConvolution2dFloat32NhwcWorkload)
{
    NeonCreateDepthWiseConvolutionWorkloadTest<DataType::Float32>(DataLayout::NHWC);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateDepthWiseConvolution2dFloat16NhwcWorkload)
{
    NeonCreateDepthWiseConvolutionWorkloadTest<DataType::Float16>(DataLayout::NHWC);
}
#endif

template <typename FullyConnectedWorkloadType, typename armnn::DataType DataType>
static void NeonCreateFullyConnectedWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto                workload = CreateFullyConnectedWorkloadTest<FullyConnectedWorkloadType,
                                   DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({3, 1, 4, 5}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({3, 7}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloat16Workload)
{
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedWorkload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloatWorkload)
{
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedWorkload, DataType::Float32>();
}

template <typename NormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto workload = CreateNormalizationWorkloadTest<NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    // Checks that outputs and inputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 5, 5, 1} : TensorShape{3, 1, 5, 5};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 5, 5, 1} : TensorShape{3, 1, 5, 5};

    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateNormalizationFloat16NchwWorkload)
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloat16NhwcWorkload)
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float16>(DataLayout::NHWC);
}
#endif

BOOST_AUTO_TEST_CASE(CreateNormalizationFloatNchwWorkload)
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloatNhwcWorkload)
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float32>(DataLayout::NHWC);
}


template <typename armnn::DataType DataType>
static void NeonCreatePooling2dWorkloadTest(DataLayout dataLayout = DataLayout::NCHW)
{
    Graph               graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto                workload = CreatePooling2dWorkloadTest<NeonPooling2dWorkload, DataType>
                                   (factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 2, 5, 5} : TensorShape{3, 5, 5, 2};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 2, 2, 4} : TensorShape{3, 2, 4, 2};

    // Checks that outputs and inputs are as we expect them (see definition of CreatePooling2dWorkloadTest).
    Pooling2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreatePooling2dFloat16Workload)
{
    NeonCreatePooling2dWorkloadTest<DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreatePooling2dFloatNchwWorkload)
{
    NeonCreatePooling2dWorkloadTest<DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloatNhwcWorkload)
{
    NeonCreatePooling2dWorkloadTest<DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dUint8NchwWorkload)
{
    NeonCreatePooling2dWorkloadTest<DataType::QuantisedAsymm8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dUint8NhwcWorkload)
{
    NeonCreatePooling2dWorkloadTest<DataType::QuantisedAsymm8>(DataLayout::NHWC);
}

template <typename armnn::DataType DataType>
static void NeonCreateReshapeWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto                workload = CreateReshapeWorkloadTest<NeonReshapeWorkload, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateReshapeWorkloadTest).
    ReshapeQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({4, 1}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({1, 4}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateReshapeFloat16Workload)
{
    NeonCreateReshapeWorkloadTest<DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateReshapeFloatWorkload)
{
    NeonCreateReshapeWorkloadTest<DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeUint8Workload)
{
    NeonCreateReshapeWorkloadTest<DataType::QuantisedAsymm8>();
}

template <typename SoftmaxWorkloadType, typename armnn::DataType DataType>
static void NeonCreateSoftmaxWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto workload = CreateSoftmaxWorkloadTest<SoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateSoftmaxWorkloadTest).
    SoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({4, 1}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({4, 1}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateSoftmaxFloat16Workload)
{
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxFloatWorkload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateSoftmaxFloatWorkload)
{
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxFloatWorkload, DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterWorkload)
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto workload = CreateSplitterWorkloadTest<NeonSplitterWorkload, DataType::Float32>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateSplitterWorkloadTest).
    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({5, 7, 7}, DataType::Float32)));

    auto outputHandle0 = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle0, TensorInfo({1, 7, 7}, DataType::Float32)));

    auto outputHandle1 = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle1, TensorInfo({2, 7, 7}, DataType::Float32)));

    auto outputHandle2 = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[2]);
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle2, TensorInfo({2, 7, 7}, DataType::Float32)));
}

BOOST_AUTO_TEST_CASE(CreateSplitterMerger)
{
    // Tests that it is possible to decide which output of the splitter layer
    // should be lined to which input of the merger layer.
    // We tested that is is possible to specify 0th output
    // of the splitter to be the 1st input to the merger, and the 1st output of the splitter to be 0th input
    // of the merger.

    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();

    auto workloads =
        CreateSplitterMergerWorkloadTest<NeonSplitterWorkload, NeonMergerWorkload,
            DataType::Float32>(factory, graph);

    auto wlSplitter = std::move(workloads.first);
    auto wlMerger = std::move(workloads.second);

    //Checks that the index of inputs/outputs matches what we declared on InputDescriptor construction.
    armnn::INeonTensorHandle* sOut0 = dynamic_cast<armnn::INeonTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::INeonTensorHandle* sOut1 = dynamic_cast<armnn::INeonTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::INeonTensorHandle* mIn0 = dynamic_cast<armnn::INeonTensorHandle*>(wlMerger->GetData().m_Inputs[0]);
    armnn::INeonTensorHandle* mIn1 = dynamic_cast<armnn::INeonTensorHandle*>(wlMerger->GetData().m_Inputs[1]);

    BOOST_TEST(sOut0);
    BOOST_TEST(sOut1);
    BOOST_TEST(mIn0);
    BOOST_TEST(mIn1);

    bool validDataPointers = (sOut0 == mIn1) && (sOut1 == mIn0);

    BOOST_TEST(validDataPointers);
}

BOOST_AUTO_TEST_CASE(CreateSingleOutputMultipleInputs)
{
    // Tests that it is possible to assign multiple (two) different layers to each of the outputs of a splitter layer.
    // We created a splitter with two outputs. That each of those outputs is used by two different activation layers

    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    std::unique_ptr<NeonSplitterWorkload> wlSplitter;
    std::unique_ptr<NeonActivationWorkload> wlActiv0_0;
    std::unique_ptr<NeonActivationWorkload> wlActiv0_1;
    std::unique_ptr<NeonActivationWorkload> wlActiv1_0;
    std::unique_ptr<NeonActivationWorkload> wlActiv1_1;

    CreateSplitterMultipleInputsOneOutputWorkloadTest<NeonSplitterWorkload,
        NeonActivationWorkload, DataType::Float32>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1,
                                                   wlActiv1_0, wlActiv1_1);

    armnn::INeonTensorHandle* sOut0 = dynamic_cast<armnn::INeonTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::INeonTensorHandle* sOut1 = dynamic_cast<armnn::INeonTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::INeonTensorHandle* activ0_0Im = dynamic_cast<armnn::INeonTensorHandle*>(wlActiv0_0->GetData().m_Inputs[0]);
    armnn::INeonTensorHandle* activ0_1Im = dynamic_cast<armnn::INeonTensorHandle*>(wlActiv0_1->GetData().m_Inputs[0]);
    armnn::INeonTensorHandle* activ1_0Im = dynamic_cast<armnn::INeonTensorHandle*>(wlActiv1_0->GetData().m_Inputs[0]);
    armnn::INeonTensorHandle* activ1_1Im = dynamic_cast<armnn::INeonTensorHandle*>(wlActiv1_1->GetData().m_Inputs[0]);


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

BOOST_AUTO_TEST_CASE(CreateMemCopyWorkloadsNeon)
{
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    CreateMemCopyWorkloads<INeonTensorHandle>(factory);
}

template <typename L2NormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateL2NormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory();
    auto workload =
            CreateL2NormalizationWorkloadTest<L2NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    L2NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 5, 20, 50, 67 } : TensorShape{ 5, 50, 67, 20 };
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 5, 20, 50, 67 } : TensorShape{ 5, 50, 67, 20 };

    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat16NchwWorkload)
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat16NhwcWorkload)
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float16>(DataLayout::NHWC);
}
#endif

BOOST_AUTO_TEST_CASE(CreateL2NormalizationNchwWorkload)
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationNhwcWorkload)
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_SUITE_END()
