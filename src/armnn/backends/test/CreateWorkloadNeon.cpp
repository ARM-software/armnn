//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "backends/NeonWorkloadFactory.hpp"
#include "backends/NeonWorkloadUtils.hpp"
#include "backends/NeonWorkloads.hpp"
#include "backends/MemCopyWorkload.hpp"
#include "backends/NeonTensorHandle.hpp"

#include "test/CreateWorkloadClNeon.hpp"

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

template <typename ActivationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateActivationWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory;
    auto workload = CreateActivationWorkloadTest<ActivationWorkloadType, DataType>
            (factory, graph);

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
    NeonCreateActivationWorkloadTest<NeonActivationFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateActivationFloat32Workload)
{
    NeonCreateActivationWorkloadTest<NeonActivationFloat32Workload, DataType::Float32>();
}

template <typename AdditionWorkloadType, typename armnn::DataType DataType>
static void NeonCreateAdditionWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory;
    auto workload = CreateAdditionWorkloadTest<AdditionWorkloadType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateAdditionWorkloadTest).
    AdditionQueueDescriptor queueDescriptor = workload->GetData();
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
    NeonCreateAdditionWorkloadTest<NeonAdditionFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateAdditionFloat32Workload)
{
    NeonCreateAdditionWorkloadTest<NeonAdditionFloat32Workload, DataType::Float32>();
}

template <typename BatchNormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateBatchNormalizationWorkloadTest()
{
    Graph                graph;
    NeonWorkloadFactory  factory;
    auto workload = CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateBatchNormalizationWorkloadTest).
    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({2, 3, 1, 1}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({2, 3, 1, 1}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat16Workload)
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat32Workload)
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationFloat32Workload, DataType::Float32>();
}

template <typename Convolution2dWorkloadType, typename armnn::DataType DataType>
static void NeonCreateConvolution2dWorkloadTest()
{
    Graph                graph;
    NeonWorkloadFactory  factory;
    auto                 workload = CreateConvolution2dWorkloadTest<Convolution2dWorkloadType,
                                    DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({2, 3, 8, 16}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle,  TensorInfo({2, 2, 2, 10}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateConvolution2dFloat16Workload)
{
    NeonCreateConvolution2dWorkloadTest<NeonConvolution2dFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloat32Workload)
{
    NeonCreateConvolution2dWorkloadTest<NeonConvolution2dFloat32Workload, DataType::Float32>();
}

template <typename FullyConnectedWorkloadType, typename armnn::DataType DataType>
static void NeonCreateFullyConnectedWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory;
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
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloat32Workload)
{
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedFloat32Workload, DataType::Float32>();
}

template <typename MultiplicationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateMultiplicationWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory;
    auto                workload = CreateMultiplicationWorkloadTest<MultiplicationWorkloadType,
                                   DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateMultiplicationWorkloadTest).
    MultiplicationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle1 = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle2 = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle1, TensorInfo({2, 3}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle2, TensorInfo({2, 3}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({2, 3}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateMultiplicationFloat16Workload)
{
    NeonCreateMultiplicationWorkloadTest<NeonMultiplicationFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloat32Workload)
{
    NeonCreateMultiplicationWorkloadTest<NeonMultiplicationFloat32Workload, DataType::Float32>();
}

template <typename NormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateNormalizationWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory;
    auto                workload = CreateNormalizationWorkloadTest<NormalizationWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({3, 5, 5, 1}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({3, 5, 5, 1}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreateNormalizationFloat16Workload)
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateNormalizationFloat32Workload)
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloat32Workload, DataType::Float32>();
}

template <typename Pooling2dWorkloadType, typename armnn::DataType DataType>
static void NeonCreatePooling2dWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory;
    auto                workload = CreatePooling2dWorkloadTest<Pooling2dWorkloadType, DataType>
                                   (factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreatePooling2dWorkloadTest).
    Pooling2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = boost::polymorphic_downcast<INeonTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(TestNeonTensorHandleInfo(inputHandle, TensorInfo({3, 2, 5, 5}, DataType)));
    BOOST_TEST(TestNeonTensorHandleInfo(outputHandle, TensorInfo({3, 2, 2, 4}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
BOOST_AUTO_TEST_CASE(CreatePooling2dFloat16Workload)
{
    NeonCreatePooling2dWorkloadTest<NeonPooling2dFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat32Workload)
{
    NeonCreatePooling2dWorkloadTest<NeonPooling2dFloat32Workload, DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreatePooling2dUint8Workload)
{
    NeonCreatePooling2dWorkloadTest<NeonPooling2dUint8Workload, DataType::QuantisedAsymm8>();
}

template <typename ReshapeWorkloadType, typename armnn::DataType DataType>
static void NeonCreateReshapeWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory;
    auto                workload = CreateReshapeWorkloadTest<ReshapeWorkloadType, DataType>(factory, graph);

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
    NeonCreateReshapeWorkloadTest<NeonReshapeFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateReshapeFloat32Workload)
{
    NeonCreateReshapeWorkloadTest<NeonReshapeFloat32Workload, DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeUint8Workload)
{
    NeonCreateReshapeWorkloadTest<NeonReshapeUint8Workload, DataType::QuantisedAsymm8>();
}

template <typename SoftmaxWorkloadType, typename armnn::DataType DataType>
static void NeonCreateSoftmaxWorkloadTest()
{
    Graph               graph;
    NeonWorkloadFactory factory;
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
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxFloat32Workload, DataType::Float16>();
}
#endif

BOOST_AUTO_TEST_CASE(CreateSoftmaxFloat32Workload)
{
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxFloat32Workload, DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterWorkload)
{
    Graph graph;
    NeonWorkloadFactory factory;
    auto workload = CreateSplitterWorkloadTest<NeonSplitterFloat32Workload, DataType::Float32>(factory, graph);

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
    NeonWorkloadFactory factory;

    auto workloads =
        CreateSplitterMergerWorkloadTest<NeonSplitterFloat32Workload, NeonMergerFloat32Workload,
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
    NeonWorkloadFactory factory;
    std::unique_ptr<NeonSplitterFloat32Workload> wlSplitter;
    std::unique_ptr<NeonActivationFloat32Workload> wlActiv0_0;
    std::unique_ptr<NeonActivationFloat32Workload> wlActiv0_1;
    std::unique_ptr<NeonActivationFloat32Workload> wlActiv1_0;
    std::unique_ptr<NeonActivationFloat32Workload> wlActiv1_1;

    CreateSplitterMultipleInputsOneOutputWorkloadTest<NeonSplitterFloat32Workload,
        NeonActivationFloat32Workload, DataType::Float32>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1,
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
    NeonWorkloadFactory    factory;
    CreateMemCopyWorkloads<INeonTensorHandle>(factory);
}

BOOST_AUTO_TEST_SUITE_END()
