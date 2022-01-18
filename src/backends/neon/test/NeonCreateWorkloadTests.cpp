//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>

#include <aclCommon/test/CreateWorkloadClNeon.hpp>

#include <neon/NeonWorkloadFactory.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>
#include <neon/workloads/NeonWorkloads.hpp>

#include <doctest/doctest.h>

TEST_SUITE("CreateWorkloadNeon")
{
namespace
{

armnn::PredicateResult CompareIAclTensorHandleShape(IAclTensorHandle* tensorHandle,
                                                    std::initializer_list<unsigned int> expectedDimensions)
{
    return CompareTensorHandleShape<IAclTensorHandle>(tensorHandle, expectedDimensions);
}

bool TestNeonTensorHandleInfo(armnn::IAclTensorHandle* handle, const armnn::TensorInfo& expectedInfo)
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
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateActivationWorkloadTest<NeonActivationWorkload, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateActivationWorkloadTest).
    ActivationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo({1, 1}, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo({1, 1}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateActivationFloat16Workload")
{
    NeonCreateActivationWorkloadTest<DataType::Float16>();
}
#endif

TEST_CASE("CreateActivationFloatWorkload")
{
    NeonCreateActivationWorkloadTest<DataType::Float32>();
}

template <typename WorkloadType,
          typename DescriptorType,
          typename LayerType,
          armnn::DataType DataType>
static void NeonCreateElementwiseWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateElementwiseWorkloadTest<WorkloadType, DescriptorType, LayerType, DataType>(factory, graph);

    DescriptorType queueDescriptor = workload->GetData();
    auto inputHandle1 = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle2 = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(inputHandle1, TensorInfo({2, 3}, DataType)));
    CHECK(TestNeonTensorHandleInfo(inputHandle2, TensorInfo({2, 3}, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo({2, 3}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateAdditionFloat16Workload")
{
    NeonCreateElementwiseWorkloadTest<NeonAdditionWorkload,
                                      AdditionQueueDescriptor,
                                      AdditionLayer,
                                      DataType::Float16>();
}
#endif

TEST_CASE("CreateAdditionFloatWorkload")
{
    NeonCreateElementwiseWorkloadTest<NeonAdditionWorkload,
                                      AdditionQueueDescriptor,
                                      AdditionLayer,
                                      DataType::Float32>();
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateSubtractionFloat16Workload")
{
    NeonCreateElementwiseWorkloadTest<NeonSubtractionWorkload,
                                      SubtractionQueueDescriptor,
                                      SubtractionLayer,
                                      DataType::Float16>();
}
#endif

TEST_CASE("CreateSubtractionFloatWorkload")
{
    NeonCreateElementwiseWorkloadTest<NeonSubtractionWorkload,
                                      SubtractionQueueDescriptor,
                                      SubtractionLayer,
                                      DataType::Float32>();
}

TEST_CASE("CreateSubtractionUint8Workload")
{
    NeonCreateElementwiseWorkloadTest<NeonSubtractionWorkload,
                                      SubtractionQueueDescriptor,
                                      SubtractionLayer,
                                      DataType::QAsymmU8>();
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateMultiplicationFloat16Workload")
{
    NeonCreateElementwiseWorkloadTest<NeonMultiplicationWorkload,
                                      MultiplicationQueueDescriptor,
                                      MultiplicationLayer,
                                      DataType::Float16>();
}
#endif

TEST_CASE("CreateMultiplicationFloatWorkload")
{
    NeonCreateElementwiseWorkloadTest<NeonMultiplicationWorkload,
                                      MultiplicationQueueDescriptor,
                                      MultiplicationLayer,
                                      DataType::Float32>();
}

TEST_CASE("CreateMultiplicationUint8Workload")
{
    NeonCreateElementwiseWorkloadTest<NeonMultiplicationWorkload,
                                      MultiplicationQueueDescriptor,
                                      MultiplicationLayer,
                                      DataType::QAsymmU8>();
}

TEST_CASE("CreateDivisionFloatWorkloadTest")
{
    NeonCreateElementwiseWorkloadTest<NeonDivisionWorkload,
                                      DivisionQueueDescriptor,
                                      DivisionLayer,
                                      armnn::DataType::Float32>();
}

template <typename BatchNormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateBatchNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>
                    (factory, graph, dataLayout);

    // Checks that outputs and inputs are as we expect them (see definition of CreateBatchNormalizationWorkloadTest).
    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 4, 4} : TensorShape{2, 4, 4, 3};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 4, 4} : TensorShape{2, 4, 4, 3};

    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateBatchNormalizationFloat16NchwWorkload")
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationWorkload, DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE("CreateBatchNormalizationFloat16NhwcWorkload")
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationWorkload, DataType::Float16>(DataLayout::NHWC);
}
#endif

TEST_CASE("CreateBatchNormalizationFloatNchwWorkload")
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationWorkload, DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreateBatchNormalizationFloatNhwcWorkload")
{
    NeonCreateBatchNormalizationWorkloadTest<NeonBatchNormalizationWorkload, DataType::Float32>(DataLayout::NHWC);
}

template <typename armnn::DataType DataType>
static void NeonCreateConvolution2dWorkloadTest(DataLayout dataLayout = DataLayout::NCHW)
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConvolution2dWorkloadTest<NeonConvolution2dWorkload, DataType>(factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 3, 8, 16} : TensorShape{2, 8, 16, 3};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{2, 2, 2, 10} : TensorShape{2, 2, 10, 2};

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle,  TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateConvolution2dFloat16NchwWorkload")
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float16>();
}

TEST_CASE("CreateConvolution2dFloat16NhwcWorkload")
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float16>(DataLayout::NHWC);
}

#endif
TEST_CASE("CreateConvolution2dFloatNchwWorkload")
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float32>();
}

TEST_CASE("CreateConvolution2dFloatNhwcWorkload")
{
    NeonCreateConvolution2dWorkloadTest<DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE("CreateConvolution2dFastMathEnabledWorkload")
{
    Graph graph;
    using ModelOptions = std::vector<BackendOptions>;
    ModelOptions modelOptions = {};
    BackendOptions cpuAcc("CpuAcc",
    {
        { "FastMathEnabled", true }
    });
    modelOptions.push_back(cpuAcc);
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager(), modelOptions);

    auto workload =
        CreateConvolution2dWorkloadFastMathTest<NeonConvolution2dWorkload, armnn::DataType::Float32>(factory,
                                                                                             graph,
                                                                                             DataLayout::NCHW,
                                                                                             modelOptions);

    ARMNN_ASSERT(workload != nullptr);
    auto conv2dWorkload = PolymorphicDowncast<NeonConvolution2dWorkload*>(workload.get());
    IgnoreUnused(conv2dWorkload);
    ARMNN_ASSERT(conv2dWorkload != nullptr);
    ARMNN_ASSERT(conv2dWorkload->GetConvolutionMethod() == arm_compute::ConvolutionMethod::WINOGRAD);
}

template <typename armnn::DataType DataType>
static void NeonCreateDepthWiseConvolutionWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateDepthwiseConvolution2dWorkloadTest<NeonDepthwiseConvolutionWorkload,
                                                             DataType>(factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    DepthwiseConvolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });

    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

TEST_CASE("CreateDepthWiseConvolution2dFloat32NhwcWorkload")
{
    NeonCreateDepthWiseConvolutionWorkloadTest<DataType::Float32>(DataLayout::NHWC);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateDepthWiseConvolution2dFloat16NhwcWorkload")
{
    NeonCreateDepthWiseConvolutionWorkloadTest<DataType::Float16>(DataLayout::NHWC);
}
#endif

template <typename FullyConnectedWorkloadType, typename armnn::DataType DataType>
static void NeonCreateFullyConnectedWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateFullyConnectedWorkloadTest<FullyConnectedWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    float inputsQScale = DataType == armnn::DataType::QAsymmU8 ? 1.0f : 0.0;
    float outputQScale = DataType == armnn::DataType::QAsymmU8 ? 2.0f : 0.0;
    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo({3, 1, 4, 5}, DataType, inputsQScale)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo({3, 7}, DataType, outputQScale)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateFullyConnectedFloat16Workload")
{
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedWorkload, DataType::Float16>();
}
#endif

TEST_CASE("CreateFullyConnectedFloatWorkload")
{
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedWorkload, DataType::Float32>();
}

TEST_CASE("CreateFullyConnectedQAsymmU8Workload")
{
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedWorkload, DataType::QAsymmU8>();
}

TEST_CASE("CreateFullyConnectedQAsymmS8Workload")
{
    NeonCreateFullyConnectedWorkloadTest<NeonFullyConnectedWorkload, DataType::QAsymmS8>();
}

template <typename NormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateNormalizationWorkloadTest<NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    // Checks that outputs and inputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 5, 5, 1} : TensorShape{3, 1, 5, 5};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 5, 5, 1} : TensorShape{3, 1, 5, 5};

    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateNormalizationFloat16NchwWorkload")
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE("CreateNormalizationFloat16NhwcWorkload")
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float16>(DataLayout::NHWC);
}
#endif

TEST_CASE("CreateNormalizationFloatNchwWorkload")
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreateNormalizationFloatNhwcWorkload")
{
    NeonCreateNormalizationWorkloadTest<NeonNormalizationFloatWorkload, DataType::Float32>(DataLayout::NHWC);
}


template <typename armnn::DataType DataType>
static void NeonCreatePooling2dWorkloadTest(DataLayout dataLayout = DataLayout::NCHW)
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreatePooling2dWorkloadTest<NeonPooling2dWorkload, DataType>(factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 2, 5, 5} : TensorShape{3, 5, 5, 2};
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? TensorShape{3, 2, 2, 4} : TensorShape{3, 2, 4, 2};

    // Checks that outputs and inputs are as we expect them (see definition of CreatePooling2dWorkloadTest).
    Pooling2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreatePooling2dFloat16Workload")
{
    NeonCreatePooling2dWorkloadTest<DataType::Float16>();
}
#endif

TEST_CASE("CreatePooling2dFloatNchwWorkload")
{
    NeonCreatePooling2dWorkloadTest<DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreatePooling2dFloatNhwcWorkload")
{
    NeonCreatePooling2dWorkloadTest<DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE("CreatePooling2dUint8NchwWorkload")
{
    NeonCreatePooling2dWorkloadTest<DataType::QAsymmU8>(DataLayout::NCHW);
}

TEST_CASE("CreatePooling2dUint8NhwcWorkload")
{
    NeonCreatePooling2dWorkloadTest<DataType::QAsymmU8>(DataLayout::NHWC);
}

static void NeonCreatePreluWorkloadTest(const armnn::TensorShape& inputShape,
                                        const armnn::TensorShape& alphaShape,
                                        const armnn::TensorShape& outputShape,
                                        armnn::DataType dataType)
{
    Graph graph;
    NeonWorkloadFactory factory =
            NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreatePreluWorkloadTest<NeonPreluWorkload>(factory,
                                                               graph,
                                                               inputShape,
                                                               alphaShape,
                                                               outputShape,
                                                               dataType);

    // Checks that outputs and inputs are as we expect them (see definition of CreateReshapeWorkloadTest).
    PreluQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto alphaHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, dataType)));
    CHECK(TestNeonTensorHandleInfo(alphaHandle, TensorInfo(alphaShape, dataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, dataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreatePreluFloat16Workload")
{
    NeonCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::Float16);
}
#endif

TEST_CASE("CreatePreluFloatWorkload")
{
    NeonCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::Float32);
}

TEST_CASE("CreatePreluUint8Workload")
{
    NeonCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::QAsymmU8);
}

template <typename armnn::DataType DataType>
static void NeonCreateReshapeWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateReshapeWorkloadTest<NeonReshapeWorkload, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateReshapeWorkloadTest).
    ReshapeQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo({4, 1}, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo({1, 4}, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateReshapeFloat16Workload")
{
    NeonCreateReshapeWorkloadTest<DataType::Float16>();
}
#endif

TEST_CASE("CreateReshapeFloatWorkload")
{
    NeonCreateReshapeWorkloadTest<DataType::Float32>();
}

TEST_CASE("CreateReshapeUint8Workload")
{
    NeonCreateReshapeWorkloadTest<DataType::QAsymmU8>();
}

template <typename ResizeWorkloadType, armnn::DataType DataType>
static void NeonCreateResizeWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory =
            NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    auto workload = CreateResizeBilinearWorkloadTest<ResizeWorkloadType, DataType>(factory, graph, dataLayout);

    auto queueDescriptor = workload->GetData();

    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    armnn::PredicateResult predResult(true);
    switch (dataLayout)
    {
        case DataLayout::NHWC:
            predResult = CompareIAclTensorHandleShape(inputHandle, { 2, 4, 4, 3 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            predResult = CompareIAclTensorHandleShape(outputHandle, { 2, 2, 2, 3 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            break;
        default: // DataLayout::NCHW
            predResult = CompareIAclTensorHandleShape(inputHandle, { 2, 3, 4, 4 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            predResult = CompareIAclTensorHandleShape(outputHandle, { 2, 3, 2, 2 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }
}

TEST_CASE("CreateResizeFloat32NchwWorkload")
{
    NeonCreateResizeWorkloadTest<NeonResizeWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreateResizeUint8NchwWorkload")
{
    NeonCreateResizeWorkloadTest<NeonResizeWorkload, armnn::DataType::QAsymmU8>(DataLayout::NCHW);
}

TEST_CASE("CreateResizeFloat32NhwcWorkload")
{
    NeonCreateResizeWorkloadTest<NeonResizeWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE("CreateResizeUint8NhwcWorkload")
{
    NeonCreateResizeWorkloadTest<NeonResizeWorkload, armnn::DataType::QAsymmU8>(DataLayout::NHWC);
}

template <typename SoftmaxWorkloadType, typename armnn::DataType DataType>
static void NeonCreateSoftmaxWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateSoftmaxWorkloadTest<SoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateSoftmaxWorkloadTest).
    SoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
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
    CHECK(TestNeonTensorHandleInfo(inputHandle, tensorInfo));
    CHECK(TestNeonTensorHandleInfo(outputHandle, tensorInfo));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateSoftmaxFloat16Workload")
{
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxWorkload, DataType::Float16>();
}
#endif

TEST_CASE("CreateSoftmaxFloatWorkload")
{
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxWorkload, DataType::Float32>();
}

TEST_CASE("CreateSoftmaxQAsymmU8Workload")
{
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxWorkload, DataType::QAsymmU8>();
}

TEST_CASE("CreateSoftmaxQAsymmS8Workload")
{
    NeonCreateSoftmaxWorkloadTest<NeonSoftmaxWorkload, DataType::QAsymmS8>();
}

template <typename SpaceToDepthWorkloadType, typename armnn::DataType DataType>
static void NeonSpaceToDepthWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory =
            NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateSpaceToDepthWorkloadTest<SpaceToDepthWorkloadType, DataType>(factory, graph);

    SpaceToDepthQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo({ 1, 2, 2, 1 }, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo({ 1, 1, 1, 4 }, DataType)));
}

TEST_CASE("CreateSpaceToDepthFloat32Workload")
{
    NeonSpaceToDepthWorkloadTest<NeonSpaceToDepthWorkload, armnn::DataType::Float32>();
}

TEST_CASE("CreateSpaceToDepthFloat16Workload")
{
    NeonSpaceToDepthWorkloadTest<NeonSpaceToDepthWorkload, armnn::DataType::Float16>();
}

TEST_CASE("CreateSpaceToDepthQAsymm8Workload")
{
    NeonSpaceToDepthWorkloadTest<NeonSpaceToDepthWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE("CreateSpaceToDepthQSymm16Workload")
{
    NeonSpaceToDepthWorkloadTest<NeonSpaceToDepthWorkload, armnn::DataType::QSymmS16>();
}

TEST_CASE("CreateSplitterWorkload")
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateSplitterWorkloadTest<NeonSplitterWorkload, DataType::Float32>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateSplitterWorkloadTest).
    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo({5, 7, 7}, DataType::Float32)));

    auto outputHandle0 = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(outputHandle0, TensorInfo({1, 7, 7}, DataType::Float32)));

    auto outputHandle1 = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[1]);
    CHECK(TestNeonTensorHandleInfo(outputHandle1, TensorInfo({2, 7, 7}, DataType::Float32)));

    auto outputHandle2 = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[2]);
    CHECK(TestNeonTensorHandleInfo(outputHandle2, TensorInfo({2, 7, 7}, DataType::Float32)));
}

TEST_CASE("CreateSplitterConcat")
{
    // Tests that it is possible to decide which output of the splitter layer
    // should be lined to which input of the concat layer.
    // We tested that is is possible to specify 0th output
    // of the splitter to be the 1st input to the concat, and the 1st output of the splitter to be 0th input
    // of the concat.

    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workloads =
        CreateSplitterConcatWorkloadTest<NeonSplitterWorkload, NeonConcatWorkload,
            DataType::Float32>(factory, graph);

    auto wlSplitter = std::move(workloads.first);
    auto wlConcat = std::move(workloads.second);

    //Checks that the index of inputs/outputs matches what we declared on InputDescriptor construction.
    armnn::IAclTensorHandle* sOut0 = dynamic_cast<armnn::IAclTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::IAclTensorHandle* sOut1 = dynamic_cast<armnn::IAclTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::IAclTensorHandle* mIn0 = dynamic_cast<armnn::IAclTensorHandle*>(wlConcat->GetData().m_Inputs[0]);
    armnn::IAclTensorHandle* mIn1 = dynamic_cast<armnn::IAclTensorHandle*>(wlConcat->GetData().m_Inputs[1]);

    CHECK(sOut0);
    CHECK(sOut1);
    CHECK(mIn0);
    CHECK(mIn1);

    bool validDataPointers = (sOut0 == mIn1) && (sOut1 == mIn0);

    CHECK(validDataPointers);
}

TEST_CASE("CreateSingleOutputMultipleInputs")
{
    // Tests that it is possible to assign multiple (two) different layers to each of the outputs of a splitter layer.
    // We created a splitter with two outputs. That each of those outputs is used by two different activation layers

    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    std::unique_ptr<NeonSplitterWorkload> wlSplitter;
    std::unique_ptr<NeonActivationWorkload> wlActiv0_0;
    std::unique_ptr<NeonActivationWorkload> wlActiv0_1;
    std::unique_ptr<NeonActivationWorkload> wlActiv1_0;
    std::unique_ptr<NeonActivationWorkload> wlActiv1_1;

    CreateSplitterMultipleInputsOneOutputWorkloadTest<NeonSplitterWorkload,
        NeonActivationWorkload, DataType::Float32>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1,
                                                   wlActiv1_0, wlActiv1_1);

    armnn::IAclTensorHandle* sOut0 = dynamic_cast<armnn::IAclTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::IAclTensorHandle* sOut1 = dynamic_cast<armnn::IAclTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::IAclTensorHandle* activ0_0Im = dynamic_cast<armnn::IAclTensorHandle*>(wlActiv0_0->GetData().m_Inputs[0]);
    armnn::IAclTensorHandle* activ0_1Im = dynamic_cast<armnn::IAclTensorHandle*>(wlActiv0_1->GetData().m_Inputs[0]);
    armnn::IAclTensorHandle* activ1_0Im = dynamic_cast<armnn::IAclTensorHandle*>(wlActiv1_0->GetData().m_Inputs[0]);
    armnn::IAclTensorHandle* activ1_1Im = dynamic_cast<armnn::IAclTensorHandle*>(wlActiv1_1->GetData().m_Inputs[0]);


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

#if defined(ARMNNREF_ENABLED)

// This test unit needs the reference backend, it's not available if the reference backend is not built

TEST_CASE("CreateMemCopyWorkloadsNeon")
{
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    CreateMemCopyWorkloads<IAclTensorHandle>(factory);
}

#endif

template <typename L2NormalizationWorkloadType, typename armnn::DataType DataType>
static void NeonCreateL2NormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload =
            CreateL2NormalizationWorkloadTest<L2NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    L2NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 5, 20, 50, 67 } : TensorShape{ 5, 50, 67, 20 };
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ?
                TensorShape{ 5, 20, 50, 67 } : TensorShape{ 5, 50, 67, 20 };

    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateL2NormalizationFloat16NchwWorkload")
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE("CreateL2NormalizationFloat16NhwcWorkload")
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float16>(DataLayout::NHWC);
}
#endif

TEST_CASE("CreateL2NormalizationNchwWorkload")
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE("CreateL2NormalizationNhwcWorkload")
{
    NeonCreateL2NormalizationWorkloadTest<NeonL2NormalizationFloatWorkload, DataType::Float32>(DataLayout::NHWC);
}

template <typename LogSoftmaxWorkloadType, typename armnn::DataType DataType>
static void NeonCreateLogSoftmaxWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateLogSoftmaxWorkloadTest<LogSoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateLogSoftmaxWorkloadTest).
    LogSoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    armnn::TensorInfo tensorInfo({4, 1}, DataType);

    CHECK(TestNeonTensorHandleInfo(inputHandle, tensorInfo));
    CHECK(TestNeonTensorHandleInfo(outputHandle, tensorInfo));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateLogSoftmaxFloat16Workload")
{
    NeonCreateLogSoftmaxWorkloadTest<NeonLogSoftmaxWorkload, DataType::Float16>();
}
#endif

TEST_CASE("CreateLogSoftmaxFloatWorkload")
{
    NeonCreateLogSoftmaxWorkloadTest<NeonLogSoftmaxWorkload, DataType::Float32>();
}

template <typename LstmWorkloadType>
static void NeonCreateLstmWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory =
            NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateLstmWorkloadTest<LstmWorkloadType>(factory, graph);

    LstmQueueDescriptor queueDescriptor = workload->GetData();

    auto inputHandle  = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[1]);

    CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo({ 2, 2 }, DataType::Float32)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo({ 2, 4 }, DataType::Float32)));
}

TEST_CASE("CreateLSTMWorkloadFloatWorkload")
{
    NeonCreateLstmWorkloadTest<NeonLstmFloatWorkload>();
}

template <typename ConcatWorkloadType, armnn::DataType DataType>
static void NeonCreateConcatWorkloadTest(std::initializer_list<unsigned int> outputShape,
                                         unsigned int concatAxis)
{
    Graph graph;
    NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConcatWorkloadTest<ConcatWorkloadType, DataType>(factory, graph, outputShape, concatAxis);

    ConcatQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle0 = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle1 = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);

    CHECK(TestNeonTensorHandleInfo(inputHandle0, TensorInfo({ 2, 3, 2, 5 }, DataType)));
    CHECK(TestNeonTensorHandleInfo(inputHandle1, TensorInfo({ 2, 3, 2, 5 }, DataType)));
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

TEST_CASE("CreateConcatDim0Float32Workload")
{
    NeonCreateConcatWorkloadTest<NeonConcatWorkload, armnn::DataType::Float32>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE("CreateConcatDim1Float32Workload")
{
    NeonCreateConcatWorkloadTest<NeonConcatWorkload, armnn::DataType::Float32>({ 2, 6, 2, 5 }, 1);
}

TEST_CASE("CreateConcatDim3Float32Workload")
{
    NeonCreateConcatWorkloadTest<NeonConcatWorkload, armnn::DataType::Float32>({ 2, 3, 2, 10 }, 3);
}

TEST_CASE("CreateConcatDim0Uint8Workload")
{
    NeonCreateConcatWorkloadTest<NeonConcatWorkload, armnn::DataType::QAsymmU8>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE("CreateConcatDim1Uint8Workload")
{
    NeonCreateConcatWorkloadTest<NeonConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 6, 2, 5 }, 1);
}

TEST_CASE("CreateConcatDim3Uint8Workload")
{
    NeonCreateConcatWorkloadTest<NeonConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 3, 2, 10 }, 3);
}

template <armnn::DataType DataType>
static void NeonCreateStackWorkloadTest(const std::initializer_list<unsigned int>& inputShape,
                                        const std::initializer_list<unsigned int>& outputShape,
                                        unsigned int axis,
                                        unsigned int numInputs)
{
    armnn::Graph graph;
    NeonWorkloadFactory factory =
            NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateStackWorkloadTest<NeonStackWorkload, DataType>(factory,
                                                                         graph,
                                                                         TensorShape(inputShape),
                                                                         TensorShape(outputShape),
                                                                         axis,
                                                                         numInputs);

    // Check inputs and output are as expected
    StackQueueDescriptor queueDescriptor = workload->GetData();
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        auto inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[i]);
        CHECK(TestNeonTensorHandleInfo(inputHandle, TensorInfo(inputShape, DataType)));
    }
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK(TestNeonTensorHandleInfo(outputHandle, TensorInfo(outputShape, DataType)));
}

TEST_CASE("CreateStackFloat32Workload")
{
    NeonCreateStackWorkloadTest<armnn::DataType::Float32>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_CASE("CreateStackFloat16Workload")
{
    NeonCreateStackWorkloadTest<armnn::DataType::Float16>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}
#endif

TEST_CASE("CreateStackUint8Workload")
{
    NeonCreateStackWorkloadTest<armnn::DataType::QAsymmU8>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

template <typename QuantizedLstmWorkloadType>
static void NeonCreateQuantizedLstmWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateQuantizedLstmWorkloadTest<QuantizedLstmWorkloadType>(factory, graph);

    QuantizedLstmQueueDescriptor queueDescriptor = workload->GetData();

    IAclTensorHandle* inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    CHECK((inputHandle->GetShape() == TensorShape({2, 2})));
    CHECK((inputHandle->GetDataType() == arm_compute::DataType::QASYMM8));

    IAclTensorHandle* cellStateInHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[1]);
    CHECK((cellStateInHandle->GetShape() == TensorShape({2, 4})));
    CHECK((cellStateInHandle->GetDataType() == arm_compute::DataType::QSYMM16));

    IAclTensorHandle* outputStateInHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[2]);
    CHECK((outputStateInHandle->GetShape() == TensorShape({2, 4})));
    CHECK((outputStateInHandle->GetDataType() == arm_compute::DataType::QASYMM8));

    IAclTensorHandle* cellStateOutHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((cellStateOutHandle->GetShape() == TensorShape({2, 4})));
    CHECK((cellStateOutHandle->GetDataType() == arm_compute::DataType::QSYMM16));

    IAclTensorHandle* outputStateOutHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[1]);
    CHECK((outputStateOutHandle->GetShape() == TensorShape({2, 4})));
    CHECK((outputStateOutHandle->GetDataType() == arm_compute::DataType::QASYMM8));
}

TEST_CASE("CreateQuantizedLstmWorkload")
{
    NeonCreateQuantizedLstmWorkloadTest<NeonQuantizedLstmWorkload>();
}

template <typename QLstmWorkloadType>
static void NeonCreateQLstmWorkloadTest()
{
    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateQLstmWorkloadTest<QLstmWorkloadType>(factory, graph);
    QLstmQueueDescriptor queueDescriptor = workload->GetData();

    IAclTensorHandle* inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    CHECK((inputHandle->GetShape() == TensorShape({2, 4})));
    CHECK((inputHandle->GetDataType() == arm_compute::DataType::QASYMM8_SIGNED));

    IAclTensorHandle* cellStateOutHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[1]);
    CHECK((cellStateOutHandle->GetShape() == TensorShape({2, 4})));
    CHECK((cellStateOutHandle->GetDataType() == arm_compute::DataType::QSYMM16));

    IAclTensorHandle* outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[2]);
    CHECK((outputHandle->GetShape() == TensorShape({2, 4})));
    CHECK((outputHandle->GetDataType() == arm_compute::DataType::QASYMM8_SIGNED));
}

TEST_CASE("CreateQLstmWorkloadTest")
{
    NeonCreateQLstmWorkloadTest<NeonQLstmWorkload>();
}

template <armnn::DataType DataType>
static void NeonCreateActivationWorkloadReplaceFunctionsTest()
{
    shared_ptr<NeonMemoryManager> memoryManager = make_shared<NeonMemoryManager>();

    Graph graph;
    NeonWorkloadFactory factory = NeonWorkloadFactoryHelper::GetFactory(memoryManager);
    // input and output are created as armnn::TensorInfo tensorInfo({1, 1}, DataType)
    auto workloadPtr = CreateActivationWorkloadTest<NeonActivationWorkload, DataType>(factory, graph);

    // new input and output tensor handlers are created and then replace in the workload
    const NeonTensorHandleFactory tensorHandleFactory(memoryManager);
    TensorInfo inputInfo({2 , 2}, DataType::Float16);
    TensorInfo outputInfo({2 , 2}, DataType::Float16);
    unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    inputHandle->Allocate();
    unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);
    outputHandle->Allocate();

    unsigned int slot = 0;
    CHECK_THROWS_AS(workloadPtr->ReplaceInputTensorHandle(inputHandle.get(), slot), UnimplementedException);
    CHECK_THROWS_AS(workloadPtr->ReplaceOutputTensorHandle(outputHandle.get(), slot), UnimplementedException);
}

TEST_CASE("NeonReplaceFunctionsfromFloat32toFloat16ActivationWorkload")
{
    NeonCreateActivationWorkloadReplaceFunctionsTest<armnn::DataType::Float32>();
}

TEST_CASE("NeonReplaceFunctionsfromUint8toFloat16ActivationWorkload")
{
    NeonCreateActivationWorkloadReplaceFunctionsTest<armnn::DataType::QAsymmU8>();
}

}
