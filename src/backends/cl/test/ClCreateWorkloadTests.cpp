//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextControlFixture.hpp"
#include "ClWorkloadFactoryHelper.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>

#include <aclCommon/test/CreateWorkloadClNeon.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClTensorHandle.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/workloads/ClWorkloads.hpp>
#include <cl/workloads/ClWorkloadUtils.hpp>

boost::test_tools::predicate_result CompareIClTensorHandleShape(IClTensorHandle*                    tensorHandle,
                                                                std::initializer_list<unsigned int> expectedDimensions)
{
    return CompareTensorHandleShape<IClTensorHandle>(tensorHandle, expectedDimensions);
}

BOOST_FIXTURE_TEST_SUITE(CreateWorkloadCl, ClContextControlFixture)

template <armnn::DataType DataType>
static void ClCreateActivationWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateActivationWorkloadTest<ClActivationWorkload, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateActivationWorkloadTest).
    ActivationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {1, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {1, 1}));
}

BOOST_AUTO_TEST_CASE(CreateActivationFloatWorkload)
{
    ClCreateActivationWorkloadTest<armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateActivationFloat16Workload)
{
    ClCreateActivationWorkloadTest<armnn::DataType::Float16>();
}

template <typename WorkloadType,
          typename DescriptorType,
          typename LayerType,
          armnn::DataType DataType>
static void ClCreateElementwiseWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateElementwiseWorkloadTest<WorkloadType, DescriptorType, LayerType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateElementwiseWorkloadTest).
    DescriptorType queueDescriptor = workload->GetData();
    auto inputHandle1 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle2 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle1, {2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle2, {2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 3}));
}

BOOST_AUTO_TEST_CASE(CreateAdditionFloatWorkload)
{
    ClCreateElementwiseWorkloadTest<ClAdditionWorkload,
                                    AdditionQueueDescriptor,
                                    AdditionLayer,
                                    armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateAdditionFloat16Workload)
{
    ClCreateElementwiseWorkloadTest<ClAdditionWorkload,
                                    AdditionQueueDescriptor,
                                    AdditionLayer,
                                    armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateSubtractionFloatWorkload)
{
    ClCreateElementwiseWorkloadTest<ClSubtractionWorkload,
                                    SubtractionQueueDescriptor,
                                    SubtractionLayer,
                                    armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSubtractionFloat16Workload)
{
    ClCreateElementwiseWorkloadTest<ClSubtractionWorkload,
                                    SubtractionQueueDescriptor,
                                    SubtractionLayer,
                                    armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloatWorkloadTest)
{
    ClCreateElementwiseWorkloadTest<ClMultiplicationWorkload,
                                    MultiplicationQueueDescriptor,
                                    MultiplicationLayer,
                                    armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloat16WorkloadTest)
{
    ClCreateElementwiseWorkloadTest<ClMultiplicationWorkload,
                                    MultiplicationQueueDescriptor,
                                    MultiplicationLayer,
                                    armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationUint8WorkloadTest)
{
    ClCreateElementwiseWorkloadTest<ClMultiplicationWorkload,
                                    MultiplicationQueueDescriptor,
                                    MultiplicationLayer,
                                    armnn::DataType::QAsymmU8>();
}

BOOST_AUTO_TEST_CASE(CreateDivisionFloatWorkloadTest)
{
    ClCreateElementwiseWorkloadTest<ClDivisionFloatWorkload,
                                    DivisionQueueDescriptor,
                                    DivisionLayer,
                                    armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateDivisionFloat16WorkloadTest)
{
    ClCreateElementwiseWorkloadTest<ClDivisionFloatWorkload,
                                    DivisionQueueDescriptor,
                                    DivisionLayer,
                                    armnn::DataType::Float16>();
}

template <typename WorkloadType, 
          typename DescriptorType,
          armnn::DataType DataType>
static void ClCreateElementwiseUnaryWorkloadTest(armnn::UnaryOperation op)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateElementwiseUnaryWorkloadTest<WorkloadType, DescriptorType, DataType>(factory, graph, op);

    DescriptorType queueDescriptor = workload->GetData();

    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 3}));
}

BOOST_AUTO_TEST_CASE(CreateRsqrtFloat32WorkloadTest)
{
    ClCreateElementwiseUnaryWorkloadTest<ClRsqrtWorkload, RsqrtQueueDescriptor, armnn::DataType::Float32>(
        UnaryOperation::Rsqrt);
}

template <typename BatchNormalizationWorkloadType, armnn::DataType DataType>
static void ClCreateBatchNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>
                    (factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateBatchNormalizationWorkloadTest).
    BatchNormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

     switch (dataLayout)
    {
        case DataLayout::NHWC:
            BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 2, 4, 4, 3 }));
            BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 2, 4, 4, 3 }));
            break;
        default: // NCHW
            BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 2, 3, 4, 4 }));
            BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 2, 3, 4, 4 }));
    }
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloatNchwWorkload)
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat16NchwWorkload)
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloatNhwcWorkload)
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationNhwcFloat16NhwcWorkload)
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float16>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateConvertFp16ToFp32Workload)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConvertFp16ToFp32WorkloadTest<ClConvertFp16ToFp32Workload>(factory, graph);

    ConvertFp16ToFp32QueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {1, 3, 2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {1, 3, 2, 3}));
    BOOST_TEST((inputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F16));
    BOOST_TEST((outputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F32));
}

BOOST_AUTO_TEST_CASE(CreateConvertFp32ToFp16Workload)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConvertFp32ToFp16WorkloadTest<ClConvertFp32ToFp16Workload>(factory, graph);

    ConvertFp32ToFp16QueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {1, 3, 2, 3}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {1, 3, 2, 3}));
    BOOST_TEST((inputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F32));
    BOOST_TEST((outputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F16));
}

template <typename Convolution2dWorkloadType, typename armnn::DataType DataType>
static void ClConvolution2dWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConvolution2dWorkloadTest<ClConvolution2dWorkload, DataType>(factory,
                                                                                       graph,
                                                                                       dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({2, 3, 8, 16})
                                                               : std::initializer_list<unsigned int>({2, 8, 16, 3});
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({2, 2, 2, 10})
                                                               : std::initializer_list<unsigned int>({2, 2, 10, 2});

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((inputHandle->GetShape() == inputShape));
    BOOST_TEST((outputHandle->GetShape() == outputShape));
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNchwWorkload)
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNhwcWorkload)
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloat16NchwWorkload)
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloat16NhwcWorkload)
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float16>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFastMathEnabledWorkload)
{
    Graph graph;

    using ModelOptions = std::vector<BackendOptions>;
    ModelOptions modelOptions = {};
    BackendOptions gpuAcc("GpuAcc",
    {
        { "FastMathEnabled", true }
    });
    modelOptions.push_back(gpuAcc);

    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager(), modelOptions);

    auto workload =
        CreateConvolution2dWorkloadFastMathTest<ClConvolution2dWorkload, armnn::DataType::Float32>(factory,
                                                                                           graph,
                                                                                           DataLayout::NCHW,
                                                                                           modelOptions);

    ARMNN_ASSERT(workload != nullptr);
    auto conv2dWorkload = PolymorphicDowncast<ClConvolution2dWorkload*>(workload.get());
    IgnoreUnused(conv2dWorkload);
    ARMNN_ASSERT(conv2dWorkload != nullptr);
    ARMNN_ASSERT(conv2dWorkload->GetConvolutionMethod() == arm_compute::ConvolutionMethod::WINOGRAD);
}

template <typename DepthwiseConvolutionWorkloadType, typename armnn::DataType DataType>
static void ClDepthwiseConvolutionWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateDepthwiseConvolution2dWorkloadTest<DepthwiseConvolutionWorkloadType, DataType>
                    (factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateDepthwiseConvolution2dWorkloadTest).
    DepthwiseConvolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned int>({ 2, 5, 5, 2 });

    BOOST_TEST((inputHandle->GetShape() == inputShape));
    BOOST_TEST((outputHandle->GetShape() == outputShape));
}

BOOST_AUTO_TEST_CASE(CreateDepthwiseConvolutionFloat32NhwcWorkload)
{
    ClDepthwiseConvolutionWorkloadTest<ClDepthwiseConvolutionWorkload, DataType::Float32>(DataLayout::NHWC);
}

template <typename Convolution2dWorkloadType, typename armnn::DataType DataType>
static void ClDirectConvolution2dWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateDirectConvolution2dWorkloadTest<ClConvolution2dWorkload, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateDirectConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {2, 3, 6, 6}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {2, 2, 6, 6}));
}

BOOST_AUTO_TEST_CASE(CreateDirectConvolution2dFloatWorkload)
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateDirectConvolution2dFloat16Workload)
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateDirectConvolution2dUint8Workload)
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::QAsymmU8>();
}

template <typename FullyConnectedWorkloadType, typename armnn::DataType DataType>
static void ClCreateFullyConnectedWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload =
        CreateFullyConnectedWorkloadTest<FullyConnectedWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateFullyConnectedWorkloadTest).
    FullyConnectedQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {3, 1, 4, 5}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {3, 7}));
}


BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloatWorkloadTest)
{
    ClCreateFullyConnectedWorkloadTest<ClFullyConnectedWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateFullyConnectedFloat16WorkloadTest)
{
    ClCreateFullyConnectedWorkloadTest<ClFullyConnectedWorkload, armnn::DataType::Float16>();
}

template <typename NormalizationWorkloadType, typename armnn::DataType DataType>
static void ClNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateNormalizationWorkloadTest<NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({3, 5, 5, 1})
                                                               : std::initializer_list<unsigned int>({3, 1, 5, 5});
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({3, 5, 5, 1})
                                                               : std::initializer_list<unsigned int>({3, 1, 5, 5});

    BOOST_TEST((inputHandle->GetShape() == inputShape));
    BOOST_TEST((outputHandle->GetShape() == outputShape));
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloat32NchwWorkload)
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloat16NchwWorkload)
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloat32NhwcWorkload)
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateNormalizationFloat16NhwcWorkload)
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float16>(DataLayout::NHWC);
}

template <typename armnn::DataType DataType>
static void ClPooling2dWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreatePooling2dWorkloadTest<ClPooling2dWorkload, DataType>(factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({3, 2, 5, 5})
                                                               : std::initializer_list<unsigned int>({3, 5, 5, 2});
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({3, 2, 2, 4})
                                                               : std::initializer_list<unsigned int>({3, 2, 4, 2});

    // Check that inputs/outputs are as we expect them (see definition of CreatePooling2dWorkloadTest).
    Pooling2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST((inputHandle->GetShape() == inputShape));
    BOOST_TEST((outputHandle->GetShape() == outputShape));
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloatNchwWorkload)
{
    ClPooling2dWorkloadTest<armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloatNhwcWorkload)
{
    ClPooling2dWorkloadTest<armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat16NchwWorkload)
{
    ClPooling2dWorkloadTest<armnn::DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat16NhwcWorkload)
{
    ClPooling2dWorkloadTest<armnn::DataType::Float16>(DataLayout::NHWC);
}

static void ClCreatePreluWorkloadTest(const armnn::TensorShape& inputShape,
                                      const armnn::TensorShape& alphaShape,
                                      const armnn::TensorShape& outputShape,
                                      armnn::DataType dataType)
{
    Graph graph;
    ClWorkloadFactory factory =
            ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreatePreluWorkloadTest<ClPreluWorkload>(factory,
                                                             graph,
                                                             inputShape,
                                                             alphaShape,
                                                             outputShape,
                                                             dataType);

    // Checks that outputs and inputs are as we expect them (see definition of CreatePreluWorkloadTest).
    PreluQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto alphaHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST((inputHandle->GetShape() == inputShape));
    BOOST_TEST((alphaHandle->GetShape() == alphaShape));
    BOOST_TEST((outputHandle->GetShape() == outputShape));
}

BOOST_AUTO_TEST_CASE(CreatePreluFloat16Workload)
{
    ClCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::Float16);
}

BOOST_AUTO_TEST_CASE(CreatePreluFloatWorkload)
{
    ClCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::Float32);
}

BOOST_AUTO_TEST_CASE(CreatePreluUint8Workload)
{
    ClCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::QAsymmU8);
}

template <typename armnn::DataType DataType>
static void ClCreateReshapeWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateReshapeWorkloadTest<ClReshapeWorkload, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateReshapeWorkloadTest).
    ReshapeQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {4, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {1, 4}));
}

BOOST_AUTO_TEST_CASE(CreateReshapeFloatWorkload)
{
    ClCreateReshapeWorkloadTest<armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeFloat16Workload)
{
    ClCreateReshapeWorkloadTest<armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeUint8Workload)
{
    ClCreateReshapeWorkloadTest<armnn::DataType::QAsymmU8>();
}

template <typename SoftmaxWorkloadType, typename armnn::DataType DataType>
static void ClSoftmaxWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateSoftmaxWorkloadTest<SoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of ClSoftmaxFloatWorkload).
    SoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

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

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {4, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {4, 1}));
}


BOOST_AUTO_TEST_CASE(CreateSoftmaxFloat32WorkloadTest)
{
    ClSoftmaxWorkloadTest<ClSoftmaxWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxFloat16WorkloadTest)
{
    ClSoftmaxWorkloadTest<ClSoftmaxWorkload, armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxQAsymmU8Workload)
{
    ClSoftmaxWorkloadTest<ClSoftmaxWorkload, armnn::DataType::QAsymmU8>();
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxQAsymmS8Workload)
{
    ClSoftmaxWorkloadTest<ClSoftmaxWorkload, armnn::DataType::QAsymmS8>();
}

template <typename armnn::DataType DataType>
static void ClSplitterWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateSplitterWorkloadTest<ClSplitterWorkload, DataType>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateSplitterWorkloadTest).
    SplitterQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {5, 7, 7}));

    auto outputHandle1 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle1, {2, 7, 7}));

    auto outputHandle2 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[2]);
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle2, {2, 7, 7}));

    auto outputHandle0 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle0, {1, 7, 7}));
}

BOOST_AUTO_TEST_CASE(CreateSplitterFloatWorkload)
{
    ClSplitterWorkloadTest<armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterFloat16Workload)
{
    ClSplitterWorkloadTest<armnn::DataType::Float16>();
}

template <typename armnn::DataType DataType>
static void ClSplitterConcatTest()
{
    // Tests that it is possible to decide which output of the splitter layer
    // should be lined to which input of the concat layer.
    // We test that is is possible to specify 0th output
    // of the splitter to be the 1st input to the concat and the 1st output of the splitter  to be 0th input
    // of the concat.

    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workloads =
        CreateSplitterConcatWorkloadTest<ClSplitterWorkload, ClConcatWorkload, DataType>
            (factory, graph);

    auto wlSplitter = std::move(workloads.first);
    auto wlConcat = std::move(workloads.second);

    //Checks that the index of inputs/outputs matches what we declared on InputDescriptor construction.
    armnn::ClSubTensorHandle* sOut0 = dynamic_cast<armnn::ClSubTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
    armnn::ClSubTensorHandle* sOut1 = dynamic_cast<armnn::ClSubTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
    armnn::ClSubTensorHandle* mIn0 = dynamic_cast<armnn::ClSubTensorHandle*>(wlConcat->GetData().m_Inputs[0]);
    armnn::ClSubTensorHandle* mIn1 = dynamic_cast<armnn::ClSubTensorHandle*>(wlConcat->GetData().m_Inputs[1]);

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

BOOST_AUTO_TEST_CASE(CreateSplitterConcatFloatWorkload)
{
    ClSplitterConcatTest<armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSplitterConcatFloat16Workload)
{
    ClSplitterConcatTest<armnn::DataType::Float16>();
}


BOOST_AUTO_TEST_CASE(CreateSingleOutputMultipleInputs)
{
    // Test that it is possible to assign multiple (two) different layers to each of the outputs of a splitter layer.
    // We create a splitter with two outputs. That each of those outputs is used by two different activation layers.

    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    std::unique_ptr<ClSplitterWorkload> wlSplitter;
    std::unique_ptr<ClActivationWorkload> wlActiv0_0;
    std::unique_ptr<ClActivationWorkload> wlActiv0_1;
    std::unique_ptr<ClActivationWorkload> wlActiv1_0;
    std::unique_ptr<ClActivationWorkload> wlActiv1_1;

    CreateSplitterMultipleInputsOneOutputWorkloadTest<ClSplitterWorkload,
        ClActivationWorkload, armnn::DataType::Float32>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1,
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

#if defined(ARMNNREF_ENABLED)

// This test unit needs the reference backend, it's not available if the reference backend is not built

BOOST_AUTO_TEST_CASE(CreateMemCopyWorkloadsCl)
{
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    CreateMemCopyWorkloads<IClTensorHandle>(factory);
}

#endif

template <typename L2NormalizationWorkloadType, typename armnn::DataType DataType>
static void ClL2NormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload =
            CreateL2NormalizationWorkloadTest<L2NormalizationWorkloadType, DataType>(factory, graph, dataLayout);

    // Checks that inputs/outputs are as we expect them (see definition of CreateNormalizationWorkloadTest).
    L2NormalizationQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 5, 20, 50, 67 })
                                                               : std::initializer_list<unsigned int>({ 5, 50, 67, 20 });
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned int>({ 5, 20, 50, 67 })
                                                               : std::initializer_list<unsigned int>({ 5, 50, 67, 20 });

    BOOST_TEST((inputHandle->GetShape() == inputShape));
    BOOST_TEST((outputHandle->GetShape() == outputShape));
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloatNchwWorkload)
{
    ClL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloatNhwcWorkload)
{
    ClL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat16NchwWorkload)
{
    ClL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat16NhwcWorkload)
{
    ClL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float16>(DataLayout::NHWC);
}

template <typename LogSoftmaxWorkloadType, typename armnn::DataType DataType>
static void ClCreateLogSoftmaxWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
            ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateLogSoftmaxWorkloadTest<LogSoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of CreateLogSoftmaxWorkloadTest).
    LogSoftmaxQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {4, 1}));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, {4, 1}));
}

BOOST_AUTO_TEST_CASE(CreateLogSoftmaxFloat32WorkloadTest)
{
    ClCreateLogSoftmaxWorkloadTest<ClLogSoftmaxWorkload, armnn::DataType::Float32>();
}

template <typename LstmWorkloadType>
static void ClCreateLstmWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateLstmWorkloadTest<LstmWorkloadType>(factory, graph);

    LstmQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 2, 2 }));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 2, 4 }));
}

BOOST_AUTO_TEST_CASE(CreateLSTMWorkloadFloatWorkload)
{
    ClCreateLstmWorkloadTest<ClLstmFloatWorkload>();
}

template <typename ResizeWorkloadType, typename armnn::DataType DataType>
static void ClResizeWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateResizeBilinearWorkloadTest<ResizeWorkloadType, DataType>(factory, graph, dataLayout);

    auto queueDescriptor = workload->GetData();

    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 2, 4, 4, 3 }));
            BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 2, 2, 2, 3 }));
            break;
        case DataLayout::NCHW:
        default:
            BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 2, 3, 4, 4 }));
            BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 2, 3, 2, 2 }));
    }
}

BOOST_AUTO_TEST_CASE(CreateResizeFloat32NchwWorkload)
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateResizeFloat16NchwWorkload)
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateResizeUint8NchwWorkload)
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::QAsymmU8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateResizeFloat32NhwcWorkload)
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateResizeFloat16NhwcWorkload)
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float16>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateResizeUint8NhwcWorkload)
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::QAsymmU8>(DataLayout::NHWC);
}

template <typename MeanWorkloadType, typename armnn::DataType DataType>
static void ClMeanWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateMeanWorkloadTest<MeanWorkloadType, DataType>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreateMeanWorkloadTest).
    MeanQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    // The first dimension (batch size) in both input and output is singular thus it has been reduced by ACL.
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, {  1, 3, 7, 4 }));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 1, 4 }));
}

BOOST_AUTO_TEST_CASE(CreateMeanFloat32Workload)
{
    ClMeanWorkloadTest<ClMeanWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateMeanFloat16Workload)
{
    ClMeanWorkloadTest<ClMeanWorkload, armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateMeanUint8Workload)
{
    ClMeanWorkloadTest<ClMeanWorkload, armnn::DataType::QAsymmU8>();
}

template <typename ConcatWorkloadType, armnn::DataType DataType>
static void ClCreateConcatWorkloadTest(std::initializer_list<unsigned int> outputShape,
                                       unsigned int concatAxis)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConcatWorkloadTest<ConcatWorkloadType, DataType>(factory, graph, outputShape, concatAxis);

    ConcatQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle0  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle1  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle0, { 2, 3, 2, 5 }));
    BOOST_TEST(CompareIClTensorHandleShape(inputHandle1, { 2, 3, 2, 5 }));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, outputShape));
}

BOOST_AUTO_TEST_CASE(CreateConcatDim0Float32Workload)
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::Float32>({ 4, 3, 2, 5 }, 0);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim1Float32Workload)
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::Float32>({ 2, 6, 2, 5 }, 1);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim3Float32Workload)
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::Float32>({ 2, 3, 2, 10 }, 3);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim0Uint8Workload)
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::QAsymmU8>({ 4, 3, 2, 5 }, 0);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim1Uint8Workload)
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 6, 2, 5 }, 1);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim3Uint8Workload)
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 3, 2, 10 }, 3);
}

template <typename SpaceToDepthWorkloadType, typename armnn::DataType DataType>
static void ClSpaceToDepthWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory =
            ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateSpaceToDepthWorkloadTest<SpaceToDepthWorkloadType, DataType>(factory, graph);

    SpaceToDepthQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(CompareIClTensorHandleShape(inputHandle, { 1, 2, 2, 1 }));
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, { 1, 1, 1, 4 }));
}

BOOST_AUTO_TEST_CASE(CreateSpaceToDepthFloat32Workload)
{
    ClSpaceToDepthWorkloadTest<ClSpaceToDepthWorkload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSpaceToDepthFloat16Workload)
{
    ClSpaceToDepthWorkloadTest<ClSpaceToDepthWorkload, armnn::DataType::Float16>();
}

BOOST_AUTO_TEST_CASE(CreateSpaceToDepthQAsymm8Workload)
{
    ClSpaceToDepthWorkloadTest<ClSpaceToDepthWorkload, armnn::DataType::QAsymmU8>();
}

BOOST_AUTO_TEST_CASE(CreateSpaceToDepthQSymm16Workload)
{
    ClSpaceToDepthWorkloadTest<ClSpaceToDepthWorkload, armnn::DataType::QSymmS16>();
}

template <armnn::DataType DataType>
static void ClCreateStackWorkloadTest(const std::initializer_list<unsigned int>& inputShape,
                                      const std::initializer_list<unsigned int>& outputShape,
                                      unsigned int axis,
                                      unsigned int numInputs)
{
    armnn::Graph graph;
    ClWorkloadFactory factory =
            ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateStackWorkloadTest<ClStackWorkload, DataType>(factory,
                                                                       graph,
                                                                       TensorShape(inputShape),
                                                                       TensorShape(outputShape),
                                                                       axis,
                                                                       numInputs);

    // Check inputs and output are as expected
    StackQueueDescriptor queueDescriptor = workload->GetData();
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        auto inputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[i]);
        BOOST_TEST(CompareIClTensorHandleShape(inputHandle, inputShape));
    }
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST(CompareIClTensorHandleShape(outputHandle, outputShape));
}

BOOST_AUTO_TEST_CASE(CreateStackFloat32Workload)
{
    ClCreateStackWorkloadTest<armnn::DataType::Float32>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

BOOST_AUTO_TEST_CASE(CreateStackFloat16Workload)
{
    ClCreateStackWorkloadTest<armnn::DataType::Float16>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

BOOST_AUTO_TEST_CASE(CreateStackUint8Workload)
{
    ClCreateStackWorkloadTest<armnn::DataType::QAsymmU8>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}


template <typename QLstmWorkloadType>
static void ClCreateQLstmWorkloadTest()
{
    Graph graph;
    ClWorkloadFactory factory = ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateQLstmWorkloadTest<QLstmWorkloadType>(factory, graph);
    QLstmQueueDescriptor queueDescriptor = workload->GetData();

    IAclTensorHandle* inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    BOOST_TEST((inputHandle->GetShape() == TensorShape({2, 4})));
    BOOST_TEST((inputHandle->GetDataType() == arm_compute::DataType::QASYMM8_SIGNED));

    IAclTensorHandle* cellStateOutHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST((cellStateOutHandle->GetShape() == TensorShape({2, 4})));
    BOOST_TEST((cellStateOutHandle->GetDataType() == arm_compute::DataType::QSYMM16));

    IAclTensorHandle* outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[2]);
    BOOST_TEST((outputHandle->GetShape() == TensorShape({2, 4})));
    BOOST_TEST((outputHandle->GetDataType() == arm_compute::DataType::QASYMM8_SIGNED));
}

BOOST_AUTO_TEST_CASE(CreateQLstmWorkloadTest)
{
    ClCreateQLstmWorkloadTest<ClQLstmWorkload>();
}

template <typename QuantizedLstmWorkloadType>
static void ClCreateQuantizedLstmWorkloadTest()
{
    using namespace armnn::armcomputetensorutils;

    Graph graph;
    ClWorkloadFactory factory =
            ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateQuantizedLstmWorkloadTest<QuantizedLstmWorkloadType>(factory, graph);

    QuantizedLstmQueueDescriptor queueDescriptor = workload->GetData();

    IAclTensorHandle* inputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[0]);
    BOOST_TEST((inputHandle->GetShape() == TensorShape({2, 2})));
    BOOST_TEST((inputHandle->GetDataType() == arm_compute::DataType::QASYMM8));

    IAclTensorHandle* cellStateInHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[1]);
    BOOST_TEST((cellStateInHandle->GetShape() == TensorShape({2, 4})));
    BOOST_TEST((cellStateInHandle->GetDataType() == arm_compute::DataType::QSYMM16));

    IAclTensorHandle* outputStateInHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Inputs[2]);
    BOOST_TEST((outputStateInHandle->GetShape() == TensorShape({2, 4})));
    BOOST_TEST((outputStateInHandle->GetDataType() == arm_compute::DataType::QASYMM8));

    IAclTensorHandle* cellStateOutHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((cellStateOutHandle->GetShape() == TensorShape({2, 4})));
    BOOST_TEST((cellStateOutHandle->GetDataType() == arm_compute::DataType::QSYMM16));

    IAclTensorHandle* outputStateOutHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[1]);
    BOOST_TEST((outputStateOutHandle->GetShape() == TensorShape({2, 4})));
    BOOST_TEST((outputStateOutHandle->GetDataType() == arm_compute::DataType::QASYMM8));
}

BOOST_AUTO_TEST_CASE(CreateQuantizedLstmWorkload)
{
    ClCreateQuantizedLstmWorkloadTest<ClQuantizedLstmWorkload>();
}

BOOST_AUTO_TEST_SUITE_END()
