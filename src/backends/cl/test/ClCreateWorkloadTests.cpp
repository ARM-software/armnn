//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextControlFixture.hpp"
#include "ClWorkloadFactoryHelper.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/TensorHelpers.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <aclCommon/test/CreateWorkloadClNeon.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClImportTensorHandle.hpp>
#include <cl/ClImportTensorHandleFactory.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/workloads/ClWorkloads.hpp>
#include <cl/workloads/ClWorkloadUtils.hpp>

#include <doctest/doctest.h>

armnn::PredicateResult CompareIClTensorHandleShape(IClTensorHandle* tensorHandle,
                                                   std::initializer_list<unsigned int> expectedDimensions)
{
    return CompareTensorHandleShape<IClTensorHandle>(tensorHandle, expectedDimensions);
}

TEST_SUITE("CreateWorkloadCl")
{
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

    auto predResult = CompareIClTensorHandleShape(inputHandle, {1, 1});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());

    predResult = CompareIClTensorHandleShape(outputHandle, {1, 1});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateActivationFloatWorkload")
{
    ClCreateActivationWorkloadTest<armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateActivationFloat16Workload")
{
    ClCreateActivationWorkloadTest<armnn::DataType::Float16>();
}

template <typename WorkloadType,
          armnn::DataType DataType>
static void ClCreateElementwiseWorkloadTest(BinaryOperation binaryOperator)
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateElementwiseBinaryWorkloadTest<WorkloadType, DataType>(factory, graph, binaryOperator);

    // Checks that inputs/outputs are as we expect them (see definition of CreateElementwiseWorkloadTest).
    auto queueDescriptor = workload->GetData();
    auto inputHandle1 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle2 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    auto predResult = CompareIClTensorHandleShape(inputHandle1, {2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(inputHandle2, {2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateAdditionFloatWorkload")
{
    ClCreateElementwiseWorkloadTest<ClAdditionWorkload,
                                    armnn::DataType::Float32>(BinaryOperation::Add);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateAdditionFloat16Workload")
{
    ClCreateElementwiseWorkloadTest<ClAdditionWorkload,
                                    armnn::DataType::Float16>(BinaryOperation::Add);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSubtractionFloatWorkload")
{
    ClCreateElementwiseWorkloadTest<ClSubtractionWorkload,
                                    armnn::DataType::Float32>(BinaryOperation::Sub);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSubtractionFloat16Workload")
{
    ClCreateElementwiseWorkloadTest<ClSubtractionWorkload,
                                    armnn::DataType::Float16>(BinaryOperation::Sub);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateMultiplicationFloatWorkloadTest")
{
    ClCreateElementwiseWorkloadTest<ClMultiplicationWorkload,
                                    armnn::DataType::Float32>(BinaryOperation::Mul);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateMultiplicationFloat16WorkloadTest")
{
    ClCreateElementwiseWorkloadTest<ClMultiplicationWorkload,
                                    armnn::DataType::Float16>(BinaryOperation::Mul);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateMultiplicationUint8WorkloadTest")
{
    ClCreateElementwiseWorkloadTest<ClMultiplicationWorkload,
                                    armnn::DataType::QAsymmU8>(BinaryOperation::Mul);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateDivisionFloatWorkloadTest")
{
    ClCreateElementwiseWorkloadTest<ClDivisionWorkload,
                                    armnn::DataType::Float32>(BinaryOperation::Div);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateDivisionFloat16WorkloadTest")
{
    ClCreateElementwiseWorkloadTest<ClDivisionWorkload,
                                    armnn::DataType::Float16>(BinaryOperation::Div);
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

    auto predResult = CompareIClTensorHandleShape(inputHandle, {2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());

    predResult = CompareIClTensorHandleShape(outputHandle, {2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateRsqrtFloat32WorkloadTest")
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

    armnn::PredicateResult predResult(true);
    switch (dataLayout)
    {
        case DataLayout::NHWC:
            predResult = CompareIClTensorHandleShape(inputHandle, { 2, 4, 4, 3 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            predResult = CompareIClTensorHandleShape(outputHandle, { 2, 4, 4, 3 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            break;
        default: // NCHW
            predResult = CompareIClTensorHandleShape(inputHandle, { 2, 3, 4, 4 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            predResult = CompareIClTensorHandleShape(outputHandle, { 2, 3, 4, 4 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateBatchNormalizationFloatNchwWorkload")
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateBatchNormalizationFloat16NchwWorkload")
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateBatchNormalizationFloatNhwcWorkload")
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateBatchNormalizationNhwcFloat16NhwcWorkload")
{
    ClCreateBatchNormalizationWorkloadTest<ClBatchNormalizationFloatWorkload,
                                           armnn::DataType::Float16>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvertFp16ToFp32Workload")
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConvertFp16ToFp32WorkloadTest<ClConvertFp16ToFp32Workload>(factory, graph);

    ConvertFp16ToFp32QueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    auto predResult = CompareIClTensorHandleShape(inputHandle, {1, 3, 2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {1, 3, 2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    CHECK((inputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F16));
    CHECK((outputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F32));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvertFp32ToFp16Workload")
{
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload = CreateConvertFp32ToFp16WorkloadTest<ClConvertFp32ToFp16Workload>(factory, graph);

    ConvertFp32ToFp16QueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);

    auto predResult = CompareIClTensorHandleShape(inputHandle, {1, 3, 2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {1, 3, 2, 3});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    CHECK((inputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F32));
    CHECK((outputHandle->GetTensor().info()->data_type() == arm_compute::DataType::F16));
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
    CHECK((inputHandle->GetShape() == inputShape));
    CHECK((outputHandle->GetShape() == outputShape));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvolution2dFloatNchwWorkload")
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvolution2dFloatNhwcWorkload")
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvolution2dFloat16NchwWorkload")
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvolution2dFloat16NhwcWorkload")
{
    ClConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float16>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvolution2dFastMathEnabledWorkload")
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

TEST_CASE_FIXTURE(ClContextControlFixture, "ClReplaceInputOutputConvolution2dWorkload")
{
    // Create Convolution2dWorkload with ClTensorHandle input and output
    // Then replace the input and output with ClImportTensorHandle
    Graph graph;
    ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    auto workload =
        CreateConvolution2dWorkloadTest<ClConvolution2dWorkload, DataType::Float32>(factory,
                                                                                    graph,
                                                                                    DataLayout::NHWC);

    TensorShape inputShape  = std::initializer_list<unsigned int>({2, 8, 16, 3});
    TensorShape outputShape = std::initializer_list<unsigned int>({2, 2, 10, 2});

    // Checks that outputs and inputs are as we expect them (see definition of CreateConvolution2dWorkloadTest).
    Convolution2dQueueDescriptor queueDescriptor = workload->GetData();
    auto inputHandle  = PolymorphicDowncast<ITensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicDowncast<ITensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((inputHandle->GetShape() == inputShape));
    CHECK((outputHandle->GetShape() == outputShape));
    // The input and output handles are created correctly as ClTensorHandle
    CHECK((dynamic_cast<ClTensorHandle*>(inputHandle) != nullptr));
    CHECK((dynamic_cast<ClTensorHandle*>(outputHandle) != nullptr));

    // Replace with ImportTensorHandle
    ClImportTensorHandleFactory importFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));

    TensorInfo inputInfo({ 2, 8, 16, 3 }, DataType::Float32);
    TensorInfo outputInfo({ 2, 2, 10, 2 }, DataType::Float32);

    // create TensorHandle for memory import
    auto inputImportHandle = importFactory.CreateTensorHandle(inputInfo);
    auto outputImportHandle = importFactory.CreateTensorHandle(outputInfo);

    // Calling ReplaceInputTensorHandle and ReplaceOutputTensorHandle does not throw exception
    // as Reconfigure function is implemented
    workload->ReplaceInputTensorHandle(inputImportHandle.get(), 0);
    workload->ReplaceOutputTensorHandle(outputImportHandle.get(), 0);

    // Correctly replaced with the import handles with correct information
    queueDescriptor = workload->GetData();
    auto replacedInputHandle  = PolymorphicDowncast<ITensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto replacedOutputHandle = PolymorphicDowncast<ITensorHandle*>(queueDescriptor.m_Outputs[0]);
    CHECK((replacedInputHandle->GetShape() == inputShape));
    CHECK((replacedOutputHandle->GetShape() == outputShape));

    CHECK((inputImportHandle.get() == replacedInputHandle));
    CHECK((inputImportHandle.get() == replacedInputHandle));

    CHECK((dynamic_cast<ClTensorHandle*>(replacedInputHandle) == nullptr));
    CHECK((dynamic_cast<ClImportTensorHandle*>(replacedInputHandle) != nullptr));
    CHECK((dynamic_cast<ClTensorHandle*>(replacedOutputHandle) == nullptr));
    CHECK((dynamic_cast<ClImportTensorHandle*>(replacedOutputHandle) != nullptr));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConvolution2dClCompiledContextWorkload")
{
    using namespace armnn;

    const DataType inputType  = DataType::QAsymmU8;
    const DataType kernelType = DataType::QSymmS8;
    const DataType biasType   = DataType::Signed32;

    TensorInfo inputInfo ({ 1, 3, 1, 2 }, inputType, 0.5f, 128);
    TensorInfo outputInfo({ 1, 3, 1, 3 }, inputType, 1.0f, 128);

    const std::vector<float> quantScales{ 0.5f, 0.75f, 1.0f };
    constexpr unsigned int quantDimension = 0;

    TensorInfo kernelInfo({ 3, 1, 1, 2 }, kernelType, quantScales, quantDimension);

    const std::vector<float> biasQuantScales{ 0.25f, 0.375f, 0.5f };
    TensorInfo biasInfo({ 3 }, biasType, biasQuantScales, quantDimension);

    std::vector<uint8_t> inputData =
    {
        138, 108, 138, 108, 138, 108
    };

    std::vector<int8_t> kernelData =
    {
        1, 2, 1, 2, 1, 2
    };

    std::vector<int32_t> biasData =
    {
        4, 4, 4
    };

    std::vector<uint8_t> expectedOutputData =
    {
        121, 118, 115, 121, 118, 115, 121, 118, 115
    };


    Convolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = DataLayout::NHWC;

    auto memoryManager = ClWorkloadFactoryHelper::GetMemoryManager();
    auto clMemoryManager = armnn::PolymorphicPointerDowncast<armnn::ClMemoryManager>(memoryManager);
    auto tensorHandleFactory = ClWorkloadFactoryHelper::GetTensorHandleFactory(memoryManager);

    std::unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelInfo);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = tensorHandleFactory.CreateTensorHandle(biasInfo);
    std::unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);


    WorkloadInfo workloadInfo;

    Convolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, kernelInfo, weightsHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, biasInfo, biasHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    // Initialize our m_CLCompileContext using default device and context
    auto context = arm_compute::CLKernelLibrary::get().context();
    auto device  = arm_compute::CLKernelLibrary::get().get_device();
    auto clCompileContext = arm_compute::CLCompileContext(context, device);



    // Check built programs are empty in context
    CHECK(clCompileContext.get_built_programs().empty());

    auto workload = std::make_unique<ClConvolution2dWorkload>(queueDescriptor,
                                                              workloadInfo,
                                                              clMemoryManager->GetIntraLayerManager(),
                                                              clCompileContext);
    ARMNN_ASSERT(workload != nullptr);
    // Check built programs are not empty in context
    CHECK(!clCompileContext.get_built_programs().empty());
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

    CHECK((inputHandle->GetShape() == inputShape));
    CHECK((outputHandle->GetShape() == outputShape));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateDepthwiseConvolutionFloat32NhwcWorkload")
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
    auto predResult = CompareIClTensorHandleShape(inputHandle, {2, 3, 6, 6});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {2, 2, 6, 6});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateDirectConvolution2dFloatWorkload")
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateDirectConvolution2dFloat16Workload")
{
    ClDirectConvolution2dWorkloadTest<ClConvolution2dWorkload, armnn::DataType::Float16>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateDirectConvolution2dUint8Workload")
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
    auto predResult = CompareIClTensorHandleShape(inputHandle, {3, 1, 4, 5});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {3, 7});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}


TEST_CASE_FIXTURE(ClContextControlFixture, "CreateFullyConnectedFloatWorkloadTest")
{
    ClCreateFullyConnectedWorkloadTest<ClFullyConnectedWorkload, armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateFullyConnectedFloat16WorkloadTest")
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

    CHECK((inputHandle->GetShape() == inputShape));
    CHECK((outputHandle->GetShape() == outputShape));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateNormalizationFloat32NchwWorkload")
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateNormalizationFloat16NchwWorkload")
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateNormalizationFloat32NhwcWorkload")
{
    ClNormalizationWorkloadTest<ClNormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateNormalizationFloat16NhwcWorkload")
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

    CHECK((inputHandle->GetShape() == inputShape));
    CHECK((outputHandle->GetShape() == outputShape));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreatePooling2dFloatNchwWorkload")
{
    ClPooling2dWorkloadTest<armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreatePooling2dFloatNhwcWorkload")
{
    ClPooling2dWorkloadTest<armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreatePooling2dFloat16NchwWorkload")
{
    ClPooling2dWorkloadTest<armnn::DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreatePooling2dFloat16NhwcWorkload")
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

    CHECK((inputHandle->GetShape() == inputShape));
    CHECK((alphaHandle->GetShape() == alphaShape));
    CHECK((outputHandle->GetShape() == outputShape));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreatePreluFloat16Workload")
{
    ClCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::Float16);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreatePreluFloatWorkload")
{
    ClCreatePreluWorkloadTest({ 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 }, DataType::Float32);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreatePreluUint8Workload")
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

    auto predResult = CompareIClTensorHandleShape(inputHandle, {4, 1});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {1, 4});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateReshapeFloatWorkload")
{
    ClCreateReshapeWorkloadTest<armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateReshapeFloat16Workload")
{
    ClCreateReshapeWorkloadTest<armnn::DataType::Float16>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateReshapeUint8Workload")
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

    auto predResult = CompareIClTensorHandleShape(inputHandle, {4, 1});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {4, 1});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}


TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSoftmaxFloat32WorkloadTest")
{
    ClSoftmaxWorkloadTest<ClSoftmaxWorkload, armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSoftmaxFloat16WorkloadTest")
{
    ClSoftmaxWorkloadTest<ClSoftmaxWorkload, armnn::DataType::Float16>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSoftmaxQAsymmU8Workload")
{
    ClSoftmaxWorkloadTest<ClSoftmaxWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSoftmaxQAsymmS8Workload")
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
    auto predResult = CompareIClTensorHandleShape(inputHandle, {5, 7, 7});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());

    auto outputHandle1 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[1]);
    predResult = CompareIClTensorHandleShape(outputHandle1, {2, 7, 7});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());

    auto outputHandle2 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[2]);
    predResult = CompareIClTensorHandleShape(outputHandle2, {2, 7, 7});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());

    auto outputHandle0 = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    predResult = CompareIClTensorHandleShape(outputHandle0, {1, 7, 7});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSplitterFloatWorkload")
{
    ClSplitterWorkloadTest<armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSplitterFloat16Workload")
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

    CHECK(sOut0);
    CHECK(sOut1);
    CHECK(mIn0);
    CHECK(mIn1);

    //Fliped order of inputs/outputs.
    bool validDataPointers = (sOut0 == mIn1) && (sOut1 == mIn0);
    CHECK(validDataPointers);


    //Also make sure that the inputs are subtensors of one tensor and outputs are sub tensors of another tensor.
    bool validSubTensorParents = (mIn0->GetTensor().parent() == mIn1->GetTensor().parent())
                                    && (sOut0->GetTensor().parent() == sOut1->GetTensor().parent());

    CHECK(validSubTensorParents);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSplitterConcatFloatWorkload")
{
    ClSplitterConcatTest<armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSplitterConcatFloat16Workload")
{
    ClSplitterConcatTest<armnn::DataType::Float16>();
}


TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSingleOutputMultipleInputs")
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

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateMemCopyWorkloadsCl")
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

    CHECK((inputHandle->GetShape() == inputShape));
    CHECK((outputHandle->GetShape() == outputShape));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateL2NormalizationFloatNchwWorkload")
{
    ClL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateL2NormalizationFloatNhwcWorkload")
{
    ClL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateL2NormalizationFloat16NchwWorkload")
{
    ClL2NormalizationWorkloadTest<ClL2NormalizationFloatWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateL2NormalizationFloat16NhwcWorkload")
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

    auto predResult = CompareIClTensorHandleShape(inputHandle, {4, 1});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {4, 1});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateLogSoftmaxFloat32WorkloadTest")
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
    auto predResult = CompareIClTensorHandleShape(inputHandle, {2, 2});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, {2, 4});
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateLSTMWorkloadFloatWorkload")
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

    armnn::PredicateResult predResult(true);
    switch (dataLayout)
    {
        case DataLayout::NHWC:
            predResult = CompareIClTensorHandleShape(inputHandle, { 2, 4, 4, 3 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            predResult = CompareIClTensorHandleShape(outputHandle, { 2, 2, 2, 3 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            break;
        default: // DataLayout::NCHW
            predResult = CompareIClTensorHandleShape(inputHandle, { 2, 3, 4, 4 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
            predResult = CompareIClTensorHandleShape(outputHandle, { 2, 3, 2, 2 });
            CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateResizeFloat32NchwWorkload")
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float32>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateResizeFloat16NchwWorkload")
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float16>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateResizeUint8NchwWorkload")
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::QAsymmU8>(DataLayout::NCHW);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateResizeFloat32NhwcWorkload")
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float32>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateResizeFloat16NhwcWorkload")
{
    ClResizeWorkloadTest<ClResizeWorkload, armnn::DataType::Float16>(DataLayout::NHWC);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateResizeUint8NhwcWorkload")
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
    auto predResult = CompareIClTensorHandleShape(inputHandle, {  1, 3, 7, 4 });
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, { 1, 4 });
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateMeanFloat32Workload")
{
    ClMeanWorkloadTest<ClMeanWorkload, armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateMeanFloat16Workload")
{
    ClMeanWorkloadTest<ClMeanWorkload, armnn::DataType::Float16>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateMeanUint8Workload")
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

    auto predResult = CompareIClTensorHandleShape(inputHandle0, { 2, 3, 2, 5 });
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(inputHandle1, { 2, 3, 2, 5 });
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, outputShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConcatDim0Float32Workload")
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::Float32>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConcatDim1Float32Workload")
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::Float32>({ 2, 6, 2, 5 }, 1);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConcatDim3Float32Workload")
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::Float32>({ 2, 3, 2, 10 }, 3);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConcatDim0Uint8Workload")
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::QAsymmU8>({ 4, 3, 2, 5 }, 0);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConcatDim1Uint8Workload")
{
    ClCreateConcatWorkloadTest<ClConcatWorkload, armnn::DataType::QAsymmU8>({ 2, 6, 2, 5 }, 1);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateConcatDim3Uint8Workload")
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

    auto predResult = CompareIClTensorHandleShape(inputHandle, { 1, 2, 2, 1 });
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    predResult = CompareIClTensorHandleShape(outputHandle, { 1, 1, 1, 4 });
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSpaceToDepthFloat32Workload")
{
    ClSpaceToDepthWorkloadTest<ClSpaceToDepthWorkload, armnn::DataType::Float32>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSpaceToDepthFloat16Workload")
{
    ClSpaceToDepthWorkloadTest<ClSpaceToDepthWorkload, armnn::DataType::Float16>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSpaceToDepthQAsymm8Workload")
{
    ClSpaceToDepthWorkloadTest<ClSpaceToDepthWorkload, armnn::DataType::QAsymmU8>();
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateSpaceToDepthQSymm16Workload")
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
        auto predResult1 = CompareIClTensorHandleShape(inputHandle, inputShape);
        CHECK_MESSAGE(predResult1.m_Result, predResult1.m_Message.str());
    }
    auto outputHandle = PolymorphicDowncast<IClTensorHandle*>(queueDescriptor.m_Outputs[0]);
    auto predResult2 = CompareIClTensorHandleShape(outputHandle, outputShape);
    CHECK_MESSAGE(predResult2.m_Result, predResult2.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateStackFloat32Workload")
{
    ClCreateStackWorkloadTest<armnn::DataType::Float32>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateStackFloat16Workload")
{
    ClCreateStackWorkloadTest<armnn::DataType::Float16>({ 3, 4, 5 }, { 3, 4, 2, 5 }, 2, 2);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateStackUint8Workload")
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
    CHECK((inputHandle->GetShape() == TensorShape({2, 4})));
    CHECK((inputHandle->GetDataType() == arm_compute::DataType::QASYMM8_SIGNED));

    IAclTensorHandle* cellStateOutHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[1]);
    CHECK((cellStateOutHandle->GetShape() == TensorShape({2, 4})));
    CHECK((cellStateOutHandle->GetDataType() == arm_compute::DataType::QSYMM16));

    IAclTensorHandle* outputHandle = PolymorphicDowncast<IAclTensorHandle*>(queueDescriptor.m_Outputs[2]);
    CHECK((outputHandle->GetShape() == TensorShape({2, 4})));
    CHECK((outputHandle->GetDataType() == arm_compute::DataType::QASYMM8_SIGNED));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateQLstmWorkloadTest")
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

TEST_CASE_FIXTURE(ClContextControlFixture, "CreateQuantizedLstmWorkload")
{
    ClCreateQuantizedLstmWorkloadTest<ClQuantizedLstmWorkload>();
}

template <armnn::DataType DataType>
static void ClCreateActivationWorkloadReplaceFunctionsTest()
{
    std::shared_ptr<ClMemoryManager> memoryManager = std::make_shared<ClMemoryManager>(
            std::make_unique<arm_compute::CLBufferAllocator>());

    Graph graph;
    ClWorkloadFactory factory = ClWorkloadFactoryHelper::GetFactory(memoryManager);
    // input and output are created as armnn::TensorInfo tensorInfo({1, 1}, DataType)
    auto workloadPtr = CreateActivationWorkloadTest<ClActivationWorkload, DataType>(factory, graph);

    // new input and output tensor handlers are created and then replace in the workload
    const ClTensorHandleFactory tensorHandleFactory(memoryManager);
    TensorInfo inputInfo({2 , 2}, DataType::Float16);
    TensorInfo outputInfo({2 , 2}, DataType::Float16);
    unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo, true);
    inputHandle->Manage();
    inputHandle->Allocate();
    unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo, true);
    outputHandle->Manage();
    outputHandle->Allocate();

    unsigned int slot = 0;
    CHECK_THROWS_AS(workloadPtr->ReplaceInputTensorHandle(inputHandle.get(), slot), UnimplementedException);
    CHECK_THROWS_AS(workloadPtr->ReplaceOutputTensorHandle(outputHandle.get(), slot), UnimplementedException);
}

TEST_CASE("ClReplaceFunctionsfromFloat32toFloat16ActivationWorkload")
{
    ClCreateActivationWorkloadReplaceFunctionsTest<armnn::DataType::Float32>();
}

}
