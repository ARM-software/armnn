//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NormalizationTestImpl.hpp"

#include <armnn/Exceptions.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/ILayerSupport.hpp>
#include <armnn/BackendHelper.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

LayerTestResult<float,4> SimpleNormalizationTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::NormalizationAlgorithmChannel normChannel,
    armnn::NormalizationAlgorithmMethod normMethod)
{
    IgnoreUnused(memoryManager);
    const unsigned int inputHeight = 2;
    const unsigned int inputWidth = 2;
    const unsigned int inputChannels = 1;
    const unsigned int inputNum = 2;

    unsigned int outputHeight = inputHeight;
    unsigned int outputWidth = inputWidth;
    unsigned int outputChannels = inputChannels;
    unsigned int outputNum = inputNum;

    unsigned int inputShape[] = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[] = { outputNum, outputChannels, outputHeight, outputWidth };

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input =
    {
        // Batch #0
        1.0f, 2.0f,
        3.0f, 4.0f,
        // Batch #1
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<float> expectedOutput(outputTensorInfo.GetNumElements());

    float alpha = 1.f;
    float beta = 1.f;
    float kappa = 1.f;
    uint32_t normSize = 3;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::NormalizationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_NormChannelType = normChannel;
    data.m_Parameters.m_NormMethodType = normMethod;
    data.m_Parameters.m_NormSize = normSize;
    data.m_Parameters.m_Alpha = alpha;
    data.m_Parameters.m_Beta = beta;
    data.m_Parameters.m_K = kappa;
    data.m_Parameters.m_DataLayout = armnn::DataLayout::NCHW;

    armnn::PassthroughTensorHandle refHandle(outputTensorInfo, expectedOutput.data());
    armnn::NormalizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, &refHandle);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Normalization,
                                                                                data,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    switch (normMethod)
    {
        case armnn::NormalizationAlgorithmMethod::LocalBrightness:
        {
            switch (normChannel)
            {
                case armnn::NormalizationAlgorithmChannel::Within:
                {
                    // When normalising within channels, the 3x3 kernel covers the entire 2x2 input at every index.
                    // Therefore, all output values should equal the inputs, but divided by:
                    // pow((kappa + (accumulatedScale * alpha)), beta)
                    // ...where accumulatedScale is the sum of every element squared.
                    float divisor[inputNum];

                    float accumulatedScale1 = 0.0f;
                    for (size_t i = 0; i < input.size()/2; ++i)
                    {
                        accumulatedScale1 += input[i]*input[i];
                    }

                    float accumulatedScale2 = 0.0f;
                    for (size_t i = input.size()/2; i < input.size(); ++i)
                    {
                        accumulatedScale2 += input[i]*input[i];
                    }

                    divisor[0] = powf((kappa + accumulatedScale1 * alpha), beta);
                    divisor[1] = powf((kappa + accumulatedScale2 * alpha), beta);

                    std::vector<float> output;
                    unsigned int divisorIndex = 0;
                    for (size_t i = 0; i < input.size(); ++i)
                    {
                        if (i == input.size()/2)
                        {
                            divisorIndex++;
                        }
                        output.emplace_back(input[i]/divisor[divisorIndex]);
                    }

                    expectedOutput = output;
                    break;
                }
                case armnn::NormalizationAlgorithmChannel::Across:
                {
                    // When normalising across channels, all output values should equal the inputs, but multiplied by:
                    // pow((kappa + (accumulatedScale * alpha)), -beta)
                    // ...where accumulatedScale is the sum of the inputs for adjacent channels for this element squared
                    // ...where adjacent channels means within half the normSize for the channel
                    // The test data has only one channel, so this is simplified below.
                    std::vector<float> outputVector;

                    for (unsigned int i = 0; i < input.size(); ++i)
                    {
                        float accumulatedScale = input[i]*input[i];
                        float scale = powf((kappa + accumulatedScale * alpha), -beta);
                        outputVector.push_back(input[i] * scale);
                    }
                    expectedOutput = outputVector;
                    break;
                }
                default:
                {
                    throw armnn::UnimplementedException("Unsupported normalisation channel type, "
                                                        "only Across and Within are supported");
                }
            }
            break;
        }
        case armnn::NormalizationAlgorithmMethod::LocalContrast: // NOTE: intentional fallthrough.
        default:
        {
            throw armnn::UnimplementedException("Unsupported normalisation method type, "
                                                "only LocalBrightness is supported");
        }
    }

    return LayerTestResult<float, 4>(actualOutput,
                                     expectedOutput,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
}

LayerTestResult<float,4> SimpleNormalizationNhwcTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::NormalizationAlgorithmChannel normChannel,
    armnn::NormalizationAlgorithmMethod normMethod)
{
    const unsigned int inputHeight = 2;
    const unsigned int inputWidth = 2;
    const unsigned int inputChannels = 1;
    const unsigned int inputNum = 2;

    unsigned int outputHeight = inputHeight;
    unsigned int outputWidth = inputWidth;
    unsigned int outputChannels = inputChannels;
    unsigned int outputNum = inputNum;

    unsigned int inputShape[] = { inputNum, inputHeight, inputWidth, inputChannels };
    unsigned int outputShape[] = { outputNum, outputHeight, outputWidth, outputChannels };

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input =
    {
        // Batch #0
        1.0f, 2.0f,
        3.0f, 4.0f,
        // Batch #1
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<float> expectedOutput(outputTensorInfo.GetNumElements());

    float alpha = 1.f;
    float beta = 1.f;
    float kappa = 1.f;
    uint32_t normSize = 3;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::NormalizationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_NormChannelType = normChannel;
    data.m_Parameters.m_NormMethodType = normMethod;
    data.m_Parameters.m_NormSize = normSize;
    data.m_Parameters.m_Alpha = alpha;
    data.m_Parameters.m_Beta = beta;
    data.m_Parameters.m_K = kappa;
    data.m_Parameters.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::PassthroughTensorHandle refHandle(outputTensorInfo, expectedOutput.data());
    armnn::NormalizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, &refHandle);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Normalization,
                                                                                data,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    switch (normMethod)
    {
        case armnn::NormalizationAlgorithmMethod::LocalBrightness:
        {
            switch (normChannel)
            {
                case armnn::NormalizationAlgorithmChannel::Across:
                {
                    expectedOutput = { 0.5f, 0.400000006f, 0.300000012f, 0.235294119f,
                                       0.192307696f, 0.16216217f, 0.140000001f, 0.123076923f };
                    break;
                }
                default:
                {
                    throw armnn::UnimplementedException("Unsupported normalisation channel type, "
                                                        "Only Cross-map is supported for NHWC layout");
                }
            }
            break;
        }
        case armnn::NormalizationAlgorithmMethod::LocalContrast: // NOTE: intentional fallthrough.
        default:
        {
            throw armnn::UnimplementedException("Unsupported normalisation method type, "
                                                "only LocalBrightness is supported");
        }
    }

    return LayerTestResult<float, 4>(actualOutput,
                                     expectedOutput,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
}

LayerTestResult<float,4> CompareNormalizationTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::NormalizationAlgorithmChannel normChannel,
    armnn::NormalizationAlgorithmMethod normMethod)
{
    constexpr unsigned int inputNum = 5;
    constexpr unsigned int inputChannels = 3;
    constexpr unsigned int inputHeight = 32;
    constexpr unsigned int inputWidth = 24;

    constexpr unsigned int outputNum = inputNum;
    constexpr unsigned int outputChannels = inputChannels;
    constexpr unsigned int outputHeight = inputHeight;
    constexpr unsigned int outputWidth = inputWidth;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = {inputNum, inputChannels, inputHeight, inputWidth};
    unsigned int outputShape[] = {outputNum, outputChannels, outputHeight, outputWidth};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    LayerTestResult<float,4> ret(outputTensorInfo);

    auto input = MakeRandomTensor<float>(inputTensorInfo, 111234);

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<float> expectedOutput(outputTensorInfo.GetNumElements());

    constexpr float alpha = 1.f;
    constexpr float beta = 1.f;
    constexpr float kappa = 1.f;
    constexpr uint32_t normSize = 5;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::NormalizationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_NormChannelType = normChannel;
    data.m_Parameters.m_NormMethodType  = normMethod;
    data.m_Parameters.m_NormSize        = normSize;
    data.m_Parameters.m_Alpha           = alpha;
    data.m_Parameters.m_Beta            = beta;
    data.m_Parameters.m_K               = kappa;

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo);

    armnn::NormalizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    // Don't execute if Normalization is not supported for the method and channel types, as an exception will be raised.
    armnn::BackendId backend = workloadFactory.GetBackendId();
    auto handle = armnn::GetILayerSupportByBackendId(backend);
    ret.m_Supported = handle.IsNormalizationSupported(inputTensorInfo, outputTensorInfo, data.m_Parameters);

    if (!ret.m_Supported)
    {
        return ret;
    }

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::Normalization, data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef
            = refWorkloadFactory.CreateWorkload(armnn::LayerType::Normalization, refData, refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());
    CopyDataToITensorHandle(inputHandleRef.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    workloadRef->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    CopyDataFromITensorHandle(expectedOutput.data(), outputHandleRef.get());
    ret.m_ActualData = actualOutput;
    ret.m_ExpectedData = expectedOutput;

    return ret;
}

LayerTestResult<float,4> AcrossChannelNormalizationTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::NormalizationAlgorithmChannel normChannel,
    armnn::NormalizationAlgorithmMethod normMethod)
{
    const unsigned int inputHeight = 1;
    const unsigned int inputWidth = 2;
    const unsigned int inputChannels = 3;
    const unsigned int inputNum = 2;

    unsigned int outputHeight = inputHeight;
    unsigned int outputWidth = inputWidth;
    unsigned int outputChannels = inputChannels;
    unsigned int outputNum = inputNum;

    unsigned int inputShape[] = { inputNum, inputHeight, inputWidth, inputChannels };
    unsigned int outputShape[] = { outputNum, outputHeight, outputWidth, outputChannels };

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input =
    {
        // Batch #0
        -2.1f, 2.6f, 1.7f, 1.2f, -1.0f, 0.7f,
        // Batch #1
        -2.1f, 2.6f, 1.7f, 1.2f, -1.0f, 0.7f,
    };

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<float> expectedOutput(outputTensorInfo.GetNumElements());

    float alpha = 4.f;
    float beta  = 0.5f;
    float kappa = 9.f;
    uint32_t normSize = 5;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::NormalizationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_NormChannelType = normChannel;
    data.m_Parameters.m_NormMethodType = normMethod;
    data.m_Parameters.m_NormSize = normSize;
    data.m_Parameters.m_Alpha = alpha;
    data.m_Parameters.m_Beta = beta;
    data.m_Parameters.m_K = kappa;
    data.m_Parameters.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::PassthroughTensorHandle refHandle(outputTensorInfo, expectedOutput.data());
    armnn::NormalizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, &refHandle);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Normalization,
                                                                                data,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    switch (normMethod)
    {
        case armnn::NormalizationAlgorithmMethod::LocalBrightness:
        {
            switch (normChannel)
            {
                case armnn::NormalizationAlgorithmChannel::Across:
                {
                    expectedOutput = { -0.259993f, 0.321897f, 0.210471f, 0.263625f, -0.219687f, 0.153781f,
                                       -0.259993f, 0.321897f, 0.210471f, 0.263625f, -0.219687f, 0.153781f, };
                    break;
                }
                default:
                {
                    throw armnn::UnimplementedException("Unsupported normalisation channel type, "
                                                        "only Across and Within are supported");
                }
            }
            break;
        }
        case armnn::NormalizationAlgorithmMethod::LocalContrast: // NOTE: intentional fallthrough.
        default:
        {
            throw armnn::UnimplementedException("Unsupported normalisation method type, "
                                                "only LocalBrightness is supported");
        }
    }

    return LayerTestResult<float, 4>(actualOutput,
                                     expectedOutput,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
}

} // anonymous namespace

LayerTestResult<float,4> SimpleNormalizationAcrossTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    auto normChannel = armnn::NormalizationAlgorithmChannel::Across;
    return SimpleNormalizationTestImpl(workloadFactory, memoryManager,  tensorHandleFactory, normChannel, normMethod);
}

LayerTestResult<float,4> SimpleNormalizationWithinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    auto normChannel = armnn::NormalizationAlgorithmChannel::Within;
    return SimpleNormalizationTestImpl(workloadFactory, memoryManager,  tensorHandleFactory, normChannel, normMethod);
}

LayerTestResult<float,4> SimpleNormalizationAcrossNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    auto normChannel = armnn::NormalizationAlgorithmChannel::Across;
    return SimpleNormalizationNhwcTestImpl(
            workloadFactory, memoryManager,  tensorHandleFactory, normChannel, normMethod);
}

LayerTestResult<float,4> CompareNormalizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::NormalizationAlgorithmChannel normChannel,
    armnn::NormalizationAlgorithmMethod normMethod)
{
    return CompareNormalizationTestImpl(
            workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory,
            normChannel, normMethod);
}

LayerTestResult<float,4> AcrossChannelNormalizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto normMethod = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    auto normChannel = armnn::NormalizationAlgorithmChannel::Across;
    return AcrossChannelNormalizationTestImpl(workloadFactory,
                                               memoryManager,
                                               tensorHandleFactory,
                                               normChannel,
                                               normMethod);
}
