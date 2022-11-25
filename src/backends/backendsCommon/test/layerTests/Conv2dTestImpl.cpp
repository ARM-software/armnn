//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Conv2dTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <armnnUtils/TensorUtils.hpp>

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/Permute.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <string>

//
// Static data
//

// 2-channel bias used by a number of Conv2d tests.
static std::vector<float> Bias2({0, 2});

static std::vector<float> Bias4({1, 2, 3, 4});

static std::vector<float> Bias8({1, 2, 3, 4, 1, 2, 3, 4});

// 3-channel 16x8 image used as common input data for a number of Conv2d tests.
static std::vector<float> ConvInput3x8x16({
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
});

using namespace armnnUtils;

//
// Helper templates
//

// Helper template that returns either Bias2 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> GetBias2(bool biasEnabled, float qScale)
{
    if(biasEnabled)
    {
        return QuantizedVector<T>(Bias2, qScale, 0);
    }
    else
    {
        return std::vector<T>();
    }
}

// Helper template that returns either Bias4 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> GetBias4(bool biasEnabled, float qScale)
{
    if(biasEnabled)
    {
        return QuantizedVector<T>(Bias4, qScale, 0);
    }
    else
    {
        return std::vector<T>();
    }
}

// Helper template that returns either Bias8 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> GetBias8(bool biasEnabled, float qScale)
{
    if(biasEnabled)
    {
        return QuantizedVector<T>(Bias8, qScale, 0);
    }
    else
    {
        return std::vector<T>();
    }
}

// Helper template that returns either Bias4 or an empty vector depending on whether bias is enabled.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> GetBias(bool biasEnabled, float qScale, armnn::TensorInfo outputInfo, armnn::DataLayout layout)
{
    const armnnUtils::DataLayoutIndexed dataLayoutIndexed(layout);
    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int outputChannels = outputInfo.GetShape()[channelsIndex];

    switch (outputChannels)
    {
        case 2:
        default:
        {
            return GetBias2<ArmnnType>(biasEnabled, qScale);
        }
        case 4:
        {
            return GetBias4<ArmnnType>(biasEnabled, qScale);
        }
        case 8:
        {
            return GetBias8<ArmnnType>(biasEnabled, qScale);
        }
    }
}

//
// Implementation templates
//

// Mapping from input type to bias type for fully connected layers.
// float => float, uint8_t => int32_t
template<typename T>
struct FullyConnectedBiasTypeForInputType;

template<>
struct FullyConnectedBiasTypeForInputType<float>
{
    using Type = float;
};

template<>
struct FullyConnectedBiasTypeForInputType<uint8_t>
{
    using Type = int32_t;
};

// Modifies a std::vector in-place using a specified bias.
template<typename T, typename B>
void ApplyBias(std::vector<T>& v, float vScale, int32_t vOffset,
    const std::vector<B>& bias, float bScale, int32_t bOffset, uint32_t w, uint32_t h)
{
    ARMNN_ASSERT_MSG((armnn::IsQuantizedType<T>() && vScale != 0.0f) || (!armnn::IsQuantizedType<T>()),
                     "Invalid type and parameter combination.");
    ARMNN_ASSERT_MSG((armnn::IsQuantizedType<B>() && bScale != 0.0f) || (!armnn::IsQuantizedType<B>()),
                     "Invalid type and parameter combination.");

    // Note we need to dequantize and re-quantize the image value and the bias.
    for (uint32_t i = 0; i < bias.size(); ++i)
    {
        float dBias = SelectiveDequantize(bias[i], bScale, bOffset);
        for (uint32_t y = 0; y < h; ++y)
        {
            for (uint32_t x = 0; x < w; ++x)
            {
                uint32_t offset = (i * h + y) * w + x;
                ARMNN_ASSERT(offset < v.size());
                T& outRef = v[offset];
                float dOutput = SelectiveDequantize(outRef, vScale, vOffset);
                outRef = SelectiveQuantize<T>(dOutput + dBias, vScale, vOffset);
            }
        }
    }
}

//
// Convolution2d implementations
//

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>, typename B = armnn::ResolveType<ArmnnBType>>
LayerTestResult<T, 4> SimpleConvolution2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const std::vector<T>& originalInput,
    const std::vector<T>& originalKernel,
    const std::vector<B>& bias,
    const std::vector<T>& originalOutputExpected,
    const armnn::TensorShape& originalInputShape,
    const armnn::TensorShape& originalKernelShape,
    const armnn::TensorShape& originalOutputExpectedShape,
    float qScale,
    int32_t qOffset,
    const armnn::DataLayout layout = armnn::DataLayout::NCHW,
    uint32_t padLeft = 0,
    uint32_t padTop = 0,
    uint32_t padRight = 0,
    uint32_t padBottom = 0,
    uint32_t strideX = 1,
    uint32_t strideY = 1,
    uint32_t dilationX = 1,
    uint32_t dilationY = 1)
{
    armnn::IgnoreUnused(memoryManager);
    unsigned int inputHeight    = armnn::numeric_cast<unsigned int>(originalInputShape[2]);
    unsigned int inputWidth     = armnn::numeric_cast<unsigned int>(originalInputShape[3]);
    unsigned int inputChannels  = armnn::numeric_cast<unsigned int>(originalInputShape[1]);
    unsigned int inputNum       = armnn::numeric_cast<unsigned int>(originalInputShape[0]);

    unsigned int outputHeight   = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[2]);
    unsigned int outputWidth    = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[3]);
    unsigned int outputChannels = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[1]);
    unsigned int outputNum      = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[0]);

    unsigned int kernelHeight   = armnn::numeric_cast<unsigned int>(originalKernelShape[2]);
    unsigned int kernelWidth    = armnn::numeric_cast<unsigned int>(originalKernelShape[3]);
    unsigned int kernelChannels = armnn::numeric_cast<unsigned int>(originalKernelShape[1]);
    unsigned int kernelDepthMul = armnn::numeric_cast<unsigned int>(originalKernelShape[0]);

    bool biasEnabled = bias.size() > 0;

    // This function currently assumes 1 batch of input/output (and duplicates this into 2 batches).
    ARMNN_ASSERT(inputNum == 1);
    ARMNN_ASSERT(outputNum == 1);

    // If a bias is used, its size must equal the number of output channels.
    ARMNN_ASSERT(!biasEnabled || bias.size() == outputChannels);

    // Note these tensors will use two (identical) batches.
    armnn::TensorInfo inputTensorInfo =
            armnnUtils::GetTensorInfo(2*inputNum, inputChannels, inputHeight, inputWidth, layout, ArmnnType);
    armnn::TensorInfo outputTensorInfo =
            armnnUtils::GetTensorInfo(2*outputNum, outputChannels, outputHeight, outputWidth, layout, ArmnnType);
    armnn::TensorInfo kernelDesc =
            armnnUtils::GetTensorInfo(kernelDepthMul, kernelChannels, kernelHeight, kernelWidth, layout, ArmnnType);
    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, ArmnnBType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
        kernelDesc.SetQuantizationScale(qScale);
        kernelDesc.SetQuantizationOffset(qOffset);
        biasDesc.SetQuantizationScale(qScale*qScale);
        biasDesc.SetQuantizationOffset(0);
    }

    // Construct input data - two batches of the same input image.
    std::vector<T> inputImage;
    inputImage.assign(originalInput.data(), originalInput.data() + 1*inputChannels*inputHeight*inputWidth);
    std::vector<T> inputData;
    inputData.insert(inputData.end(), inputImage.begin(), inputImage.end());
    inputData.insert(inputData.end(), inputImage.begin(), inputImage.end());

    // at this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(T));
        inputData = tmp;
    }

    std::vector<T> outputImage;
    outputImage.assign(originalOutputExpected.data(),
            originalOutputExpected.data() + outputChannels*outputHeight*outputWidth);

    // Apply bias to output image if it is enabled.
    if(biasEnabled)
    {
        std::vector<T> biasV;
        biasV.assign(bias.data(), bias.data() + outputChannels);
        ApplyBias(outputImage, outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
            biasV, biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
            outputWidth, outputHeight);
    }

    // Data will be copied from outputHandle
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    // Construct expected output data - two identical images.
    std::vector<T> expectedOutput;
    expectedOutput.insert(expectedOutput.end(), outputImage.begin(), outputImage.end());
    expectedOutput.insert(expectedOutput.end(), outputImage.begin(), outputImage.end());

    // at this point if we require it permute the expected output
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(expectedOutput.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, expectedOutput.data(), tmp.data(), sizeof(T));
        expectedOutput = tmp;
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelDesc);

    armnn::Convolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;

    // Permute the kernel if necessary
    std::vector<T> kernel = originalKernel;
    if (layout == armnn::DataLayout::NHWC)
    {
        armnnUtils::Permute(kernelDesc.GetShape(), NCHWToNHWC, originalKernel.data(), kernel.data(), sizeof(T));
    }

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelDesc, weightsHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;
    if (biasEnabled)
    {
        biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
        AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    }
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = layout;
    data.m_Parameters.m_DilationX = dilationX;
    data.m_Parameters.m_DilationY = dilationY;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Convolution2d,
                                                                                data,
                                                                                info);
    inputHandle->Allocate();
    outputHandle->Allocate();
    weightsHandle->Allocate();

    if (biasEnabled)
    {
        biasHandle->Allocate();
        CopyDataToITensorHandle(biasHandle.get(), bias.data());
    }

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());
    CopyDataToITensorHandle(weightsHandle.get(), kernel.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>, typename B = armnn::ResolveType<ArmnnBType>,
         armnn::DataType OutType = ArmnnType, typename O = armnn::ResolveType<OutType>>
LayerTestResult<O, 4> SimpleConvolution2dNhwcTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const std::vector<T>& input,
    const std::vector<T>& kernel,
    const std::vector<B>& bias,
    const std::vector<O>& outputExpected,
    const armnn::TensorShape& inputShape,
    const armnn::TensorShape& kernelShape,
    const armnn::TensorShape& outputExpectedShape,
    const armnn::DataLayout dataLayout,
    float qScale,
    int32_t qOffset,
    uint32_t padLeft = 1,
    uint32_t padTop = 1,
    uint32_t padRight = 1,
    uint32_t padBottom = 1,
    uint32_t strideX  = 1,
    uint32_t strideY  = 1)
{
    armnn::IgnoreUnused(qScale, qOffset);
    unsigned int inputNum       = armnn::numeric_cast<unsigned int>(inputShape[0]);
    unsigned int inputChannels  = armnn::numeric_cast<unsigned int>(inputShape[3]);
    unsigned int inputHeight    = armnn::numeric_cast<unsigned int>(inputShape[1]);
    unsigned int inputWidth     = armnn::numeric_cast<unsigned int>(inputShape[2]);

    unsigned int kernelChanMul  = armnn::numeric_cast<unsigned int>(kernelShape[0]);
    unsigned int kernelChannels = armnn::numeric_cast<unsigned int>(kernelShape[3]);
    unsigned int kernelHeight   = armnn::numeric_cast<unsigned int>(kernelShape[1]);
    unsigned int kernelWidth    = armnn::numeric_cast<unsigned int>(kernelShape[2]);

    unsigned int outputNum      = armnn::numeric_cast<unsigned int>(outputExpectedShape[0]);
    unsigned int outputChannels = armnn::numeric_cast<unsigned int>(outputExpectedShape[3]);
    unsigned int outputHeight   = armnn::numeric_cast<unsigned int>(outputExpectedShape[1]);
    unsigned int outputWidth    = armnn::numeric_cast<unsigned int>(outputExpectedShape[2]);

    bool biasEnabled = bias.size() > 0;

    // Creates the tensors.
    armnn::TensorInfo inputTensorInfo({inputNum, inputHeight, inputWidth, inputChannels}, ArmnnType);
    armnn::TensorInfo outputTensorInfo({outputNum, outputHeight, outputWidth, outputChannels},
                                       OutType);
    armnn::TensorInfo kernelDesc({kernelChanMul, kernelHeight, kernelWidth, kernelChannels}, ArmnnType);
    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, ArmnnBType);

    // Construct the input data.
    std::vector<T> inputData;
    inputData.assign(input.data(), input.data() + inputHeight*inputWidth*inputChannels);

    // Construct the output data, with bias applied, as appropriate.
    std::vector<O> outputData;
    outputData.assign(outputExpected.data(), outputExpected.data() + outputHeight*outputWidth*outputChannels);

    std::vector<O> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;

//    armnn::ScopedTensorHandle weightsTensor(kernelDesc);
//    AllocateAndCopyDataToITensorHandle(&weightsTensor, kernel.data());

//    armnn::ScopedTensorHandle biasTensor(biasDesc);

    armnn::Convolution2dQueueDescriptor data;

    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelDesc, weightsHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    if (biasEnabled)
    {
        biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
        AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    }

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Convolution2d,
                                                                                data,
                                                                                info);
    inputHandle->Allocate();
    outputHandle->Allocate();
    weightsHandle->Allocate();

    if (biasEnabled)
    {
        biasHandle->Allocate();
        CopyDataToITensorHandle(biasHandle.get(), bias.data());
    }

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());
    CopyDataToITensorHandle(weightsHandle.get(), kernel.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<O, 4>(actualOutput,
                                 outputData,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T,4> Convolution1dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled)
{
    using B = armnn::ResolveType<ArmnnBType>;
    // Until we have a specialist 1D convolution layer, we can fake one using
    // 2D convolution with the final dimension set to 1.
    // I don't anticipate this being particularly slow, given that convolution is implemented
    // as a matrix multiplication, at which point dimension doesn't matter.

    unsigned int batchSize      = 1;
    unsigned int inputChannels  = 2;
    unsigned int outputChannels = 3;
    unsigned int inputSize      = 5; // The 1D size (could view as 'width' or 'height').
    unsigned int kernelSize     = 3;
    unsigned int padSize        = 2;
    unsigned int stride         = 1;
    unsigned int outputSize     = 7; // (inputSize + 2 * padSize - kernelSize + 1) / stride.

    armnn::TensorInfo inputInfo({batchSize, inputChannels, inputSize, 1}, ArmnnType);
    armnn::TensorInfo outputInfo({batchSize, outputChannels, outputSize, 1}, ArmnnType);
    armnn::TensorInfo kernelInfo({outputChannels, inputChannels, kernelSize, 1}, ArmnnType);
    armnn::TensorInfo biasInfo({outputChannels}, ArmnnBType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputInfo.SetQuantizationScale(qScale);
        inputInfo.SetQuantizationOffset(qOffset);
        outputInfo.SetQuantizationScale(qScale);
        outputInfo.SetQuantizationOffset(qOffset);
        kernelInfo.SetQuantizationScale(qScale);
        kernelInfo.SetQuantizationOffset(qOffset);
        biasInfo.SetQuantizationScale(inputInfo.GetQuantizationScale()*kernelInfo.GetQuantizationScale());
        biasInfo.SetQuantizationOffset(0);
    }

    std::vector<T> inputData = QuantizedVector<T>(
        {
             5.0f, -2.0f, 2.5f, 0.0f, 1.0f,
            -3.0f,  3.2f, 5.0f, 2.0f, 3.0f,
        },
        inputInfo.GetQuantizationScale(),
        inputInfo.GetQuantizationOffset());

    std::vector<T> kernelData = QuantizedVector<T>(
        {
            1.0f,  0.0f,  0.0f,
            0.0f,  2.0f, -1.5f,

            0.0f,  0.0f,  0.0f,
            0.2f,  0.2f,  0.2f,

            0.5f,  0.0f,  0.5f,
            0.0f, -1.0f,  0.0f
        },
        kernelInfo.GetQuantizationScale(),
        kernelInfo.GetQuantizationOffset());

    std::vector<B> biasData =
        QuantizedVector<B>({ 1.0f, 0.0f, 0.0f }, biasInfo.GetQuantizationScale(), biasInfo.GetQuantizationOffset());

    std::vector<T> outputData = QuantizedVector<T>(
        {
             4.5f, -10.8f, 5.0f + 6.4f - 7.5f, -2.0f + 10.0f -3.0f, 2.5f + 4.0f - 4.5f, 6.0f, 1.0f,
            -0.6f, -0.6f + 0.64f, -0.6f + 0.64f + 1.0f, 0.64f + 1.0f + 0.4f, 1.0f + 0.4f + 0.6f, 0.4f + 0.6f, 0.6f,
             2.5f, -1.0f + 3.0f, 1.25f - 3.2f + 2.5f, -1.0f - 5.0f, 1.25f + 0.5f - 2.0f, -3.0f, 0.5f
        },
        outputInfo.GetQuantizationScale(),
        outputInfo.GetQuantizationOffset());

    std::vector<T> actualOutput(outputInfo.GetNumElements());

    // Optionally apply bias to output image.
    if(biasEnabled)
    {
        ApplyBias(outputData, outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset(),
            biasData, biasInfo.GetQuantizationScale(), biasInfo.GetQuantizationOffset(),
            1, outputSize);
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelInfo);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;

    armnn::Convolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;
//    armnn::ScopedTensorHandle weightsTensor(kernelInfo);
//    armnn::ScopedTensorHandle biasTensor(biasInfo);
//
//    AllocateAndCopyDataToITensorHandle(&weightsTensor, kernelData.data());
//    AllocateAndCopyDataToITensorHandle(&biasTensor, biasData.data());

    AddInputToWorkload(data, info, inputInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelInfo, weightsHandle.get());
    AddOutputToWorkload(data, info, outputInfo, outputHandle.get());

    data.m_Parameters.m_StrideX = 1;
    data.m_Parameters.m_StrideY = stride;
    data.m_Parameters.m_PadLeft = 0;
    data.m_Parameters.m_PadRight = 0;
    data.m_Parameters.m_PadTop = padSize;
    data.m_Parameters.m_PadBottom = padSize;
    data.m_Parameters.m_BiasEnabled = biasEnabled;

    if (biasEnabled)
    {
        biasHandle = tensorHandleFactory.CreateTensorHandle(biasInfo);
        AddInputToWorkload(data, info, biasInfo, biasHandle.get());
    }

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Convolution2d,
                                                                                data,
                                                                                info);
    inputHandle->Allocate();
    outputHandle->Allocate();
    weightsHandle->Allocate();

    if (biasEnabled)
    {
        biasHandle->Allocate();
        CopyDataToITensorHandle(biasHandle.get(), biasData.data());
    }

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());
    CopyDataToITensorHandle(weightsHandle.get(), kernelData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 outputData,
                                 outputHandle->GetShape(),
                                 outputInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x3NhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    armnn::DataLayout dataLayout)
{
    armnn::IgnoreUnused(biasEnabled);
    // Use common single-batch 5x5 image.

    armnn::TensorInfo inputDesc({ 1, 3, 4, 1 }, ArmnnType);
    std::vector<T> input =
    {
        1, 5, 2, 3,
        8, 7, 3, 6,
        3, 3, 9, 1
    };

    // Use a 2-element batch of 3-channel 3x3 kernels.
    armnn::TensorInfo kernelDesc({ 1, 3, 3, 1 }, ArmnnType);
    std::vector<T> kernel =
    {
        4, 5, 6,
        0, 0, 0,
        3, 2, 1
    };

    // Expected output is 1 batch of a 5x5 image.
    armnn::TensorInfo outputDesc({ 1, 3, 4, 1 }, ArmnnType);
    const std::vector<float> outputData =
    {
        23, 41, 33, 21,
        44, 65, 76, 52,
        82, 85, 79, 42
    };

    return SimpleConvolution2dNhwcTestImpl<ArmnnType, ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        std::vector<T>(),
        outputData,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        dataLayout,
        qScale,
        qOffset);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x3Stride2x2TestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset,
        bool biasEnabled,
        const armnn::DataLayout& dataLayout)
{
    armnn::IgnoreUnused(biasEnabled);

    // Input is a single-batch, 1 channel, 5x5 image.
    armnn::TensorInfo inputDesc({ 1, 5, 5, 1 }, ArmnnType);
    std::vector<T> input =
    {
        1, 5, 2, 3, 5,
        8, 7, 3, 6, 3,
        3, 3, 9, 1, 9,
        4, 1, 8, 1, 3,
        6, 8, 1, 9, 2
    };

    // Use a 3x3 kernel.
    armnn::TensorInfo kernelDesc({ 1, 3, 3, 1 }, ArmnnType);
    std::vector<T> kernel =
    {
        4, 5, 6,
        0, 0, 0,
        3, 2, 1
    };

    // Expected output is a single-batch, 1 channel, 3x3 image.
    armnn::TensorInfo outputDesc({ 1, 3, 3, 1 }, ArmnnType);
    std::vector<T> outputData =
    {
        23, 33, 24,
        91, 99, 48,
        26, 50, 19
    };

    uint32_t padLeft = 1;
    uint32_t padTop = 1;
    uint32_t padRight = 1;
    uint32_t padBottom = 1;
    uint32_t strideX  = 2;
    uint32_t strideY  = 2;

    return SimpleConvolution2dNhwcTestImpl<ArmnnType, ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        std::vector<T>(),
        outputData,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        dataLayout,
        qScale,
        qOffset,
        padLeft,
        padTop,
        padRight,
        padBottom,
        strideX,
        strideY);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x5TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use common single-batch 3-channel 16x8 image.
    armnn::TensorInfo inputDesc({ 1, 3, 8, 16 }, ArmnnType);
    std::vector<T> input = QuantizedVector<T>(ConvInput3x8x16, qScale, qOffset);

    // Use a 2-element batch with 3-channel 3x5 kernels.
    armnn::TensorInfo kernelDesc({ 2, 3, 5, 3 }, ArmnnType);
    std::vector<T> kernel = QuantizedVector<T>({
            1,  1, 1,
            1, -1, 1,
            1,  1, 1,
            1,  1, 1,
            1,  1, 1,

            0,  0, 0,
            0,  0, 0,
            0,  0, 0,
            0,  0, 0,
            0,  0, 0,

            2,  2, 2,
            2,  2, 2,
            2,  2, 2,
            2,  2, 2,
            2,  2, 2,


            0,  0, 0,
            0,  0, 0,
            0,  0, 0,
            0,  0, 0,
            0,  0, 0,

            1,  1, 1,
            1,  1, 1,
            1,  1, 1,
            1,  1, 1,
            1,  1, 1,

            0,  0, 0,
            0,  0, 0,
            0,  0, 0,
            0,  0, 0,
            0,  0, 0
        },
        qScale, qOffset);

    // Expected output is 2 batch elements of a 1-channel 14x4 image.
    armnn::TensorInfo outputDesc({ 1, 2, 4, 14 }, ArmnnType);
    std::vector<T> expectedOutput = QuantizedVector<T>({
            -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24,
            -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25,
            -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f,
            -23.5f, -23.5f, -23.5f,
            -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f, -23.5f,
            -23.5f, -23.5f, -23.5f,

            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        },
        qScale, qOffset);

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        qScale,
        qOffset,
        layout);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2d3x3TestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use a 3x3 kernel, which exercises ArmCompute's direct convolution path.

    // Use common single-batch 3-channel 16x8 image.
    armnn::TensorInfo inputDesc({ 1, 3, 8, 16 }, ArmnnType);
    std::vector<unsigned int> inputShape = { 1, 3, 8, 16 };
    std::vector<T> input = QuantizedVector<T>(ConvInput3x8x16, qScale, qOffset);

    // Use a 2-element batch of 3-channel 3x3 kernels.
    armnn::TensorInfo kernelDesc({ 2, 3, 3, 3 }, ArmnnType);
    std::vector<T> kernel = QuantizedVector<T>({
            1,  1, 1,
            1, -1, 1,
            1,  1, 1,

            0,  0, 0,
            0,  0, 0,
            0,  0, 0,

            2,  2, 2,
            2,  2, 2,
            2,  2, 2,


            0,  0, 0,
            0,  0, 0,
            0,  0, 0,

            1,  1, 1,
            1,  1, 1,
            1,  1, 1,

            0,  0, 0,
            0,  0, 0,
            0,  0, 0
        },
        qScale, qOffset);

    // Expected output is 1 batch of a 2-channel 14x6 image.
    armnn::TensorInfo outputDesc({ 1, 2, 6, 14 }, ArmnnType);
    std::vector<T> expectedOutput = QuantizedVector<T>({
            -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15,
            -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,
            -14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,-14.5f,

            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        },
        qScale, qOffset);

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        qScale,
        qOffset,
        layout);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout,
    float qScale,
    int32_t qOffset)
{
    // Use a single-batch 1-channel 3x3 image as input.
    armnn::TensorInfo inputDesc({ 1, 1, 3, 3 }, ArmnnType);
    std::vector<T> input =
        QuantizedVector<T>({
            11,21,31,
            12,22,32,
            13,23,33
        },
        qScale, qOffset);

    // Use 1 batch of a 1-channel 2x2 kernel.
    armnn::TensorInfo kernelDesc({ 1, 1, 2, 2 }, ArmnnType);
    std::vector<T> kernel =
        QuantizedVector<T>({
            -11,-21,
            -12,-22,
        },
        qScale, qOffset);

// Expected output is 1 batch of a 1-channel 6x8 image.
// Manually calculated like this:
//[-11*0 -21*0  -12*0 -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0 -12*0  -22*0 ..]
//[-11*0 -21*0  -12*0 -22*11 ; -11*0  -21*0  -12*11 -22*21 ; -11*0  -21*0  -12*21 -22*31 ; -11*0  -21*0 -12*31 -22*0 ..]
//[-11*0 -21*11 -12*0 -22*12 ; -11*11 -21*21 -12*12 -22*22 ; -11*21 -21*31 -12*22 -22*32 ; -11*31 -21*0 -12*32 -22*0 ..]
//[-11*0 -21*12 -12*0 -22*13 ; -11*12 -21*22 -12*13 -22*23 ; -11*22 -21*32 -12*23 -22*33 ; -11*32 -21*0 -12*33 -22*0 ..]
//[-11*0 -21*13 -12*0 -22*0  ; -11*13 -21*23 -12*0  -22*0  ; -11*23 -21*33 -12*0  -22*0  ; -11*33 -21*0 -12*0  -22*0 ..]
//[-11*0 -21*0  -12*0 -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0  -12*0  -22*0  ; -11*0  -21*0 -12*0  -22*0 ..]
//[..... .....  ..... .....  ; .....  .....  .....  .....  ; .....  .....  .....  .....  ; .....  ..... .....  ..... ..]
    armnn::TensorInfo outputDesc({ 1, 1, 8, 6 }, ArmnnType);
    std::vector<T> expectedOutput =
        QuantizedVector<T>({
               0,    0,      0,    0,    0,    0,
            -242,  -594,  -934, -372,    0,    0,
            -495, -1190, -1850, -725,    0,    0,
            -538, -1256, -1916, -748,    0,    0,
            -273, -626,  -946,  -363,    0,    0,
               0,    0,     0,     0,    0,    0,
               0,    0,     0,     0,    0,    0,
               0,    0,     0,     0,    0,    0
        },
        qScale, qOffset);

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        GetBias2<ArmnnBType>(false, qScale * qScale),
        expectedOutput,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        2,  // Padding top.
        3,  // Padding right.
        4); // Padding bottom.
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleConvolution2dAsymmetricPaddingTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout,
    float qScale,
    int32_t qOffset)
{
    // Use a single-batch 1-channel 5x5 image as input.
    armnn::TensorInfo inputDesc({ 1, 1, 5, 5 }, ArmnnType);
    std::vector<T> input =
        QuantizedVector<T>({
            11,21,31,41,51,
            12,22,32,42,52,
            13,23,33,43,53,
            14,24,34,44,54,
            15,25,35,45,55,
        }, qScale, qOffset);

    // Use 1 batch of a 1-channel 4x4 kernel.
    armnn::TensorInfo kernelDesc({ 1, 1, 4, 4 }, ArmnnType);
    std::vector<T> kernel =
        QuantizedVector<T>({
            -11,-21,-31,-41,
            -12,-22,-32,-42,
            -13,-23,-33,-43,
            -14,-24,-34,-44,
        },
        qScale, qOffset);

    // Expected output is 1 batch of a 1-channel 5x5 image.
    armnn::TensorInfo outputDesc({ 1, 1, 5, 5 }, ArmnnType);
    std::vector<T> expectedOutput =
        QuantizedVector<T>({
            -7140, -10580, -13940,  -9300, -5230,
            -9590, -14120, -18520, -12290, -6860,
            -9980, -14560, -18960, -12560, -7000,
            -7518, -10904, -14144,  -9318, -5152,
            -5032,  -7256,  -9376,  -6142, -3368,
        },
        qScale, qOffset);

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        GetBias2<ArmnnBType>(false, qScale * qScale),
        expectedOutput,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2); // Padding bottom.
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d3x3DilationTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const std::vector<float>& inputNoQuantizedValues,
    armnn::TensorInfo& inputTensorInfo,
    const std::vector<float>& kernelNoQuantizedValues,
    armnn::TensorInfo& kernelTensorInfo,
    const std::vector<float>& outputExpectedNoQuantizedValues,
    armnn::TensorInfo& outputTensorInfo,
    uint32_t dilationX,
    uint32_t dilationY,
    armnn::DataLayout layout = armnn::DataLayout::NCHW,
    uint32_t padLeft = 0,
    uint32_t padTop = 0,
    uint32_t padRight = 0,
    uint32_t padBottom = 0,
    uint32_t strideX  = 1,
    uint32_t strideY  = 1,
    bool biasEnabled = false
)
{
    float qScale;
    int32_t qOffset;
    switch (ArmnnType)
    {
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QAsymmS8:
        {
            qScale = 0.1f;
            qOffset = 128;
            break;
        }
        case armnn::DataType::QSymmS16:
        {
            qScale = 0.1f;
            qOffset = 0;
            break;
        }
        case armnn::DataType::Float32:
        default:
        {
            qScale = 0.f;
            qOffset = 0;
            break;
        }
    }

    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);
    kernelTensorInfo.SetQuantizationScale(qScale);
    kernelTensorInfo.SetQuantizationOffset(qOffset);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    auto input = QuantizedVector<T>(inputNoQuantizedValues,
                                    inputTensorInfo.GetQuantizationScale(),
                                    inputTensorInfo.GetQuantizationOffset());
    auto kernel = QuantizedVector<T>(kernelNoQuantizedValues,
                                     kernelTensorInfo.GetQuantizationScale(),
                                     kernelTensorInfo.GetQuantizationOffset());
    auto expectedOutput = QuantizedVector<T>(outputExpectedNoQuantizedValues,
                                             outputTensorInfo.GetQuantizationScale(),
                                             outputTensorInfo.GetQuantizationOffset());

    return SimpleConvolution2dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
            expectedOutput,
            inputTensorInfo.GetShape(),
            kernelTensorInfo.GetShape(),
            outputTensorInfo.GetShape(),
            qScale,
            qOffset,
            layout,
            padLeft,
            padTop,
            padRight,
            padBottom,
            strideX,
            strideY,
            dilationX,
            dilationY);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> Convolution2d3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({ 1, 1, 10, 10 }, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    armnn::TensorInfo kernelTensorInfo({ 1, 1, 3, 3}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
    {
        6., 5., 5., 5.,
        6., 5., 5., 5.,
        6., 5., 5., 5.,
        3., 2., 2., 2.
    };

    return Convolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> Convolution2d2x3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({ 1, 2, 10, 10 }, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    armnn::TensorInfo kernelTensorInfo({ 1, 2, 3, 3 }, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4 }, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
    {
        12., 10., 10., 10.,
        12., 10., 10., 10.,
        12., 10., 10., 10.,
         6.,  4.,  4.,  4.
    };

    return Convolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({ 1, 1, 10, 10 }, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    armnn::TensorInfo kernelTensorInfo({ 1, 1, 2, 2 }, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
    {
        1, 2,
        3, 4
    };

    // Since the dilation rate is 2 this will dilate the kernel to be like 3x3: d(K-1)+1 --> 2 x (2-1) + 1 = 3,
    // therefore the output will be 4x4: (I − K + 2P)/S +1 => trunc ( (10 - 3 + 2x2 ) / 3 + 1 )
    // where, dilation size = d = 2; kernel size = K = 2; input size = I = 10; padding size = P = 2; stride = S = 3
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
    {
        4,  7,  7, 3,
        6, 10, 10, 4,
        6, 10, 10, 4,
        2,  3,  3, 1
    };
    uint32_t padLeft = 1;
    uint32_t padTop = 1;
    uint32_t padRight = 1;
    uint32_t padBottom = 1;

    return Convolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            2,
            2,
            layout,
            padLeft,
            padTop,
            padRight,
            padBottom,
            3,
            3,
            biasEnabled
            );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T,4> CompareConvolution2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory)
{
    unsigned int inputHeight   = 8;
    unsigned int inputWidth    = 16;
    unsigned int inputChannels = 3;
    unsigned int inputNum      = 5;

    unsigned int kernelHeight = 3;
    unsigned int kernelWidth  = 3;

    unsigned int strideX = 2;
    unsigned int strideY = 3;
    unsigned int padX    = 1;
    unsigned int padY    = 1;

    unsigned int outputNum      = inputNum;
    unsigned int outputChannels = 2;
    unsigned int outputHeight   = (inputHeight + 2 * padY - kernelHeight + strideY) / strideY;
    unsigned int outputWidth    = (inputWidth + 2 * padX - kernelWidth + strideX) / strideX;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;
    armnn::TensorInfo kernelDesc;
    armnn::TensorInfo biasDesc;

    unsigned int inputShape[]  = {inputNum, inputChannels, inputHeight, inputWidth};
    unsigned int outputShape[] = {outputNum, outputChannels, outputHeight, outputWidth};
    unsigned int kernelShape[] = {outputChannels, inputChannels, kernelHeight, kernelWidth};
    unsigned int biasShape[]   = {outputChannels};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);
    kernelDesc = armnn::TensorInfo(4, kernelShape, ArmnnType);
    biasDesc = armnn::TensorInfo(1, biasShape, ArmnnType);

    auto input  = MakeRandomTensor<T>(inputTensorInfo, 124908);
    auto kernel = MakeRandomTensor<T>(kernelDesc, 891234);
    auto bias   = MakeRandomTensor<T>(biasDesc, 1028);

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Convolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelDesc, weightsHandle.get());
    AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    AllocateAndCopyDataToITensorHandle(weightsHandle.get(), kernel.data());
    AllocateAndCopyDataToITensorHandle(biasHandle.get(), bias.data());

    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padX;
    data.m_Parameters.m_PadRight = padX;
    data.m_Parameters.m_PadTop = padY;
    data.m_Parameters.m_PadBottom = padY;
    data.m_Parameters.m_BiasEnabled = true;

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandleRef = refTensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> biasHandleRef = refTensorHandleFactory.CreateTensorHandle(biasDesc);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo);

    armnn::Convolution2dQueueDescriptor refData = data;
    armnn::WorkloadInfo                 refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadInput(refData, refInfo, 1, kernelDesc, weightsHandleRef.get());
    SetWorkloadInput(refData, refInfo, 2, biasDesc, biasHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::Convolution2d, data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef
            = refWorkloadFactory.CreateWorkload(armnn::LayerType::Convolution2d, refData, refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();
    weightsHandleRef->Allocate();
    biasHandleRef->Allocate();

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());
    CopyDataToITensorHandle(inputHandleRef.get(), input.data());
    CopyDataToITensorHandle(weightsHandleRef.get(), kernel.data());
    CopyDataToITensorHandle(biasHandleRef.get(), bias.data());

    ExecuteWorkload(*workload, memoryManager);

    workloadRef->PostAllocationConfigure();
    workloadRef->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    CopyDataFromITensorHandle(expectedOutput.data(), outputHandleRef.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

LayerTestResult<float, 4> Convolution2d3x3Stride2x2BFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout& dataLayout)
{
    // BFloat16 input and weight, Float32 output
    armnn::IgnoreUnused(biasEnabled);

    // Input is a single-batch, 1 channel, 5x5 image.
    armnn::TensorInfo inputDesc({ 1, 5, 5, 1 }, armnn::DataType::BFloat16);

    std::vector<armnn::BFloat16> inputValues = armnnUtils::QuantizedVector<armnn::BFloat16>(
        {
            10.0367984f,  // 10.0625
             2.0380895f,  // 2.03125
            15.0420157f,  // 15.0625
            22.0675631f,  // 22.125
             8.0938920f,  // 8.125
             5.0476106f,  // 5.0625
            80.1035490f,  // 80
           100.1260370f,  // 100
            55.0461647f,  // 55
           120.0883828f,  // 120
             9.1159540f,  // 9.125
            90.0498519f,  // 90
           200.0104630f,  // 200
            30.0154114f,  // 30
            75.00137681f, // 75
            30.0344238f,  // 30
            25.0356445f,  // 25
           130.0495605f,  // 130
            60.0683594f,  // 60
            35.0991211f,  // 35
             8.0461426f,  // 8.0625
            12.0996094f,  // 12.125
            98.1269530f,  // 98
           125.0393066f,  // 125
             5.103516f    // 5.0937
       },
        1.0f, 0);

    // Use a 3x3 kernel.
    armnn::TensorInfo kernelDesc({1, 3, 3, 1}, armnn::DataType::BFloat16);

    std::vector<armnn::BFloat16> kernelValues = armnnUtils::QuantizedVector<armnn::BFloat16>(
        {
            -0.126184f, // -0.125977
            -0.150468f, // -0.150391
            -0.101412f, // -0.101562
            -0.0586369f,// -0.0585938
            -0.0865864f,// -0.0864258
            -0.0435089f,// -0.043457
            0.0347555f, // 0.034668
            0.0323111f, // 0.0322266
            0.0385381f  // 0.0385742
         },
        1.0f, 0);

    // Expected output is a single-batch, 1 channel, 3x3 image.
    armnn::TensorInfo outputDesc({ 1, 3, 3, 1 }, armnn::DataType::Float32);

    // Expected output (with results if calculated as FP32 in the comments)
    const std::vector<float> outputData =
        {
            2.296875f, //  2.29240716
            5.75f,     //  5.75851926
            3.78125f,  //  3.79855026
            -11.625f,  // -11.65498118
            -47.25f,   // -47.27316893
            -30.0f,    // -30.04771684
            -8.25f,    //  -8.28126168
            -43.5f,    // -43.46531337
            -20.625f   // -20.63477281
        };

    uint32_t padLeft = 1;
    uint32_t padTop = 1;
    uint32_t padRight = 1;
    uint32_t padBottom = 1;
    uint32_t strideX  = 2;
    uint32_t strideY  = 2;

    return SimpleConvolution2dNhwcTestImpl
        <armnn::DataType::BFloat16, armnn::DataType::Float32, armnn::BFloat16, float, armnn::DataType::Float32, float>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputValues,
        kernelValues,
        std::vector<float>(),
        outputData,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        dataLayout,
        1.0f,
        0,
        padLeft,
        padTop,
        padRight,
        padBottom,
        strideX,
        strideY);
}

LayerTestResult<float, 4> Convolution2d3x3Stride2x2BFloat16SmallValueTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout& dataLayout)
{
    // BFloat16 input and weight, Float32 output
    armnn::IgnoreUnused(biasEnabled);

    // Input is a single-batch, 1 channel, 5x5 image.
    armnn::TensorInfo inputDesc({1, 5, 5, 1}, armnn::DataType::BFloat16);

    std::vector<armnn::BFloat16> inputValues = armnnUtils::QuantizedVector<armnn::BFloat16>(
        {
            0.0367984f,  // 0.0368652
            0.0380895f,  // 0.0380859
            0.0420157f,  // 0.0419922
            0.0675631f,  // 0.0673828
            0.0938920f,  // 0.09375
            0.0476106f,  // 0.0476074
            0.1035490f,  // 0.103516
            0.1260370f,  // 0.125977
            0.0461647f,  // 0.0461426
            0.0883828f,  // 0.0883789
            0.1159540f,  // 0.115723
            0.0498519f,  // 0.0498047
            0.0104630f,  // 0.010437
            0.0154114f,  // 0.0154419
            0.00137681f, // 0.00137329
            0.0344238f,  // 0.0344616
            0.0356445f,  // 0.0355693
            0.0495605f,  // 0.0495018
            0.0683594f,  // 0.0683308
            0.0991211f,  // 0.0988837
            0.0461426f,  // 0.0461838
            0.0996094f,  // 0.0997546
            0.1269530f,  // 0.127099
            0.0393066f,  // 0.0392791
            0.103516f    // 0.103641
       },
        1.0f, 0);

    // Use a 3x3 kernel.
    armnn::TensorInfo kernelDesc({1, 3, 3, 1}, armnn::DataType::BFloat16);

    std::vector<armnn::BFloat16> kernelValues = armnnUtils::QuantizedVector<armnn::BFloat16>(
        {
            -0.126184f, // -0.125977
            -0.150468f, // -0.150391
            -0.101412f, // -0.101562
            -0.0586369f,// -0.0585938
            -0.0865864f,// -0.0864258
            -0.0435089f,// -0.043457
            0.0347555f, // 0.034668
            0.0323111f, // 0.0322266
            0.0385381f  // 0.0385742
         },
        1.0f, 0);

    // Expected output is a single-batch, 1 channel, 3x3 image.
    armnn::TensorInfo outputDesc({1, 3, 3, 1}, armnn::DataType::Float32);

    // Expected output (with results if calculated as FP32 in the comments)
    const std::vector<float> outputData =
        {
             0.000686645508f, // 0.000685
             0.000640869141f, // 0.000639
            -0.00759887695f,  // -0.007631
            -0.02734375f,     // -0.027388
            -0.0356445312f,   // -0.035737
            -0.0145874023f,   // -0.014568
            -0.0170898438f,   // -0.017124
            -0.0373535156f,   // -0.037431
            -0.0346679688f    // -0.034808
        };

    uint32_t padLeft = 1;
    uint32_t padTop = 1;
    uint32_t padRight = 1;
    uint32_t padBottom = 1;
    uint32_t strideX  = 2;
    uint32_t strideY  = 2;

    return SimpleConvolution2dNhwcTestImpl
        <armnn::DataType::BFloat16, armnn::DataType::Float32, armnn::BFloat16, float, armnn::DataType::Float32, float>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputValues,
        kernelValues,
        std::vector<float>(),
        outputData,
        inputDesc.GetShape(),
        kernelDesc.GetShape(),
        outputDesc.GetShape(),
        dataLayout,
        1.0f,
        0,
        padLeft,
        padTop,
        padRight,
        padBottom,
        strideX,
        strideY);
}

//
// DepthwiseConvolution2d implementations
//

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>, typename B = armnn::ResolveType<ArmnnBType>>
LayerTestResult<T, 4> DepthwiseConvolution2dAsymmetricTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const std::vector<T>& input,
    const std::vector<T>& kernel,
    const std::vector<B>& bias,
    const std::vector<T>& outputExpected,
    const armnn::TensorShape& inputShape,
    const armnn::TensorShape& kernelShape,
    const armnn::TensorShape& outputExpectedShape,
    float qScale,
    int32_t qOffset,
    const armnn::DataLayout layout,
    uint32_t padLeft = 0,
    uint32_t padTop = 0,
    uint32_t padRight = 0,
    uint32_t padBottom = 0,
    uint32_t strideX = 1,
    uint32_t strideY = 1)
{
    unsigned int inputNum       = armnn::numeric_cast<unsigned int>(inputShape[0]);
    unsigned int inputChannels  = armnn::numeric_cast<unsigned int>(inputShape[1]);
    unsigned int inputHeight    = armnn::numeric_cast<unsigned int>(inputShape[2]);
    unsigned int inputWidth     = armnn::numeric_cast<unsigned int>(inputShape[3]);
    unsigned int kernelHeight   = armnn::numeric_cast<unsigned int>(kernelShape[1]);
    unsigned int kernelWidth    = armnn::numeric_cast<unsigned int>(kernelShape[2]);
    unsigned int kernelChannels = armnn::numeric_cast<unsigned int>(kernelShape[3]);
    unsigned int outputNum      = armnn::numeric_cast<unsigned int>(outputExpectedShape[0]);
    unsigned int outputChannels = armnn::numeric_cast<unsigned int>(outputExpectedShape[1]);
    unsigned int outputHeight   = armnn::numeric_cast<unsigned int>(outputExpectedShape[2]);
    unsigned int outputWidth    = armnn::numeric_cast<unsigned int>(outputExpectedShape[3]);

    // If a bias is used, its size must equal the number of output channels.
    bool biasEnabled = bias.size() > 0;
    ARMNN_ASSERT(!biasEnabled || bias.size() == outputChannels);

    // Creates the tensors.
    armnn::TensorInfo inputTensorInfo =
            armnnUtils::GetTensorInfo(inputNum, inputChannels, inputHeight, inputWidth, layout, ArmnnType);
    armnn::TensorInfo outputTensorInfo =
            armnnUtils::GetTensorInfo(outputNum, outputChannels, outputHeight, outputWidth, layout, ArmnnType);
    armnn::TensorInfo kernelDesc({1, kernelHeight, kernelWidth, kernelChannels}, ArmnnType);
    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, ArmnnBType);

    // Set quantization parameters if the requested type is a quantized type.
    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
        kernelDesc.SetQuantizationScale(qScale);
        kernelDesc.SetQuantizationOffset(qOffset);
        biasDesc.SetQuantizationScale(qScale*qScale);
        biasDesc.SetQuantizationOffset(0);
    }

    // Construct the input data.
    std::vector<T> inputData;
    inputData.assign(input.data(), input.data() + inputChannels*inputHeight*inputWidth);

    // At this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(T));
        inputData = tmp;
    }

    std::vector<T> kernelData;
    kernelData.assign(kernel.data(), kernel.data() + kernelHeight * kernelWidth * outputChannels);
    if (workloadFactory.GetBackendId() == armnn::BackendId("GpuAcc") ||
        workloadFactory.GetBackendId() == armnn::BackendId("CpuAcc"))
    {
        if (layout == armnn::DataLayout::NCHW)
        {
            std::vector<T> tmp(kernelData.size());
            kernelDesc.SetShape(armnnUtils::Permuted(kernelDesc.GetShape(), {0, 2, 3, 1}));
            armnnUtils::Permute(kernelDesc.GetShape(), {0, 2, 3, 1}, kernelData.data(), tmp.data(), sizeof(T));
            kernelData = tmp;
        }
    }

    // Construct the output data, with bias applied, as appropriate.
    std::vector<T> outputData;
    outputData.assign(outputExpected.data(), outputExpected.data() + outputChannels*outputHeight*outputWidth);
    if (biasEnabled)
    {
        std::vector<T> biasV;
        biasV.assign(bias.data(), bias.data() + outputChannels);
        ApplyBias(outputData, outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
            biasV, biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
            outputWidth, outputHeight);
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    // At this point if we require it permute the expected output
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp.data(), sizeof(T));
        outputData = tmp;
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;

    AllocateAndCopyDataToITensorHandle(weightsHandle.get(), kernelData.data()); // required for ConstantTensor

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelDesc, weightsHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    armnn::ScopedTensorHandle biasTensor(biasDesc);
    if (biasEnabled)
    {
        AllocateAndCopyDataToITensorHandle(&biasTensor, bias.data());

        biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
        AllocateAndCopyDataToITensorHandle(biasHandle.get(), bias.data());
        AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    }

    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = layout;

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::DepthwiseConvolution2d, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 outputData,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dDepthMul1TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    using B = armnn::ResolveType<ArmnnBType>;

    unsigned int inputHeight = 3;
    unsigned int inputWidth = 3;
    unsigned int inputChannels = 2;
    unsigned int inputNum = 1;

    unsigned int kernelHeight = 3;
    unsigned int kernelWidth = 3;

    unsigned int outputHeight = 1;
    unsigned int outputWidth = 1;
    unsigned int outputChannels = inputChannels;
    unsigned int outputNum = inputNum;

    armnn::TensorInfo inputTensorInfo =
            armnnUtils::GetTensorInfo(inputNum, inputChannels, inputHeight, inputWidth, layout, ArmnnType);
    armnn::TensorInfo outputTensorInfo =
            armnnUtils::GetTensorInfo(outputNum, outputChannels, outputHeight, outputWidth, layout, ArmnnType);
    armnn::TensorInfo kernelDesc({1, kernelHeight, kernelWidth, outputChannels},
                                 ArmnnType);
    armnn::TensorInfo biasDesc({ outputChannels }, ArmnnBType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
        kernelDesc.SetQuantizationScale(qScale);
        kernelDesc.SetQuantizationOffset(qOffset);
        biasDesc.SetQuantizationScale(qScale*qScale);
        biasDesc.SetQuantizationOffset(0);
    }
    std::vector<T> inputData = std::vector<T>(
            QuantizedVector<T>({
                1.f, 2.f, 1.f,
                2.f, 1.f, 2.f,
                1.f, 2.f, 1.f,

                1.f, 2.f, 1.f,
                2.f, 1.f, 2.f,
                1.f, 2.f, 1.f,
            },
            inputTensorInfo.GetQuantizationScale(),
            inputTensorInfo.GetQuantizationOffset()));

    // at this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(T));
        inputData = tmp;
    }

    std::vector<B> biasV(QuantizedVector<B>({ 0, 2 },
                                            biasDesc.GetQuantizationScale(),
                                            biasDesc.GetQuantizationOffset()));

    std::vector<T> kernelData = std::vector<T>(
            QuantizedVector<T>({
                 1.f, 0.f,  1.f,
                 0.f, 0.f,  0.f,
                -1.f, 0.f, -1.f,

                 1.f, 0.f,  1.f,
                 0.f, 0.f,  0.f,
                -1.f, 0.f, -1.f,
            },
            kernelDesc.GetQuantizationScale(),
            kernelDesc.GetQuantizationOffset()));

    if (workloadFactory.GetBackendId() == armnn::BackendId("GpuAcc") ||
        workloadFactory.GetBackendId() == armnn::BackendId("CpuAcc"))
    {
        if (layout == armnn::DataLayout::NCHW)
        {
            std::vector<T> tmp(kernelData.size());
            kernelDesc.SetShape(armnnUtils::Permuted(kernelDesc.GetShape(), {0, 2, 3, 1}));
            armnnUtils::Permute(kernelDesc.GetShape(), {0, 2, 3, 1}, kernelData.data(), tmp.data(), sizeof(T));
            kernelData = tmp;
        }
    }

    // Manually calculated.
    std::vector<T> outputImage(
        QuantizedVector<T>({ 0.f, 0.f },
                           outputTensorInfo.GetQuantizationScale(),
                           outputTensorInfo.GetQuantizationOffset())
    );

    // Optionally apply bias to output image.
    if(biasEnabled)
    {
        ApplyBias(outputImage, outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
                  biasV, biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
                  outputWidth, outputHeight);
    }

    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(outputImage.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputImage.data(), tmp.data(), sizeof(T));
        outputImage = tmp;
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;

    AllocateAndCopyDataToITensorHandle(weightsHandle.get(), kernelData.data()); // required for ConstantTensor

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelDesc, weightsHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    armnn::ScopedTensorHandle biasTensor(biasDesc);
    if (biasEnabled)
    {
        AllocateAndCopyDataToITensorHandle(&biasTensor, biasV.data());

        biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
        AllocateAndCopyDataToITensorHandle(biasHandle.get(), biasV.data());
        AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    }

    data.m_Parameters.m_StrideX = 1;
    data.m_Parameters.m_StrideY = 1;
    data.m_Parameters.m_PadLeft = 0;
    data.m_Parameters.m_PadRight = 0;
    data.m_Parameters.m_PadTop = 0;
    data.m_Parameters.m_PadBottom = 0;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = layout;

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::DepthwiseConvolution2d, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 outputImage,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    using B = armnn::ResolveType<ArmnnBType>;

    unsigned int depthMultiplier = 2;

    unsigned int inputHeight    = 8;
    unsigned int inputWidth     = 16;
    unsigned int inputChannels  = 2;
    unsigned int inputBatchSize = 1;

    unsigned int kernelHeight = 5;
    unsigned int kernelWidth  = 3;

    unsigned int outputHeight    = inputHeight - kernelHeight + 1 + 2;
    unsigned int outputWidth     = (inputWidth - kernelWidth + 1)/2;
    unsigned int outputChannels  = inputChannels * depthMultiplier;
    unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo = armnnUtils::GetTensorInfo(
            inputBatchSize, inputChannels, inputHeight, inputWidth, layout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnnUtils::GetTensorInfo(
            outputBatchSize, outputChannels, outputHeight, outputWidth, layout, ArmnnType);
    armnn::TensorInfo kernelDesc({1, kernelHeight, kernelWidth, outputChannels},
                                 ArmnnType);
    armnn::TensorInfo biasDesc({outputChannels}, ArmnnBType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
        kernelDesc.SetQuantizationScale(qScale);
        kernelDesc.SetQuantizationOffset(qOffset);
        biasDesc.SetQuantizationScale(qScale*qScale);
        biasDesc.SetQuantizationOffset(0);
    }

    // NOTE: originalInputData is in NCHW format
    std::vector<T> originalInputData = std::vector<T>(
            QuantizedVector<T>({
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
            },
            inputTensorInfo.GetQuantizationScale(),
            inputTensorInfo.GetQuantizationOffset()));

    std::vector<T> inputData = originalInputData;
    // at this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (layout == armnn::DataLayout::NHWC)
    {
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC,
                            originalInputData.data(), inputData.data(), sizeof(T));
    }

    std::vector<B> biasV = QuantizedVector<B>({ 0, 2, 1, -1 },
                                              biasDesc.GetQuantizationScale(),
                                              biasDesc.GetQuantizationOffset());

    std::vector<T> kernelData = std::vector<T>(
            QuantizedVector<T>({
                1,  1, 1,
                1, -1, 1,
                1,  1, 1,
                1,  1, 1,
                1,  1, 1,

                2,  2, 2,
                2,  2, 2,
                2,  2, 2,
                2,  2, 2,
                2,  2, 2,

                0,  0, 0,
                0, -1, 0,
                0,  0, 0,
                0,  0, 0,
                0,  0, 0,

                0,  0, 0,
                0,  0, 0,
                0,  1, 0,
                0,  0, 0,
                0,  0, 0
            },
            kernelDesc.GetQuantizationScale(),
            kernelDesc.GetQuantizationOffset()));

    if (workloadFactory.GetBackendId() == armnn::BackendId("GpuAcc") ||
        workloadFactory.GetBackendId() == armnn::BackendId("CpuAcc"))
    {
        if (layout == armnn::DataLayout::NCHW)
        {
            std::vector<T> tmp(kernelData.size());
            kernelDesc.SetShape(armnnUtils::Permuted(kernelDesc.GetShape(), {0, 2, 3, 1}));
            armnnUtils::Permute(kernelDesc.GetShape(), {0, 2, 3, 1}, kernelData.data(), tmp.data(), sizeof(T));
            kernelData = tmp;
        }
    }

    // Manually calculated.
    std::vector<T> originalOutputImage = std::vector<T>(
        QuantizedVector<T>({
               3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
               5,   5,   5,   5,   5,   5,   5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5,
             5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5,   5,   5,   5,   5,   5,   5,   5,
             2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,
             4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,   6,   6,   6,   6,   6,   6,   6,
               6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
               1,   3,   0,   0,   0,   0,   0,   2,   4,   0,   0,   0,   0,   0,
               2,   4,   0,   0,   0,   0,   0,   2,   4,   0,   0,   0,   0,   0,
               2,   4,   0,   0,   0,   0,   0,   2,   4,   0,   0,   0,   0,   0,
               2,   4,   0,   0,   0,   0,   0,   3,   5,   0,   0,   0,   0,   0,
               3,   5,   0,   0,   0,   0,   0,   3,   5,   0,   0,   0,   0,   0,
               3,   5,   0,   0,   0,   0,   0,   3,   5,   0,   0,   0,   0,   0
        },
        outputTensorInfo.GetQuantizationScale(),
        outputTensorInfo.GetQuantizationOffset()));

    // Optionally apply bias to output image.
    if(biasEnabled)
    {
        ApplyBias(originalOutputImage,
                  outputTensorInfo.GetQuantizationScale(),
                  outputTensorInfo.GetQuantizationOffset(),
                  biasV,
                  biasDesc.GetQuantizationScale(),
                  biasDesc.GetQuantizationOffset(),
                  outputWidth,
                  outputHeight);
    }

    std::vector<T> outputImage = originalOutputImage;
    if (layout == armnn::DataLayout::NHWC)
    {
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC,
                            originalOutputImage.data(), outputImage.data(), sizeof(T));
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;

    AllocateAndCopyDataToITensorHandle(weightsHandle.get(), kernelData.data()); // required for ConstantTensor

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelDesc, weightsHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    armnn::ScopedTensorHandle biasTensor(biasDesc);
    if (biasEnabled)
    {
        AllocateAndCopyDataToITensorHandle(&biasTensor, biasV.data());

        biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
        AllocateAndCopyDataToITensorHandle(biasHandle.get(), biasV.data());
        AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    }

    data.m_Parameters.m_StrideX = 2;
    data.m_Parameters.m_StrideY = 1;
    data.m_Parameters.m_PadLeft = 0;
    data.m_Parameters.m_PadRight = 0;
    data.m_Parameters.m_PadTop = 1;
    data.m_Parameters.m_PadBottom = 1;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = layout;

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::DepthwiseConvolution2d, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 outputImage,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());

}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
        typename T = armnn::ResolveType<ArmnnType>, typename B = armnn::ResolveType<ArmnnBType>>
LayerTestResult<T, 4> DepthwiseConvolution2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const std::vector<T>& originalInput,
    const std::vector<T>& originalKernel,
    const std::vector<B>& bias,
    const std::vector<T>& originalOutputExpected,
    const armnn::TensorShape& originalInputShape,
    const armnn::TensorShape& originalKernelShape,
    const armnn::TensorShape& originalOutputExpectedShape,
    float qScale,
    int32_t qOffset,
    const armnn::DataLayout layout = armnn::DataLayout::NCHW,
    uint32_t padLeft = 0,
    uint32_t padTop = 0,
    uint32_t padRight = 0,
    uint32_t padBottom = 0,
    uint32_t strideX = 1,
    uint32_t strideY = 1,
    uint32_t dilationX = 1,
    uint32_t dilationY = 1)
{
    unsigned int inputHeight    = armnn::numeric_cast<unsigned int>(originalInputShape[2]);
    unsigned int inputWidth     = armnn::numeric_cast<unsigned int>(originalInputShape[3]);
    unsigned int inputChannels  = armnn::numeric_cast<unsigned int>(originalInputShape[1]);
    unsigned int inputNum       = armnn::numeric_cast<unsigned int>(originalInputShape[0]);

    unsigned int outputHeight   = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[2]);
    unsigned int outputWidth    = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[3]);
    unsigned int outputChannels = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[1]);
    unsigned int outputNum      = armnn::numeric_cast<unsigned int>(originalOutputExpectedShape[0]);

    unsigned int kernelHeight   = armnn::numeric_cast<unsigned int>(originalKernelShape[1]);
    unsigned int kernelWidth    = armnn::numeric_cast<unsigned int>(originalKernelShape[2]);
    unsigned int kernelChannels = armnn::numeric_cast<unsigned int>(originalKernelShape[3]);

    bool biasEnabled = bias.size() > 0;

    // This function currently assumes 1 batch of input/output (and duplicates this into 2 batches).
    ARMNN_ASSERT(inputNum == 1);
    ARMNN_ASSERT(outputNum == 1);

    // If a bias is used, its size must equal the number of output channels.
    ARMNN_ASSERT(!biasEnabled || bias.size() == outputChannels);


    // Note these tensors will use two (identical) batches.
    armnn::TensorInfo inputTensorInfo =
            armnnUtils::GetTensorInfo(2*inputNum, inputChannels, inputHeight, inputWidth, layout, ArmnnType);
    armnn::TensorInfo outputTensorInfo =
            armnnUtils::GetTensorInfo(2*outputNum, outputChannels, outputHeight, outputWidth, layout, ArmnnType);

    // Kernel must be NCHW layout always, independently of the layout of the input and output for depthwise convolution.
    armnn::TensorInfo kernelDesc({1, kernelHeight, kernelWidth, kernelChannels}, ArmnnType);

    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, ArmnnBType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
        kernelDesc.SetQuantizationScale(qScale);
        kernelDesc.SetQuantizationOffset(qOffset);
        biasDesc.SetQuantizationScale(qScale*qScale);
        biasDesc.SetQuantizationOffset(0);
    }

    std::vector<T> kernelData;
    kernelData.assign(originalKernel.data(), originalKernel.data() + kernelHeight*kernelWidth*outputChannels);
    if (workloadFactory.GetBackendId() == armnn::BackendId("GpuAcc") ||
        workloadFactory.GetBackendId() == armnn::BackendId("CpuAcc"))
    {
        if (layout == armnn::DataLayout::NCHW)
        {
            std::vector<T> tmp(kernelData.size());
            kernelDesc.SetShape(armnnUtils::Permuted(kernelDesc.GetShape(), {0, 2, 3, 1}));
            armnnUtils::Permute(kernelDesc.GetShape(), {0, 2, 3, 1}, kernelData.data(), tmp.data(), sizeof(T));
            kernelData = tmp;
        }
    }

    // Construct input data
    std::vector<T> input;
    input.assign(originalInput.data(), originalInput.data() + 1*inputChannels*inputHeight*inputWidth);
    std::vector<T> inputData;
    inputData.insert(inputData.end(), input.begin(), input.end());
    inputData.insert(inputData.end(), input.begin(), input.end());

    // at this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(T));
        inputData = tmp;
    }

    std::vector<T> output;
    output.assign(originalOutputExpected.data(),
                       originalOutputExpected.data() + outputChannels*outputHeight*outputWidth);

    // Apply bias to output data if it is enabled.
    if(biasEnabled)
    {
        std::vector<T> biasV;
        biasV.assign(bias.data(), bias.data() + outputChannels);
        ApplyBias(output, outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
                  biasV, biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
                  outputWidth, outputHeight);
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    // Construct expected output data
    std::vector<T> outputData;
    outputData.insert(outputData.end(), output.begin(), output.end());
    outputData.insert(outputData.end(), output.begin(), output.end());

    // at this point if we require it permute the expected output
    if (layout == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp.data(), sizeof(T));
        outputData = tmp;
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;

    AllocateAndCopyDataToITensorHandle(weightsHandle.get(), kernelData.data()); // required for ConstantTensor

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, kernelDesc, weightsHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    armnn::ScopedTensorHandle biasTensor(biasDesc);
    if (biasEnabled)
    {
        AllocateAndCopyDataToITensorHandle(&biasTensor, bias.data());

        biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
        AllocateAndCopyDataToITensorHandle(biasHandle.get(), bias.data());
        AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    }

    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = layout;
    data.m_Parameters.m_DilationX = dilationX;
    data.m_Parameters.m_DilationY = dilationY;

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::DepthwiseConvolution2d, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 outputData,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dAsymmetricTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    // Use a single-batch 2-channel 5x5 image as input.
    armnn::TensorInfo inputTensorInfo({ 1, 2, 5, 5 }, ArmnnType);
    auto input = QuantizedVector<T>(
         {
             0,  1,  2,  3,  4,
             5,  6,  7,  8,  9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,

            25, 26, 27, 28, 29,
            30, 31, 32, 33, 34,
            35, 36, 37, 38, 39,
            40, 41, 42, 43, 44,
            45, 46, 47, 48, 49
        },
        inputTensorInfo.GetQuantizationScale(),
        inputTensorInfo.GetQuantizationOffset());

    // Use a depth multiplier of 1 on a 2-channel 4x4 kernel.
    // Weights layout for depthwise: [1,H,W,I*M]
    armnn::TensorInfo kernelTensorInfo({ 1, 4, 4, 2 }, ArmnnType);
    auto kernel = QuantizedVector<T>({
            32, 31, 30, 29,
            28, 27, 26, 25,
            24, 23, 22, 21,
            20, 19, 18, 17,

            16, 15, 14, 13,
            12, 11, 10,  9,
             8,  7,  6,  5,
             4,  3,  2,  1
        },
        kernelTensorInfo.GetQuantizationScale(),
        kernelTensorInfo.GetQuantizationOffset());

    // Expected output is 1 batch of a 2-channel 5x5 image.
    // Calculated using the python tensorflow library with strideX=1, strideY=1.
    armnn::TensorInfo outputTensorInfo({ 1, 2, 5, 5 }, ArmnnType);
    auto expectedOutput = QuantizedVector<T>(
         {
             396, 664, 820, 756, 602, 1016, 1608, 1880, 1652, 1268, 1976, 2968, 3240, 2732,
             2028, 2628, 3808, 4060, 3312, 2390, 2596, 3700, 3900, 3130, 2226, 2817, 4186,
             4330, 3609, 2651, 5414, 7864, 8120, 6626, 4780, 6314, 9144, 9400, 7646, 5500,
             6759, 9610, 9850, 7875, 5579, 5935, 8348, 8540, 6757, 4742
        },
        outputTensorInfo.GetQuantizationScale(),
        outputTensorInfo.GetQuantizationOffset());

    return DepthwiseConvolution2dAsymmetricTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        inputTensorInfo.GetShape(),
        kernelTensorInfo.GetShape(),
        outputTensorInfo.GetShape(),
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2,  // Padding bottom.
        1,  // strideX
        1); // strideY
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dNhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled)
{
    auto layout = armnn::DataLayout::NHWC;

    armnn::TensorInfo inputTensorInfo({ 1, 2, 5, 5}, ArmnnType);
    auto input = QuantizedVector<T>(
         {
             0,  1,  2,  3,  4,
             5,  6,  7,  8,  9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,

            25, 26, 27, 28, 29,
            30, 31, 32, 33, 34,
            35, 36, 37, 38, 39,
            40, 41, 42, 43, 44,
            45, 46, 47, 48, 49
        },
        inputTensorInfo.GetQuantizationScale(),
        inputTensorInfo.GetQuantizationOffset());

    armnn::TensorInfo kernelTensorInfo({ 1, 4, 4, 2 }, ArmnnType);
    auto kernel = QuantizedVector<T>({
             32, 31, 30, 29,
             28, 27, 26, 25,
             24, 23, 22, 21,
             20, 19, 18, 17,

             16, 15, 14, 13,
             12, 11, 10,  9,
              8,  7,  6,  5,
              4,  3,  2,  1
        },
        kernelTensorInfo.GetQuantizationScale(),
        kernelTensorInfo.GetQuantizationOffset());

    armnn::TensorInfo outputTensorInfo({ 1, 2, 5, 5}, ArmnnType);
    auto expectedOutput = QuantizedVector<T>(
         {
             396,664,820,756,602,
             1016,1608,1880,1652,1268,
             1976,2968,3240,2732,2028,
             2628,3808,4060,3312,2390,
             2596,3700,3900,3130,2226,

             2817,4186,4330,3609,2651,
             5414,7864,8120,6626,4780,
             6314,9144,9400,7646,5500,
             6759,9610,9850,7875,5579,
             5935,8348,8540,6757,4742
        },
        outputTensorInfo.GetQuantizationScale(),
        outputTensorInfo.GetQuantizationOffset());

    return DepthwiseConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        inputTensorInfo.GetShape(),
        kernelTensorInfo.GetShape(),
        outputTensorInfo.GetShape(),
        qScale,
        qOffset,
        layout,
        1,  // Padding left.
        1,  // Padding top.
        2,  // Padding right.
        2,  // Padding bottom.
        1,  // strideX
        1);  // strideY
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    bool biasEnabled)
{
    auto layout = armnn::DataLayout::NHWC;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 9, 9 }, ArmnnType);
    auto input = QuantizedVector<T>(
         {
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        },
        inputTensorInfo.GetQuantizationScale(),
        inputTensorInfo.GetQuantizationOffset());

    armnn::TensorInfo kernelTensorInfo({ 1, 3, 3, 1}, ArmnnType);
    auto kernel = QuantizedVector<T>({
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        },
        kernelTensorInfo.GetQuantizationScale(),
        kernelTensorInfo.GetQuantizationOffset());

    uint32_t padLeft = 0;
    uint32_t padTop = 0;
    uint32_t padRight = 0;
    uint32_t padBottom = 0;
    uint32_t strideX  = 1;
    uint32_t strideY  = 1;
    uint32_t dilationX  = 3;
    uint32_t dilationY  = 3;

    // Since the dilation rate is 3 this will reduce the size of the output from 9x9 to 3x3 of all 5s.
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, ArmnnType);
    auto expectedOutput = QuantizedVector<T>(
        {
            5, 5, 5,
            5, 5, 5,
            5, 5, 5
        },
        outputTensorInfo.GetQuantizationScale(),
        outputTensorInfo.GetQuantizationOffset());

    return DepthwiseConvolution2dTestImpl<ArmnnType, ArmnnBType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        input,
        kernel,
        GetBias2<ArmnnBType>(biasEnabled, qScale * qScale),
        expectedOutput,
        inputTensorInfo.GetShape(),
        kernelTensorInfo.GetShape(),
        outputTensorInfo.GetShape(),
        qScale,
        qOffset,
        layout,
        padLeft,
        padTop,
        padRight,
        padBottom,
        strideX,
        strideY,
        dilationX,
        dilationY);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2d3x3DilationTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const std::vector<float>& inputNoQuantizedValues,
        armnn::TensorInfo& inputTensorInfo,
        const std::vector<float>& kernelNoQuantizedValues,
        armnn::TensorInfo& kernelTensorInfo,
        const std::vector<float>& outputExpectedNoQuantizedValues,
        armnn::TensorInfo& outputTensorInfo,
        uint32_t dilationX,
        uint32_t dilationY,
        armnn::DataLayout layout = armnn::DataLayout::NCHW,
        bool biasEnabled = false)
{
    float qScale;
    int32_t qOffset;
    switch (ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        {
            qScale = 0.1f;
            qOffset = 128;
            break;
        }
        case armnn::DataType::QSymmS16:
        {
            qScale = 0.1f;
            qOffset = 0;
            break;
        }
        case armnn::DataType::Float32:
        default:
        {
            qScale = 0.f;
            qOffset = 0;
            break;
        }
    }

    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);
    kernelTensorInfo.SetQuantizationScale(qScale);
    kernelTensorInfo.SetQuantizationOffset(qOffset);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    auto input = QuantizedVector<T>(inputNoQuantizedValues,
                                    inputTensorInfo.GetQuantizationScale(),
                                    inputTensorInfo.GetQuantizationOffset());
    auto kernel = QuantizedVector<T>(kernelNoQuantizedValues,
                                     kernelTensorInfo.GetQuantizationScale(),
                                     kernelTensorInfo.GetQuantizationOffset());
    auto expectedOutput = QuantizedVector<T>(outputExpectedNoQuantizedValues,
                                             outputTensorInfo.GetQuantizationScale(),
                                             outputTensorInfo.GetQuantizationOffset());

    uint32_t padLeft = 0;
    uint32_t padTop = 0;
    uint32_t padRight = 0;
    uint32_t padBottom = 0;
    uint32_t strideX  = 1;
    uint32_t strideY  = 1;

    return DepthwiseConvolution2dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBias<ArmnnBType>(biasEnabled, qScale * qScale, outputTensorInfo, layout),
            expectedOutput,
            inputTensorInfo.GetShape(),
            kernelTensorInfo.GetShape(),
            outputTensorInfo.GetShape(),
            qScale,
            qOffset,
            layout,
            padLeft,
            padTop,
            padRight,
            padBottom,
            strideX,
            strideY,
            dilationX,
            dilationY);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2d3x3Dilation3x3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 1, 10, 10}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };

    armnn::TensorInfo kernelTensorInfo({ 1, 3, 3, 1}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
            {
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9
            };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 1, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    6., 5., 5., 5.,
                    3., 2., 2., 2.
            };

    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2d2x3x3Dilation3x3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 2, 10, 10}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };

    armnn::TensorInfo kernelTensorInfo({ 1, 3, 3, 2}, ArmnnType);
    std::vector<float> kernelNoQuantizedValues =
            {
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,

                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9
            };

    // Since the dilation rate is 3 this will dilate the kernel to be like 7x7,
    // therefore the output will be 2x4x4: (I−K+2P)/S +1 => (10-7 +0)/1 +1
    armnn::TensorInfo outputTensorInfo({ 1, 2, 4, 4}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                    2, 9, 9, 9, 2, 9, 9, 9, 2, 9, 9, 9, 5, 3, 3, 3, 3,

                    1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 6, 4, 4, 4
            };

    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            3,
            3,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2dMult4Test(
            armnn::IWorkloadFactory& workloadFactory,
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
            const armnn::ITensorHandleFactory& tensorHandleFactory,
            bool biasEnabled,
            const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 2, 3, 3}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,

                    21.0, 22.0, 23.0,
                    24.0, 25.0, 26.0,
                    27.0, 28.0, 29.0
            };

    armnn::TensorInfo kernelTensorInfo({ 1, 2, 2, 8}, ArmnnType);

    std::vector<float> kernelNoQuantizedValues =
            {
                    0.25f, 0.25f,
                    0.25f, 0.25f,

                    0.25f, 0.25f,
                    0.25f, 0.25f,

                    0.0f , 0.0f,
                    0.0f , 0.1f,

                    0.0f , 0.0f,
                    0.0f , 0.1f,

                    0.2f , 0.0f,
                    0.0f , 0.0f,

                    0.2f , 0.0f,
                    0.0f , 0.0f,

                    0.0f , 0.3f,
                    0.0f , 0.0f,

                    0.0f , 0.3f,
                    0.0f , 0.0f
            };

    armnn::TensorInfo outputTensorInfo({ 1, 8, 2, 2}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                      4.5f,  4.5f,  4.5f,   4.5f,   5.5f,  5.5f,  5.5f,   5.5f,
                      2.5f,  2.5f,  2.5f,   2.5f,   3.5f,  3.5f,  3.5f,   3.5f,
                    10.05f, 10.5f, 11.4f, 11.85f, 12.75f, 13.3f, 14.4f, 14.95f,
                     5.25f,  5.5f,  6.0f,  6.25f,  7.45f,  7.8f,  8.5f,  8.85f
            };


    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            1,
            1,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> DepthwiseConvolution2dMult2Test(
            armnn::IWorkloadFactory& workloadFactory,
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
            const armnn::ITensorHandleFactory& tensorHandleFactory,
            bool biasEnabled,
            const armnn::DataLayout layout)
{
    armnn::TensorInfo inputTensorInfo({1, 2, 3, 3}, ArmnnType);
    std::vector<float> inputNoQuantizedValues =
            {
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,
                    10.0, 10.0, 10.0,

                    21.0, 22.0, 23.0,
                    24.0, 25.0, 26.0,
                    27.0, 28.0, 29.0
            };

    armnn::TensorInfo kernelTensorInfo({ 1, 2, 2, 4}, ArmnnType);

    std::vector<float> kernelNoQuantizedValues =
            {
                    0.25f, 0.25f,
                    0.25f, 0.25f,

                    0.2f , 0.0f,
                    0.0f , 0.0f,

                    0.0f , 0.0f,
                    0.0f , 0.1f,

                    0.0f , 0.3f,
                    0.0f , 0.0f

            };

    armnn::TensorInfo outputTensorInfo({ 1, 4, 2, 2}, ArmnnType);
    std::vector<float> outputExpectedNoQuantizedValues =
            {
                     4.5f, 4.5f, 4.5f,  4.5f,
                     5.5f, 5.5f, 5.5f,  5.5f,
                    5.25f, 5.5f, 6.0f, 6.25f,
                    7.65f, 8.0f, 8.7f, 9.05f
            };


    return DepthwiseConvolution2d3x3DilationTestCommon<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputNoQuantizedValues,
            inputTensorInfo,
            kernelNoQuantizedValues,
            kernelTensorInfo,
            outputExpectedNoQuantizedValues,
            outputTensorInfo,
            1,
            1,
            layout,
            biasEnabled);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> CompareDepthwiseConvolution2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    const armnnUtils::DataLayoutIndexed& layout)
{
    unsigned int inputHeight = 8;
    unsigned int inputWidth = 16;
    unsigned int inputChannels = 3;
    unsigned int inputNum = 5;

    unsigned int kernelHeight = 3;
    unsigned int kernelWidth = 3;
    unsigned int channelMultiplier = 1;

    unsigned int strideX = 2;
    unsigned int strideY = 3;
    unsigned int padX = 1;
    unsigned int padY = 1;

    unsigned int outputNum = inputNum;
    unsigned int outputChannels = inputChannels * channelMultiplier;
    unsigned int outputHeight = (inputHeight + 2 * padY - kernelHeight + strideY) / strideY;
    unsigned int outputWidth = (inputWidth + 2 * padX - kernelWidth + strideX) / strideX;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;
    armnn::TensorInfo kernelDesc;
    armnn::TensorInfo biasDesc;

    std::vector<unsigned int> inputShape;
    std::vector<unsigned int> outputShape;
    std::vector<unsigned int> kernelShape{ 1, kernelHeight, kernelWidth, outputChannels };
    std::vector<unsigned int> biasShape{ outputChannels };
    switch (layout.GetDataLayout())
    {
        case armnn::DataLayout::NCHW:
            inputShape =  { inputNum, inputChannels, inputHeight, inputWidth };
            outputShape = { outputNum, outputChannels, outputHeight, outputWidth };
            break;
        case armnn::DataLayout ::NHWC:
            inputShape =  { inputNum, inputHeight, inputWidth, inputChannels };
            outputShape = { outputNum, outputHeight, outputWidth, outputChannels };
            break;
        default:
            throw armnn::InvalidArgumentException("unknown data layout ["
                                                  + std::to_string(static_cast<int>(layout.GetDataLayout())) + "]");
    }

    float inputsQScale = armnn::IsQuantizedType<T>() ? 1.0f : 0;
    float outputQScale = armnn::IsQuantizedType<T>() ? 2.0f : 0;
    int32_t qOffset = 0;

    inputTensorInfo = armnn::TensorInfo(4, inputShape.data(), ArmnnType, inputsQScale, qOffset);
    outputTensorInfo = armnn::TensorInfo(4, outputShape.data(), ArmnnType, outputQScale, qOffset);
    kernelDesc = armnn::TensorInfo(4, kernelShape.data(), ArmnnType, inputsQScale, qOffset);
    biasDesc = armnn::TensorInfo(1, biasShape.data(), armnn::GetBiasDataType(ArmnnType), inputsQScale, qOffset);

    auto input  = MakeRandomTensor<T>(inputTensorInfo, 124908, 0.0f, 255.0f);
    auto kernel = MakeRandomTensor<T>(kernelDesc, 891234, 0.0f, 255.0f);
    auto bias   = MakeRandomTensor<typename FullyConnectedBiasTypeForInputType<T>::Type>(biasDesc, 1028, 0.0f, 255.0f);

    armnn::TensorInfo aclKernelDescriptor = kernelDesc;
    std::vector<T> aclKernelData;
    aclKernelData.assign(kernel.data(), kernel.data() + kernelHeight * kernelWidth * outputChannels);
    if (workloadFactory.GetBackendId() == armnn::BackendId("GpuAcc") ||
        workloadFactory.GetBackendId() == armnn::BackendId("CpuAcc"))
    {
        if (layout == armnn::DataLayout::NCHW)
        {
            std::vector<T> tmp(kernel.size());
            aclKernelDescriptor.SetShape(armnnUtils::Permuted(kernelDesc.GetShape(), {0, 2, 3, 1}));
            armnnUtils::Permute(kernelDesc.GetShape(), {0, 2, 3, 1}, kernel.data(), tmp.data(), sizeof(T));
            aclKernelData = tmp;
        }
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(aclKernelDescriptor);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = tensorHandleFactory.CreateTensorHandle(biasDesc);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddInputToWorkload(data, info, aclKernelDescriptor, weightsHandle.get());
    AddInputToWorkload(data, info, biasDesc, biasHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    AllocateAndCopyDataToITensorHandle(weightsHandle.get(), aclKernelData.data());
    AllocateAndCopyDataToITensorHandle(biasHandle.get(), bias.data());

    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padX;
    data.m_Parameters.m_PadRight = padX;
    data.m_Parameters.m_PadTop = padY;
    data.m_Parameters.m_PadBottom = padY;
    data.m_Parameters.m_BiasEnabled = true;
    data.m_Parameters.m_DataLayout = layout.GetDataLayout();

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandleRef = refTensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> biasHandleRef = refTensorHandleFactory.CreateTensorHandle(biasDesc);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadInput(refData, refInfo, 1, kernelDesc, weightsHandleRef.get());
    SetWorkloadInput(refData, refInfo, 2, biasDesc, biasHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::DepthwiseConvolution2d, data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef
            = refWorkloadFactory.CreateWorkload(armnn::LayerType::DepthwiseConvolution2d, refData, refInfo);

    outputHandleRef->Allocate();
    weightsHandleRef->Allocate();
    biasHandleRef->Allocate();
    inputHandleRef->Allocate();

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());
    CopyDataToITensorHandle(inputHandleRef.get(), input.data());
    CopyDataToITensorHandle(weightsHandleRef.get(), kernel.data());
    CopyDataToITensorHandle(biasHandleRef.get(), bias.data());

    ExecuteWorkload(*workload, memoryManager);

    workloadRef->PostAllocationConfigure();
    workloadRef->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    CopyDataFromITensorHandle(expectedOutput.data(), outputHandleRef.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

//
// Explicit template specializations
//
template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
Convolution2d3x3Dilation3x3Test<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
Convolution2d2x3x3Dilation3x3Test<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory&,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
    const armnn::ITensorHandleFactory&,
    bool,
    armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
DepthwiseConvolution2d3x3Dilation3x3Test<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
DepthwiseConvolution2d2x3x3Dilation3x3Test<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
        armnn::IWorkloadFactory&,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory&,
        bool,
        armnn::DataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
DepthwiseConvolution2dMult4Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2dMult4Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
DepthwiseConvolution2dMult2Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthwiseConvolution2dMult2Test<armnn::DataType::Float32, armnn::DataType::Float32>(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout);

//
// Implementation functions
//

LayerTestResult<float, 4> SimpleConvolution2d3x5Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x5TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.f, 0, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x5Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x5TestCommon<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled)
{
    return SimpleConvolution2d3x3NhwcTestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        biasEnabled,
        armnn::DataLayout::NHWC);
}

LayerTestResult<float, 4> SimpleConvolution2d3x3Stride2x2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3Stride2x2TestCommon<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        0.f,
        0,
        biasEnabled,
        layout);
}

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<int16_t, 4> SimpleConvolution2d3x5QSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x5TestCommon<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<int16_t, 4> SimpleConvolution2d3x3QSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return SimpleConvolution2d3x3TestCommon<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout layout)
{
    return SimpleConvolution2dAsymmetricPaddingTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, layout, 0.0f, 0);
}

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout layout)
{
    return Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTestCommon
            <armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, layout, 0.0f, 0);
}

LayerTestResult<float, 4> Convolution1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled)
{
    return Convolution1dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0, biasEnabled);
}

LayerTestResult<uint8_t, 4> Convolution1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled)
{
    return Convolution1dTestImpl<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 128, biasEnabled);
}

LayerTestResult<uint8_t, 4> Convolution2dPerAxisQuantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
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

    if (layout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputInfo, inputData);
        PermuteTensorNhwcToNchw(kernelInfo, kernelData);
        PermuteTensorNhwcToNchw(outputInfo, expectedOutputData);
    }

    std::vector<uint8_t> actualOutput(outputInfo.GetNumElements());

    Convolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = layout;

    std::unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);
    std::unique_ptr<armnn::ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelInfo);
    std::unique_ptr<armnn::ITensorHandle> biasHandle = nullptr;

    WorkloadInfo workloadInfo;
//    ScopedTensorHandle weightTensor(kernelInfo);
//    ScopedTensorHandle biasTensor(biasInfo);
//
//    AllocateAndCopyDataToITensorHandle(&weightTensor, kernelData.data());
//    AllocateAndCopyDataToITensorHandle(&biasTensor, biasData.data());

    Convolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, kernelInfo, weightsHandle.get());

    if (descriptor.m_BiasEnabled)
    {
        biasHandle = tensorHandleFactory.CreateTensorHandle(biasInfo);
        AddInputToWorkload(queueDescriptor, workloadInfo, biasInfo, biasHandle.get());
    }

    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    std::unique_ptr<IWorkload> workload= workloadFactory.CreateWorkload(armnn::LayerType::Convolution2d,
                                                                        queueDescriptor,
                                                                        workloadInfo);
    inputHandle->Allocate();
    outputHandle->Allocate();
    weightsHandle->Allocate();

    if (descriptor.m_BiasEnabled)
    {
        biasHandle->Allocate();
        CopyDataToITensorHandle(biasHandle.get(), biasData.data());
    }
    CopyDataToITensorHandle(inputHandle.get(), inputData.data());
    CopyDataToITensorHandle(weightsHandle.get(), kernelData.data());


    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<uint8_t, 4>(actualOutput,
                                       expectedOutputData,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());
}

LayerTestResult<float,4> CompareConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory)
{
    return CompareConvolution2dTestImpl<armnn::DataType::Float32>(
            workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory);
}

LayerTestResult<float, 4> DepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled)
{
    return DepthwiseConvolution2dNhwcTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0, biasEnabled);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul64Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo({ 1, 1, 2, 2 }, armnn::DataType::Float32);
    std::vector<float> input = { 1.f, 2.f, 3.f, 4.f };

    std::vector<float> kernelData;
    std::vector<float> singleDepthKernel{ 1.f, -1.f, -1.f, 1.f };
    for (unsigned int i = 0; i < 64; ++i)
    {
        kernelData.insert(kernelData.end(), singleDepthKernel.begin(), singleDepthKernel.end());
    }
    armnn::TensorInfo kernelTensorInfo({ 64, 1, 2, 2 }, armnn::DataType::Float32);

    // permute from [O,1,H,W] --> [1,H,W,O]
    armnn::PermutationVector permutationVector {3,0,1,2};
    kernelTensorInfo = armnnUtils::Permuted(kernelTensorInfo, permutationVector);
    std::vector<float> kernelPermuted(kernelTensorInfo.GetNumElements());
    armnnUtils::Permute(kernelTensorInfo.GetShape(), permutationVector,
                        kernelData.data(), kernelPermuted.data(),
                        GetDataTypeSize(kernelTensorInfo.GetDataType()));

    std::vector<float> expectedOutputData(64, 0.f);
    armnn::TensorInfo outputTensorInfo({ 1, 64, 1, 1 }, armnn::DataType::Float32);

    return DepthwiseConvolution2dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernelPermuted,
            std::vector<float>(),
            expectedOutputData,
            inputTensorInfo.GetShape(),
            kernelTensorInfo.GetShape(),
            outputTensorInfo.GetShape(),
            0.f,
            0,
            armnn::DataLayout::NCHW);
}

LayerTestResult<float, 4> DepthwiseConvolution2dAsymmetricTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dAsymmetricTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dDepthMul1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<float, 4> SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            0.f,
            0,
            false);
}

LayerTestResult<int16_t, 4> DepthwiseConvolution2dInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dTestImpl<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<int16_t, 4> DepthwiseConvolution2dDepthMul1Int16Test(
                armnn::IWorkloadFactory& workloadFactory,
                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                const armnn::ITensorHandleFactory& tensorHandleFactory,
                bool biasEnabled,
                const armnn::DataLayout layout)
{
    return DepthwiseConvolution2dDepthMul1TestImpl<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
        workloadFactory, memoryManager, tensorHandleFactory, 0.5f, 50, biasEnabled, layout);
}

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dPerAxisQuantTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout layout)
{
    using namespace armnn;

    const DataType inputType  = DataType::QAsymmU8;
    const DataType kernelType = DataType::QSymmS8;
    const DataType biasType   = DataType::Signed32;

    TensorInfo inputInfo ({ 1, 3, 3, 2 }, inputType, 0.5f, 128); // N H W C
    TensorInfo outputInfo({ 1, 2, 2, 4 }, inputType, 1.0f, 128); // N H W C

    const std::vector<float> quantScales{ 1.0f, 0.5f, 1.0f, 0.5f };
    const unsigned int quantDimension = 3;
    TensorInfo kernelInfo({ 1, 2, 2, 4 }, kernelType, quantScales, quantDimension); // [1, H, W, I*M]

    const std::vector<float> biasQuantScales{ 0.5f, 0.25f, 0.5f, 0.25f };
    constexpr unsigned int biasQuantDimension = 0;
    TensorInfo biasInfo({ 4 }, biasType, biasQuantScales, biasQuantDimension);

    std::vector<uint8_t> inputData =
    {
        129, 130,
        129, 130,
        129, 130,
        129, 130,
        129, 130,
        129, 130,
        129, 130,
        129, 130,
        129, 130
    };

    std::vector<int8_t> kernelData =
    {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };

    if (workloadFactory.GetBackendId() == armnn::BackendId("GpuAcc") ||
        workloadFactory.GetBackendId() == armnn::BackendId("CpuAcc"))
    {
        if (layout == armnn::DataLayout::NCHW)
        {
            std::vector<int8_t> tmp(kernelData.size());
            kernelInfo.SetShape(armnnUtils::Permuted(kernelInfo.GetShape(), {0, 2, 3, 1}));
            armnnUtils::Permute(kernelInfo.GetShape(), {0, 2, 3, 1}, kernelData.data(), tmp.data(), sizeof(int8_t));
            kernelData = tmp;
        }
    }

    std::vector<int32_t> biasData =
    {
        4, 4, 4, 4
    };

    std::vector<uint8_t> expectedOutputData =
    {
        132, 130, 134, 131,
        132, 130, 134, 131,
        132, 130, 134, 131,
        132, 130, 134, 131
    };

    if (layout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputInfo, inputData);
        PermuteTensorNhwcToNchw(outputInfo, expectedOutputData);
    }

    std::vector<uint8_t> actualOutput(outputInfo.GetNumElements());

    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_DilationX   = 1;
    descriptor.m_DilationY   = 1;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = layout;

    std::unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<ITensorHandle> weightsHandle = tensorHandleFactory.CreateTensorHandle(kernelInfo);
    std::unique_ptr<ITensorHandle> biasHandle = tensorHandleFactory.CreateTensorHandle(biasInfo);
    std::unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    DepthwiseConvolution2dQueueDescriptor queueDescriptor;
    WorkloadInfo workloadInfo;

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, kernelInfo, weightsHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, biasInfo, biasHandle.get());

    AllocateAndCopyDataToITensorHandle(weightsHandle.get(), kernelData.data());
    AllocateAndCopyDataToITensorHandle(biasHandle.get(), biasData.data());

    queueDescriptor.m_Parameters = descriptor;

    std::unique_ptr<IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::DepthwiseConvolution2d,
                                                                         queueDescriptor,
                                                                         workloadInfo);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    LayerTestResult<uint8_t, 4> ret(outputInfo);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<uint8_t, 4>(actualOutput,
                                       expectedOutputData,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());
}

LayerTestResult<float, 4> CompareDepthwiseConvolution2dFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    const armnn::DataLayout layout)
{
    return CompareDepthwiseConvolution2dTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory, layout);
}

LayerTestResult<uint8_t, 4> CompareDepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    const armnn::DataLayout layout)
{
    return CompareDepthwiseConvolution2dTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory, layout);
}
