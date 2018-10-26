//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>

#include <test/TensorHelpers.hpp>
#include "QuantizeHelper.hpp"

#include <backends/CpuTensorHandle.hpp>
#include <backends/WorkloadFactory.hpp>
#include "Permute.hpp"
#include <boost/numeric/conversion/cast.hpp>

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
    BOOST_ASSERT_MSG((armnn::IsQuantizedType<T>() && vScale != 0.0f) || (!armnn::IsQuantizedType<T>()),
                     "Invalid type and parameter combination.");
    BOOST_ASSERT_MSG((armnn::IsQuantizedType<B>() && bScale != 0.0f) || (!armnn::IsQuantizedType<B>()),
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
                BOOST_ASSERT(offset < v.size());
                T& outRef = v[offset];
                float dOutput = SelectiveDequantize(outRef, vScale, vOffset);
                outRef = SelectiveQuantize<T>(dOutput + dBias, vScale, vOffset);
            }
        }
    }
}

template<typename T>
armnn::TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int height,
                                unsigned int width,
                                const armnn::DataLayoutIndexed& layout)
{
    switch (layout.GetDataLayout())
    {
        case armnn::DataLayout::NCHW:
            return armnn::TensorInfo({numberOfBatches, numberOfChannels, height, width}, armnn::GetDataType<T>());
        case armnn::DataLayout ::NHWC:
            return armnn::TensorInfo({numberOfBatches, height, width, numberOfChannels}, armnn::GetDataType<T>());
        default:
            throw armnn::InvalidArgumentException("unknown data layout ["
                + std::to_string(static_cast<int>(layout.GetDataLayout())) + "]");
    }
}



template<typename T, typename B>
LayerTestResult<T, 4> SimpleConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                  const boost::multi_array<T, 4>& originalInput,
                                                  const boost::multi_array<T, 4>& originalKernel,
                                                  const boost::multi_array<B, 1>& bias,
                                                  const boost::multi_array<T, 4>& originalOutputExpected,
                                                  float qScale,
                                                  int32_t qOffset,
                                                  const armnn::DataLayoutIndexed& layout = armnn::DataLayout::NCHW,
                                                  uint32_t padLeft = 0,
                                                  uint32_t padTop = 0,
                                                  uint32_t padRight = 0,
                                                  uint32_t padBottom = 0)
{
    unsigned int inputHeight   = boost::numeric_cast<unsigned int>(originalInput.shape()[2]);
    unsigned int inputWidth    = boost::numeric_cast<unsigned int>(originalInput.shape()[3]);
    unsigned int inputChannels = boost::numeric_cast<unsigned int>(originalInput.shape()[1]);
    unsigned int inputNum      = boost::numeric_cast<unsigned int>(originalInput.shape()[0]);

    unsigned int outputHeight   = boost::numeric_cast<unsigned int>(originalOutputExpected.shape()[2]);
    unsigned int outputWidth    = boost::numeric_cast<unsigned int>(originalOutputExpected.shape()[3]);
    unsigned int outputChannels = boost::numeric_cast<unsigned int>(originalOutputExpected.shape()[1]);
    unsigned int outputNum      = boost::numeric_cast<unsigned int>(originalOutputExpected.shape()[0]);

    unsigned int kernelHeight = boost::numeric_cast<unsigned int>(originalKernel.shape()[2]);
    unsigned int kernelWidth = boost::numeric_cast<unsigned int>(originalKernel.shape()[3]);
    unsigned int kernelChannels = boost::numeric_cast<unsigned int>(originalKernel.shape()[1]);
    unsigned int kernelDepthMul = boost::numeric_cast<unsigned int>(originalKernel.shape()[0]);

    bool biasEnabled = bias.size() > 0;

    // This function currently assumes 1 batch of input/output (and duplicates this into 2 batches).
    BOOST_ASSERT(inputNum == 1);
    BOOST_ASSERT(outputNum == 1);

    // If a bias is used, its size must equal the number of output channels.
    BOOST_ASSERT(!biasEnabled || bias.size() == outputChannels);


    // Note these tensors will use two (identical) batches.
    // NOTE: if layout is unknown we will get an exception at this point.
    armnn::TensorInfo inputTensorInfo = GetTensorInfo<T>(2*inputNum, inputChannels, inputHeight, inputWidth, layout);
    armnn::TensorInfo outputTensorInfo = GetTensorInfo<T>(
            2*outputNum, outputChannels, outputHeight, outputWidth, layout);
    armnn::TensorInfo kernelDesc = GetTensorInfo<T>(kernelDepthMul, kernelChannels, kernelHeight, kernelWidth, layout);
    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, armnn::GetDataType<B>());

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

    LayerTestResult<T, 4> ret(outputTensorInfo);

    // Construct input data - two batches of the same input image.
    std::vector<T> inputImage;
    inputImage.assign(originalInput.data(), originalInput.data() + 1*inputChannels*inputHeight*inputWidth);
    std::vector<T> inputData;
    inputData.insert(inputData.end(), inputImage.begin(), inputImage.end());
    inputData.insert(inputData.end(), inputImage.begin(), inputImage.end());

    // at this point if we require it permute the input data
    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (layout.GetDataLayout() == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;
    }

    auto batchedInput = MakeTensor<T, 4>(inputTensorInfo, inputData);

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

    // Construct expected output data - two identical images.
    std::vector<T> outputData;
    outputData.insert(outputData.end(), outputImage.begin(), outputImage.end());
    outputData.insert(outputData.end(), outputImage.begin(), outputImage.end());

    // at this point if we require it permute the expected output
    if (layout.GetDataLayout() == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp.data());
        outputData = tmp;
    }
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputData);

    // Todo: nontrivial padding and strides.
    uint32_t                    strideX  = 1;
    uint32_t                    strideY  = 1;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Convolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);
    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);
    // Permute the kernel if necessary
    boost::multi_array<T, 4> kernel = boost::multi_array<T, 4>(originalKernel);
    if (layout.GetDataLayout() == armnn::DataLayout::NHWC)
    {
        armnnUtils::Permute(kernelDesc.GetShape(), NCHWToNHWC, originalKernel.data(), kernel.data());
    }
    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);

    if(biasEnabled)
    {
        AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);
    }

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor; // Still set this whether or not bias is enabled - can be a source of bugs.
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = layout.GetDataLayout();

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConvolution2d(data, info);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &batchedInput[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<typename T, typename B>
LayerTestResult<T, 4> SimpleConvolution2dNhwcTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                      const boost::multi_array<T, 4>& input,
                                                      const boost::multi_array<T, 4>& kernel,
                                                      const boost::multi_array<B, 1>& bias,
                                                      const boost::multi_array<T, 4>& outputExpected,
                                                      armnn::DataLayout dataLayout,
                                                      float qScale,
                                                      int32_t qOffset,
                                                      uint32_t padLeft = 1,
                                                      uint32_t padTop = 1,
                                                      uint32_t padRight = 1,
                                                      uint32_t padBottom = 1,
                                                      uint32_t strideX  = 1,
                                                      uint32_t strideY  = 1)
{
    unsigned int inputNum       = boost::numeric_cast<unsigned int>(input.shape()[0]);
    unsigned int inputChannels  = boost::numeric_cast<unsigned int>(input.shape()[3]);
    unsigned int inputHeight    = boost::numeric_cast<unsigned int>(input.shape()[1]);
    unsigned int inputWidth     = boost::numeric_cast<unsigned int>(input.shape()[2]);

    unsigned int kernelChanMul  = boost::numeric_cast<unsigned int>(kernel.shape()[0]);
    unsigned int kernelChannels = boost::numeric_cast<unsigned int>(kernel.shape()[3]);
    unsigned int kernelHeight   = boost::numeric_cast<unsigned int>(kernel.shape()[1]);
    unsigned int kernelWidth    = boost::numeric_cast<unsigned int>(kernel.shape()[2]);

    unsigned int outputNum      = boost::numeric_cast<unsigned int>(outputExpected.shape()[0]);
    unsigned int outputChannels = boost::numeric_cast<unsigned int>(outputExpected.shape()[3]);
    unsigned int outputHeight   = boost::numeric_cast<unsigned int>(outputExpected.shape()[1]);
    unsigned int outputWidth    = boost::numeric_cast<unsigned int>(outputExpected.shape()[2]);

    bool biasEnabled = bias.size() > 0;

    // Creates the tensors.
    armnn::TensorInfo inputTensorInfo({inputNum, inputHeight, inputWidth, inputChannels}, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo({outputNum, outputHeight, outputWidth, outputChannels},
                                       armnn::GetDataType<T>());
    armnn::TensorInfo kernelDesc({kernelChanMul, kernelHeight, kernelWidth, kernelChannels}, armnn::GetDataType<T>());
    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, armnn::GetDataType<B>());

    // Construct the input data.
    std::vector<T> inputData;
    inputData.assign(input.data(), input.data() + inputHeight*inputWidth*inputChannels);
    auto batchedInput = MakeTensor<T, 4>(inputTensorInfo, inputData);

    // Construct the output data, with bias applied, as appropriate.
    std::vector<T> outputData;
    outputData.assign(outputExpected.data(), outputExpected.data() + outputHeight*outputWidth*outputChannels);

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);
    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);

    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);

    armnn::Convolution2dQueueDescriptor data;

    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor; // Still set this whether or not bias is enabled - can be a source of bugs.
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
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConvolution2d(data, info);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &batchedInput[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<typename T, typename B>
LayerTestResult<T, 4> DepthwiseConvolution2dAsymmetricTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                               const boost::multi_array<T, 4>& input,
                                                               const boost::multi_array<T, 4>& originalKernel,
                                                               const boost::multi_array<B, 1>& bias,
                                                               const boost::multi_array<T, 4>& outputExpected,
                                                               float qScale,
                                                               int32_t qOffset,
                                                               const armnn::DataLayoutIndexed& layout,
                                                               uint32_t padLeft = 0,
                                                               uint32_t padTop = 0,
                                                               uint32_t padRight = 0,
                                                               uint32_t padBottom = 0,
                                                               uint32_t strideX = 1,
                                                               uint32_t strideY = 1)
{
    unsigned int inputNum       = boost::numeric_cast<unsigned int>(input.shape()[0]);
    unsigned int inputChannels  = boost::numeric_cast<unsigned int>(input.shape()[1]);
    unsigned int inputHeight    = boost::numeric_cast<unsigned int>(input.shape()[2]);
    unsigned int inputWidth     = boost::numeric_cast<unsigned int>(input.shape()[3]);
    unsigned int kernelChanMul  = boost::numeric_cast<unsigned int>(originalKernel.shape()[0]);
    unsigned int kernelChannels = boost::numeric_cast<unsigned int>(originalKernel.shape()[1]);
    unsigned int kernelHeight   = boost::numeric_cast<unsigned int>(originalKernel.shape()[2]);
    unsigned int kernelWidth    = boost::numeric_cast<unsigned int>(originalKernel.shape()[3]);
    unsigned int outputNum      = boost::numeric_cast<unsigned int>(outputExpected.shape()[0]);
    unsigned int outputChannels = boost::numeric_cast<unsigned int>(outputExpected.shape()[1]);
    unsigned int outputHeight   = boost::numeric_cast<unsigned int>(outputExpected.shape()[2]);
    unsigned int outputWidth    = boost::numeric_cast<unsigned int>(outputExpected.shape()[3]);

    // If a bias is used, its size must equal the number of output channels.
    bool biasEnabled = bias.size() > 0;
    BOOST_ASSERT(!biasEnabled || bias.size() == outputChannels);

    // Creates the tensors.
    armnn::TensorInfo inputTensorInfo = GetTensorInfo<T>(inputNum, inputChannels, inputHeight, inputWidth, layout);
    armnn::TensorInfo outputTensorInfo = GetTensorInfo<T>(outputNum, outputChannels, outputHeight, outputWidth, layout);
    armnn::TensorInfo kernelDesc = GetTensorInfo<T>(kernelChanMul, kernelChannels, kernelHeight, kernelWidth, layout);
    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, armnn::GetDataType<B>());

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
    if (layout.GetDataLayout() == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data());
        inputData = tmp;
    }

    auto batchedInput = MakeTensor<T, 4>(inputTensorInfo, inputData);

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

    LayerTestResult<T, 4> ret(outputTensorInfo);

    // At this point if we require it permute the expected output
    if (layout.GetDataLayout() == armnn::DataLayout::NHWC)
    {
        std::vector<T> tmp(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp.data());
        outputData = tmp;
    }

    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);

    // Permute the kernel if necessary
    boost::multi_array<T, 4> kernel = boost::multi_array<T, 4>(originalKernel);
    if (layout.GetDataLayout() == armnn::DataLayout::NHWC)
    {
        armnnUtils::Permute(kernelDesc.GetShape(), NCHWToNHWC, originalKernel.data(), kernel.data());
    }

    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);

    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);
    if (biasEnabled)
    {
        AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);
    }

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor; // Still set this whether or not bias is enabled - it can be a source of bugs.
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_DataLayout = layout.GetDataLayout();

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDepthwiseConvolution2d(data, info);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &batchedInput[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<typename T, typename B>
LayerTestResult<T, 4> DepthwiseConvolution2dDepthMul1TestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                              float qScale,
                                                              int32_t qOffset,
                                                              bool biasEnabled)
{
    unsigned int inputHeight = 3;
    unsigned int inputWidth = 3;
    unsigned int inputChannels = 2;
    unsigned int inputNum = 1;

    unsigned int kernelHeight = 3;
    unsigned int kernelWidth = 3;
    unsigned int kernelChannels = inputChannels;

    unsigned int outputHeight = 1;
    unsigned int outputWidth = 1;
    unsigned int outputChannels = kernelChannels;
    unsigned int outputNum = inputNum;

    armnn::TensorInfo inputTensorInfo({ inputNum, inputChannels, inputHeight, inputWidth }, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo({ outputNum, outputChannels, outputHeight, outputWidth },
        armnn::GetDataType<T>());
    armnn::TensorInfo kernelDesc({ 1, outputChannels, kernelHeight, kernelWidth }, armnn::GetDataType<T>());
    armnn::TensorInfo biasDesc({ outputChannels }, armnn::GetDataType<B>());

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

    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset(), {
            1.f, 2.f, 1.f,
            2.f, 1.f, 2.f,
            1.f, 2.f, 1.f,

            1.f, 2.f, 1.f,
            2.f, 1.f, 2.f,
            1.f, 2.f, 1.f,
        })));

    std::vector<B> biasV(QuantizedVector<B>(biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
                                            {0, 2}));
    auto bias = MakeTensor<B, 1>(biasDesc, biasV);

    auto kernel = MakeTensor<T, 4>(kernelDesc, std::vector<T>(
        QuantizedVector<T>(kernelDesc.GetQuantizationScale(), kernelDesc.GetQuantizationOffset(), {
            1.f, 0.f,  1.f,
            0.f, 0.f,  0.f,
           -1.f, 0.f, -1.f,

            1.f, 0.f,  1.f,
            0.f, 0.f,  0.f,
           -1.f, 0.f, -1.f,
        })));

    // Manually calculated.
    std::vector<T> outputImage(
        QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                           outputTensorInfo.GetQuantizationOffset(),
                           {0.f, 0.f})
    );

    // Optionally apply bias to output image.
    if(biasEnabled)
    {
        ApplyBias(outputImage, outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
                  biasV, biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
                  outputWidth, outputHeight);
    }

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputImage);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);
    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);

    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);
    AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor; // Still set this whether or not bias is enabled.
    data.m_Parameters.m_StrideX = 1;
    data.m_Parameters.m_StrideY = 1;
    data.m_Parameters.m_PadLeft = 0;
    data.m_Parameters.m_PadRight = 0;
    data.m_Parameters.m_PadTop = 0;
    data.m_Parameters.m_PadBottom = 0;
    data.m_Parameters.m_BiasEnabled = biasEnabled;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDepthwiseConvolution2d(data, info);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<typename T, typename B>
LayerTestResult<T, 4> DepthwiseConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                     float qScale,
                                                     int32_t qOffset,
                                                     bool biasEnabled)
{
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

    armnn::TensorInfo inputTensorInfo({inputBatchSize, inputChannels, inputHeight, inputWidth},
                                      armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo({outputBatchSize, outputChannels, outputHeight, outputWidth},
        armnn::GetDataType<T>());
    armnn::TensorInfo kernelDesc({depthMultiplier, inputChannels, kernelHeight, kernelWidth}, armnn::GetDataType<T>());
    armnn::TensorInfo biasDesc({outputChannels}, armnn::GetDataType<B>());

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

    auto input = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset(), {
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
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        })));

    std::vector<B> biasV(QuantizedVector<B>(biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
        {0, 2, 1, -1}));
    auto bias = MakeTensor<B, 1>(biasDesc, biasV);

    auto kernel = MakeTensor<T, 4>(kernelDesc, std::vector<T>(
        QuantizedVector<T>(kernelDesc.GetQuantizationScale(), kernelDesc.GetQuantizationOffset(), {
            1, 1, 1,
            1, -1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,

            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,

            0, 0, 0,
            0, -1, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,

            0, 0, 0,
            0, 0, 0,
            0, 1, 0,
            0, 0, 0,
            0, 0, 0
        })));

    // Manually calculated.
    std::vector<T> outputImage = std::vector<T>(
        QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(), {
            3.5f,  3.5f,  3.5f,  3.5f,  3.5f,  3.5f,  3.5f,
            6.0f,  6.0f,  6.0f,  6.0f,  6.0f,  6.0f,  6.0f,
            5.0f,  5.0f,  5.0f,  5.0f,  5.0f,  5.0f,  5.0f,
            6.5f,  6.5f,  6.5f,  6.5f,  6.5f,  6.5f,  6.5f,
            6.5f,  6.5f,  6.5f,  6.5f,  6.5f,  6.5f,  6.5f,
            5.0f,  5.0f,  5.0f,  5.0f,  5.0f,  5.0f,  5.0f,

            -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
            -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
            -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
            -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,

            8.0f,  8.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            10.0f, 10.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            10.0f, 10.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            10.0f, 10.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            10.0f, 10.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            8.0f,  8.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,

            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
            0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f
        }));

    // Optionally apply bias to output image.
    if(biasEnabled)
    {
        ApplyBias(outputImage, outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
            biasV, biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset(),
            outputWidth, outputHeight);
    }

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputImage);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);
    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);

    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);
    AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor; // Still set this whether or not bias is enabled.
    data.m_Parameters.m_StrideX = 2;
    data.m_Parameters.m_StrideY = 1;
    data.m_Parameters.m_PadLeft = 0;
    data.m_Parameters.m_PadRight = 0;
    data.m_Parameters.m_PadTop = 1;
    data.m_Parameters.m_PadBottom = 1;
    data.m_Parameters.m_BiasEnabled = biasEnabled;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDepthwiseConvolution2d(data, info);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<typename T, typename B>
LayerTestResult<T, 4> DepthwiseConvolution2dNhwcTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                         const boost::multi_array<T, 4>& input,
                                                         const boost::multi_array<T, 4>& kernel,
                                                         const boost::multi_array<B, 1>& bias,
                                                         const boost::multi_array<T, 4>& outputExpected,
                                                         float qScale,
                                                         int32_t qOffset,
                                                         uint32_t padLeft = 0,
                                                         uint32_t padTop = 0,
                                                         uint32_t padRight = 0,
                                                         uint32_t padBottom = 0,
                                                         uint32_t strideX = 1,
                                                         uint32_t strideY = 1)
{
    unsigned int inputNum       = boost::numeric_cast<unsigned int>(input.shape()[0]);
    unsigned int inputChannels  = boost::numeric_cast<unsigned int>(input.shape()[3]);
    unsigned int inputHeight    = boost::numeric_cast<unsigned int>(input.shape()[1]);
    unsigned int inputWidth     = boost::numeric_cast<unsigned int>(input.shape()[2]);

    unsigned int kernelChanMul  = boost::numeric_cast<unsigned int>(kernel.shape()[0]);
    unsigned int kernelChannels = boost::numeric_cast<unsigned int>(kernel.shape()[3]);
    unsigned int kernelHeight   = boost::numeric_cast<unsigned int>(kernel.shape()[1]);
    unsigned int kernelWidth    = boost::numeric_cast<unsigned int>(kernel.shape()[2]);

    unsigned int outputNum      = boost::numeric_cast<unsigned int>(outputExpected.shape()[0]);
    unsigned int outputChannels = boost::numeric_cast<unsigned int>(outputExpected.shape()[3]);
    unsigned int outputHeight   = boost::numeric_cast<unsigned int>(outputExpected.shape()[1]);
    unsigned int outputWidth    = boost::numeric_cast<unsigned int>(outputExpected.shape()[2]);

    // Creates the tensors.
    armnn::TensorInfo inputTensorInfo({inputNum, inputHeight, inputWidth, inputChannels}, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo({outputNum, outputHeight, outputWidth, outputChannels},
                                       armnn::GetDataType<T>());
    armnn::TensorInfo kernelDesc({kernelChanMul, kernelHeight, kernelWidth, kernelChannels}, armnn::GetDataType<T>());
    armnn::TensorInfo biasDesc({static_cast<unsigned int>(bias.size())}, armnn::GetDataType<B>());

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
    inputData.assign(input.data(), input.data() + inputHeight*inputWidth*inputChannels);
    auto batchedInput = MakeTensor<T, 4>(inputTensorInfo, inputData);

    // Construct the output data, with bias applied, as appropriate.
    std::vector<T> outputData;
    outputData.assign(outputExpected.data(), outputExpected.data() + outputHeight*outputWidth*outputChannels);

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);
    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);

    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor; // Still set this whether or not bias is enabled - it can be a source of bugs.
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDepthwiseConvolution2d(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &batchedInput[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<typename T>
LayerTestResult<T,4> Convolution1dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                           float qScale,
                                           int32_t qOffset,
                                           bool biasEnabled)
{
    using B = typename FullyConnectedBiasTypeForInputType<T>::Type;

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

    armnn::TensorInfo inputInfo({batchSize, inputChannels, inputSize, 1}, armnn::GetDataType<T>());
    armnn::TensorInfo outputInfo({batchSize, outputChannels, outputSize, 1}, armnn::GetDataType<T>());
    armnn::TensorInfo kernelInfo({outputChannels, inputChannels, kernelSize, 1}, armnn::GetDataType<T>());
    armnn::TensorInfo biasInfo({outputChannels}, armnn::GetDataType<B>());

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

    std::vector<T> inputData(
        QuantizedVector<T>(inputInfo.GetQuantizationScale(), inputInfo.GetQuantizationOffset(), {
            5.0f, -2.0f, 2.5f, 0.0f, 1.0f,
            -3.0f, 3.2f, 5.0f, 2.0f, 3.0f,
        }));

    std::vector<T> kernelData(
        QuantizedVector<T>(kernelInfo.GetQuantizationScale(), kernelInfo.GetQuantizationOffset(), {
            1.0f, 0.0f, 0.0f,
            0.0f, 2.0f, -1.5f,

            0.0f, 0.0f, 0.0f,
            0.2f, 0.2f, 0.2f,

            0.5f, 0.0f, 0.5f,
            0.0f, -1.0f, 0.0f
        }));

    std::vector<B> biasData(
        QuantizedVector<B>(biasInfo.GetQuantizationScale(), biasInfo.GetQuantizationOffset(), {
            1.0f, 0.0f, 0.0f
        }));

    std::vector<T> outputData(
        QuantizedVector<T>(outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset(), {
            4.5f, -10.8f, 5.0f + 6.4f - 7.5f, -2.0f + 10.0f -3.0f, 2.5f + 4.0f - 4.5f, 6.0f, 1.0f,
            -0.6f, -0.6f + 0.64f, -0.6f + 0.64f + 1.0f, 0.64f + 1.0f + 0.4f, 1.0f + 0.4f + 0.6f, 0.4f + 0.6f, 0.6f,
            2.5f, -1.0f + 3.0f, 1.25f - 3.2f + 2.5f, -1.0f - 5.0f, 1.25f + 0.5f - 2.0f, -3.0f, 0.5f
        }));

    // Optionally apply bias to output image.
    if(biasEnabled)
    {
        ApplyBias(outputData, outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset(),
            biasData, biasInfo.GetQuantizationScale(), biasInfo.GetQuantizationOffset(),
            1, outputSize);
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputInfo);

    armnn::Convolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle         weightsTensor(kernelInfo);
    armnn::ScopedCpuTensorHandle         biasTensor(biasInfo);

    AllocateAndCopyDataToITensorHandle(&weightsTensor, kernelData.data());
    AllocateAndCopyDataToITensorHandle(&biasTensor, biasData.data());

    AddInputToWorkload(data, info, inputInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputInfo, outputHandle.get());

    data.m_Weight         = &weightsTensor;
    data.m_Bias           = &biasTensor;
    data.m_Parameters.m_StrideX        = 1;
    data.m_Parameters.m_StrideY        = stride;
    data.m_Parameters.m_PadLeft        = 0;
    data.m_Parameters.m_PadRight       = 0;
    data.m_Parameters.m_PadTop         = padSize;
    data.m_Parameters.m_PadBottom      = padSize;
    data.m_Parameters.m_BiasEnabled    = biasEnabled;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConvolution2d(data, info);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    workloadFactory.Finalize();
    workload->Execute();

    // Output
    LayerTestResult<T,4> ret(outputInfo);
    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());
    ret.outputExpected = MakeTensor<T, 4>(outputInfo, outputData);
    return ret;
}



template<typename T>
LayerTestResult<T,4> CompareConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                armnn::IWorkloadFactory& refWorkloadFactory)
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

    unsigned int inputShape[]    = {inputNum, inputChannels, inputHeight, inputWidth};
    unsigned int outputShape[]   = {outputNum, outputChannels, outputHeight, outputWidth};
    unsigned int kernelShape[]   = {outputChannels, inputChannels, kernelHeight, kernelWidth};
    unsigned int biasShape[]     = {outputChannels};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::GetDataType<T>());
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::GetDataType<T>());
    kernelDesc = armnn::TensorInfo(4, kernelShape, armnn::GetDataType<T>());
    biasDesc = armnn::TensorInfo(1, biasShape, armnn::GetDataType<T>());

    LayerTestResult<T,4> ret(outputTensorInfo);

    auto input  = MakeRandomTensor<T, 4>(inputTensorInfo, 124908);
    auto kernel = MakeRandomTensor<T, 4>(kernelDesc, 891234);
    auto bias   = MakeRandomTensor<T, 1>(biasDesc, 1028);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Convolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);
    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);

    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);
    AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor;
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padX;
    data.m_Parameters.m_PadRight = padX;
    data.m_Parameters.m_PadTop = padY;
    data.m_Parameters.m_PadBottom = padY;
    data.m_Parameters.m_BiasEnabled = true;

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refWorkloadFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refWorkloadFactory.CreateTensorHandle(inputTensorInfo);

    armnn::Convolution2dQueueDescriptor refData = data;
    armnn::WorkloadInfo               refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload  = workloadFactory.CreateConvolution2d(data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateConvolution2d(refData, refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);
    CopyDataToITensorHandle(inputHandleRef.get(), &input[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();
    refWorkloadFactory.Finalize();
    workloadRef->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());
    CopyDataFromITensorHandle(&ret.outputExpected[0][0][0][0], outputHandleRef.get());

    return ret;
}

template<typename T>
LayerTestResult<T, 4> CompareDepthwiseConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
    armnn::IWorkloadFactory& refWorkloadFactory)
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

    unsigned int inputShape[] = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[] = { outputNum, outputChannels, outputHeight, outputWidth };
    unsigned int kernelShape[] = { channelMultiplier, inputChannels, kernelHeight, kernelWidth };
    unsigned int biasShape[] = { outputChannels };

    float inputsQScale = armnn::IsQuantizedType<T>() ? 1.0f : 0;
    float outputQScale = armnn::IsQuantizedType<T>() ? 2.0f : 0;
    int32_t qOffset = 0;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::GetDataType<T>(), inputsQScale, qOffset);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::GetDataType<T>(), outputQScale, qOffset);
    kernelDesc = armnn::TensorInfo(4, kernelShape, armnn::GetDataType<T>(), inputsQScale, qOffset);
    biasDesc = armnn::TensorInfo(1, biasShape, armnn::GetBiasDataType(armnn::GetDataType<T>()), inputsQScale, qOffset);

    LayerTestResult<T, 4> ret(outputTensorInfo);

    auto input = MakeRandomTensor<T, 4>(inputTensorInfo, 124908, 0.0f, 255.0f);
    auto kernel = MakeRandomTensor<T, 4>(kernelDesc, 891234, 0.0f, 255.0f);
    auto bias = MakeRandomTensor<typename FullyConnectedBiasTypeForInputType<T>::Type, 1>(biasDesc, 1028, 0.0f, 255.0f);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle weightsTensor(kernelDesc);
    armnn::ScopedCpuTensorHandle biasTensor(biasDesc);

    AllocateAndCopyDataToITensorHandle(&weightsTensor, &kernel[0][0][0][0]);
    AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor;
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_PadLeft = padX;
    data.m_Parameters.m_PadRight = padX;
    data.m_Parameters.m_PadTop = padY;
    data.m_Parameters.m_PadBottom = padY;
    data.m_Parameters.m_BiasEnabled = true;

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refWorkloadFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refWorkloadFactory.CreateTensorHandle(inputTensorInfo);

    armnn::DepthwiseConvolution2dQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDepthwiseConvolution2d(data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateDepthwiseConvolution2d(refData, refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);
    CopyDataToITensorHandle(inputHandleRef.get(), &input[0][0][0][0]);

    workloadFactory.Finalize();
    workload->Execute();
    refWorkloadFactory.Finalize();
    workloadRef->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());
    CopyDataFromITensorHandle(&ret.outputExpected[0][0][0][0], outputHandleRef.get());

    return ret;
}
