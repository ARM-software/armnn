//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Conv3dTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

using namespace armnnUtils;

//
// Helper templates
//

// Helper template that returns a quantized bias depending on the number of output channels.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> GetBiasData(bool biasEnabled, float qScale, armnn::TensorInfo outputInfo, armnn::DataLayout layout)
{
    if(!biasEnabled)
    {
        return std::vector<T>();
    }
    else
    {
        const armnnUtils::DataLayoutIndexed dataLayoutIndexed(layout);
        const unsigned int outputChannels = outputInfo.GetShape()[dataLayoutIndexed.GetChannelsIndex()];

        switch (outputChannels)
        {
            case 1:
            {
                return QuantizedVector<T>({2}, qScale, 0);
            }
            case 2:
            default:
            {
                return QuantizedVector<T>({0, 2}, qScale, 0);
            }
        }
    }
}

// Modifies a std::vector in-place using a specified bias.
template<typename T, typename B>
void ApplyBiasToData(std::vector<T>& v, const std::vector<B>& bias,
                     float vScale, int32_t vOffset,
                     float bScale, int32_t bOffset)
{
    ARMNN_ASSERT_MSG((armnn::IsQuantizedType<T>() && vScale != 0.0f) || (!armnn::IsQuantizedType<T>()),
                     "Invalid type and parameter combination.");
    ARMNN_ASSERT_MSG((armnn::IsQuantizedType<B>() && bScale != 0.0f) || (!armnn::IsQuantizedType<B>()),
                     "Invalid type and parameter combination.");

    for (uint32_t i = 0; i < bias.size(); ++i)
    {
        for (size_t j = i; j < v.size(); j+=bias.size())
        {
            // Note we need to dequantize and re-quantize the image value and the bias.
            float dBias = SelectiveDequantize(bias[i], bScale, bOffset);

            T& outRef = v[j];
            float dOutput = SelectiveDequantize(outRef, vScale, vOffset);
            outRef = SelectiveQuantize<T>(dOutput + dBias, vScale, vOffset);
        }
    }
}

// Set the quantization scale and offset values for data types.
template<armnn::DataType ArmnnType>
void SetScaleOffset(float& qScale, int32_t& qOffset)
{
    switch (ArmnnType)
    {
        case armnn::DataType::QAsymmU8:
        {
            qScale = 0.1f;
            qOffset = 128;
            break;
        }
        case armnn::DataType::QAsymmS8:
        {
            qScale = 0.1f;
            qOffset = 64;
            break;
        }
        case armnn::DataType::QSymmS16:
        {
            qScale = 0.1f;
            qOffset = 0;
            break;
        }
        case armnn::DataType::BFloat16:
        case armnn::DataType::Float16:
        case armnn::DataType::Float32:
        default:
        {
            qScale = 1.f;
            qOffset = 0;
            break;
        }
    }
}

// Create a vector from 0 to size and quantize (if required).
template <typename T>
std::vector<T> CreateQuantizedData(int32_t size, float qScale, int32_t qOffset)
{
    std::vector<float> data;
    for (int32_t i = 0; i < size; ++i)
    {
        data.push_back(static_cast<float>(i));
    }

    return QuantizedVector<T>(data, qScale, qOffset);
}

// Create a vector from 0 to size divided and then quantized (if required) to create smaller floating point values.
template <typename T>
std::vector<T> CreateSmallQuantizedData(int32_t size, float divisor, float qScale, int32_t qOffset)
{
    std::vector<float> data;
    for (int32_t i = 0; i < size; ++i)
    {
        float value = static_cast<float>(i);
        data.push_back(value/divisor);
    }

    return QuantizedVector<T>(data, qScale, qOffset);;
}

//
// Convolution3d implementations
//

template<armnn::DataType ArmnnType,
        armnn::DataType ArmnnBType,
        typename T = armnn::ResolveType<ArmnnType>,
        typename B = armnn::ResolveType<ArmnnBType>>
LayerTestResult<T, 5> SimpleConvolution3dTestImpl(
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
        const armnn::DataLayout dataLayout,
        float qScale,
        int32_t qOffset,
        uint32_t strideX   = 1,
        uint32_t strideY   = 1,
        uint32_t strideZ   = 1,
        uint32_t dilationX = 1,
        uint32_t dilationY = 1,
        uint32_t dilationZ = 1,
        uint32_t padLeft   = 0,
        uint32_t padTop    = 0,
        uint32_t padRight  = 0,
        uint32_t padBottom = 0,
        uint32_t padFront  = 0,
        uint32_t padBack   = 0)
{
    unsigned int inputNum       = armnn::numeric_cast<unsigned int>(inputShape[0]);
    unsigned int inputDepth     = armnn::numeric_cast<unsigned int>(inputShape[1]);
    unsigned int inputHeight    = armnn::numeric_cast<unsigned int>(inputShape[2]);
    unsigned int inputWidth     = armnn::numeric_cast<unsigned int>(inputShape[3]);
    unsigned int inputChannels  = armnn::numeric_cast<unsigned int>(inputShape[4]);

    // Conv3d weights/kernel layout: [D,H,W,I,O]
    unsigned int kernelDepth        = armnn::numeric_cast<unsigned int>(kernelShape[0]);
    unsigned int kernelHeight       = armnn::numeric_cast<unsigned int>(kernelShape[1]);
    unsigned int kernelWidth        = armnn::numeric_cast<unsigned int>(kernelShape[2]);
    unsigned int kernelInChannels   = armnn::numeric_cast<unsigned int>(kernelShape[3]);
    unsigned int kernelOutChannels  = armnn::numeric_cast<unsigned int>(kernelShape[4]);

    unsigned int outputNum      = armnn::numeric_cast<unsigned int>(outputExpectedShape[0]);
    unsigned int outputDepth    = armnn::numeric_cast<unsigned int>(outputExpectedShape[1]);
    unsigned int outputHeight   = armnn::numeric_cast<unsigned int>(outputExpectedShape[2]);
    unsigned int outputWidth    = armnn::numeric_cast<unsigned int>(outputExpectedShape[3]);
    unsigned int outputChannels = armnn::numeric_cast<unsigned int>(outputExpectedShape[4]);

    bool biasEnabled = bias.size() > 0;

    // If a bias is used, its size must equal the number of output channels.
    ARMNN_ASSERT(!biasEnabled || bias.size() == outputChannels);

    // Creates the tensors.
    armnn::TensorInfo inputTensorInfo({inputNum, inputDepth, inputHeight, inputWidth, inputChannels}, ArmnnType);
    armnn::TensorInfo outputTensorInfo({outputNum, outputDepth, outputHeight, outputWidth, outputChannels}, ArmnnType);
    armnn::TensorInfo kernelDesc({kernelDepth, kernelHeight, kernelWidth, kernelInChannels, kernelOutChannels},
                                 ArmnnType);
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

    // Construct the input data.
    std::vector<T> inputData;
    inputData.assign(input.data(), input.data() + inputNum*inputDepth*inputHeight*inputWidth*inputChannels);

    // Construct the output data and apply bias if needed.
    std::vector<T> outputData;
    outputData.assign(outputExpected.data(), outputExpected.data() +
        outputNum*outputDepth*outputHeight*outputWidth*outputChannels);

    if (biasEnabled)
    {
        ApplyBiasToData(outputData, bias,
                        outputTensorInfo.GetQuantizationScale(), outputTensorInfo.GetQuantizationOffset(),
                        biasDesc.GetQuantizationScale(), biasDesc.GetQuantizationOffset());
    }

    // Permute input and output if data layout is NCDHW.
    if (dataLayout == armnn::DataLayout::NCDHW)
    {
        PermuteTensorNdhwcToNcdhw(inputTensorInfo, inputData);
        PermuteTensorNdhwcToNcdhw(outputTensorInfo, outputData);
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> input0Handle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> input1Handle = tensorHandleFactory.CreateTensorHandle(kernelDesc);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::Convolution3dQueueDescriptor data;
    data.m_Parameters.m_StrideX = strideX;
    data.m_Parameters.m_StrideY = strideY;
    data.m_Parameters.m_StrideZ = strideZ;
    data.m_Parameters.m_PadLeft = padLeft;
    data.m_Parameters.m_PadRight = padRight;
    data.m_Parameters.m_PadTop = padTop;
    data.m_Parameters.m_PadBottom = padBottom;
    data.m_Parameters.m_PadFront = padFront;
    data.m_Parameters.m_PadBack = padBack;
    data.m_Parameters.m_DilationX = dilationX;
    data.m_Parameters.m_DilationY = dilationY;
    data.m_Parameters.m_DilationZ = dilationZ;
    data.m_Parameters.m_DataLayout = dataLayout;
    data.m_Parameters.m_BiasEnabled = biasEnabled;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, input0Handle.get());
    AddInputToWorkload(data, info, kernelDesc, input1Handle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::ITensorHandle> input2Handle = nullptr;
    if (biasEnabled)
    {
        input2Handle = tensorHandleFactory.CreateTensorHandle(biasDesc);
        AddInputToWorkload(data, info, biasDesc, input2Handle.get());
    }

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Convolution3d,
                                                                                data,
                                                                                info);
    input0Handle->Allocate();
    input1Handle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(input0Handle.get(), inputData.data());
    CopyDataToITensorHandle(input1Handle.get(), kernel.data());
    if (biasEnabled)
    {
        input2Handle->Allocate();
        CopyDataToITensorHandle(input2Handle.get(), bias.data());
    }

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 5>(actualOutput,
                                 outputData,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType,
        armnn::DataType ArmnnBType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> SimpleConvolution3d3x3x3TestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    float qScale;
    int32_t qOffset;
    SetScaleOffset<ArmnnType>(qScale, qOffset);

    armnn::TensorInfo inputDesc({ 1, 5, 5, 5, 1 }, ArmnnType);
    std::vector<T> input = CreateQuantizedData<T>(125, qScale, qOffset);

    armnn::TensorInfo kernelDesc({ 3, 3, 3, 1, 1 }, ArmnnType);
    std::vector<T> kernel = QuantizedVector<T>(
    {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        0, 0, 0,
        0, 1, 0,
        0, 0, 0,

        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    },
    qScale, qOffset);

    armnn::TensorInfo outputDesc({ 1, 3, 3, 3, 1 }, ArmnnType);
    std::vector<T> outputData = QuantizedVector<T>(
    {
        589, 608, 627,
        684, 703, 722,
        779, 798, 817,

        1064, 1083, 1102,
        1159, 1178, 1197,
        1254, 1273, 1292,

        1539, 1558, 1577,
        1634, 1653, 1672,
        1729, 1748, 1767
    },
    qScale, qOffset);

    return SimpleConvolution3dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<ArmnnBType>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset
    );
}

template<armnn::DataType ArmnnType,
        armnn::DataType ArmnnBType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> Convolution3d2x2x2Strides3x5x5TestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    float qScale;
    int32_t qOffset;
    SetScaleOffset<ArmnnType>(qScale, qOffset);

    armnn::TensorInfo inputDesc({ 1, 3, 10, 10, 1 }, ArmnnType);
    std::vector<T> input = CreateQuantizedData<T>(300, qScale, qOffset);

    armnn::TensorInfo kernelDesc({ 3, 5, 5, 1, 1 }, ArmnnType);
    std::vector<T> kernel = QuantizedVector<T>(
    {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
    },
    qScale, qOffset);

    armnn::TensorInfo outputDesc({ 1, 1, 3, 3, 1 }, ArmnnType);
    std::vector<T> outputData = QuantizedVector<T>(
    {
        11650, 11800, 11950,

        13150, 13300, 13450,

        14650, 14800, 14950
    },
    qScale, qOffset);

    return SimpleConvolution3dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<ArmnnBType>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset,
            2, // strideX
            2, // strideY
            2  // strideZ
    );
}

template<armnn::DataType ArmnnType,
        armnn::DataType ArmnnBType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> Convolution3d2x2x2Dilation2x2x2TestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    float qScale;
    int32_t qOffset;
    SetScaleOffset<ArmnnType>(qScale, qOffset);

    armnn::TensorInfo inputDesc({ 1, 5, 5, 5, 2 }, ArmnnType);
    std::vector<T> input = CreateQuantizedData<T>(250, qScale, qOffset);

    armnn::TensorInfo kernelDesc({ 2, 2, 2, 2, 2 }, ArmnnType);
    std::vector<T> kernel = QuantizedVector<T>(
    {
        -1, -1,  -1, -1,  -1, -1,  -1, -1,  -1, -1,  -1,  1,   1,  1,  -1, -1,
         1,  1,  -1,  1,  -1,  1,  -1,  1,  -1, -1,  -1,  1,  -1,  1,  -1,  1,
    },
    qScale, qOffset);

    // Since the dilation rate is 3 this will dilate the kernel to be 4x4,
    // therefore the output will be 2x2
    armnn::TensorInfo outputDesc({ 1, 2, 2, 2, 2 }, ArmnnType);
    std::vector<T> outputData = QuantizedVector<T>(
    {
        -1124, 974,
        -1148, 978,

        -1244, 994,
        -1268, 998,

        -1724, 1074,
        -1748, 1078,

        -1844, 1094,
        -1868, 1098
    },
    qScale, qOffset);

    return SimpleConvolution3dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<ArmnnBType>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset,
            1, // strideX
            1, // strideY
            1, // strideZ
            3, // dilationX
            3, // dilationY
            3 // dilationZ
    );
}

template<armnn::DataType ArmnnType,
        armnn::DataType ArmnnBType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> Convolution3dPaddingSame3x3x3TestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    float qScale;
    int32_t qOffset;
    SetScaleOffset<ArmnnType>(qScale, qOffset);

    armnn::TensorInfo inputDesc({ 1, 5, 5, 5, 1 }, ArmnnType);
    std::vector<T> input = CreateQuantizedData<T>(125, qScale, qOffset);

    armnn::TensorInfo kernelDesc({ 3, 3, 3, 1, 1 }, ArmnnType);
    std::vector<T> kernel = QuantizedVector<T>(
    {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        0, 0, 0,
        0, 0, 0,
        0, 0, 0,

        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    },
    qScale, qOffset);

    armnn::TensorInfo outputDesc({ 1, 5, 5, 5, 1 }, ArmnnType);
    std::vector<T> outputData = QuantizedVector<T>(
    {
        112, 171, 177, 183, 124,
        183, 279, 288, 297, 201,
        213, 324, 333, 342, 231,
        243, 369, 378, 387, 261,
        172, 261, 267, 273, 184,

        224, 342, 354, 366, 248,
        366, 558, 576, 594, 402,
        426, 648, 666, 684, 462,
        486, 738, 756, 774, 522,
        344, 522, 534, 546, 368,

        424, 642,  654,  666,  448,
        666, 1008, 1026, 1044, 702,
        726, 1098, 1116, 1134, 762,
        786, 1188, 1206, 1224, 822,
        544, 822,  834,  846,  568,
        624, 942,  954,  966,  648,

        966,  1458, 1476, 1494, 1002,
        1026, 1548, 1566, 1584, 1062,
        1086, 1638, 1656, 1674, 1122,
        744,  1122, 1134, 1146, 768,
        312,  471,  477,  483,  324,
        483,  729,  738,  747,  501,
        513,  774,  783,  792,  531,
        543,  819,  828,  837,  561,
        372,  561,  567,  573,  384
    },
    qScale, qOffset);

    return SimpleConvolution3dTestImpl<ArmnnType, ArmnnBType>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<ArmnnBType>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset,
            1, // strideX
            1, // strideY
            1, // strideZ
            1, // dilationX
            1, // dilationY
            1, // dilationZ
            1, // padLeft
            1, // padTop
            1, // padRight
            1, // padBottom
            1, // padFront
            1 // padBack
    );
}

LayerTestResult<float, 5> Convolution3dStrideDilationPadding3x3x3TestCommonFloat32(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    float qScale = 0.f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputDesc({ 1, 3, 10, 10, 2 }, armnn::DataType::Float32);
    std::vector<float> input = CreateSmallQuantizedData<float>(600, 100.0f, qScale, qOffset);

    armnn::TensorInfo kernelDesc({ 3, 3, 3, 2, 2 }, armnn::DataType::Float32);
    std::vector<float> kernel = CreateSmallQuantizedData<float>(108, 100.0f, qScale, qOffset);

    // Since the dilation rate is 2 this will dilate the kernel to be 5x5: d(K-1)+1 --> 2 x (3-1) + 1 = 5,
    // therefore the output will be 1x4x4: (I − K + 2P)/S +1 => trunc((10 - 3 + 2x2 )/3 + 1))
    // where, dilation size = d = 2; kernel size = K = 3; input size = I = 10; padding size = P = 2; stride = S = 3
    armnn::TensorInfo outputDesc({ 1, 1, 4, 4, 2 }, armnn::DataType::Float32);
    std::vector<float> outputData =
    {
        12.0312f, 12.2268f, 17.7512f, 18.0494f,
        18.176f,  18.4814f, 5.6912f,  5.7938f,
        19.1664f, 19.5078f, 28.119f,  28.6383f,
        28.6914f, 29.2215f, 8.9094f,  9.0873f,

        23.1264f, 23.5398f, 33.843f,  34.4703f,
        34.4154f, 35.0535f, 10.6734f, 10.8873f,
        6.2712f,  6.417f,   9.0718f,  9.2929f,
        9.2194f,  9.4441f,  2.7862f,  2.8615f
    };

    return SimpleConvolution3dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<armnn::DataType::Float32>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset,
            3, // strideX
            3, // strideY
            3, // strideZ
            2, // dilationX
            2, // dilationY
            2, // dilationZ
            1, // padLeft
            1, // padTop
            1, // padRight
            1, // padBottom
            1, // padFront
            1 // padBack
    );
}

LayerTestResult<float, 5> Convolution3d2x2x2Stride3x3x3SmallTestCommonFloat32(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    float qScale = 0.f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputDesc({ 1, 3, 10, 10, 1 }, armnn::DataType::Float32);
    std::vector<float> input = CreateSmallQuantizedData<float>(300, 100.0f, qScale, qOffset);

    armnn::TensorInfo kernelDesc({ 3, 3, 3, 1, 1 }, armnn::DataType::Float32);
    std::vector<float> kernel =
    {
         0.125977f,  0.150391f,  0.101562f,
         0.0585938f, 0.0864258f, 0.043457f,
         0.034668f,  0.0322266f, 0.0385742f,

         0.125977f,  0.150391f, -0.101562f,
        -0.0585938f,-0.0864258f,-0.043457f,
        -0.0104630f, 0.0154114f, 0.0013768f,

         0.0344238f, 0.035644f,  0.0495605f,
         0.0683594f, 0.099121f, -0.0461426f,
        -0.0996094f,-0.126953f, -0.043457f,
    };

    armnn::TensorInfo outputDesc({ 1, 1, 4, 4, 1 }, armnn::DataType::Float32);
    std::vector<float> outputData =
    {
        -0.08156067f, -0.06891209f, -0.05589598f, -0.04310101f,
        0.04584253f,   0.05855697f,  0.07129729f,  0.08325434f,
        0.17304349f,   0.18521416f,  0.19818866f,  0.21096253f,
        0.29965734f,   0.312698f,    0.32547557f,  0.33818722f
    };

    return SimpleConvolution3dTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<armnn::DataType::Float32>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset,
            2, // strideX
            2, // strideY
            2  // strideZ
    );
}

LayerTestResult<armnn::Half, 5> Convolution3d2x3x3TestCommonFloat16(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    using namespace half_float::literal;

    float qScale = 0.f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputDesc({ 1, 2, 3, 3, 2 }, armnn::DataType::Float16);
    const std::vector<armnn::Half> input =
    {
        1._h,  2._h,  3._h,
        4._h,  5._h,  6._h,

        7._h,  8._h,  9._h,
        10._h, 11._h, 12._h,

        13._h, 14._h, 15._h,
        16._h, 17._h, 18._h,

        19._h, 20._h, 21._h,
        22._h, 23._h, 24._h,

        25._h, 26._h, 27._h,
        28._h, 29._h, 30._h,

        31._h, 32._h, 33._h,
        34._h, 35._h, 36._h
    };

    armnn::TensorInfo kernelDesc({ 2, 2, 2, 2, 2 }, armnn::DataType::Float16);
    std::vector<armnn::Half> kernel =
    {
        -1._h, -1._h,  -1._h, -1._h,  -1._h, -1._h,  -1._h, -1._h,
        -1._h, -1._h,  -1._h,  1._h,   1._h,  1._h,  -1._h, -1._h,
         1._h,  1._h,  -1._h,  1._h,  -1._h,  1._h,  -1._h,  1._h,
        -1._h, -1._h,  -1._h,  1._h,  -1._h,  1._h,  -1._h,  1._h,
    };

    armnn::TensorInfo outputDesc({ 1, 1, 2, 2, 2 }, armnn::DataType::Float16);
    std::vector<armnn::Half> outputData =
    {
        -176._h,  128._h,
        -200._h,  132._h,

        -248._h,  140._h,
        -272._h,  144._h
    };

    return SimpleConvolution3dTestImpl<armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<armnn::DataType::Float16>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset
    );
}

LayerTestResult<armnn::Half, 5> Convolution3d2x2x2SmallTestCommonFloat16(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    using namespace half_float::literal;

    float qScale = 0.f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputDesc({ 1, 2, 4, 4, 1 }, armnn::DataType::Float16);
    const std::vector<armnn::Half> input =
    {
        0.0367984_h, 0.0380895_h, 0.0420157_h,  0.0675631_h,
        0.0938920_h, 0.0476106_h, 0.1035490_h,  0.1260370_h,
        0.0461647_h, 0.0883828_h, 0.1159540_h,  0.0498519_h,
        0.0104630_h, 0.0154114_h, 0.00137681_h, 0.0344238_h,

        0.0356445_h, 0.0495605_h, 0.0683594_h,  0.0991211_h,
        0.0461426_h, 0.0996094_h, 0.1269530_h,  0.0393066_h,
        0.103516_h,  0.032544_h,  0.124334_h,   0.0564566_h,
        0.0123544_h, 0.0461647_h, 0.0883828_h,  0.1159540_h,
    };

    armnn::TensorInfo kernelDesc({ 2, 2, 2, 1, 1 }, armnn::DataType::Float16);
    std::vector<armnn::Half> kernel =
    {
        -0.126184_h, -0.150468_h,
        -0.101412_h, -0.0586369_h,

        -0.0435089_h, 0.0347555_h,
         0.0323111_h, 0.0385381_h
    };

    armnn::TensorInfo outputDesc({ 1, 1, 3, 3, 1 }, armnn::DataType::Float16);
    std::vector<armnn::Half> outputData =
    {
        -0.01718917_h, -0.01370182_h, -0.02727737_h,

        -0.02282543_h, -0.03144084_h, -0.04468598_h,

        -0.02228982_h, -0.02244923_h, -0.02042268_h
    };

    return SimpleConvolution3dTestImpl<armnn::DataType::Float16, armnn::DataType::Float16>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            input,
            kernel,
            GetBiasData<armnn::DataType::Float16>(biasEnabled, qScale * qScale, outputDesc, dataLayout),
            outputData,
            inputDesc.GetShape(),
            kernelDesc.GetShape(),
            outputDesc.GetShape(),
            dataLayout,
            qScale,
            qOffset
    );
}

LayerTestResult<float, 5> SimpleConvolution3d3x3x3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return SimpleConvolution3d3x3x3TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int8_t, 5> SimpleConvolution3d3x3x3Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return SimpleConvolution3d3x3x3TestCommon<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<uint8_t, 5> SimpleConvolution3d3x3x3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return SimpleConvolution3d3x3x3TestCommon<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int16_t, 5> SimpleConvolution3d3x3x3Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return SimpleConvolution3d3x3x3TestCommon<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}


LayerTestResult<float, 5> Convolution3d2x2x2Strides3x5x5Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Strides3x5x5TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int8_t, 5> Convolution3d2x2x2Strides3x5x5Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Strides3x5x5TestCommon<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<uint8_t, 5> Convolution3d2x2x2Strides3x5x5Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Strides3x5x5TestCommon<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int16_t, 5> Convolution3d2x2x2Strides3x5x5Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Strides3x5x5TestCommon<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<float, 5> Convolution3d2x2x2Dilation2x2x2Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Dilation2x2x2TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int8_t, 5> Convolution3d2x2x2Dilation2x2x2Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Dilation2x2x2TestCommon<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<uint8_t, 5> Convolution3d2x2x2Dilation2x2x2Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Dilation2x2x2TestCommon<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int16_t, 5> Convolution3d2x2x2Dilation2x2x2Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Dilation2x2x2TestCommon<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<float, 5> Convolution3dPaddingSame3x3x3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3dPaddingSame3x3x3TestCommon<armnn::DataType::Float32, armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int8_t, 5> Convolution3dPaddingSame3x3x3Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3dPaddingSame3x3x3TestCommon<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<uint8_t, 5> Convolution3dPaddingSame3x3x3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3dPaddingSame3x3x3TestCommon<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<int16_t, 5> Convolution3dPaddingSame3x3x3Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3dPaddingSame3x3x3TestCommon<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<float, 5> Convolution3dStrideDilationPadding3x3x3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3dStrideDilationPadding3x3x3TestCommonFloat32(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<float, 5> Convolution3d2x2x2Stride3x3x3SmallFloat32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2Stride3x3x3SmallTestCommonFloat32(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<armnn::Half, 5> Convolution3d2x3x3Float16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x3x3TestCommonFloat16(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}

LayerTestResult<armnn::Half, 5> Convolution3d2x2x2SmallFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout)
{
    return Convolution3d2x2x2SmallTestCommonFloat16(
            workloadFactory, memoryManager, tensorHandleFactory, biasEnabled, dataLayout);
}
