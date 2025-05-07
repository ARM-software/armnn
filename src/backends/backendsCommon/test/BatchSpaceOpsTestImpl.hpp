//
// Copyright © 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/Descriptors.hpp>
#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>
#include <optional>

// helper function that computes the SpaceToBatch Output Shape (assumes NHWC)
inline armnn::TensorShape ComputeSpaceToBatchOutputShape(const armnn::TensorShape& inputShape,
                                                         const std::vector<unsigned int>& blockShape,
                                                         const std::vector<std::pair<unsigned int,
                                                                                     unsigned int>>& padList)
{
    unsigned int batch = inputShape[0] * blockShape[0] * blockShape[1];

    unsigned int height = (inputShape[1] + padList[0].first + padList[0].second) / blockShape[0];

    unsigned int width = (inputShape[2] + padList[1].first + padList[1].second) / blockShape[1];

    unsigned int channels = inputShape[3];

    return armnn::TensorShape({ batch, height, width, channels });
}

// helper function that computes the BatchToSpace Output Shape (assumes NHWC)
inline armnn::TensorShape ComputeBatchToSpaceOutputShape(const armnn::TensorShape& inputShape,
                                                         const std::vector<unsigned int>& blockShape,
                                                         const std::vector<std::pair<unsigned int,
                                                                                     unsigned int>>& cropList)
{
    unsigned int batch = inputShape[0] / (blockShape[0] * blockShape[1]);

    unsigned int height = inputShape[1] * blockShape[0] - cropList[0].first - cropList[0].second;

    unsigned int width = inputShape[2] * blockShape[1] - cropList[1].first - cropList[1].second;

    unsigned int channels = inputShape[3];

    return armnn::TensorShape({ batch, height, width, channels });
}

// Helper function for creating input data that ramps from 0 upwards In the case of int8/uint8 clamps to max values
template<typename T>
inline void generateInputDataQuantized(std::vector<T> &data,
                                       unsigned int numInputElements,
                                       const float qScale,
                                       const int32_t qOffset)
{
    /* 
        create a float vector the same size as the input data 
        and fill it with floating point representation of index i
    */
    data.resize(numInputElements);
    
    for (unsigned int i = 0; i < numInputElements; ++i)
    {
        float val = static_cast<float>(i);
        if constexpr (std::is_same<T, float>::value)
        {
            data[i] = val;
        }
        else
        {
            // clamp to int8_t range
            val = std::clamp(val, -128.f, 127.f);
            int32_t quantized = static_cast<int32_t>(std::round(val / qScale)) + qOffset;
            quantized = std::clamp(quantized, -128, 127);
            data[i] = static_cast<T>(quantized);
        }
    }
}

// Helper function that calculates the expected output data for a SpaceToBatch Layer
template<typename T>
std::vector<T> GenerateExpectedSpaceToBatchOutputNHWC(const std::vector<T>& inputData,
                                                      const std::vector<unsigned int>& inputShape,
                                                      const std::vector<unsigned int>& blockShape,
                                                      const std::vector<std::pair<unsigned int, unsigned int>>& padList,
                                                      int32_t qOffset = 0)
{
    // this function assumes 4d shape ie. nhwc and generates the resultant output data for that operator
    const unsigned int N = inputShape[0];
    const unsigned int H = inputShape[1];
    const unsigned int W = inputShape[2];
    const unsigned int C = inputShape[3];

    const unsigned int blockHeight = blockShape[0];
    const unsigned int blockWidth = blockShape[1];

    const unsigned int padTop    = padList[0].first;
    const unsigned int padBottom = padList[0].second;
    const unsigned int padLeft   = padList[1].first;
    const unsigned int padRight  = padList[1].second;

    const unsigned int paddedHeight = H + padTop + padBottom;
    const unsigned int paddedWidth = W + padLeft + padRight;

    const unsigned int outBatch = N * blockHeight * blockWidth;
    const unsigned int outH = paddedHeight / blockHeight;
    const unsigned int outW = paddedWidth / blockWidth;

    const size_t paddedSize = static_cast<size_t>(N) * paddedHeight * paddedWidth * C;
    std::vector<T> paddedInput(paddedSize, static_cast<T>(qOffset));

    // Fill padded input over multiple loops that iterate over NHWC
    for (unsigned int n = 0; n < N; ++n)
    {
        for (unsigned int h = 0; h < H; ++h)
        {
            for (unsigned int w = 0; w < W; ++w)
            {
                for (unsigned int c = 0; c < C; ++c)
                {
                    const size_t inIdx = static_cast<size_t>(((n * H + h) * W + w) * C + c);
                    const size_t paddedHIdx = static_cast<size_t>(h + padTop);
                    const size_t paddedWIdx = static_cast<size_t>(w + padLeft);
                    const size_t paddedIdx = static_cast<size_t>(
                                            ((n * paddedHeight + paddedHIdx) * paddedWidth + paddedWIdx) * C + c);
                    paddedInput[paddedIdx] = inputData[inIdx];
                }
            }
        }
    }

    const size_t outputSize = static_cast<size_t>(outBatch) * outH * outW * C;
    std::vector<T> output(outputSize, static_cast<T>(qOffset));

    // interleaves blocks of input data into batch format i.e. [blockRow][blockCol][batch] order
    for (unsigned int blockRow = 0; blockRow < blockHeight; ++blockRow)
    {
        for (unsigned int blockCol = 0; blockCol < blockWidth; ++blockCol)
        {
            for (unsigned int batch = 0; batch < N; ++batch)
            {
                const unsigned int outputBatch = (blockRow * blockWidth + blockCol) * N + batch;

                for (unsigned int outputRow = 0; outputRow < outH; ++outputRow)
                {
                    for (unsigned int outputCol = 0; outputCol < outW; ++outputCol)
                    {
                        const unsigned int inH = outputRow * blockHeight + blockRow;
                        const unsigned int inW = outputCol * blockWidth + blockCol;

                        for (unsigned int c = 0; c < C; ++c)
                        {
                            const size_t paddedIdx = static_cast<size_t>((
                                                    (batch * paddedHeight + inH) * paddedWidth + inW) * C + c);
                            const size_t outIdx = static_cast<size_t>(
                                                    ((outputBatch * outH + outputRow) * outW + outputCol) * C + c);
                            output[outIdx] = paddedInput[paddedIdx];
                        }
                    }
                }
            }
        }
    }

    return output;
}

// Helper function that calculates the expected output data for a BatchToSpace Layer
// Essentially the inverse of the SpaceToBatchOutputCalculation
template<typename T>
std::vector<T> GenerateExpectedBatchToSpaceOutputNHWC(const std::vector<T>& inputData,
                                                      const std::vector<unsigned int>& inputShape,
                                                      const std::vector<unsigned int>& blockShape,
                                                      const std::vector<std::pair<unsigned int,
                                                                                  unsigned int>>& cropList,
                                                      int32_t qOffset = 0)
{
    // this function assumes 4d shape ie. nhwc and generates the resultant output data for that operator
    const unsigned int inBatch = inputShape[0];
    const unsigned int inH = inputShape[1];
    const unsigned int inW = inputShape[2];
    const unsigned int C = inputShape[3];

    const unsigned int blockHeight = blockShape[0];
    const unsigned int blockWidth = blockShape[1];

    const unsigned int cropTop = cropList[0].first;
    const unsigned int cropBottom = cropList[0].second;
    const unsigned int cropLeft = cropList[1].first;
    const unsigned int cropRight = cropList[1].second;

    const unsigned int outBatch = inBatch / (blockHeight * blockWidth);
    const unsigned int croppedHeight = inH * blockHeight - cropTop - cropBottom;
    const unsigned int croppedWidth = inW * blockWidth  - cropLeft - cropRight;

    const size_t reshapedSize = static_cast<size_t>(outBatch) * inH * blockHeight * inW * blockWidth * C;
    std::vector<T> reshaped(reshapedSize, static_cast<T>(qOffset));

    // De-interleave batch into spatial blocks
    for (unsigned int blockRow = 0; blockRow < blockHeight; ++blockRow)
    {
        for (unsigned int blockCol = 0; blockCol < blockWidth; ++blockCol)
        {
            for (unsigned int batch = 0; batch < outBatch; ++batch)
            {
                const unsigned int inputBatch = (blockRow * blockWidth + blockCol) * outBatch + batch;

                for (unsigned int h = 0; h < inH; ++h)
                {
                    for (unsigned int w = 0; w < inW; ++w)
                    {
                        const unsigned int outH = h * blockHeight + blockRow;
                        const unsigned int outW = w * blockWidth + blockCol;

                        for (unsigned int c = 0; c < C; ++c)
                        {
                            const size_t inIdx = static_cast<size_t>(((inputBatch * inH + h) * inW + w) * C + c);
                            const size_t reshapedIdx = static_cast<size_t>(((batch * (inH * blockHeight) + outH)
                                                                             * (inW * blockWidth) + outW) * C + c);
                            reshaped[reshapedIdx] = inputData[inIdx];
                        }
                    }
                }
            }
        }
    }

    // Apply cropping to data
    std::vector<T> output(static_cast<size_t>(outBatch) * croppedHeight * croppedWidth * C, static_cast<T>(qOffset));

    for (unsigned int b = 0; b < outBatch; ++b)
    {
        for (unsigned int h = 0; h < croppedHeight; ++h)
        {
            for (unsigned int w = 0; w < croppedWidth; ++w)
            {
                for (unsigned int c = 0; c < C; ++c)
                {
                    const unsigned int reshapedH = h + cropTop;
                    const unsigned int reshapedW = w + cropLeft;

                    const size_t reshapedIdx = static_cast<size_t>(((b * (inH * blockHeight) + reshapedH)
                                                                     * (inW * blockWidth) + reshapedW) * C + c);
                    const size_t outIdx = static_cast<size_t>(((b * croppedHeight + h) * croppedWidth + w) * C + c);

                    output[outIdx] = reshaped[reshapedIdx];
                }
            }
        }
    }

    return output;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
armnn::INetworkPtr CreateSpaceToBatchNdNetwork(const armnn::TensorShape& inputShape,
                                              const std::vector<unsigned int>& blockShape,
                                              const std::vector<std::pair<unsigned int, unsigned int>>& paddings,
                                              const armnn::TensorShape& outputShape)
{
    using namespace armnn;
    INetworkPtr network = INetwork::Create();

    // Configure SpaceToBatchND layer descriptor
    SpaceToBatchNdDescriptor desc(blockShape, paddings);
    desc.m_DataLayout = DataLayout::NHWC;

    // Add layers to the network
    IConnectableLayer* inputLayer  = network->AddInputLayer(0, "input");
    IConnectableLayer* stbLayer   = network->AddSpaceToBatchNdLayer(desc, "spaceToBatchND");
    IConnectableLayer* outputLayer = network->AddOutputLayer(0, "output");

    // Set tensor info for input and output
    const float qScale  = 1.0f;
    const int32_t qOffset = 0;
    TensorInfo inputInfo(inputShape, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo(outputShape, ArmnnType, qScale, qOffset, true);

    Connect(inputLayer, stbLayer, inputInfo, 0, 0);
    Connect(stbLayer, outputLayer, outputInfo, 0, 0);
    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
armnn::INetworkPtr CreateBatchToSpaceNdNetwork(const armnn::TensorShape& inputShape,
                                              const std::vector<unsigned int>& blockShape,
                                              const std::vector<std::pair<unsigned int, unsigned int>>& cropList,
                                              const armnn::TensorShape& outputShape)
{
    using namespace armnn;
    INetworkPtr network = INetwork::Create();

    // Configure BatchToSpaceND layer descriptor
    BatchToSpaceNdDescriptor desc(blockShape, cropList);
    desc.m_DataLayout = DataLayout::NHWC;

    // Add layers to the network
    IConnectableLayer* inputLayer = network->AddInputLayer(0, "input");
    IConnectableLayer* btsLayer = network->AddBatchToSpaceNdLayer(desc, "batchToSpaceND");
    IConnectableLayer* outputLayer = network->AddOutputLayer(0, "output");

    // Set tensor info for input and output
    const float qScale  = 1.0f;
    const int32_t qOffset = 0;
    TensorInfo inputInfo(inputShape, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo(outputShape, ArmnnType, qScale, qOffset, true);

    Connect(inputLayer, btsLayer, inputInfo, 0, 0);
    Connect(btsLayer, outputLayer, outputInfo, 0, 0);
    return network;
}
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void SpaceToBatchNdEndToEnd(const std::vector<unsigned int>& inputDims,
                            const std::vector<unsigned int>& blockShape,
                            const std::vector<std::pair<unsigned int, unsigned int>>& padList,
                            const std::vector<armnn::BackendId>& backends,
                            const std::optional<armnn::TensorShape>& expectedOutputShapeOverride = std::nullopt,
                            const std::optional<std::vector<T>>& expectedOutputOverride = std::nullopt)
{
    using namespace armnn;

    TensorShape inputShape(static_cast<unsigned int>(inputDims.size()), inputDims.data());
    TensorShape outputShape;
    if (expectedOutputShapeOverride.has_value())
    {
        /* 
           This was intended as an override for higher dimension tensors where we cannot use the output shape 
           computation function to determine the shape
           in its current form ComputeSpaceToBatchOUtputShape only works for NHWC layout
        */
        outputShape = expectedOutputShapeOverride.value();
    }
    else
    {
        outputShape = ComputeSpaceToBatchOutputShape(inputShape, blockShape, padList);
    }

    INetworkPtr network = CreateSpaceToBatchNdNetwork<ArmnnType>(inputShape, blockShape, padList, outputShape);
    CHECK(network);  

    const float qScale  = 1.0f;
    const int32_t qOffset = 0;
    unsigned int numInputElements = inputShape.GetNumElements();

    std::vector<T> inputData(numInputElements);

    generateInputDataQuantized(inputData, numInputElements, qScale, qOffset);

    std::vector<unsigned int> inputShapeInts = { static_cast<unsigned int>(inputShape[0]),
                                                 static_cast<unsigned int>(inputShape[1]),
                                                 static_cast<unsigned int>(inputShape[2]),
                                                 static_cast<unsigned int>(inputShape[3])
                                               };

    std::vector<unsigned int> blockShapeInts = { static_cast<unsigned int>(blockShape[0]),
                                                 static_cast<unsigned int>(blockShape[1])
                                               };


    std::vector<std::pair<unsigned int, unsigned int>> padListInts = {
        { static_cast<unsigned int>(padList[0].first), static_cast<unsigned int>(padList[0].second) },
        { static_cast<unsigned int>(padList[1].first), static_cast<unsigned int>(padList[1].second) }
    };

    std::vector<T> expectedOutput;

    if (expectedOutputOverride.has_value())
    {
        /* 
           This was intended as an override for higher dimension tensors where we cannot use
           GenerateExpectedSpaceToBatchOutputNHWC to determine the expected output i.e. tensors rank 4+
        */
        expectedOutput = expectedOutputOverride.value();
    }
    else
    {
        // Generate expected output for space to batch
        expectedOutput = GenerateExpectedSpaceToBatchOutputNHWC<T>(inputData,
                                                                   inputShapeInts,
                                                                   blockShapeInts,
                                                                   padListInts);
    }
    std::map<int, std::vector<T>> inputTensorData    = { {0, inputData} };
    std::map<int, std::vector<T>> expectedOutputData = { {0, expectedOutput} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void BatchToSpaceNdEndToEnd(const std::vector<unsigned int>& inputDims,
                            const std::vector<unsigned int>& blockShape,
                            const std::vector<std::pair<unsigned int, unsigned int>>& cropList,
                            const std::vector<armnn::BackendId>& backends,
                            const std::optional<armnn::TensorShape>& expectedOutputShapeOverride = std::nullopt,
                            const std::optional<std::vector<T>>& expectedOutputOverride = std::nullopt)
{
    using namespace armnn;

    TensorShape inputShape(static_cast<unsigned int>(inputDims.size()), inputDims.data());
    TensorShape outputShape;
    if (expectedOutputShapeOverride.has_value())
    {
        /* 
           This was intended as an override for higher dimension tensors where we cannot use the output shape 
           computation function to determine the shape
           in its current form ComputeBatchToSpaceOutputShape only works for NHWC layout
        */
        outputShape = expectedOutputShapeOverride.value();
    }
    else
    {
        outputShape = ComputeBatchToSpaceOutputShape(inputShape, blockShape, cropList);
    }

    INetworkPtr network = CreateBatchToSpaceNdNetwork<ArmnnType>(inputShape, blockShape, cropList, outputShape);
    CHECK(network);  

    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    unsigned int numInputElements = inputShape.GetNumElements();
    std::vector<T> inputData(numInputElements);

    generateInputDataQuantized(inputData, numInputElements, qScale, qOffset);


    std::vector<unsigned int> inputShapeInts = { static_cast<unsigned int>(inputShape[0]),
                                                 static_cast<unsigned int>(inputShape[1]),
                                                 static_cast<unsigned int>(inputShape[2]),
                                                 static_cast<unsigned int>(inputShape[3])
                                               };

    std::vector<unsigned int> blockShapeInts = { static_cast<unsigned int>(blockShape[0]),
                                                 static_cast<unsigned int>(blockShape[1])
                                               };


    std::vector<std::pair<unsigned int, unsigned int>> cropListInts = {
        { static_cast<unsigned int>(cropList[0].first), static_cast<unsigned int>(cropList[0].second) },
        { static_cast<unsigned int>(cropList[1].first), static_cast<unsigned int>(cropList[1].second) }
    };

    std::vector<T> expectedOutput;

    if (expectedOutputOverride.has_value())
    {
        /* 
           This was intended as an override for higher dimension tensors where we cannot use
           GenerateExpectedBatchToSpaceOutputNHWC to determine the expected output i.e. tensors rank 4+
        */
        expectedOutput = expectedOutputOverride.value();
    }
    else
    {
        // Generate expected output for batch to space
        expectedOutput = GenerateExpectedBatchToSpaceOutputNHWC<T>(inputData,
                                                                   inputShapeInts,
                                                                   blockShapeInts,
                                                                   cropList);
    }
    std::map<int, std::vector<T>> inputTensorData    = { {0, inputData} };
    std::map<int, std::vector<T>> expectedOutputData = { {0, expectedOutput} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}
