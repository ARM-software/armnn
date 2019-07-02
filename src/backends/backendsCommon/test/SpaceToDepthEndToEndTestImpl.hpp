//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ResolveType.hpp"
#include "DataLayoutIndexed.hpp"
#include "EndToEndTestImpl.hpp"

#include "armnn/INetwork.hpp"

#include "backendsCommon/test/CommonTestUtils.hpp"

#include <Permute.hpp>
#include <boost/test/unit_test.hpp>

#include <vector>

namespace
{
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void PermuteDataToNCHW(const std::vector<armnn::BackendId>& backends,
                       const armnn::DataLayout& dataLayout,
                       TensorInfo& tensorInfo,
                       std::vector<T>& data)
{
    const armnn::PermutationVector NHWCToNCHW = {0, 2, 3, 1};

    tensorInfo = armnnUtils::Permuted(tensorInfo, NHWCToNCHW);

    std::vector<T> tmp(data.size());
    armnnUtils::Permute(tensorInfo.GetShape(), NHWCToNCHW, data.data(), tmp.data(), sizeof(T));

    data = tmp;
}

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreateSpaceToDepthNetwork(const armnn::TensorShape& inputShape,
                                             const armnn::TensorShape& outputShape,
                                             const armnn::DataLayout dataLayout,
                                             unsigned int blockSize,
                                             const float qScale = 1.0f,
                                             const int32_t qOffset = 0)
{
    using namespace armnn;
    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset);

    armnnUtils::DataLayoutIndexed dimensionIndices(dataLayout);
    if (inputShape[dimensionIndices.GetHeightIndex()] % blockSize!=0
        || inputShape[dimensionIndices.GetWidthIndex()] % blockSize!=0)
    {
        throw InvalidArgumentException("Input shape must be divisible by block size in all spatial dimensions");
    }

    SpaceToDepthDescriptor spaceToDepthDesc;
    spaceToDepthDesc.m_BlockSize = blockSize;
    spaceToDepthDesc.m_DataLayout = dataLayout;

    IConnectableLayer* SpaceToDepth = net->AddSpaceToDepthLayer(spaceToDepthDesc, "SpaceToDepth");
    IConnectableLayer* input        = net->AddInputLayer(0, "input");
    Connect(input, SpaceToDepth, inputTensorInfo, 0, 0);

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
    Connect(SpaceToDepth, output, outputTensorInfo, 0, 0);

    return net;
}

void SpaceToDepthEndToEnd(const std::vector<armnn::BackendId>& backends,
                          const armnn::DataLayout& dataLayout,
                          TensorInfo& inputTensorInfo,
                          TensorInfo& outputTensorInfo,
                          std::vector<float>& inputData,
                          std::vector<float>& expectedOutputData,
                          const unsigned int blockSize)
{
    using namespace armnn;

    if (dataLayout == armnn::DataLayout::NCHW)
    {
        PermuteDataToNCHW<armnn::DataType::Float32>(backends, dataLayout, inputTensorInfo, inputData);
        PermuteDataToNCHW<armnn::DataType::Float32>(backends, dataLayout, outputTensorInfo, expectedOutputData);
    }

    // Builds up the structure of the network
    INetworkPtr net = CreateSpaceToDepthNetwork<armnn::DataType::Float32>(
            inputTensorInfo.GetShape(),
            outputTensorInfo.GetShape(),
            dataLayout,
            blockSize);

    BOOST_TEST_CHECKPOINT("Create a network");

    std::map<int, std::vector<float>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<float>> expectedOutputTensorData = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<armnn::DataType::Float32, armnn::DataType::Float32>(
            move(net),
            inputTensorData,
            expectedOutputTensorData,
            backends);
}

void SpaceToDepthNHWCEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends)
{
    const unsigned int blockSize = 2;

    armnn::TensorShape inputShape{1, 2, 2, 1};
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);

    armnn::TensorShape outputShape{1, 1, 1, 4};
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         armnn::DataLayout::NHWC,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}

void SpaceToDepthNCHWEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends)
{
    const unsigned int blockSize = 2;

    armnn::TensorShape inputShape{1, 2, 2, 1};
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);

    armnn::TensorShape outputShape{1, 1, 1, 4};
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         armnn::DataLayout::NCHW,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}

void SpaceToDepthNHWCEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends)
{
    const unsigned int blockSize = 2;

    armnn::TensorShape inputShape{1, 2, 2, 2};
    armnn::TensorShape outputShape{1, 1, 1, 8};

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);

    std::vector<float> inputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         armnn::DataLayout::NHWC,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}

void SpaceToDepthNCHWEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends)
{
    const unsigned int blockSize = 2;

    armnn::TensorShape inputShape{1, 2, 2, 2};
    armnn::TensorShape outputShape{1, 1, 1, 8};

    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);


    std::vector<float> inputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         armnn::DataLayout::NCHW,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}

} // anonymous namespace
