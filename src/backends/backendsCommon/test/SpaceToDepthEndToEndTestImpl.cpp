//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToDepthEndToEndTestImpl.hpp"
#include "ResolveType.hpp"
#include "EndToEndTestImpl.hpp"

#include <armnn/INetwork.hpp>

#include <armnnUtils/Permute.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>

#include <TestUtils.hpp>

#include <doctest/doctest.h>

namespace
{

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

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);

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
                          armnn::TensorInfo& inputTensorInfo,
                          armnn::TensorInfo& outputTensorInfo,
                          std::vector<float>& inputData,
                          std::vector<float>& expectedOutputData,
                          const unsigned int blockSize)
{
    using namespace armnn;

    if (dataLayout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw<float>(inputTensorInfo, inputData);
        PermuteTensorNhwcToNchw<float>(outputTensorInfo, expectedOutputData);
    }

    // Builds up the structure of the network
    INetworkPtr net = CreateSpaceToDepthNetwork<DataType::Float32>(
            inputTensorInfo.GetShape(),
            outputTensorInfo.GetShape(),
            dataLayout,
            blockSize);

    CHECK(net);

    std::map<int, std::vector<float>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<float>> expectedOutputTensorData = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<DataType::Float32, DataType::Float32>(
            move(net),
            inputTensorData,
            expectedOutputTensorData,
            backends);
}

} // anonymous namespace

void SpaceToDepthNhwcEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const unsigned int blockSize = 2;

    TensorShape inputShape{1, 2, 2, 1};
    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);

    TensorShape outputShape{1, 1, 1, 4};
    TensorInfo outputTensorInfo(outputShape, DataType::Float32);

    std::vector<float> inputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         DataLayout::NHWC,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}

void SpaceToDepthNchwEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const unsigned int blockSize = 2;

    TensorShape inputShape{1, 2, 2, 1};
    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);

    TensorShape outputShape{1, 1, 1, 4};
    TensorInfo outputTensorInfo(outputShape, DataType::Float32);

    std::vector<float> inputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         DataLayout::NCHW,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}

void SpaceToDepthNhwcEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const unsigned int blockSize = 2;

    TensorShape inputShape{1, 2, 2, 2};
    TensorShape outputShape{1, 1, 1, 8};

    TensorInfo outputTensorInfo(outputShape, DataType::Float32);
    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);

    std::vector<float> inputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         DataLayout::NHWC,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}

void SpaceToDepthNchwEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const unsigned int blockSize = 2;

    TensorShape inputShape{1, 2, 2, 2};
    TensorShape outputShape{1, 1, 1, 8};

    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);
    TensorInfo outputTensorInfo(outputShape, DataType::Float32);


    std::vector<float> inputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    SpaceToDepthEndToEnd(defaultBackends,
                         DataLayout::NCHW,
                         inputTensorInfo,
                         outputTensorInfo,
                         inputData,
                         expectedOutputData,
                         blockSize);
}
