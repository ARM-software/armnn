//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "EndToEndTestImpl.hpp"
#include "LogSoftmaxEndToEndTestImpl.hpp"

#include <armnn/INetwork.hpp>

#include <TestUtils.hpp>

#include <doctest/doctest.h>

namespace {

template <typename armnn::DataType DataType>
armnn::INetworkPtr CreateLogSoftmaxNetwork(const armnn::TensorShape& inputShape,
                                           const armnn::TensorShape& outputShape,
                                           const float beta,
                                           const int axis,
                                           const float qScale = 1.0f,
                                           const int32_t qOffset = 0)
{
    using namespace armnn;

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);

    LogSoftmaxDescriptor logSoftmaxDesc;
    logSoftmaxDesc.m_Beta = beta;
    logSoftmaxDesc.m_Axis = axis;

    IConnectableLayer* logSoftmax = net->AddLogSoftmaxLayer(logSoftmaxDesc, "Log_Softmax");
    IConnectableLayer* input = net->AddInputLayer(0, "input");
    Connect(input, logSoftmax, inputTensorInfo, 0, 0);

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
    Connect(logSoftmax, output, outputTensorInfo, 0, 0);

    return net;
}

void LogSoftmaxEndToEnd(const std::vector<armnn::BackendId>& backends,
                        armnn::TensorInfo& inputTensorInfo,
                        armnn::TensorInfo& outputTensorInfo,
                        std::vector<float>& inputData,
                        std::vector<float>& expectedOutputData,
                        const float beta,
                        const int axis)
{
    using namespace armnn;

    // Builds up the structure of the network
    INetworkPtr net = CreateLogSoftmaxNetwork<DataType::Float32>(inputTensorInfo.GetShape(),
                                                                 outputTensorInfo.GetShape(),
                                                                 beta,
                                                                 axis);

    CHECK(net);

    std::map<int, std::vector<float>> inputTensorData = { {0, inputData} };
    std::map<int, std::vector<float>> expectedOutputTensorData = { {0, expectedOutputData} };

    EndToEndLayerTestImpl<DataType::Float32, DataType::Float32>(move(net),
                                                                inputTensorData,
                                                                expectedOutputTensorData,
                                                                backends);
}

} // anonymous namespace

void LogSoftmaxEndToEndTest(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const float beta = 10.0f; // non-default beta
    const int axis   = 3;     // positive axis

    const TensorShape inputShape{1, 1, 2, 4};
    TensorInfo inputTensorInfo(inputShape, DataType::Float32);

    const TensorShape outputShape{1, 1, 2, 4};
    TensorInfo outputTensorInfo(outputShape, DataType::Float32);

    std::vector<float> inputData = std::vector<float>({
        0.0f, -0.6f, 0.2f, 0.4f,
        0.3f, -0.2f, 1.0f, 0.1f
    });

    std::vector<float> expectedOutputData = std::vector<float>({
        -4.14297f, -10.14297f, -2.14297f, -0.14297f,
        -7.00104f, -12.00104f, -0.00104087f, -9.00104f
    });

    LogSoftmaxEndToEnd(defaultBackends,
                       inputTensorInfo,
                       outputTensorInfo,
                       inputData,
                       expectedOutputData,
                       beta,
                       axis);
}