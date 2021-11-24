//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "InstanceNormalizationEndToEndTestImpl.hpp"
#include "EndToEndTestImpl.hpp"
#include "ResolveType.hpp"

#include <armnnUtils/Permute.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/INetwork.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>

#include <TestUtils.hpp>

#include <doctest/doctest.h>

namespace
{

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreateInstanceNormalizationNetwork(const armnn::TensorShape& inputShape,
                                                      const armnn::TensorShape& outputShape,
                                                      const armnn::DataLayout dataLayout,
                                                      const float gamma,
                                                      const float beta,
                                                      const float eps,
                                                      const float qScale = 1.0f,
                                                      const int32_t qOffset = 0)
{
    using namespace armnn;

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);

    InstanceNormalizationDescriptor instanceNormalizationDesc;
    instanceNormalizationDesc.m_Gamma = gamma;
    instanceNormalizationDesc.m_Beta  = beta;
    instanceNormalizationDesc.m_Eps   = eps;
    instanceNormalizationDesc.m_DataLayout = dataLayout;

    IConnectableLayer* instanceNormalization = net->AddInstanceNormalizationLayer(instanceNormalizationDesc,
                                                                                  "InstanceNormalization");
    IConnectableLayer* input = net->AddInputLayer(0, "input");
    Connect(input, instanceNormalization, inputTensorInfo, 0, 0);

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
    Connect(instanceNormalization, output, outputTensorInfo, 0, 0);

    return net;
}

void InstanceNormalizationEndToEnd(const std::vector<armnn::BackendId>& backends,
                                   const armnn::DataLayout& dataLayout,
                                   armnn::TensorInfo& inputTensorInfo,
                                   armnn::TensorInfo& outputTensorInfo,
                                   std::vector<float>& inputData,
                                   std::vector<float>& expectedOutputData,
                                   const float gamma,
                                   const float beta,
                                   const float eps)
{
    using namespace armnn;

    if (dataLayout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw<float>(inputTensorInfo, inputData);
        PermuteTensorNhwcToNchw<float>(outputTensorInfo, expectedOutputData);
    }

    // Builds up the structure of the network
    INetworkPtr net = CreateInstanceNormalizationNetwork<DataType::Float32>(inputTensorInfo.GetShape(),
                                                                            outputTensorInfo.GetShape(),
                                                                            dataLayout,
                                                                            gamma,
                                                                            beta,
                                                                            eps);

    CHECK(net);

    std::map<int, std::vector<float>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<float>> expectedOutputTensorData = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<DataType::Float32, DataType::Float32>(move(net),
                                                                inputTensorData,
                                                                expectedOutputTensorData,
                                                                backends);
}

} // anonymous namespace

void InstanceNormalizationNhwcEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const float eps       = 0.0001f;
    const float beta      = 0.0f;
    const float gamma     = 1.0f;

    TensorShape inputShape{2, 2, 2, 2};
    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);

    TensorShape outputShape{2, 2, 2, 2};
    TensorInfo outputTensorInfo(outputShape, DataType::Float32);

    std::vector<float> inputData = std::vector<float>(
    {
        // Batch 0, Height 0, Width 0 x Channel (2)
        0.f,  1.f,
        // Batch 0, Height 0, Width 1 x Channel (2)
        0.f,  2.f,

        // Batch 0, Height 1, Width 0 x Channel (2)
        0.f,  2.f,
        // Batch 0, Height 1, Width 1 x Channel (2)
        0.f,  4.f,

        // Batch 1, Height 0, Width 0 x Channel (2)
        1.f, -1.f,
        // Batch 1, Height 0, Width 1 x Channel (2)
        -1.f,  2.f,

        // Batch 1, Height 1, Width 0 x Channel (2)
        -1.f, -2.f,
        // Batch 1, Height 1, Width 1 x Channel (2)
        1.f,  4.f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        // Batch 0, Height 0, Width 0 x Channel (2)
        0.f, -1.1470304f,
        // Batch 0, Height 0, Width 1 x Channel (2)
        0.f, -0.22940612f,
        // Batch 0, Height 1, Width 0 x Channel (2)
        0.f, -0.22940612f,
        // Batch 0, Height 1, Width 1 x Channel (2)
        0.f,  1.6058424f,

        // Batch 1, Height 0, Width 0 x Channel (2)
        0.99995005f, -0.7337929f,
        // Batch 1, Height 0, Width 1 x Channel (2)
        -0.99995005f,  0.52413774f,

        // Batch 1, Height 1, Width 0 x Channel (2)
        -0.99995005f, -1.1531031f,
        // Batch 1, Height 1, Width 1 x Channel (2)
        0.99995005f,  1.3627582f
    });

    InstanceNormalizationEndToEnd(defaultBackends,
                                  DataLayout::NHWC,
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  gamma,
                                  beta,
                                  eps);
}

void InstanceNormalizationNchwEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const float eps       = 0.0001f;
    const float beta      = 0.0f;
    const float gamma     = 1.0f;

    TensorShape inputShape{2, 2, 2, 2};
    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);

    TensorShape outputShape{2, 2, 2, 2};
    TensorInfo outputTensorInfo(outputShape, DataType::Float32);

    std::vector<float> inputData = std::vector<float>(
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            0.f,  1.f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            0.f,  2.f,

            // Batch 0, Height 1, Width 0 x Channel (2)
            0.f,  2.f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            0.f,  4.f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            1.f, -1.f,
            // Batch 1, Height 0, Width 1 x Channel (2)
            -1.f,  2.f,

            // Batch 1, Height 1, Width 0 x Channel (2)
            -1.f, -2.f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            1.f,  4.f
        });

    std::vector<float> expectedOutputData = std::vector<float>(
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            0.f, -1.1470304f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            0.f, -0.22940612f,
            // Batch 0, Height 1, Width 0 x Channel (2)
            0.f, -0.22940612f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            0.f,  1.6058424f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            0.99995005f, -0.7337929f,
            // Batch 1, Height 0, Width 1 x Channel (2)
            -0.99995005f,  0.52413774f,

            // Batch 1, Height 1, Width 0 x Channel (2)
            -0.99995005f, -1.1531031f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            0.99995005f,  1.3627582f
        });


    InstanceNormalizationEndToEnd(defaultBackends,
                                  DataLayout::NCHW,
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  gamma,
                                  beta,
                                  eps);
}

void InstanceNormalizationNhwcEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const float eps        = 0.0001f;
    const float beta       = 10.0f;
    const float gamma      = 2.0f;

    TensorShape inputShape{2, 2, 2, 2};
    TensorShape outputShape{2, 2, 2, 2};

    TensorInfo outputTensorInfo(outputShape, DataType::Float32);
    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);

    std::vector<float> inputData = std::vector<float>(
    {
        // Batch 0, Height 0, Width 0 x Channel (2)
        0.f,  1.f,
        // Batch 0, Height 0, Width 1 x Channel (2)
        0.f,  2.f,

        // Batch 0, Height 1, Width 0 x Channel (2)
        0.f,  2.f,
        // Batch 0, Height 1, Width 1 x Channel (2)
        0.f,  4.f,

        // Batch 1, Height 0, Width 0 x Channel (2)
        1.f, -1.f,
        // Batch 1, Height 0, Width 1 x Channel (2)
        -1.f,  2.f,

        // Batch 1, Height 1, Width 0 x Channel (2)
        -1.f, -2.f,
        // Batch 1, Height 1, Width 1 x Channel (2)
        1.f,  4.f
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        // Batch 0, Height 0, Width 0 x Channel (2)
        10.f,     7.7059393f,
        // Batch 0, Height 0, Width 1 x Channel (2)
        10.f,     9.541187f,

        // Batch 0, Height 1, Width 0 x Channel (2)
        10.f,     9.541187f,
        // Batch 0, Height 1, Width 1 x Channel (2)
        10.f,     13.211685f,

        // Batch 1, Height 0, Width 0 x Channel (2)
        11.9999f, 8.532414f,
        // Batch 1, Height 0, Width 1 x Channel (2)
        8.0001f,  11.048275f,

        // Batch 1, Height 1, Width 0 x Channel (2)
        8.0001f,  7.693794f,
        // Batch 1, Height 1, Width 1 x Channel (2)
        11.9999f, 12.725516f
    });

    InstanceNormalizationEndToEnd(defaultBackends,
                                  DataLayout::NHWC,
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  gamma,
                                  beta,
                                  eps);
}

void InstanceNormalizationNchwEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends)
{
    using namespace armnn;

    const float eps        = 0.0001f;
    const float beta       = 10.0f;
    const float gamma      = 2.0f;

    TensorShape inputShape{2, 2, 2, 2};
    TensorShape outputShape{2, 2, 2, 2};

    TensorInfo outputTensorInfo(outputShape, DataType::Float32);
    TensorInfo inputTensorInfo(inputShape, DataType::Float32, 0.0f, 0, true);

    std::vector<float> inputData = std::vector<float>(
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            0.f,  1.f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            0.f,  2.f,

            // Batch 0, Height 1, Width 0 x Channel (2)
            0.f,  2.f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            0.f,  4.f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            1.f, -1.f,
            // Batch 1, Height 0, Width 1 x Channel (2)
            -1.f,  2.f,

            // Batch 1, Height 1, Width 0 x Channel (2)
            -1.f, -2.f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            1.f,  4.f
        });

    std::vector<float> expectedOutputData = std::vector<float>(
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            10.f,     7.7059393f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            10.f,     9.541187f,

            // Batch 0, Height 1, Width 0 x Channel (2)
            10.f,     9.541187f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            10.f,     13.211685f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            11.9999f, 8.532414f,
            // Batch 1, Height 0, Width 1 x Channel (2)
            8.0001f,  11.048275f,

            // Batch 1, Height 1, Width 0 x Channel (2)
            8.0001f,  7.693794f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            11.9999f, 12.725516f
        });

    InstanceNormalizationEndToEnd(defaultBackends,
                                  DataLayout::NCHW,
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  gamma,
                                  beta,
                                  eps);
}
