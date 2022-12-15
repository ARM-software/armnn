//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>

#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

using namespace armnn;

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreatePooling2dNetwork(const armnn::TensorShape& inputShape,
                                          const armnn::TensorShape& outputShape,
                                          PaddingMethod padMethod = PaddingMethod::Exclude,
                                          PoolingAlgorithm poolAlg = PoolingAlgorithm::Max,
                                          const float qScale = 1.0f,
                                          const int32_t qOffset = 0)
{
    INetworkPtr network(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset, true);

    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = poolAlg;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = padMethod;
    descriptor.m_DataLayout = DataLayout::NHWC;

    IConnectableLayer* pool = network->AddPooling2dLayer(descriptor, "pool");
    IConnectableLayer* input = network->AddInputLayer(0, "input");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, pool, inputTensorInfo, 0, 0);
    Connect(pool, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void MaxPool2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                       PaddingMethod padMethod = PaddingMethod::Exclude)
{
    const TensorShape& inputShape = { 1, 3, 3, 1 };
    const TensorShape& outputShape = { 1, 3, 3, 1 };

    INetworkPtr network = CreatePooling2dNetwork<ArmnnType>(inputShape, outputShape, padMethod);

    CHECK(network);

    std::vector<T> inputData{ 1, 2, 3,
                              4, 5, 6,
                              7, 8, 9 };
    std::vector<T> expectedOutput{ 5, 6, 6,
                                   8, 9, 9,
                                   8, 9, 9 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void MaxPool2dEndToEndFloat16(const std::vector<armnn::BackendId>& backends)
{
    using namespace half_float::literal;
    using Half = half_float::half;

    const TensorShape& inputShape = { 1, 3, 3, 1 };
    const TensorShape& outputShape = { 1, 3, 3, 1 };

    INetworkPtr network = CreatePooling2dNetwork<ArmnnType>(inputShape, outputShape);
    CHECK(network);

    std::vector<Half> inputData{ 1._h, 2._h, 3._h,
                                 4._h, 5._h, 6._h,
                                 7._h, 8._h, 9._h };
    std::vector<Half> expectedOutput{ 5._h, 6._h, 6._h,
                                      8._h, 9._h, 9._h,
                                      8._h, 9._h, 9._h };

    std::map<int, std::vector<Half>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<Half>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void AvgPool2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                       PaddingMethod padMethod = PaddingMethod::Exclude)
{
    const TensorShape& inputShape =  { 1, 3, 3, 1 };
    const TensorShape& outputShape =  { 1, 3, 3, 1 };

    INetworkPtr network = CreatePooling2dNetwork<ArmnnType>(
        inputShape, outputShape, padMethod, PoolingAlgorithm::Average);
    CHECK(network);

    std::vector<T> inputData{ 1, 2, 3,
                              4, 5, 6,
                              7, 8, 9 };
    std::vector<T> expectedOutput;
    if (padMethod == PaddingMethod::Exclude)
    {
        expectedOutput  = { 3.f , 3.5f, 4.f ,
                            4.5f, 5.f , 5.5f,
                            6.f , 6.5f, 7.f  };
    }
    else
    {
        expectedOutput  = { 1.33333f, 2.33333f, 1.77778f,
                            3.f     , 5.f     , 3.66667f,
                            2.66667f, 4.33333f, 3.11111f };
    }

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                inputTensorData,
                                                expectedOutputData,
                                                backends,
                                                0.00001f);
}

template<armnn::DataType ArmnnType>
void AvgPool2dEndToEndFloat16(const std::vector<armnn::BackendId>& backends,
                              PaddingMethod padMethod = PaddingMethod::IgnoreValue)
{
    using namespace half_float::literal;
    using Half = half_float::half;

    const TensorShape& inputShape =  { 1, 3, 3, 1 };
    const TensorShape& outputShape =  { 1, 3, 3, 1 };

    INetworkPtr network = CreatePooling2dNetwork<ArmnnType>(
        inputShape, outputShape, padMethod, PoolingAlgorithm::Average);
    CHECK(network);

    std::vector<Half> inputData{ 1._h, 2._h, 3._h,
                                 4._h, 5._h, 6._h,
                                 7._h, 8._h, 9._h };
    std::vector<Half> expectedOutput{ 1.33333_h, 2.33333_h, 1.77778_h,
                                      3._h     , 5._h     , 3.66667_h,
                                      2.66667_h, 4.33333_h, 3.11111_h };

    std::map<int, std::vector<Half>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<Half>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                inputTensorData,
                                                expectedOutputData,
                                                backends,
                                                0.00001f);
}

} // anonymous namespace
