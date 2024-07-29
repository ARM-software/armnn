//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/INetwork.hpp>

#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreateReduceNetwork(const armnn::TensorShape& inputShape,
                                        const armnn::TensorShape& outputShape,
                                        const armnn::ReduceDescriptor& descriptor,
                                        const float qScale = 1.0f,
                                        const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);


    IConnectableLayer* reduce = network->AddReduceLayer(descriptor, "reduce");
    IConnectableLayer* input   = network->AddInputLayer(0, "input");
    IConnectableLayer* output  = network->AddOutputLayer(0, "output");

    Connect(input, reduce, inputTensorInfo, 0, 0);
    Connect(reduce, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReduceEndToEnd2d(const std::vector<armnn::BackendId>& backends,
                       ReduceOperation reduceOperation,
                       bool keepDims = false)
{
    using namespace armnn;

    ReduceDescriptor descriptor;
    descriptor.m_KeepDims        = keepDims;
    descriptor.m_vAxis           = { 0 };
    descriptor.m_ReduceOperation = reduceOperation;

    TensorShape inputShape  = { 2, 3 };
    TensorShape outputShape = inputShape;

    if (keepDims)
    {
        outputShape[descriptor.m_vAxis[0]] = 1;
    }
    else
    {
        outputShape = { 3 };
    }

    INetworkPtr network = CreateReduceNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> floatInputData  =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    std::vector<float> floatOutputData;

    switch(reduceOperation)
    {
        case ReduceOperation::Sum:
            floatOutputData = 
            {
                5.0f, 7.0f, 9.0f
            };
            break;
        case ReduceOperation::Mean:
            floatOutputData =
            {
                5.0f/2.f, 7.0f/2.f, 9.0f/2.f
            };
            break;
        default:
            throw armnn::Exception("ReduceEndToEnd2d: Reduce Operation not implemented.");
    }

    std::vector<T> inputData  = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData    = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReduceEndToEnd3d(const std::vector<armnn::BackendId>& backends,
                      ReduceOperation reduceOperation,
                      bool keepDims = false)
{
    using namespace armnn;

    ReduceDescriptor descriptor;
    descriptor.m_KeepDims          = keepDims;
    descriptor.m_vAxis             = { 1 };
    descriptor.m_ReduceOperation   = reduceOperation;

    TensorShape inputShape  = { 2, 2, 3 };
    TensorShape outputShape = inputShape;

    if (keepDims)
    {
        outputShape[descriptor.m_vAxis[0]] = 1;
    }
    else
    {
        outputShape = { 2, 3 };
    }

    INetworkPtr network = CreateReduceNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> floatInputData  =
    {
         1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,

         7.0f,  8.0f,  9.0f,
        10.0f, 11.0f, 12.0f
    };

    std::vector<float> floatOutputData;

    switch(reduceOperation)
    {
        case ReduceOperation::Sum:
            floatOutputData =
            {
                5.0f,  7.0f,  9.0f,
                17.0f, 19.0f, 21.0f
            };
            break;
        case ReduceOperation::Mean:
            floatOutputData =
            {
                5.0f/2.f,  7.0f/2.f,  9.0f/2.f,
                17.0f/2.f, 19.0f/2.f, 21.0f/2.f
            };
            break;
        default:
            throw armnn::Exception("ReduceEndToEnd3d: Reduce Operation not implemented.");
    }

    std::vector<T> inputData  = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData    = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReduceEndToEnd4d(const std::vector<armnn::BackendId>& backends,
                      ReduceOperation reduceOperation,
                      bool keepDims = false)
{
    using namespace armnn;

    ReduceDescriptor descriptor;
    descriptor.m_KeepDims        = keepDims;
    descriptor.m_vAxis           = { 3 };
    descriptor.m_ReduceOperation = reduceOperation;

    TensorShape inputShape  = { 1, 1, 1, 5 };
    TensorShape outputShape = inputShape;

    if (keepDims)
    {
        outputShape[descriptor.m_vAxis[0]] = 1;
    }
    else
    {
        outputShape = { 1, 1, 1 };
    }

    INetworkPtr network = CreateReduceNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> floatInputData = 
    {
        5.0f, 2.0f, 8.0f, 10.0f, 9.0f
    };

    std::vector<float> floatOutputData;

    switch(reduceOperation)
    {
        case ReduceOperation::Sum:
            floatOutputData = { 34.0f };
            break;
        case ReduceOperation::Mean:
            floatOutputData = { 34.0f/5.f };
            break;
        default:
            throw armnn::Exception("ReduceEndToEnd4d: Reduce Operation not implemented.");
    }

    std::vector<T> inputData  = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData    = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReduceEndToEndEmptyAxis(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape  = { 2, 1, 3 };
    const TensorShape& outputShape = {       1 };

    ReduceDescriptor descriptor;

    INetworkPtr network = CreateReduceNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> floatInputData  =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    std::vector<float> floatOutputData = { 21.0f };

    std::vector<T> inputData  = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData    = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}
} // anonymous namespace
