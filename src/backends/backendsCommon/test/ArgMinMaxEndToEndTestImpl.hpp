//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>


namespace
{

armnn::INetworkPtr CreateArgMinMaxNetwork(const armnn::TensorInfo& inputTensorInfo,
                                          const armnn::TensorInfo& outputTensorInfo,
                                          armnn::ArgMinMaxFunction function,
                                          int axis)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Function = function;
    descriptor.m_Axis = axis;

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* argMinMaxLayer  = network->AddArgMinMaxLayer(descriptor, "ArgMinMax");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, argMinMaxLayer, inputTensorInfo, 0, 0);
    Connect(argMinMaxLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMinMaxEndToEndImpl(const armnn::TensorShape& inputShape,
                           const armnn::TensorShape& outputShape,
                           const std::vector<float>& inputData,
                           const std::vector<int32_t>& expectedOutputData,
                           armnn::ArgMinMaxFunction function,
                           int axis,
                           const std::vector<armnn::BackendId>& backends)
{
    const float qScale  = armnn::IsQuantizedType<T>() ? 2.0f : 1.0f;
    const int32_t qOffset = armnn::IsQuantizedType<T>() ? 2 : 0;

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset, true);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    // quantize data
    std::vector<T> qInputData = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);

    armnn::INetworkPtr network = CreateArgMinMaxNetwork(inputTensorInfo,
                                                        outputTensorInfo,
                                                        function,
                                                        axis);

    EndToEndLayerTestImpl<ArmnnType, armnn::DataType::Signed32>(std::move(network),
                                                                { { 0, qInputData } },
                                                                { { 0, expectedOutputData } },
                                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMaxEndToEndSimple(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 1, 1, 5 };
    const armnn::TensorShape outputShape{ 1, 1, 1 };

    std::vector<float> inputData({ 6.0f, 2.0f, 8.0f, 10.0f, 9.0f });
    std::vector<int32_t> expectedOutputData({ 3 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Max,
                                     -1,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMinEndToEndSimple(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 1, 1, 5 };
    const armnn::TensorShape outputShape{ 1, 1, 1 };

    std::vector<float> inputData({ 6.0f, 2.0f, 8.0f, 10.0f, 9.0f });
    std::vector<int32_t> expectedOutputData({ 1 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Min,
                                     3,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMaxAxis0EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 3, 2, 1, 4 };
    const armnn::TensorShape outputShape{ 2, 1, 4 };

    std::vector<float> inputData({    1.0f,   2.0f,   3.0f,   4.0f,
                                      8.0f,   7.0f,   6.0f,   6.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f });

    std::vector<int32_t> expectedOutputData({ 1, 2, 1, 2,
                                              1, 1, 1, 1 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Max,
                                     0,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMinAxis0EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 3, 2, 1, 4 };
    const armnn::TensorShape outputShape{ 2, 1, 4 };

    std::vector<float> inputData({    1.0f,   2.0f,   3.0f,   4.0f,
                                      8.0f,   7.0f,   6.0f,   6.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f });

    std::vector<int32_t> expectedOutputData({ 0, 0, 0, 0,
                                              0, 0, 0, 0 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Min,
                                     0,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMaxAxis1EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 2, 4 };

    std::vector<float> inputData({    1.0f,   2.0f,   3.0f,   4.0f,
                                      8.0f,   7.0f,   6.0f,   6.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f });

    std::vector<int32_t> expectedOutputData({ 1, 2, 1, 2,
                                              1, 1, 1, 1 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Max,
                                     1,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMinAxis1EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 2, 4 };

    std::vector<float> inputData({    1.0f,   2.0f,   3.0f,   4.0f,
                                      8.0f,   7.0f,   6.0f,   6.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f });

    std::vector<int32_t> expectedOutputData({ 0, 0, 0, 0,
                                              0, 0, 0, 0 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Min,
                                     1,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMaxAxis2EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 3, 4 };

    std::vector<float> inputData({    1.0f,   2.0f,   3.0f,   4.0f,
                                      8.0f,   7.0f,   6.0f,   6.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f });

    std::vector<int32_t> expectedOutputData({ 1, 1, 1, 1,
                                              1, 1, 1, 1,
                                              1, 0, 1, 0});

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Max,
                                     2,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMinAxis2EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 3, 4 };

    std::vector<float> inputData({    1.0f,   2.0f,   3.0f,   4.0f,
                                      8.0f,   7.0f,   6.0f,   6.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f });

    std::vector<int32_t> expectedOutputData({ 0, 0, 0, 0,
                                              0, 0, 0, 0,
                                              0, 1, 0, 1 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Min,
                                     2,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMaxAxis3EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 3, 2 };

    std::vector<float> inputData({    1.0f,   3.0f,   6.0f,   7.0f,
                                      8.0f,   7.0f,   6.0f,   6.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f });

    std::vector<int32_t> expectedOutputData({ 3, 0,
                                              2, 0,
                                              3, 3});

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Max,
                                     3,
                                     backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ArgMinAxis3EndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 3, 2 };

    std::vector<float> inputData({    1.0f,   3.0f,   6.0f,   7.0f,
                                     18.0f,  16.0f,  14.0f,  12.0f,
                                    100.0f,  20.0f, 300.0f,  40.0f,
                                    500.0f, 476.0f, 450.0f, 426.0f,
                                     10.0f, 200.0f,  30.0f, 400.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f });

    std::vector<int32_t> expectedOutputData({ 0, 3,
                                              1, 3,
                                              0, 0 });

    ArgMinMaxEndToEndImpl<ArmnnType>(inputShape,
                                     outputShape,
                                     inputData,
                                     expectedOutputData,
                                     armnn::ArgMinMaxFunction::Min,
                                     3,
                                     backends);
}

} // anonymous namespace
