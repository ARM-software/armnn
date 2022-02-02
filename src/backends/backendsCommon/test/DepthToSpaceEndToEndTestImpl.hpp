//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ResolveType.hpp>


#include <armnnUtils/QuantizeHelper.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>

namespace
{

armnn::INetworkPtr CreateDepthToSpaceNetwork(const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo,
                                             const armnn::DepthToSpaceDescriptor& descriptor)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    IConnectableLayer* input        = network->AddInputLayer(0, "input");
    IConnectableLayer* depthToSpace = network->AddDepthToSpaceLayer(descriptor, "depthToSpace");
    IConnectableLayer* output       = network->AddOutputLayer(0, "output");

    Connect(input, depthToSpace, inputInfo, 0, 0);
    Connect(depthToSpace, output, outputInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DepthToSpaceEndToEndImpl(const std::vector<armnn::BackendId>& backends,
                              const DepthToSpaceDescriptor& descriptor,
                              const armnn::TensorShape& nhwcInputShape,
                              const armnn::TensorShape& nhwcOutputShape,
                              const std::vector<float>& floatInputData,
                              const std::vector<float>& floatExpectedOutputData)
{
    using namespace armnn;

    TensorInfo inputInfo(nhwcInputShape, ArmnnType);
    inputInfo.SetConstant(true);
    TensorInfo outputInfo(nhwcOutputShape, ArmnnType);

    constexpr float   qScale  = 0.25f;
    constexpr int32_t qOffset = 128;

    // Set quantization parameters for quantized types
    if (IsQuantizedType<T>())
    {
        inputInfo.SetQuantizationScale(qScale);
        inputInfo.SetQuantizationOffset(qOffset);
        outputInfo.SetQuantizationScale(qScale);
        outputInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> inputData          = armnnUtils::QuantizedVector<T>(floatInputData, qScale, qOffset);
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData, qScale, qOffset);

    // Permute tensors from NHWC to NCHW (if needed)
    if (descriptor.m_DataLayout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputInfo, inputData);
        PermuteTensorNhwcToNchw(outputInfo, expectedOutputData);
    }

    INetworkPtr network = CreateDepthToSpaceNetwork(inputInfo, outputInfo, descriptor);
    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                { { 0, inputData } },
                                                { { 0, expectedOutputData } },
                                                backends);
}

} // anonymous namespace

template<armnn::DataType ArmnnType>
void DepthToSpaceEndToEnd(const std::vector<armnn::BackendId>& defaultBackends,
                          armnn::DataLayout dataLayout)
{
    using namespace armnn;

    TensorShape inputShape  = { 2, 2, 2, 4 };
    TensorShape outputShape = { 2, 4, 4, 1 };

    std::vector<float> inputData =
    {
         1.f,  2.f,  3.f,  4.f,
         5.f,  6.f,  7.f,  8.f,
         9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f,

        17.f, 18.f, 19.f, 20.f,
        21.f, 22.f, 23.f, 24.f,
        25.f, 26.f, 27.f, 28.f,
        29.f, 30.f, 31.f, 32.f
    };

    std::vector<float> expectedOutputData =
    {
         1.f,  2.f,  5.f,  6.f,
         3.f,  4.f,  7.f,  8.f,
         9.f, 10.f, 13.f, 14.f,
        11.f, 12.f, 15.f, 16.f,

        17.f, 18.f, 21.f, 22.f,
        19.f, 20.f, 23.f, 24.f,
        25.f, 26.f, 29.f, 30.f,
        27.f, 28.f, 31.f, 32.f
    };

    DepthToSpaceEndToEndImpl<ArmnnType>(defaultBackends,
                                        DepthToSpaceDescriptor(2, dataLayout),
                                        inputShape,
                                        outputShape,
                                        inputData,
                                        expectedOutputData);
}
