//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

using namespace armnn;

namespace
{

inline void StridedSliceInvalidSliceEndToEndTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // Configure a strided slice with a stride the same size as the input but with a ShrinkAxisMask on the first
    // dim of the output to make it too small to hold the specified slice.
    StridedSliceDescriptor descriptor;
    descriptor.m_Begin          = {0, 0};
    descriptor.m_End            = {2, 3};
    descriptor.m_Stride         = {1, 1};
    descriptor.m_BeginMask      = 0;
    descriptor.m_EndMask        = 0;
    descriptor.m_ShrinkAxisMask = 1;
    IConnectableLayer* stridedSlice = net->AddStridedSliceLayer(descriptor);

    IConnectableLayer* output0 = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(stridedSlice->GetInputSlot(0));
    stridedSlice->GetOutputSlot(0).Connect(output0->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 2, 3 }, DataType::Float32, 0.0f, 0, true));
    stridedSlice->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 3 }, DataType::Float32));

    // Attempt to optimize the network and check that the correct exception is thrown
    CHECK_THROWS_AS(Optimize(*net, backends, runtime->GetDeviceSpec()), armnn::LayerValidationException);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
inline void StridedSlice3DMaskedEndToEndTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    StridedSliceDescriptor descriptor;

    descriptor.m_Begin  = {0, 0, 0};
    descriptor.m_End    = {1, 1, 3};
    descriptor.m_Stride = {1, 1, 1};
    descriptor.m_BeginMask      = 5;
    descriptor.m_EndMask        = 5;
    descriptor.m_ShrinkAxisMask = 2;
    descriptor.m_DataLayout     = DataLayout::NHWC;
    descriptor.m_NewAxisMask    = 0;

    IConnectableLayer* stridedSlice = net->AddStridedSliceLayer(descriptor);

    IConnectableLayer* output0 = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(stridedSlice->GetInputSlot(0));
    stridedSlice->GetOutputSlot(0).Connect(output0->GetInputSlot(0));

    unsigned int inputShape[]  = {1, 2, 3};
    unsigned int outputShape[] = {1, 3};

    auto inputTensorInfo = armnn::TensorInfo(3, inputShape, ArmnnType);
    auto outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);
    inputTensorInfo.SetConstant(true);

    const float qScale = 1.0f;
    const int32_t qOffset = 0;
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);

        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    stridedSlice->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Creates structures for input & output.
    std::vector<float> inputData = std::vector<float>(
            {
                    1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 1.0f, 1.0f
            });

    std::vector<T> inputTensorData = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(outputExpected, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensorMap = {{0, inputTensorData}};
    std::map<int, std::vector<T>> expectedOutputMap = {{0, expectedOutput}};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorMap,
                                                expectedOutputMap,
                                                backends,
                                                1);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
inline void StridedSlice4DEndToEndTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    StridedSliceDescriptor descriptor;

    descriptor.m_Begin  = {1, 0, 0, 0};
    descriptor.m_End    = {2, 2, 3, 1};
    descriptor.m_Stride = {1, 1, 1, 1};
    descriptor.m_BeginMask      = 0;
    descriptor.m_EndMask        = 0;
    descriptor.m_ShrinkAxisMask = 0;
    descriptor.m_DataLayout     = DataLayout::NHWC;
    descriptor.m_NewAxisMask    = 0;

    IConnectableLayer* stridedSlice = net->AddStridedSliceLayer(descriptor);

    IConnectableLayer* output0 = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(stridedSlice->GetInputSlot(0));
    stridedSlice->GetOutputSlot(0).Connect(output0->GetInputSlot(0));

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {1, 2, 3, 1};

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);
    inputTensorInfo.SetConstant(true);

    const float qScale = 1.0f;
    const int32_t qOffset = 0;
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);

        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    stridedSlice->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Creates structures for input & output.
    std::vector<float> inputData = std::vector<float>(
        {
            1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

            3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

            5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
        });

    std::vector<float> outputExpected = std::vector<float>(
        {
                3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f
        });

    std::vector<T> inputTensorData = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(outputExpected, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensorMap = {{0, inputTensorData}};
    std::map<int, std::vector<T>> expectedOutputMap = {{0, expectedOutput}};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorMap,
                                                expectedOutputMap,
                                                backends,
                                                1);
}

}