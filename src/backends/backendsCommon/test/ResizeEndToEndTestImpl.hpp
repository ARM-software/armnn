//
// Copyright © 2017, 2019-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnnUtils/Permute.hpp>

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <CommonTestUtils.hpp>

#include <map>
#include <vector>

namespace
{

armnn::INetworkPtr CreateResizeNetwork(const armnn::ResizeDescriptor& descriptor,
                                       const armnn::TensorInfo& inputInfo,
                                       const armnn::TensorInfo& outputInfo)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());
    IConnectableLayer* input  = network->AddInputLayer(0, "input");
    IConnectableLayer* resize = network->AddResizeLayer(descriptor, "resize");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, resize, inputInfo, 0, 0);
    Connect(resize, output, outputInfo, 0, 0);

    return network;
}

template <armnn::DataType ArmnnType>
void ResizeEndToEnd(const std::vector<armnn::BackendId>& backends,
                    armnn::DataLayout dataLayout,
                    armnn::ResizeMethod resizeMethod,
                    bool alignCorners = false,
                    bool halfPixel    = false,
                    float tolerance   = 0.000001f)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    constexpr unsigned int inputWidth  = 3u;
    constexpr unsigned int inputHeight = inputWidth;

    constexpr unsigned int outputWidth  = 5u;
    constexpr unsigned int outputHeight = outputWidth;

    TensorShape inputShape  = MakeTensorShape(1, 1, inputHeight, inputWidth, dataLayout);
    TensorShape outputShape = MakeTensorShape(1, 1, outputHeight, outputWidth, dataLayout);

    const float   qScale  = IsQuantizedType<T>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? 50    : 0;

    TensorInfo inputInfo(inputShape, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<float> inputData =
    {
       1.f, 2.f, 3.f,
       4.f, 5.f, 6.f,
       7.f, 8.f, 9.f
    };

    if (alignCorners && halfPixel)
    {
        throw InvalidArgumentException("alignCorners and halfPixel cannot be true simultaneously ");
    }

    std::vector<float> expectedOutputData;
    switch(resizeMethod)
    {
        case ResizeMethod::Bilinear:
        {
            expectedOutputData =
            {
                1.0f, 1.6f, 2.2f, 2.8f, 3.0f,
                2.8f, 3.4f, 4.0f, 4.6f, 4.8f,
                4.6f, 5.2f, 5.8f, 6.4f, 6.6f,
                6.4f, 7.0f, 7.6f, 8.2f, 8.4f,
                7.0f, 7.6f, 8.2f, 8.8f, 9.0f
            };
            break;
        }
        case ResizeMethod::NearestNeighbor:
        {
            if (alignCorners)
            {
                expectedOutputData =
                {
                    1.f, 2.f, 2.f, 3.f, 3.f,
                    4.f, 5.f, 5.f, 6.f, 6.f,
                    4.f, 5.f, 5.f, 6.f, 6.f,
                    7.f, 8.f, 8.f, 9.f, 9.f,
                    7.f, 8.f, 8.f, 9.f, 9.f
                };
            }
            else
            {
                if (halfPixel)
                {
                    expectedOutputData =
                    {
                        1.f, 1.f, 2.f, 3.f, 3.f,
                        1.f, 1.f, 2.f, 3.f, 3.f,
                        4.f, 4.f, 5.f, 6.f, 6.f,
                        7.f, 7.f, 8.f, 9.f, 9.f,
                        7.f, 7.f, 8.f, 9.f, 9.f
                    };
                }
                else
                {
                    expectedOutputData =
                    {
                        1.f, 1.f, 2.f, 2.f, 3.f,
                        1.f, 1.f, 2.f, 2.f, 3.f,
                        4.f, 4.f, 5.f, 5.f, 6.f,
                        4.f, 4.f, 5.f, 5.f, 6.f,
                        7.f, 7.f, 8.f, 8.f, 9.f
                    };
                }
            }
            break;
        }
        default:
        {
            throw InvalidArgumentException("Unrecognized resize method");
        }
    }

    ResizeDescriptor descriptor;
    descriptor.m_TargetWidth  = outputWidth;
    descriptor.m_TargetHeight = outputHeight;
    descriptor.m_Method       = resizeMethod;
    descriptor.m_DataLayout   = dataLayout;
    descriptor.m_AlignCorners = alignCorners;
    descriptor.m_HalfPixelCenters = halfPixel;


    // swizzle data if needed
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        constexpr size_t dataTypeSize = sizeof(float);
        const armnn::PermutationVector nchwToNhwc = { 0, 3, 1, 2 };

        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputInfo.GetShape(), nchwToNhwc, inputData.data(), tmp.data(), dataTypeSize);
        inputData = tmp;
    }

    // quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    INetworkPtr network = CreateResizeNetwork(descriptor, inputInfo, outputInfo);

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                { { 0, qInputData } },
                                                { { 0, qExpectedOutputData } },
                                                backends,
                                                tolerance);
}

} // anonymous namespace

template <armnn::DataType ArmnnType>
void ResizeBilinearEndToEnd(const std::vector<armnn::BackendId>& backends,
                            armnn::DataLayout dataLayout,
                            float tolerance = 0.000001f)
{
    ResizeEndToEnd<ArmnnType>(backends, dataLayout, armnn::ResizeMethod::Bilinear, false, false, tolerance);
}

template<armnn::DataType ArmnnType>
void ResizeNearestNeighborEndToEnd(const std::vector<armnn::BackendId>& backends,
                                   armnn::DataLayout dataLayout,
                                   bool alignCorners = false,
                                   bool halfPixel = false)
{
    ResizeEndToEnd<ArmnnType>(backends, dataLayout, armnn::ResizeMethod::NearestNeighbor, alignCorners, halfPixel);
}
