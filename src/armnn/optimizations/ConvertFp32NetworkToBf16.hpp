//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "NetworkUtils.hpp"
#include "Optimization.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{
namespace optimizations
{

template <typename LayerT>
inline LayerT* ConvertWeight(Layer* l)
{
    LayerT* layer = PolymorphicDowncast<LayerT*>(l);
    if ((layer->GetType() == LayerType::Convolution2d || layer->GetType() == LayerType::FullyConnected)
         && layer->m_Weight)
    {
        const TensorInfo& info = layer->m_Weight->GetTensorInfo();

        if (info.GetDataType() == DataType::Float32)
        {
            std::vector<BFloat16> newValues(info.GetNumElements());

            armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(
                    layer->m_Weight->template GetConstTensor<float>(),
                    info.GetNumElements(),
                    newValues.data());

            TensorInfo newInfo(info);
            newInfo.SetDataType(DataType::BFloat16);
            ConstTensor newInput(newInfo, newValues);
            layer->m_Weight.reset(new ScopedTensorHandle(newInput));
        }
    }
    return layer;
}

class ConvertFp32NetworkToBf16Impl
{
public:

    void Run(Graph& graph, Layer& layer) const
    {
        // Only convert Float32 To BFloat16 for the Input of Convolution2d layer and FullyConnected layer.
        // And also convert weight data type from Float32 to Bfloat16.
        // Do not convert bias data type.
        if (layer.GetType() == LayerType::Convolution2d)
        {
            if (layer.GetDataType() == DataType::Float32)
            {
                InsertConvertFp32ToBf16LayersBefore(graph,layer);
                ConvertWeight<Convolution2dLayer>(&layer);
            }
        }
        else if (layer.GetType() == LayerType::FullyConnected)
        {
            if (layer.GetDataType() == DataType::Float32)
            {
                InsertConvertFp32ToBf16LayersBefore(graph,layer);
                ConvertWeight<FullyConnectedLayer>(&layer);
            }
        }
    }

protected:
    ConvertFp32NetworkToBf16Impl() = default;
    ~ConvertFp32NetworkToBf16Impl() = default;
};

using Fp32NetworkToBf16Converter = OptimizeForType<Layer, ConvertFp32NetworkToBf16Impl>;

} // namespace optimizations
} // namespace armnn
