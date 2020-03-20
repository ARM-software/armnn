//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"
#include "NetworkUtils.hpp"

namespace armnn
{
namespace optimizations
{

class ConvertFp32NetworkToBf16Impl
{
public:
    void Run(Graph& graph, Layer& layer) const
    {
        if(layer.GetType() == LayerType::Input)
        {
            // if the outputs of this layer are DataType::Float32
            // add a ConvertFloat32ToBFloat16 layer after each of the outputs
            if (layer.GetDataType() == DataType::Float32)
            {
                InsertConvertFp32ToBf16LayersAfter(graph, layer);
            }
        }
        else if (layer.GetType() == LayerType::Output)
        {
            // if the inputs of this layer are DataType::Float32
            // add a ConvertBFloat16ToFloat32 layer before each of the inputs
            if (layer.GetDataType() == DataType::Float32)
            {
                // NOTE: We need to call InsertConvertBf16ToFp32LayersBefore with expectCorrectInputType = false
                // here, otherwise it will expect the inputs to be DataType::BFloat16
                InsertConvertBf16ToFp32LayersBefore(graph, layer, false);
            }
        }
        else if (layer.GetType() != LayerType::ConvertFp32ToBf16 && layer.GetType() != LayerType::ConvertBf16ToFp32)
        {
            // if the inputs/outputs of this layer are DataType::Float32
            // change the data type for all inputs and outputs to DataType::BFloat16
            for (auto&& input = layer.BeginInputSlots(); input != layer.EndInputSlots(); ++input)
            {
                // if it is connected to OutputSlot of the InputLayer do not change the DataType of connection
                // InputSlots of the current layer will be updated when conversion layer is inserted after InputLayer
                Layer& base = input->GetConnectedOutputSlot()->GetOwningLayer();
                if (base.GetType() != LayerType::Input)
                {
                    TensorInfo convertInfo = input->GetConnection()->GetTensorInfo();
                    if (convertInfo.GetDataType() == DataType::Float32)
                    {
                        convertInfo.SetDataType(DataType::BFloat16);
                        input->GetConnection()->SetTensorInfo(convertInfo);
                    }
                }
            }

            // change outputs to DataType::BFloat16
            for (auto&& output = layer.BeginOutputSlots(); output != layer.EndOutputSlots(); ++output)
            {
                TensorInfo convertInfo = output->GetTensorInfo();
                if (convertInfo.GetDataType() == DataType::Float32)
                {
                    convertInfo.SetDataType(DataType::BFloat16);
                    output->SetTensorInfo(convertInfo);
                }
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
