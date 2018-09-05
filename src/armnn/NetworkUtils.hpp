//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Graph.hpp"

namespace armnn
{

inline std::vector<ConvertFp16ToFp32Layer*> InsertConvertFp16ToFp32LayersBefore(Graph& graph, Layer& layer)
{
    std::vector<ConvertFp16ToFp32Layer*> convertLayers;
    convertLayers.reserve(layer.GetNumInputSlots());

    for (auto&& inputSlot = layer.BeginInputSlots(); inputSlot != layer.EndInputSlots(); ++inputSlot)
    {
        // Insert FP16 to FP32 converter layer before the layer
        const std::string name =
            std::string("convert_fp16_to_fp32-" + std::to_string(inputSlot->GetSlotIndex()) + "-") + layer.GetName();
        ConvertFp16ToFp32Layer* convertLayer =
            graph.InsertNewLayer<ConvertFp16ToFp32Layer>(*inputSlot, name.c_str());

        // Sets output tensor info for the convert layer
        TensorInfo convertInfo = convertLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
        convertInfo.SetDataType(DataType::Float32);

        convertLayer->GetOutputSlot().SetTensorInfo(convertInfo);

        convertLayers.emplace_back(convertLayer);
    }

    // Sets the output tensor info for the unsupported layer
    auto UpdateTensorInfo = [](auto& outputSlot)
    {
        // Copy original tensor info and change data type to FP32
        TensorInfo newTensorInfo = outputSlot.GetTensorInfo();
        newTensorInfo.SetDataType(DataType::Float32);

        outputSlot.SetTensorInfo(newTensorInfo);
    };

    std::for_each(layer.BeginOutputSlots(), layer.EndOutputSlots(), UpdateTensorInfo);

    return convertLayers;
}

inline std::vector<ConvertFp32ToFp16Layer*> InsertConvertFp32ToFp16LayersAfter(Graph& graph, Layer& layer)
{
    std::vector<ConvertFp32ToFp16Layer*> convertLayers;
    convertLayers.reserve(layer.GetNumOutputSlots());

    int index = 0;
    // Change outputs to DataType::Float16
    for (auto&& outputSlot = layer.BeginOutputSlots(); outputSlot != layer.EndOutputSlots(); ++outputSlot)
    {
        BOOST_ASSERT(outputSlot->GetTensorInfo().GetDataType() == DataType::Float32);

        // Insert FP32 to FP16 converter layer after the layer
        const std::string name =
            std::string("convert_fp32_to_fp16-" + std::to_string(index++) + "-") + layer.GetName();
        ConvertFp32ToFp16Layer* convertLayer =
            graph.InsertNewLayer<ConvertFp32ToFp16Layer>(*outputSlot, name.c_str());

        // Sets output tensor info for the convert layer.
        TensorInfo convertInfo = convertLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
        convertInfo.SetDataType(DataType::Float16);

        convertLayer->GetOutputSlot().SetTensorInfo(convertInfo);

        convertLayers.emplace_back(convertLayer);
    }

    return convertLayers;
}

} //namespace armnn