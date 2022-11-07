//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkUtils.hpp"

#include "SubgraphViewSelector.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/BackendRegistry.hpp>

namespace armnn
{

namespace
{

void UpdateOutputSlotToFp32(OutputSlot& outputSlot)
{
    const TensorInfo& origTensorInfo = outputSlot.GetTensorInfo();
    TensorInfo newTensorInfo(origTensorInfo);
    newTensorInfo.SetDataType(DataType::Float32);
    outputSlot.SetTensorInfo(newTensorInfo);
}

void ChangeOutputFp16ToFp32(Layer& layer)
{
    for (auto&& outputSlot = layer.BeginOutputSlots(); outputSlot != layer.EndOutputSlots(); ++outputSlot)
    {
        if (outputSlot->GetTensorInfo().GetDataType() == DataType::Float16)
        {
            UpdateOutputSlotToFp32(*outputSlot);
        }
    }
}

} // anonymous namespace

std::vector<ConvertFp16ToFp32Layer*> InsertConvertFp16ToFp32LayersBefore(Graph& graph,
                                                                         Layer& layer,
                                                                         bool expectCorrectInputType)
{
    std::vector<ConvertFp16ToFp32Layer*> convertLayers;
    convertLayers.reserve(layer.GetNumInputSlots());

    // Insert a ConvertFp16ToFp32Layer before each input slot
    for (auto&& inputSlot = layer.BeginInputSlots(); inputSlot != layer.EndInputSlots(); ++inputSlot)
    {
        bool allowInsert = true;
        if (expectCorrectInputType)
        {
            // Only insert ConvertFp16ToFp32Layer before FP16 input slots
            OutputSlot* connectedOutputSlot = inputSlot->GetConnectedOutputSlot();
            allowInsert =
                connectedOutputSlot && connectedOutputSlot->GetTensorInfo().GetDataType() == DataType::Float16;
        }

        if (allowInsert)
        {
            const std::string name =
                std::string("convert_fp16_to_fp32-" + std::to_string(inputSlot->GetSlotIndex()) + "-") +
                layer.GetName();
            ConvertFp16ToFp32Layer* convertLayer =
                graph.InsertNewLayer<ConvertFp16ToFp32Layer>(*inputSlot, name.c_str());

            TensorInfo convertInfo = convertLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
            convertInfo.SetDataType(DataType::Float32);

            convertLayer->GetOutputSlot().SetTensorInfo(convertInfo);

            convertLayers.emplace_back(convertLayer);
        }
    }

    return convertLayers;
}

std::vector<ConvertFp32ToFp16Layer*> InsertConvertFp32ToFp16LayersAfter(Graph& graph, Layer& layer)
{
    const unsigned int numOutputSlots = layer.GetNumOutputSlots();

    std::vector<ConvertFp32ToFp16Layer*> convertLayers;
    convertLayers.reserve(numOutputSlots);

    // Update FP16 output slots to FP32 on current layer
    ChangeOutputFp16ToFp32(layer);

    // Insert a ConvertFp32ToFp16Layer after each FP32 output slot
    for (unsigned int slotIndex = 0u; slotIndex < numOutputSlots; ++slotIndex)
    {
        OutputSlot& outputSlot = layer.GetOutputSlot(slotIndex);
        if(outputSlot.GetTensorInfo().GetDataType() == DataType::Float32)
        {
            const std::string name =
                std::string("convert_fp32_to_fp16-" + std::to_string(slotIndex) + "-") + layer.GetName();
            ConvertFp32ToFp16Layer* convertLayer =
                graph.InsertNewLayer<ConvertFp32ToFp16Layer>(outputSlot, name.c_str());

            TensorInfo convertInfo = convertLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
            convertInfo.SetDataType(DataType::Float16);

            convertLayer->GetOutputSlot().SetTensorInfo(convertInfo);

            convertLayers.emplace_back(convertLayer);
        }
    }

    return convertLayers;
}

std::vector<DebugLayer*> InsertDebugLayerAfter(Graph& graph, Layer& layer, bool toFile)
{
    std::vector<DebugLayer*> debugLayers;
    debugLayers.reserve(layer.GetNumOutputSlots());

    // Connect a DebugLayer to each output slot of the layer
    uint32_t outputSlotIdx = 0;
    for (auto outputSlot = layer.BeginOutputSlots(); outputSlot != layer.EndOutputSlots(); ++outputSlot)
    {
        const std::string debugName = std::string("DebugLayerAfter") + layer.GetNameStr() + "_" +
            std::to_string(outputSlotIdx);

        DebugLayer* debugLayer =
            graph.InsertNewLayer<DebugLayer>(*outputSlot, debugName.c_str(), toFile);

        // Sets output tensor info for the debug layer.
        ARMNN_ASSERT(debugLayer->GetInputSlot(0).GetConnectedOutputSlot() == &(*outputSlot));
        TensorInfo debugInfo = debugLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();

        debugLayer->GetOutputSlot().SetTensorInfo(debugInfo);

        // NOTE: It is OK to do this because DebugLayer is only supported on CpuRef
        debugLayer->SetBackendId(Compute::CpuRef);

        debugLayers.emplace_back(debugLayer);

        ++outputSlotIdx;
    }

    return debugLayers;
}

} // namespace armnn
