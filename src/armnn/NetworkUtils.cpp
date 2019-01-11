//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkUtils.hpp"

#include "SubGraphSelector.hpp"

#include <armnn/Exceptions.hpp>

#include <backendsCommon/BackendRegistry.hpp>

namespace armnn
{

std::vector<ConvertFp16ToFp32Layer*> InsertConvertFp16ToFp32LayersBefore(Graph& graph, Layer& layer)
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

std::vector<ConvertFp32ToFp16Layer*> InsertConvertFp32ToFp16LayersAfter(Graph& graph, Layer& layer)
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

std::vector<DebugLayer*> InsertDebugLayerAfter(Graph& graph, Layer& layer)
{
    std::vector<DebugLayer*> debugLayers;
    debugLayers.reserve(layer.GetNumOutputSlots());

    // Change outputs to DataType::Float16
    for (auto&& outputSlot = layer.BeginOutputSlots(); outputSlot != layer.EndOutputSlots(); ++outputSlot)
    {
        // Insert debug layer after the layer
        const std::string name =
            std::string("DebugLayerAfter") + layer.GetName();

        const DebugDescriptor descriptor;

        DebugLayer* debugLayer =
            graph.InsertNewLayer<DebugLayer>(*outputSlot, descriptor, name.c_str());

        // Sets output tensor info for the debug layer.
        TensorInfo debugInfo = debugLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();

        debugLayer->GetOutputSlot().SetTensorInfo(debugInfo);

        // NOTE: It is OK to do this because DebugLayer is only supported on CpuRef
        debugLayer->SetBackendId(Compute::CpuRef);

        debugLayers.emplace_back(debugLayer);
    }

    return debugLayers;
}

PreCompiledLayer* CreatePreCompiledLayer(Graph& graph,
                                         const SubGraph& subGraph,
                                         unsigned int subGraphIndex,
                                         const IBackendInternalUniquePtr& backendObjPtr)
{
    BOOST_ASSERT(backendObjPtr);

    IBackendInternal::ISubGraphConverterPtr converter =
            backendObjPtr->CreateSubGraphConverter(std::make_shared<SubGraph>(subGraph));
    if (!converter)
    {
        return nullptr;
    }

    try
    {
        // Attempt to convert and compile sub-graph
        auto preCompiledObject = converter->GetOutput();
    }
    catch (std::exception&)
    {
        return nullptr;
    }

    // Create pre-compiled layer
    std::string name = "pre-compiled" + std::to_string(subGraphIndex);
    PreCompiledLayer* preCompiledLayer = graph.AddLayer<PreCompiledLayer>(
        PreCompiledDescriptor(subGraph.GetNumInputSlots(), subGraph.GetNumOutputSlots()), name.c_str());

    // Copy output tensor infos from sub-graph
    for (unsigned int i = 0u; i < subGraph.GetNumOutputSlots(); i++)
    {
        preCompiledLayer->GetOutputSlot(i).SetTensorInfo(subGraph.GetOutputSlot(i)->GetTensorInfo());
    }

    // Assign pre-compiled object to layer
    preCompiledLayer->SetPreCompiledObject(converter->GetOutput());

    // Set the backend-id for the pre-compiled layer
    BackendId backendId = backendObjPtr->GetId();
    preCompiledLayer->SetBackendId(backendId);

    return preCompiledLayer;
}

} // namespace armnn
