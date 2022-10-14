//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkUtils.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>
#include <BFloat16.hpp>
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

void ChangeOutputBf16ToFp32(Layer& layer)
{
    for (auto&& outputSlot = layer.BeginOutputSlots(); outputSlot != layer.EndOutputSlots(); ++outputSlot)
    {
        if (outputSlot->GetTensorInfo().GetDataType() == DataType::BFloat16)
        {
            UpdateOutputSlotToFp32(*outputSlot);
        }
    }
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

std::vector<ConvertBf16ToFp32Layer*> InsertConvertBf16ToFp32LayersBefore(Graph& graph,
                                                                         Layer& layer,
                                                                         bool expectCorrectInputType)
{
    std::vector<ConvertBf16ToFp32Layer*> convertLayers;
    convertLayers.reserve(layer.GetNumInputSlots());

    // Insert a ConvertBf16ToFp32Layer before each input slot
    for (auto&& inputSlot = layer.BeginInputSlots(); inputSlot != layer.EndInputSlots(); ++inputSlot)
    {
        bool allowInsert = true;
        if (expectCorrectInputType)
        {
            // Only insert ConvertBf16ToFp32Layer before BF16 input slots
            OutputSlot* connectedOutputSlot = inputSlot->GetConnectedOutputSlot();
            allowInsert =
                connectedOutputSlot && connectedOutputSlot->GetTensorInfo().GetDataType() == DataType::BFloat16;
        }

        if (allowInsert)
        {
            const std::string name =
                std::string("convert_bf16_to_fp32-" + std::to_string(inputSlot->GetSlotIndex()) + "-") +
                layer.GetName();
            ConvertBf16ToFp32Layer* convertLayer =
                graph.InsertNewLayer<ConvertBf16ToFp32Layer>(*inputSlot, name.c_str());

            TensorInfo convertInfo = convertLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
            convertInfo.SetDataType(DataType::Float32);

            convertLayer->GetOutputSlot().SetTensorInfo(convertInfo);

            convertLayers.emplace_back(convertLayer);
        }
    }

    return convertLayers;
}

std::vector<ConvertFp32ToBf16Layer*> InsertConvertFp32ToBf16LayersBefore(Graph& graph,
                                                                         Layer& layer,
                                                                         bool expectCorrectInputType)
{
    std::vector<ConvertFp32ToBf16Layer*> convertLayers;
    convertLayers.reserve(layer.GetNumInputSlots());

    // Insert a ConvertFp32ToBf16Layer before each input slot
    for (auto&& inputSlot = layer.BeginInputSlots(); inputSlot != layer.EndInputSlots(); ++inputSlot)
    {
        bool allowInsert = true;

        if ((layer.GetType() == LayerType::Convolution2d ||
             layer.GetType() == LayerType::FullyConnected ||
             layer.GetType() == LayerType::DepthwiseConvolution2d)
                && inputSlot->GetSlotIndex() == 2)
        {
            // Refrain from reducing bias to Bf16
            continue;
        }
        if (expectCorrectInputType)
        {
            // Only insert ConvertFp32ToBf16Layer before FP32 input slots
            OutputSlot* connectedOutputSlot = inputSlot->GetConnectedOutputSlot();
            allowInsert =
                connectedOutputSlot && connectedOutputSlot->GetTensorInfo().GetDataType() == DataType::Float32;
        }

        if (allowInsert)
        {
            const std::string name =
                std::string("convert_fp32_to_bf16-" + std::to_string(inputSlot->GetSlotIndex()) + "-") +
                layer.GetName();
            ConvertFp32ToBf16Layer* convertLayer =
                graph.InsertNewLayer<ConvertFp32ToBf16Layer>(*inputSlot, name.c_str());

            TensorInfo convertInfo = convertLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
            convertInfo.SetDataType(DataType::BFloat16);

            convertLayer->GetOutputSlot().SetTensorInfo(convertInfo);

            convertLayers.emplace_back(convertLayer);
        }
    }

    return convertLayers;
}

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

std::vector<ConvertFp32ToBf16Layer*> InsertConvertFp32ToBf16LayersAfter(Graph& graph, Layer& layer)
{
    const unsigned int numOutputSlots = layer.GetNumOutputSlots();

    std::vector<ConvertFp32ToBf16Layer*> convertLayers;
    convertLayers.reserve(numOutputSlots);

    // Update Bf16 output slots to FP32 on current layer
    ChangeOutputBf16ToFp32(layer);

    // Insert a ConvertFp32ToBf16Layer after each FP32 output slot
    for (unsigned int slotIndex = 0u; slotIndex < numOutputSlots; ++slotIndex)
    {
        OutputSlot& outputSlot = layer.GetOutputSlot(slotIndex);
        if(outputSlot.GetTensorInfo().GetDataType() == DataType::Float32)
        {
            const std::string name =
                std::string("convert_fp32_to_bf16-" + std::to_string(slotIndex) + "-") + layer.GetName();
            ConvertFp32ToBf16Layer* convertLayer =
                graph.InsertNewLayer<ConvertFp32ToBf16Layer>(outputSlot, name.c_str());

            TensorInfo convertInfo = convertLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
            convertInfo.SetDataType(DataType::BFloat16);

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

bool RevertConstantWeightsToFP32(Layer* layer)
{
    if (layer->GetType() == LayerType::Convolution2d || layer->GetType() == LayerType::FullyConnected)
    {
        // Revert Weights on Constant Layer to FP32 so they can be accessed by Conv2d or FullyConnected
        // This prevents a conversion layer being added in during backend assignment which blocks
        // the RedirectMembersToConstantInputs backward compatibility workaround/optimization.
        auto constantLayerInfo = layer->GetInputSlot(1).GetConnection()->GetTensorInfo();

        if (constantLayerInfo.IsConstant() && constantLayerInfo.GetDataType() == DataType::BFloat16)
        {
            std::vector<float> newValues(constantLayerInfo.GetNumElements());

            auto weightLayer = PolymorphicDowncast<ConstantLayer*>(
                    &layer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer());
            armnnUtils::FloatingPointConverter::ConvertBFloat16ToFloat32(
                    weightLayer->m_LayerOutput->GetConstTensor<BFloat16>(),
                    constantLayerInfo.GetNumElements(),
                    newValues.data());

            TensorInfo newInfo(constantLayerInfo.GetShape(), DataType::Float32);
            newInfo.SetConstant(true);
            ConstTensor newInput(newInfo, newValues);
            weightLayer->m_LayerOutput.reset(new ScopedTensorHandle(newInput));
            weightLayer->GetOutputSlot(0).SetTensorInfo(newInfo);

            // Connect Conv2d/FullyConnected to InputLayer directly leaving out
            // the ConversionLayer to be cleaned up later
            auto& conversionLayer = layer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
            auto actualInputOutputSlot = conversionLayer.GetInputSlot(0).GetConnection();

            auto& conversionLayerOutputSlot =
                    layer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer().GetOutputSlot(0);
            auto& conversionLayerInputSlot =
                    layer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer().GetInputSlot(0);
            actualInputOutputSlot->Disconnect(conversionLayerInputSlot);
            conversionLayerOutputSlot.Disconnect(layer->GetInputSlot(0));

            actualInputOutputSlot->Connect(layer->GetInputSlot(0));

            return true;
        }
    }
    return false;
}

} // namespace armnn
