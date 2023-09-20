//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <aclCommon/ArmComputeSubgraphUtils.hpp>

namespace armnn
{

// Changes shapes of the form [1, 1, ..., W] to [ W ]
inline bool CollapseLeadingUnitDimensions(const TensorInfo& in, TensorInfo& out)
{
    unsigned int numDimensions = in.GetNumDimensions();
    for (unsigned int i = 0; i < (numDimensions-1); ++i)
    {
        if (in.GetShape()[i] != 1)
        {
            return false;
        }
    }

    unsigned int w = in.GetShape()[numDimensions-1];
    out = in;
    out.SetShape({w});

    return true;
}

//
// Build slot and tensor info lists for Add/Mul/Add replacement
//
template<typename SlotListType>
void BuildAddMulAddSlotLists(bool handleReLu,
                             bool multipleOutputs,
                             std::vector<SlotListType>& inputLayersSlotLists,
                             std::vector<SlotListType>& outputLayersSlotLists)
{
    // Build input slot list
    inputLayersSlotLists.push_back({0, 1});     // Add
    inputLayersSlotLists.push_back({1});        // Mul
    inputLayersSlotLists.push_back({1});        // Add
    if (handleReLu)
    {
        inputLayersSlotLists.push_back({});     // Relu
    }

    // Build output slot list
    if (multipleOutputs)
    {
        outputLayersSlotLists.push_back({0});   // Add
    }
    else
    {
        outputLayersSlotLists.push_back({});    // Add
    }
    outputLayersSlotLists.push_back({});        // Mul
    if (handleReLu)
    {
        outputLayersSlotLists.push_back({});    // Add
        outputLayersSlotLists.push_back({0});   // Relu
    }
    else
    {
        outputLayersSlotLists.push_back({0});   // Add
    }
}

inline void GetFusedName(Layer *layerList[4], std::string& fusedName)
{
    // Build the fused name string
    fusedName = "fused";
    for (unsigned int layerIdx = 0; layerIdx< 4; ++layerIdx)
    {
        if (! layerList[layerIdx])
        {
            break;
        }
        fusedName += "-";
        fusedName += layerList[layerIdx]->GetNameStr();
    }
}

template<typename Type>
bool BuildAddMulAddTensorInfoLists(Type* layerList[4],
                                   unsigned int& numInputs,
                                   unsigned int& numOutputs,
                                   std::vector<TensorInfo>& inputInfos,
                                   std::vector<TensorInfo>& outputInfos,
                                   const ActivationDescriptor*& activationDescriptor,
                                   bool& fuseReLu)
{
    ARMNN_THROW_INVALIDARG_IF_FALSE(layerList[0]);
    ARMNN_THROW_INVALIDARG_IF_FALSE(layerList[1]);
    ARMNN_THROW_INVALIDARG_IF_FALSE(layerList[2]);

    ARMNN_THROW_INVALIDARG_IF_FALSE(IsSequenceLayerType(*layerList[0], BinaryOperation::Add));
    ARMNN_THROW_INVALIDARG_IF_FALSE(IsSequenceLayerType(*layerList[1], BinaryOperation::Mul));
    ARMNN_THROW_INVALIDARG_IF_FALSE(IsSequenceLayerType(*layerList[2], BinaryOperation::Add));

    fuseReLu = (layerList[3] != nullptr);
    if (fuseReLu)
    {
        activationDescriptor = &PolymorphicDowncast<ActivationLayer *>(layerList[3])->GetParameters();
        ARMNN_THROW_INVALIDARG_IF_FALSE((activationDescriptor->m_Function == ActivationFunction::ReLu) ||
                     (activationDescriptor->m_Function == ActivationFunction::BoundedReLu));
    }

    numInputs = 0;
    numOutputs = 0;

    // Ensure that there are 6 input slots in the add/mul/add layers
    // we are going to replace
    unsigned int layerIdx = 0;
    unsigned int inputSlotCount = 0;
    for (layerIdx = 0; layerIdx < 3; ++layerIdx)
    {
        for (unsigned int slotIdx = 0; slotIdx < layerList[layerIdx]->GetNumInputSlots(); ++slotIdx)
        {
            InputSlot* inputSlot = &layerList[layerIdx]->GetInputSlot(slotIdx);
            OutputSlot* outputSlot = inputSlot->GetConnectedOutputSlot();
            if (outputSlot)
            {
                if (layerIdx == 0)
                {
                    // Always count the input connections of the first add
                    inputInfos.push_back(inputSlot->GetTensorInfo());
                    numInputs++;
                }
                else
                {
                    // For subsequent layers, we skip connections to the previous layers in the counting
                    if (&outputSlot->GetOwningLayer() != layerList[layerIdx-1])
                    {
                        TensorInfo inputSlotInfo = inputSlot->GetTensorInfo();
                        if (numInputs == 2 || numInputs == 3)
                        {
                            // Workaround the broadcast optimization to collapse shapes such as
                            // [1, 1, 1, 2] to [2] as required by backend
                            if (CollapseLeadingUnitDimensions(inputSlot->GetTensorInfo(), inputSlotInfo))
                            {
                                OutputSlot* previousLayerSlot = inputSlot->GetConnectedOutputSlot();
                                if (previousLayerSlot)
                                {
                                    if (previousLayerSlot->GetOwningLayer().GetType() == LayerType::Constant)
                                    {
                                        // First update the TensorInfo in the constant owning layer
                                        previousLayerSlot->SetTensorInfo(inputSlotInfo);
                                        // Then update the TensorInfo in the workload for the owning layer
                                        ConstantLayer* layer = PolymorphicDowncast<ConstantLayer*>(
                                                &previousLayerSlot->GetOwningLayer());
                                        layer->m_LayerOutput
                                                = std::make_unique<ScopedTensorHandle>(
                                                ConstTensor(inputSlotInfo,
                                                            layer->m_LayerOutput.get()->GetConstTensor<void>()));
                                    }
                                }
                            }
                        }
                        inputInfos.push_back(inputSlotInfo);
                        numInputs++;
                    }
                }
                inputSlotCount++;
            }
        }
    }

    // Check the input counts
    bool validInputCount = (inputSlotCount == 6) && (inputInfos.size() == 4);
    if (! validInputCount)
    {
        return false;
    }

    const unsigned int maxIdx = (fuseReLu) ? 4 : 3;
    for (layerIdx = 0; layerIdx < maxIdx; ++layerIdx)
    {
        for (unsigned int slotIdx = 0; slotIdx < layerList[layerIdx]->GetNumOutputSlots(); ++slotIdx)
        {
            OutputSlot* outputSlot = &layerList[layerIdx]->GetOutputSlot(slotIdx);

            for (unsigned int connectionIdx = 0; connectionIdx < outputSlot->GetNumConnections(); ++connectionIdx)
            {
                InputSlot* inputSlot = outputSlot->GetConnection(connectionIdx);
                if (layerIdx < (maxIdx-1))
                {
                    if (&inputSlot->GetOwningLayer() != layerList[layerIdx+1])
                    {
                        outputInfos.push_back(outputSlot->GetTensorInfo());
                        numOutputs++;
                    }
                }
                else if (layerList[layerIdx] != nullptr)
                {
                    outputInfos.push_back(outputSlot->GetTensorInfo());
                    numOutputs++;
                }
            }
        }
    }

    // Check the output count
    bool validOutputCount = (outputInfos.size() > 0);
    if (! validOutputCount)
    {
        return false;
    }

    return true;
}

}
