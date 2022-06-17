//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "arm-armnn-sl"

#include "ModelToINetworkTransformer.hpp"
#include "CanonicalUtils.hpp"
#include "Converter.hpp"

#include <log/log.h>
#include <type_traits>

namespace armnn_driver
{

ModelToINetworkTransformer::ModelToINetworkTransformer(
    const std::vector<armnn::BackendId>& backends,
    const Model& model,
    const std::set<unsigned int>& forcedUnsupportedOperations)
    : m_Data(backends)
    , m_Model(model)
    , m_ForcedUnsupportedOperations(forcedUnsupportedOperations)
    , m_ConversionResult(ConversionResult::Success)
{
    try
    {
        Convert();
    }
    catch (std::exception& e)
    {
        m_ConversionResult = ConversionResult::UnsupportedFeature;
        VLOG(DRIVER) << "ModelToINetworkTransformer: Unexpected exception: " << e.what();
        assert(false);
    }
}

void ModelToINetworkTransformer::Convert()
{
    VLOG(DRIVER) << "ModelToINetworkTransformer: Convert()";
    //VLOG(DRIVER) << "ModelToINetworkTransformer: Convert(): " << GetModelSummary(m_Model).c_str();

    // map the memory pool into shared pointers
    m_Data.m_MemPools.clear();
    if (!setRunTimePoolInfosFromCanonicalMemories(&m_Data.m_MemPools, m_Model.pools))
    {
        VLOG(DRIVER) << "Setting of run time pool infos from Hidl Memories has failed." << __func__;
        m_ConversionResult = ConversionResult::ErrorMappingPools;
        return;
    }

    using NetworkOptions = std::vector<armnn::BackendOptions>;
    NetworkOptions networkOptions;
    armnn::BackendOptions shapeInferenceMethodOption("ShapeInferenceMethod",
                                                    {
                                                            { "InferAndValidate", true }
                                                    });

    networkOptions.push_back(shapeInferenceMethodOption);

    // Create armnn::INetwork
    m_Data.m_Network = armnn::INetwork::Create(networkOptions);

    // add operations to it
    // track which layer outputs each operand
    VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): m_OutputSlotForOperand";
    m_Data.m_OutputSlotForOperand = std::vector<armnn::IOutputSlot*>(m_Model.main.operands.size(), nullptr);
    try
    {
        VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): for m_Model.inputIndexes.size()";
        for (uint32_t i = 0; i < m_Model.main.inputIndexes.size(); i++)
        {
            VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): m_Model.inputIndexes[i]";
            // inputs in android nn are represented by operands
            uint32_t inputIndex = m_Model.main.inputIndexes[i];
            VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): m_Model.operands[inputIndex]";
            const Operand& operand = m_Model.main.operands[inputIndex];
            VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): GetTensorInfoForOperand(operand)";

            const armnn::TensorInfo& tensor = GetTensorInfoForOperand(operand);
            const std::string layerName = "Input_" + std::to_string(i);
            VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): m_Data.m_Network->AddInputLayer(...)";
            armnn::IConnectableLayer* layer = m_Data.m_Network->AddInputLayer(i, layerName.c_str());

            VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): layer->GetOutputSlot(0)";
            armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
            VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): outputSlot.SetTensorInfo(...)";
            outputSlot.SetTensorInfo(GetTensorInfoForOperand(operand));

            VLOG(DRIVER) << "ModelToINetworkTransformer::Convert(): store for later layers";
            // store for later layers
            m_Data.m_OutputSlotForOperand[inputIndex] = &outputSlot;
        }
    }
    catch (UnsupportedOperand<OperandType>& e)
    {
        VLOG(DRIVER) <<  __func__ << "Operand type: " <<  e.m_type << " is not supported in ArmnnDriver";
        m_ConversionResult = ConversionResult::UnsupportedFeature;
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        Fail("%s: Failed to convert input operand to TensorShape: %s", __func__, e.what());
        m_ConversionResult = ConversionResult::UnsupportedFeature;
    }
    bool UnsupportedDynamicOperation = false;
    for (uint32_t operationIdx = 0; operationIdx < m_Model.main.operations.size(); operationIdx++)
    {
        const auto& operation = m_Model.main.operations[operationIdx];

        bool ok = true;
        if (m_ForcedUnsupportedOperations.find(operationIdx) != m_ForcedUnsupportedOperations.end())
        {
            Fail("%s: Operation at index %i has been forced to be unsupported.", __func__, operationIdx);
            ok = false;
        }

        if (ok)
        {
            try
            {
                ok = Converter::ConvertOperation(operation, m_Model, m_Data);
            }
            catch (UnsupportedOperand<OperandType>& e)
            {
                VLOG(DRIVER) << __func__ << "Operation type: " << e.m_type << "is not supported in ArmnnDriver";
                ok = false;
            }
            catch (const armnn::InvalidArgumentException& e)
            {
                Fail("%s: Failed to convert operation in %s", __func__, e.what());
                ok = false;
            }
        }

        // Store whether this operation was successfully converted.
        m_OperationSupported.emplace(operationIdx, ok);

        // Any single operation failing will fail the entire conversion.
        // We still need to continue and check the other ones.
        if (!ok)
        {
            if (m_Data.m_DynamicInputsEncountered)
            {
                Fail("%s: The unsupported operation at index %i has dynamic inputs.", __func__, operationIdx);
                UnsupportedDynamicOperation = true;
            }

            m_ConversionResult = ConversionResult::UnsupportedFeature;
        }
        m_Data.m_DynamicInputsEncountered = false;
    }

    // Due to the NNAPI partitioner not supporting partition boundaries of unknown size,
    // any operations who's outputs connect to an unsupported operation with with dynamic inputs
    // will cause a failure.

    // The simplest solution to this problem is to not support any operations in a model containing
    // an unsupported operation with with dynamic inputs.
    if (UnsupportedDynamicOperation)
    {
        Fail("%s: Unsupported operation with dynamic inputs found. Retroactively setting all operations to unsupported",
             __func__);
        for (auto& operation : m_OperationSupported)
        {
            operation.second = false;
        }
    }

    try
    {
        if (m_ConversionResult == ConversionResult::Success)
        {
            for (uint32_t i = 0; i < m_Model.main.outputIndexes.size(); i++)
            {
                // outputs in android nn are represented by operands
                uint32_t outputIndex = m_Model.main.outputIndexes[i];
                const auto& operand = m_Model.main.operands[outputIndex];
                const armnn::TensorInfo& tensor = GetTensorInfoForOperand(operand);
                const std::string layerName = "Output_" + std::to_string(i);
                armnn::IConnectableLayer* layer = m_Data.m_Network->AddOutputLayer(i, layerName.c_str());

                assert(m_Data.m_OutputSlotForOperand[outputIndex]);
                m_Data.m_OutputSlotForOperand[outputIndex]->Connect(layer->GetInputSlot(0));
            }
        }
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        Fail("%s: Failed to convert output operand to TensorShape: %s", __func__, e.what());
        m_ConversionResult = ConversionResult::UnsupportedFeature;
    }
}

bool ModelToINetworkTransformer::IsOperationSupported(uint32_t operationIndex) const
{
    std::map<uint32_t, bool>::const_iterator it = m_OperationSupported.find(operationIndex);
    assert(it != m_OperationSupported.end());
    return it->second;
}

} // armnn_driver
