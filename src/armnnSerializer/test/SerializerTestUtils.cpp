//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SerializerTestUtils.hpp"
#include "../Serializer.hpp"

#include <doctest/doctest.h>

using armnnDeserializer::IDeserializer;

LayerVerifierBase::LayerVerifierBase(const std::string& layerName,
                                     const std::vector<armnn::TensorInfo>& inputInfos,
                                     const std::vector<armnn::TensorInfo>& outputInfos)
                                     : m_LayerName(layerName)
                                     , m_InputTensorInfos(inputInfos)
                                     , m_OutputTensorInfos(outputInfos)
{}

void LayerVerifierBase::ExecuteStrategy(const armnn::IConnectableLayer* layer,
                     const armnn::BaseDescriptor& descriptor,
                     const std::vector<armnn::ConstTensor>& constants,
                     const char* name,
                     const armnn::LayerBindingId id)
{
    armnn::IgnoreUnused(descriptor, constants, id);
    switch (layer->GetType())
    {
        case armnn::LayerType::Input: break;
        case armnn::LayerType::Output: break;
        default:
        {
            VerifyNameAndConnections(layer, name);
        }
    }
}


void LayerVerifierBase::VerifyNameAndConnections(const armnn::IConnectableLayer* layer, const char* name)
{
    CHECK(std::string(name) == m_LayerName.c_str());

    CHECK(layer->GetNumInputSlots() == m_InputTensorInfos.size());
    CHECK(layer->GetNumOutputSlots() == m_OutputTensorInfos.size());

    for (unsigned int i = 0; i < m_InputTensorInfos.size(); i++)
    {
        const armnn::IOutputSlot* connectedOutput = layer->GetInputSlot(i).GetConnection();
        CHECK(connectedOutput);

        const armnn::TensorInfo& connectedInfo = connectedOutput->GetTensorInfo();
        CHECK(connectedInfo.GetShape() == m_InputTensorInfos[i].GetShape());
        CHECK(
            GetDataTypeName(connectedInfo.GetDataType()) == GetDataTypeName(m_InputTensorInfos[i].GetDataType()));

        CHECK(connectedInfo.GetQuantizationScale() == m_InputTensorInfos[i].GetQuantizationScale());
        CHECK(connectedInfo.GetQuantizationOffset() == m_InputTensorInfos[i].GetQuantizationOffset());
    }

    for (unsigned int i = 0; i < m_OutputTensorInfos.size(); i++)
    {
        const armnn::TensorInfo& outputInfo = layer->GetOutputSlot(i).GetTensorInfo();
        CHECK(outputInfo.GetShape() == m_OutputTensorInfos[i].GetShape());
        CHECK(GetDataTypeName(outputInfo.GetDataType()) == GetDataTypeName(m_OutputTensorInfos[i].GetDataType()));

        CHECK(outputInfo.GetQuantizationScale() == m_OutputTensorInfos[i].GetQuantizationScale());
        CHECK(outputInfo.GetQuantizationOffset() == m_OutputTensorInfos[i].GetQuantizationOffset());
    }
}

void LayerVerifierBase::VerifyConstTensors(const std::string& tensorName,
                                           const armnn::ConstTensor* expectedPtr,
                                           const armnn::ConstTensor* actualPtr)
{
    if (expectedPtr == nullptr)
    {
        CHECK_MESSAGE(actualPtr == nullptr, (tensorName + " should not exist"));
    }
    else
    {
        CHECK_MESSAGE(actualPtr != nullptr, (tensorName + " should have been set"));
        if (actualPtr != nullptr)
        {
            const armnn::TensorInfo& expectedInfo = expectedPtr->GetInfo();
            const armnn::TensorInfo& actualInfo = actualPtr->GetInfo();

            CHECK_MESSAGE(expectedInfo.GetShape() == actualInfo.GetShape(),
                          (tensorName + " shapes don't match"));
            CHECK_MESSAGE(
                    GetDataTypeName(expectedInfo.GetDataType()) == GetDataTypeName(actualInfo.GetDataType()),
                    (tensorName + " data types don't match"));

            CHECK_MESSAGE(expectedPtr->GetNumBytes() == actualPtr->GetNumBytes(),
                          (tensorName + " (GetNumBytes) data sizes do not match"));
            if (expectedPtr->GetNumBytes() == actualPtr->GetNumBytes())
            {
                //check the data is identical
                const char* expectedData = static_cast<const char*>(expectedPtr->GetMemoryArea());
                const char* actualData = static_cast<const char*>(actualPtr->GetMemoryArea());
                bool same = true;
                for (unsigned int i = 0; i < expectedPtr->GetNumBytes(); ++i)
                {
                    same = expectedData[i] == actualData[i];
                    if (!same)
                    {
                        break;
                    }
                }
                CHECK_MESSAGE(same, (tensorName + " data does not match"));
            }
        }
    }
}

void CompareConstTensor(const armnn::ConstTensor& tensor1, const armnn::ConstTensor& tensor2)
{
    CHECK(tensor1.GetShape() == tensor2.GetShape());
    CHECK(GetDataTypeName(tensor1.GetDataType()) == GetDataTypeName(tensor2.GetDataType()));

    switch (tensor1.GetDataType())
    {
        case armnn::DataType::Float32:
            CompareConstTensorData<const float*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::Boolean:
            CompareConstTensorData<const uint8_t*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        case armnn::DataType::QSymmS8:
            CompareConstTensorData<const int8_t*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        case armnn::DataType::Signed32:
            CompareConstTensorData<const int32_t*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        default:
            // Note that Float16 is not yet implemented
            MESSAGE("Unexpected datatype");
            CHECK(false);
    }
}

armnn::INetworkPtr DeserializeNetwork(const std::string& serializerString)
{
    std::vector<std::uint8_t> const serializerVector{serializerString.begin(), serializerString.end()};
    return IDeserializer::Create()->CreateNetworkFromBinary(serializerVector);
}

std::string SerializeNetwork(const armnn::INetwork& network)
{
    armnnSerializer::ISerializerPtr serializer = armnnSerializer::ISerializer::Create();

    serializer->Serialize(network);

    std::stringstream stream;
    serializer->SaveSerializedToStream(stream);

    std::string serializerString{stream.str()};
    return serializerString;
}
