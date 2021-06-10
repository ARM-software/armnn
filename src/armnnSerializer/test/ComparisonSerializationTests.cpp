//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Serializer.hpp"
#include "SerializerTestUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

TEST_SUITE("SerializerTests")
{
struct ComparisonModel
{
    ComparisonModel(const std::string& layerName,
                    const armnn::TensorInfo& inputInfo,
                    const armnn::TensorInfo& outputInfo,
                    armnn::ComparisonDescriptor& descriptor)
            : m_network(armnn::INetwork::Create())
    {
        armnn::IConnectableLayer* const inputLayer0 = m_network->AddInputLayer(0);
        armnn::IConnectableLayer* const inputLayer1 = m_network->AddInputLayer(1);
        armnn::IConnectableLayer* const equalLayer = m_network->AddComparisonLayer(descriptor, layerName.c_str());
        armnn::IConnectableLayer* const outputLayer = m_network->AddOutputLayer(0);

        inputLayer0->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(0));
        inputLayer1->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(1));
        equalLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        inputLayer0->GetOutputSlot(0).SetTensorInfo(inputInfo);
        inputLayer1->GetOutputSlot(0).SetTensorInfo(inputInfo);
        equalLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    }

    armnn::INetworkPtr m_network;
};

class ComparisonLayerVerifier : public LayerVerifierBase
{
public:
    ComparisonLayerVerifier(const std::string& layerName,
                            const std::vector<armnn::TensorInfo>& inputInfos,
                            const std::vector<armnn::TensorInfo>& outputInfos,
                            const armnn::ComparisonDescriptor& descriptor)
            : LayerVerifierBase(layerName, inputInfos, outputInfos)
            , m_Descriptor (descriptor) {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Input: break;
            case armnn::LayerType::Output: break;
            case armnn::LayerType::Comparison:
            {
                VerifyNameAndConnections(layer, name);
                const armnn::ComparisonDescriptor& layerDescriptor =
                        static_cast<const armnn::ComparisonDescriptor&>(descriptor);
                CHECK(layerDescriptor.m_Operation == m_Descriptor.m_Operation);
                break;
            }
            default:
            {
                throw armnn::Exception("Unexpected layer type in Comparison test model");
            }
        }
    }

private:
    armnn::ComparisonDescriptor m_Descriptor;
};

TEST_CASE("SerializeEqual")
{
    const std::string layerName("equal");

    const armnn::TensorShape shape{2, 1, 2, 4};
    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::ComparisonDescriptor descriptor (armnn::ComparisonOperation::Equal);

    ComparisonModel model(layerName, inputInfo, outputInfo, descriptor);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*model.m_network));
    CHECK(deserializedNetwork);

    ComparisonLayerVerifier verifier(layerName, { inputInfo, inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeGreater")
{
    const std::string layerName("greater");

    const armnn::TensorShape shape{2, 1, 2, 4};
    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::ComparisonDescriptor descriptor (armnn::ComparisonOperation::Greater);

    ComparisonModel model(layerName, inputInfo, outputInfo, descriptor);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*model.m_network));
    CHECK(deserializedNetwork);

    ComparisonLayerVerifier verifier(layerName, { inputInfo, inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

}
