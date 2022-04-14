//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <random>
#include <vector>

#include <cstdlib>
#include <doctest/doctest.h>

armnn::INetworkPtr DeserializeNetwork(const std::string& serializerString);

std::string SerializeNetwork(const armnn::INetwork& network);

void CompareConstTensor(const armnn::ConstTensor& tensor1, const armnn::ConstTensor& tensor2);

class LayerVerifierBase : public armnn::IStrategy
{
public:
    LayerVerifierBase(const std::string& layerName,
                      const std::vector<armnn::TensorInfo>& inputInfos,
                      const std::vector<armnn::TensorInfo>& outputInfos);

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override;

protected:
    void VerifyNameAndConnections(const armnn::IConnectableLayer* layer, const char* name);

    void VerifyConstTensors(const std::string& tensorName,
                            const armnn::ConstTensor* expectedPtr,
                            const armnn::ConstTensor* actualPtr);

private:
    std::string m_LayerName;
    std::vector<armnn::TensorInfo> m_InputTensorInfos;
    std::vector<armnn::TensorInfo> m_OutputTensorInfos;
};

template<typename Descriptor>
class LayerVerifierBaseWithDescriptor : public LayerVerifierBase
{
public:
    LayerVerifierBaseWithDescriptor(const std::string& layerName,
                                    const std::vector<armnn::TensorInfo>& inputInfos,
                                    const std::vector<armnn::TensorInfo>& outputInfos,
                                    const Descriptor& descriptor)
        : LayerVerifierBase(layerName, inputInfos, outputInfos)
        , m_Descriptor(descriptor) {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Input: break;
            case armnn::LayerType::Output: break;
            case armnn::LayerType::Constant: break;
            default:
            {
                VerifyNameAndConnections(layer, name);
                const Descriptor& internalDescriptor = static_cast<const Descriptor&>(descriptor);
                VerifyDescriptor(internalDescriptor);
                break;
            }
        }
    }

protected:
    void VerifyDescriptor(const Descriptor& descriptor)
    {
        CHECK(descriptor == m_Descriptor);
    }

    Descriptor m_Descriptor;
};

template<typename T>
void CompareConstTensorData(const void* data1, const void* data2, unsigned int numElements)
{
    T typedData1 = static_cast<T>(data1);
    T typedData2 = static_cast<T>(data2);
    CHECK(typedData1);
    CHECK(typedData2);

    for (unsigned int i = 0; i < numElements; i++)
    {
        CHECK(typedData1[i] == typedData2[i]);
    }
}


template <typename Descriptor>
class LayerVerifierBaseWithDescriptorAndConstants : public LayerVerifierBaseWithDescriptor<Descriptor>
{
public:
    LayerVerifierBaseWithDescriptorAndConstants(const std::string& layerName,
                                                const std::vector<armnn::TensorInfo>& inputInfos,
                                                const std::vector<armnn::TensorInfo>& outputInfos,
                                                const Descriptor& descriptor,
                                                const std::vector<armnn::ConstTensor>& constants)
            : LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor)
            , m_Constants(constants) {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(id);

        switch (layer->GetType())
        {
            case armnn::LayerType::Input: break;
            case armnn::LayerType::Output: break;
            case armnn::LayerType::Constant: break;
            default:
            {
                this->VerifyNameAndConnections(layer, name);
                const Descriptor& internalDescriptor = static_cast<const Descriptor&>(descriptor);
                this->VerifyDescriptor(internalDescriptor);

                for(std::size_t i = 0; i < constants.size(); i++)
                {
                    CompareConstTensor(constants[i], m_Constants[i]);
                }
            }
        }
    }

private:
    std::vector<armnn::ConstTensor> m_Constants;
};

template<typename DataType>
static std::vector<DataType> GenerateRandomData(size_t size)
{
    constexpr bool isIntegerType = std::is_integral<DataType>::value;
    using Distribution =
        typename std::conditional<isIntegerType,
                                  std::uniform_int_distribution<DataType>,
                                  std::uniform_real_distribution<DataType>>::type;

    static constexpr DataType lowerLimit = std::numeric_limits<DataType>::min();
    static constexpr DataType upperLimit = std::numeric_limits<DataType>::max();

    static Distribution distribution(lowerLimit, upperLimit);
    static std::default_random_engine generator;

    std::vector<DataType> randomData(size);
    generate(randomData.begin(), randomData.end(), []() { return distribution(generator); });

    return randomData;
}