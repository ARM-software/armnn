//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

#include <armnn/StrategyBase.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <layers/StandInLayer.hpp>

#include <sstream>
#include <vector>

TEST_SUITE("TensorflowLiteParser_Unsupported")
{
using namespace armnn;

class StandInLayerVerifier : public StrategyBase<NoThrowStrategy>
{
public:
    StandInLayerVerifier(const std::vector<TensorInfo>& inputInfos,
                         const std::vector<TensorInfo>& outputInfos)
        : m_InputInfos(inputInfos)
        , m_OutputInfos(outputInfos) {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::StandIn:
            {
                auto standInDescriptor = static_cast<const armnn::StandInDescriptor&>(descriptor);
                unsigned int numInputs = armnn::numeric_cast<unsigned int>(m_InputInfos.size());
                        CHECK(standInDescriptor.m_NumInputs    == numInputs);
                        CHECK(layer->GetNumInputSlots() == numInputs);

                unsigned int numOutputs = armnn::numeric_cast<unsigned int>(m_OutputInfos.size());
                        CHECK(standInDescriptor.m_NumOutputs    == numOutputs);
                        CHECK(layer->GetNumOutputSlots() == numOutputs);

                const StandInLayer* standInLayer = PolymorphicDowncast<const StandInLayer*>(layer);
                for (unsigned int i = 0u; i < numInputs; ++i)
                {
                    const OutputSlot* connectedSlot = standInLayer->GetInputSlot(i).GetConnectedOutputSlot();
                            CHECK(connectedSlot != nullptr);

                    const TensorInfo& inputInfo = connectedSlot->GetTensorInfo();
                            CHECK(inputInfo == m_InputInfos[i]);
                }

                for (unsigned int i = 0u; i < numOutputs; ++i)
                {
                    const TensorInfo& outputInfo = layer->GetOutputSlot(i).GetTensorInfo();
                            CHECK(outputInfo == m_OutputInfos[i]);
                }
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

private:
    std::vector<TensorInfo> m_InputInfos;
    std::vector<TensorInfo> m_OutputInfos;
};

class DummyCustomFixture : public ParserFlatbuffersFixture
{
public:
    explicit DummyCustomFixture(const std::vector<TensorInfo>& inputInfos,
                                const std::vector<TensorInfo>& outputInfos)
        : ParserFlatbuffersFixture()
        , m_StandInLayerVerifier(inputInfos, outputInfos)
    {
        const unsigned int numInputs = armnn::numeric_cast<unsigned int>(inputInfos.size());
        ARMNN_ASSERT(numInputs > 0);

        const unsigned int numOutputs = armnn::numeric_cast<unsigned int>(outputInfos.size());
        ARMNN_ASSERT(numOutputs > 0);

        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [{
                    "builtin_code": "CUSTOM",
                    "custom_code": "DummyCustomOperator"
                }],
                "subgraphs": [ {
                    "tensors": [)";

        // Add input tensors
        for (unsigned int i = 0u; i < numInputs; ++i)
        {
            const TensorInfo& inputInfo = inputInfos[i];
            m_JsonString += R"(
                    {
                        "shape": )" + GetTensorShapeAsString(inputInfo.GetShape()) + R"(,
                        "type": )" + GetDataTypeAsString(inputInfo.GetDataType()) + R"(,
                        "buffer": 0,
                        "name": "inputTensor)" + std::to_string(i) + R"(",
                        "quantization": {
                            "min": [ 0.0 ],
                            "max": [ 255.0 ],
                            "scale": [ )" + std::to_string(inputInfo.GetQuantizationScale()) + R"( ],
                            "zero_point": [ )" + std::to_string(inputInfo.GetQuantizationOffset()) + R"( ],
                        }
                    },)";
        }

        // Add output tensors
        for (unsigned int i = 0u; i < numOutputs; ++i)
        {
            const TensorInfo& outputInfo = outputInfos[i];
            m_JsonString += R"(
                    {
                        "shape": )" + GetTensorShapeAsString(outputInfo.GetShape()) + R"(,
                        "type": )" + GetDataTypeAsString(outputInfo.GetDataType()) + R"(,
                        "buffer": 0,
                        "name": "outputTensor)" + std::to_string(i) + R"(",
                        "quantization": {
                            "min": [ 0.0 ],
                            "max": [ 255.0 ],
                            "scale": [ )" + std::to_string(outputInfo.GetQuantizationScale()) + R"( ],
                            "zero_point": [ )" + std::to_string(outputInfo.GetQuantizationOffset()) + R"( ],
                        }
                    })";

            if (i + 1 < numOutputs)
            {
                m_JsonString += ",";
            }
        }

        const std::string inputIndices  = GetIndicesAsString(0u, numInputs - 1u);
        const std::string outputIndices = GetIndicesAsString(numInputs, numInputs + numOutputs - 1u);

        // Add dummy custom operator
        m_JsonString +=  R"(],
                    "inputs": )" + inputIndices + R"(,
                    "outputs": )" + outputIndices + R"(,
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": )" + inputIndices + R"(,
                            "outputs": )" + outputIndices + R"(,
                            "builtin_options_type": 0,
                            "custom_options": [ ],
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { }
                ]
            }
        )";

        ReadStringToBinary();
    }

    void RunTest()
    {
        INetworkPtr network = m_Parser->CreateNetworkFromBinary(m_GraphBinary);
        network->ExecuteStrategy(m_StandInLayerVerifier);
    }

private:
    static std::string GetTensorShapeAsString(const TensorShape& tensorShape)
    {
        std::stringstream stream;
        stream << "[ ";
        for (unsigned int i = 0u; i < tensorShape.GetNumDimensions(); ++i)
        {
            stream << tensorShape[i];
            if (i + 1 < tensorShape.GetNumDimensions())
            {
                stream << ",";
            }
            stream << " ";
        }
        stream << "]";

        return stream.str();
    }

    static std::string GetDataTypeAsString(DataType dataType)
    {
        switch (dataType)
        {
            case DataType::Float32:         return "FLOAT32";
            case DataType::QAsymmU8: return "UINT8";
            default:                        return "UNKNOWN";
        }
    }

    static std::string GetIndicesAsString(unsigned int first, unsigned int last)
    {
        std::stringstream stream;
        stream << "[ ";
        for (unsigned int i = first; i <= last ; ++i)
        {
            stream << i;
            if (i + 1 <= last)
            {
                stream << ",";
            }
            stream << " ";
        }
        stream << "]";

        return stream.str();
    }

    StandInLayerVerifier m_StandInLayerVerifier;
};

class DummyCustom1Input1OutputFixture : public DummyCustomFixture
{
public:
    DummyCustom1Input1OutputFixture()
        : DummyCustomFixture({ TensorInfo({ 1, 1 }, DataType::Float32) },
                             { TensorInfo({ 2, 2 }, DataType::Float32) }) {}
};

class DummyCustom2Inputs1OutputFixture : public DummyCustomFixture
{
public:
    DummyCustom2Inputs1OutputFixture()
        : DummyCustomFixture({ TensorInfo({ 1, 1 }, DataType::Float32), TensorInfo({ 2, 2 }, DataType::Float32) },
                             { TensorInfo({ 3, 3 }, DataType::Float32) }) {}
};

TEST_CASE_FIXTURE(DummyCustom1Input1OutputFixture, "UnsupportedCustomOperator1Input1Output")
{
    RunTest();
}

TEST_CASE_FIXTURE(DummyCustom2Inputs1OutputFixture, "UnsupportedCustomOperator2Inputs1Output")
{
    RunTest();
}

}
