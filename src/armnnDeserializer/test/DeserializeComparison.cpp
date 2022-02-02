//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <string>

TEST_SUITE("Deserializer_Comparison")
{
#define DECLARE_SIMPLE_COMPARISON_FIXTURE(operation, dataType) \
struct Simple##operation##dataType##Fixture : public SimpleComparisonFixture \
{ \
    Simple##operation##dataType##Fixture() \
        : SimpleComparisonFixture(#dataType, #operation) {} \
};

#define DECLARE_SIMPLE_COMPARISON_TEST_CASE(operation, dataType) \
DECLARE_SIMPLE_COMPARISON_FIXTURE(operation, dataType) \
TEST_CASE_FIXTURE(Simple##operation##dataType##Fixture, #operation#dataType) \
{ \
    using T = armnn::ResolveType<armnn::DataType::dataType>; \
    constexpr float   qScale  = 1.f; \
    constexpr int32_t qOffset = 0; \
    RunTest<4, armnn::DataType::dataType, armnn::DataType::Boolean>( \
        0, \
        {{ "InputLayer0", armnnUtils::QuantizedVector<T>(s_TestData.m_InputData0, qScale, qOffset)  }, \
         { "InputLayer1", armnnUtils::QuantizedVector<T>(s_TestData.m_InputData1, qScale, qOffset)  }}, \
        {{ "OutputLayer", s_TestData.m_Output##operation }}); \
}

struct ComparisonFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ComparisonFixture(const std::string& inputShape0,
                               const std::string& inputShape1,
                               const std::string& outputShape,
                               const std::string& inputDataType,
                               const std::string& comparisonOperation)
    {
        m_JsonString = R"(
            {
                inputIds: [0, 1],
                outputIds: [3],
                layers: [
                    {
                        layer_type: "InputLayer",
                        layer: {
                            base: {
                                layerBindingId: 0,
                                base: {
                                    index: 0,
                                    layerName: "InputLayer0",
                                    layerType: "Input",
                                    inputSlots: [{
                                        index: 0,
                                        connection: { sourceLayerIndex:0, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + inputShape0 + R"(,
                                            dataType: )" + inputDataType + R"(
                                        },
                                    }],
                                },
                            }
                        },
                    },
                    {
                        layer_type: "InputLayer",
                        layer: {
                            base: {
                                layerBindingId: 1,
                                base: {
                                      index:1,
                                      layerName: "InputLayer1",
                                      layerType: "Input",
                                      inputSlots: [{
                                          index: 0,
                                          connection: { sourceLayerIndex:0, outputSlotIndex:0 },
                                      }],
                                      outputSlots: [{
                                          index: 0,
                                          tensorInfo: {
                                              dimensions: )" + inputShape1 + R"(,
                                              dataType: )" + inputDataType + R"(
                                          },
                                      }],
                                },
                            }
                        },
                    },
                    {
                        layer_type: "ComparisonLayer",
                        layer: {
                            base: {
                                 index:2,
                                 layerName: "ComparisonLayer",
                                 layerType: "Comparison",
                                 inputSlots: [{
                                     index: 0,
                                     connection: { sourceLayerIndex:0, outputSlotIndex:0 },
                                 },
                                 {
                                     index: 1,
                                     connection: { sourceLayerIndex:1, outputSlotIndex:0 },
                                 }],
                                 outputSlots: [{
                                     index: 0,
                                     tensorInfo: {
                                         dimensions: )" + outputShape + R"(,
                                         dataType: Boolean
                                     },
                                 }],
                            },
                            descriptor: {
                                operation: )" + comparisonOperation + R"(
                            }
                        },
                    },
                    {
                        layer_type: "OutputLayer",
                        layer: {
                            base:{
                                layerBindingId: 0,
                                base: {
                                    index: 3,
                                    layerName: "OutputLayer",
                                    layerType: "Output",
                                    inputSlots: [{
                                        index: 0,
                                        connection: { sourceLayerIndex:2, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: Boolean
                                        },
                                    }],
                                }
                            }
                        },
                    }
                ]
            }
        )";
        Setup();
    }
};

struct SimpleComparisonTestData
{
    SimpleComparisonTestData()
    {
        m_InputData0 =
        {
            1.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f,
            3.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 4.f
        };

        m_InputData1 =
        {
            1.f, 1.f, 1.f, 1.f, 3.f, 3.f, 3.f, 3.f,
            5.f, 5.f, 5.f, 5.f, 4.f, 4.f, 4.f, 4.f
        };

        m_OutputEqual =
        {
            1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1
        };

        m_OutputGreater =
        {
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0
        };

        m_OutputGreaterOrEqual =
        {
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1
        };

        m_OutputLess =
        {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 0, 0, 0, 0
        };

        m_OutputLessOrEqual =
        {
            1, 1, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1
        };

        m_OutputNotEqual =
        {
            0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 0, 0, 0
        };
    }

    std::vector<float> m_InputData0;
    std::vector<float> m_InputData1;

    std::vector<uint8_t> m_OutputEqual;
    std::vector<uint8_t> m_OutputGreater;
    std::vector<uint8_t> m_OutputGreaterOrEqual;
    std::vector<uint8_t> m_OutputLess;
    std::vector<uint8_t> m_OutputLessOrEqual;
    std::vector<uint8_t> m_OutputNotEqual;
};

struct SimpleComparisonFixture : public ComparisonFixture
{
    SimpleComparisonFixture(const std::string& inputDataType,
                            const std::string& comparisonOperation)
        : ComparisonFixture("[ 2, 2, 2, 2 ]", // inputShape0
                            "[ 2, 2, 2, 2 ]", // inputShape1
                            "[ 2, 2, 2, 2 ]", // outputShape,
                            inputDataType,
                            comparisonOperation) {}

    static SimpleComparisonTestData s_TestData;
};

SimpleComparisonTestData SimpleComparisonFixture::s_TestData;

DECLARE_SIMPLE_COMPARISON_TEST_CASE(Equal,          Float32)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(Greater,        Float32)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(GreaterOrEqual, Float32)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(Less,           Float32)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(LessOrEqual,    Float32)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(NotEqual,       Float32)


DECLARE_SIMPLE_COMPARISON_TEST_CASE(Equal,          QAsymmU8)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(Greater,        QAsymmU8)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(GreaterOrEqual, QAsymmU8)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(Less,           QAsymmU8)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(LessOrEqual,    QAsymmU8)
DECLARE_SIMPLE_COMPARISON_TEST_CASE(NotEqual,       QAsymmU8)

}
