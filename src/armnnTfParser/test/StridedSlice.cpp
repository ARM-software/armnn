//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfParser/ITfParser.hpp"

#include "ParserPrototxtFixture.hpp"
#include <PrototxtConversions.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

namespace {
// helper for setting the dimensions in prototxt
void shapeHelper(const armnn::TensorShape& shape, std::string& text){
    for(unsigned int i = 0; i < shape.GetNumDimensions(); ++i) {
        text.append(R"(dim {
      size: )");
        text.append(std::to_string(shape[i]));
        text.append(R"(
    })");
    }
}

// helper for converting from integer to octal representation
void octalHelper(const std::vector<int>& content, std::string& text){
    for (unsigned int i = 0; i < content.size(); ++i)
    {
        text.append(armnnUtils::ConvertInt32ToOctalString(static_cast<int>(content[i])));
    }
}
} // namespace

struct StridedSliceFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    StridedSliceFixture(const armnn::TensorShape& inputShape,
                        const std::vector<int>& beginData,
                        const std::vector<int>& endData,
                        const std::vector<int>& stridesData,
                        int beginMask = 0,
                        int endMask = 0,
                        int ellipsisMask = 0,
                        int newAxisMask = 0,
                        int shrinkAxisMask = 0)
    {
        m_Prototext = R"(
                         node {
                           name: "input"
                           op: "Placeholder"
                           attr {
                             key: "dtype"
                             value {
                               type: DT_FLOAT
                             }
                           }
                           attr {
                             key: "shape"
                             value {
                               shape {)";
                                 shapeHelper(inputShape, m_Prototext);
                                 m_Prototext.append(R"(
                               }
                             }
                           }
                         }
                         node {
                           name: "begin"
                           op: "Const"
                           attr {
                             key: "dtype"
                             value {
                               type: DT_INT32
                             }
                           }
                           attr {
                             key: "value"
                             value {
                              tensor {
                               dtype: DT_INT32
                                 tensor_shape {
                                   dim {
                                    size: )");
                                      m_Prototext += std::to_string(beginData.size());
                                      m_Prototext.append(R"(
                                    }
                                 }
                                 tensor_content: ")");
                                   octalHelper(beginData, m_Prototext);
                                   m_Prototext.append(R"("
                               }
                             }
                           }
                         }
                         node {
                           name: "end"
                           op: "Const"
                           attr {
                             key: "dtype"
                             value {
                               type: DT_INT32
                             }
                           }
                           attr {
                             key: "value"
                             value {
                              tensor {
                               dtype: DT_INT32
                                 tensor_shape {
                                   dim {
                                    size: )");
                                      m_Prototext += std::to_string(endData.size());
                                      m_Prototext.append(R"(
                                    }
                                 }
                                 tensor_content: ")");
                                   octalHelper(endData, m_Prototext);
                                   m_Prototext.append(R"("
                               }
                             }
                           }
                         }
                         node {
                           name: "strides"
                           op: "Const"
                           attr {
                             key: "dtype"
                             value {
                               type: DT_INT32
                             }
                           }
                           attr {
                             key: "value"
                             value {
                              tensor {
                               dtype: DT_INT32
                                 tensor_shape {
                                   dim {
                                    size: )");
                                      m_Prototext += std::to_string(stridesData.size());
                                      m_Prototext.append(R"(
                                    }
                                 }
                                 tensor_content: ")");
                                   octalHelper(stridesData, m_Prototext);
                                   m_Prototext.append(R"("
                               }
                             }
                           }
                         }
                         node {
                           name: "output"
                           op: "StridedSlice"
                           input: "input"
                           input: "begin"
                           input: "end"
                           input: "strides"
                           attr {
                             key: "begin_mask"
                             value {
                               i: )");
                               m_Prototext += std::to_string(beginMask);
                               m_Prototext.append(R"(
                             }
                           }
                           attr {
                             key: "end_mask"
                             value {
                               i: )");
                                 m_Prototext += std::to_string(endMask);
                                 m_Prototext.append(R"(
                             }
                           }
                           attr {
                             key: "ellipsis_mask"
                             value {
                               i: )");
                                 m_Prototext += std::to_string(ellipsisMask);
                                 m_Prototext.append(R"(
                             }
                           }
                           attr {
                             key: "new_axis_mask"
                             value {
                               i: )");
                                 m_Prototext += std::to_string(newAxisMask);
                                 m_Prototext.append(R"(
                             }
                           }
                           attr {
                             key: "shrink_axis_mask"
                             value {
                               i: )");
                                 m_Prototext += std::to_string(shrinkAxisMask);
                                 m_Prototext.append(R"(
                             }
                           }
                         })");

        Setup({ { "input", inputShape } }, { "output" });
    }
};

struct StridedSlice4DFixture : StridedSliceFixture
{
    StridedSlice4DFixture() : StridedSliceFixture({ 3, 2, 3, 1 },  // inputShape
                                                  { 1, 0, 0, 0 },  // beginData
                                                  { 2, 2, 3, 1 },  // endData
                                                  { 1, 1, 1, 1 }   // stridesData
    ) {}
};

BOOST_FIXTURE_TEST_CASE(StridedSlice4D, StridedSlice4DFixture)
{
    RunTest<4>(
            {{"input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},
            {{"output", { 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f }}});
}

struct StridedSlice4DReverseFixture : StridedSliceFixture
{

    StridedSlice4DReverseFixture() : StridedSliceFixture({ 3, 2, 3, 1 },   // inputShape
                                                         { 1, -1, 0, 0 },  // beginData
                                                         { 2, -3, 3, 1 },  // endData
                                                         { 1, -1, 1, 1 }   // stridesData
    ) {}
};

BOOST_FIXTURE_TEST_CASE(StridedSlice4DReverse, StridedSlice4DReverseFixture)
{
    RunTest<4>(
            {{"input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},
            {{"output", { 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f }}});
}

struct StridedSliceSimpleStrideFixture : StridedSliceFixture
{
    StridedSliceSimpleStrideFixture() : StridedSliceFixture({ 3, 2, 3, 1 }, // inputShape
                                                            { 0, 0, 0, 0 }, // beginData
                                                            { 3, 2, 3, 1 }, // endData
                                                            { 2, 2, 2, 1 }  // stridesData
    ) {}
};

BOOST_FIXTURE_TEST_CASE(StridedSliceSimpleStride, StridedSliceSimpleStrideFixture)
{
    RunTest<4>(
            {{"input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},
            {{"output", { 1.0f, 1.0f,
                          5.0f, 5.0f }}});
}

struct StridedSliceSimpleRangeMaskFixture : StridedSliceFixture
{
    StridedSliceSimpleRangeMaskFixture() : StridedSliceFixture({ 3, 2, 3, 1 }, // inputShape
                                                               { 1, 1, 1, 1 }, // beginData
                                                               { 1, 1, 1, 1 }, // endData
                                                               { 1, 1, 1, 1 }, // stridesData
                                                               (1 << 4) - 1,   // beginMask
                                                               (1 << 4) - 1    // endMask
    ) {}
};

BOOST_FIXTURE_TEST_CASE(StridedSliceSimpleRangeMask, StridedSliceSimpleRangeMaskFixture)
{
    RunTest<4>(
            {{"input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},
            {{"output", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                          5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}});
}

BOOST_AUTO_TEST_SUITE_END()
