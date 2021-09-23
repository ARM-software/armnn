//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include "OnnxParserTestUtils.hpp"

TEST_SUITE("OnnxParser_Gather")
{

struct GatherMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GatherMainFixture(const std::vector<int>& indicesShape,
                      const std::vector<int>& indices,
                      const std::vector<int>& inputShape,
                      const std::vector<int>& outputShape)
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        output: "indices"
                        op_type: "Constant"
                        attribute {
                          name: "value"
                          t {
                            data_type: 7
                            )" + ConstructIndicesString(indicesShape, indices) + R"(
                            name: "value"
                          }
                          type: TENSOR
                        }
                      }
                      node {
                        input: "input"
                        input: "indices"
                        output: "output"
                        op_type: "Gather"
                        attribute {
                          name: "axis"
                          i: 0
                          type: INT
                        }
                      }
                      name: "gather-model"
                      input {
                        name: "input"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(inputShape) + R"(
                            }
                          }
                        }
                      }
                      output {
                        name: "output"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(outputShape) + R"(
                            }
                          }
                        }
                      }
                    })";
    }
    std::string ConstructIndicesString(const std::vector<int>& indicesShape, const std::vector<int>& indices)
    {
        std::string shapeStr;
        for (int i : indicesShape)
        {
            shapeStr = fmt::format(" {} dims: {}", shapeStr, i);
        }
        for (int i : indices)
        {
            shapeStr = fmt::format(" {} int64_data: {}", shapeStr, i);
        }
        return shapeStr;
    }
};

struct GatherScalarFixture : GatherMainFixture
{
    GatherScalarFixture() : GatherMainFixture({ }, { 0 }, { 8 }, { })
    {
        Setup();
    }
};

struct Gather1dFixture : GatherMainFixture
{
    Gather1dFixture() : GatherMainFixture({ 4 }, { 0, 2, 1, 5 }, { 8 }, { 4 })
    {
        Setup();
    }
};

struct Gather2dFixture : GatherMainFixture
{
    Gather2dFixture() : GatherMainFixture({ 3 }, { 1, 3, 4 }, { 5, 2 }, { 3, 2 })
    {
        Setup();
    }
};

struct Gather3dMultiIndicesFixture : GatherMainFixture
{
    Gather3dMultiIndicesFixture() : GatherMainFixture({ 2, 3 }, { 1, 2, 1, 2, 1, 0 }, { 3, 2, 3 }, { 2, 3, 2, 3 })
    {
        Setup();
    }
};

struct Gather4dFixture : GatherMainFixture
{
    Gather4dFixture() : GatherMainFixture({ 3 }, { 0, 1, 3 }, { 5, 4, 3, 2 }, { 3, 4, 3, 2 })
    {
        Setup();
    }
};

TEST_CASE_FIXTURE(GatherScalarFixture, "GatherScalarTest")
{
    RunTest<1, float>({{"input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }}},
                      {{"output", { 1.0f }}});
}

TEST_CASE_FIXTURE(Gather1dFixture, "Gather1dTest")
{
    RunTest<1, float>({{"input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }}},
                      {{"output", { 1.0f, 3.0f, 2.0f, 6.0f }}});
}

TEST_CASE_FIXTURE(Gather2dFixture, "Gather2dTest")
{
    RunTest<2, float>({{"input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f }}},
                      {{"output", { 3.0f, 4.0f, 7.0f, 8.0f, 9.0f, 10.0f }}});
}

TEST_CASE_FIXTURE(Gather3dMultiIndicesFixture, "Gather3dMultiIndicesTest")
{
    RunTest<3, float>({{"input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f }}},
                      {{"output", { 7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    1.0f,  2.0f,  3.0f,
                                    4.0f,  5.0f,  6.0f }}});
}

TEST_CASE_FIXTURE(Gather4dFixture, "Gather4dTest")
{
    RunTest<4, float>({{"input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                   11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                   16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                   21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                   26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                                   31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
                                   36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                                   41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                                   46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                                   51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
                                   56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
                                   61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
                                   66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                                   71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
                                   76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
                                   81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
                                   86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                                   91.0f, 92.0f, 93.0f, 94.0f, 95.0f,
                                   96.0f, 97.0f, 98.0f, 99.0f, 100.0f,
                                   101.0f, 102.0f, 103.0f, 104.0f, 105.0f,
                                   106.0f, 107.0f, 108.0f, 109.0f, 110.0f,
                                   111.0f, 112.0f, 113.0f, 114.0f, 115.0f,
                                   116.0f, 117.0f, 118.0f, 119.0f, 120.0f }}},
                      {{"output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f,  8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f,
                                    37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f,
                                    43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
                                    73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f,
                                    79.0f, 80.0f, 81.0f, 82.0f, 83.0f, 84.0f,
                                    85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                                    91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f }}});
}

struct GatherRawDataFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GatherRawDataFixture()
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        output: "indices"
                        op_type: "Constant"
                        attribute {
                          name: "value"
                          t {
                            dims: 3
                            data_type: 7
                            raw_data:
                      "\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000"
                            name: "value"
                          }
                          type: TENSOR
                        }
                      }
                      node {
                        input: "input"
                        input: "indices"
                        output: "output"
                        op_type: "Gather"
                        attribute {
                          name: "axis"
                          i: 0
                          type: INT
                        }
                      }
                      name: "gather-model"
                      input {
                        name: "input"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 5
                              }
                              dim {
                                dim_value: 4
                              }
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 2
                              }
                            }
                          }
                        }
                      }
                      output {
                        name: "output"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 4
                              }
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 2
                              }
                            }
                          }
                        }
                      }
                    })";
        Setup();
    }
};

TEST_CASE_FIXTURE(GatherRawDataFixture, "GatherRawDataTest")
{
    RunTest<4, float>({{"input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                   11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                   16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                   21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                   26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                                   31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
                                   36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                                   41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                                   46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                                   51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
                                   56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
                                   61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
                                   66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                                   71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
                                   76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
                                   81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
                                   86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                                   91.0f, 92.0f, 93.0f, 94.0f, 95.0f,
                                   96.0f, 97.0f, 98.0f, 99.0f, 100.0f,
                                   101.0f, 102.0f, 103.0f, 104.0f, 105.0f,
                                   106.0f, 107.0f, 108.0f, 109.0f, 110.0f,
                                   111.0f, 112.0f, 113.0f, 114.0f, 115.0f,
                                   116.0f, 117.0f, 118.0f, 119.0f, 120.0f }}},
                      {{"output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f,  8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f,
                                    37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f,
                                    43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
                                    73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f,
                                    79.0f, 80.0f, 81.0f, 82.0f, 83.0f, 84.0f,
                                    85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                                    91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f }}});
}

}
