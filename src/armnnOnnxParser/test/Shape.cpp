//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_Shape")
{
struct ShapeMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ShapeMainFixture(const std::string& inputType,
                     const std::string& outputType,
                     const std::string& outputDim,
                     const std::vector<int>& inputShape)
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "Input"
                        output: "Output"
                        op_type: "Shape"
                      }
                      name: "shape-model"
                      input {
                        name: "Input"
                        type {
                          tensor_type {
                            elem_type: )" + inputType + R"(
                            shape {
                              )" + ConstructShapeString(inputShape) + R"(
                            }
                          }
                        }
                      }
                      output {
                        name: "Output"
                        type {
                          tensor_type {
                            elem_type: )" + outputType + R"(
                            shape {
                              dim {
                                dim_value: )" + outputDim + R"(
                              }
                            }
                          }
                        }
                      }
                    }
                    opset_import {
                      version: 10
                    })";
    }
    std::string ConstructShapeString(const std::vector<int>& shape)
    {
        std::string shapeStr;
        for (int i : shape)
        {
            shapeStr = fmt::format("{} dim {{ dim_value: {} }}", shapeStr, i);
        }
        return shapeStr;
    }
};

struct ShapeValidFloatFixture : ShapeMainFixture
{
    ShapeValidFloatFixture() : ShapeMainFixture("1", "7", "4", { 1, 3, 1, 5 }) {
        Setup();
    }
};

struct ShapeValidIntFixture : ShapeMainFixture
{
    ShapeValidIntFixture() : ShapeMainFixture("7", "7", "4", { 1, 3, 1, 5 }) {
        Setup();
    }
};

struct Shape3DFixture : ShapeMainFixture
{
    Shape3DFixture() : ShapeMainFixture("1", "7", "3", { 3, 2, 3 }) {
        Setup();
    }
};

struct Shape2DFixture : ShapeMainFixture
{
    Shape2DFixture() : ShapeMainFixture("1", "7", "2", { 2, 3 }) {
        Setup();
    }
};

struct Shape1DFixture : ShapeMainFixture
{
    Shape1DFixture() : ShapeMainFixture("1", "7", "1", { 5 }) {
        Setup();
    }
};

struct ShapeInvalidFixture : ShapeMainFixture
{
    ShapeInvalidFixture() : ShapeMainFixture("1", "1", "4", { 1, 3, 1, 5 }) {}
};

TEST_CASE_FIXTURE(ShapeValidFloatFixture, "FloatValidShapeTest")
{
    RunTest<2, int>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
                                 0.0f, 1.0f, 2.0f, 3.0f, 4.0f }}}, {{"Output", { 1, 3, 1, 5 }}});
}

TEST_CASE_FIXTURE(ShapeValidIntFixture, "IntValidShapeTest")
{
    RunTest<2, int>({{"Input", { 0, 1, 2, 3, 4,
                                 4, 3, 2, 1, 0,
                                 0, 1, 2, 3, 4 }}}, {{"Output", { 1, 3, 1, 5 }}});
}

TEST_CASE_FIXTURE(Shape3DFixture, "Shape3DTest")
{
    RunTest<2, int>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
                                 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }}}, {{"Output", { 3, 2, 3 }}});
}

TEST_CASE_FIXTURE(Shape2DFixture, "Shape2DTest")
{
    RunTest<2, int>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }}}, {{"Output", { 2, 3 }}});
}

TEST_CASE_FIXTURE(Shape1DFixture, "Shape1DTest")
{
    RunTest<2, int>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }}}, {{"Output", { 5 }}});
}

TEST_CASE_FIXTURE(ShapeInvalidFixture, "IncorrectOutputDataShapeTest")
{
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

}
