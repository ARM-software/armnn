//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include "OnnxParserTestUtils.hpp"

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
                              )" + armnnUtils::ConstructTensorShapeString(inputShape) + R"(
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
};

struct ShapeFloatFixture : ShapeMainFixture
{
    ShapeFloatFixture() : ShapeMainFixture("1", "7", "4", { 1, 3, 1, 5 })
    {
        Setup();
    }
};

struct ShapeIntFixture : ShapeMainFixture
{
    ShapeIntFixture() : ShapeMainFixture("7", "7", "4", { 1, 3, 1, 5 })
    {
        Setup();
    }
};

struct Shape3DFixture : ShapeMainFixture
{
    Shape3DFixture() : ShapeMainFixture("1", "7", "3", { 3, 2, 3 })
    {
        Setup();
    }
};

struct Shape2DFixture : ShapeMainFixture
{
    Shape2DFixture() : ShapeMainFixture("1", "7", "2", { 2, 3 })
    {
        Setup();
    }
};

struct Shape1DFixture : ShapeMainFixture
{
    Shape1DFixture() : ShapeMainFixture("1", "7", "1", { 5 })
    {
        Setup();
    }
};

TEST_CASE_FIXTURE(ShapeFloatFixture, "FloatValidShapeTest")
{
    RunTest<1, int>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
                                 0.0f, 1.0f, 2.0f, 3.0f, 4.0f }}}, {{"Output", { 1, 3, 1, 5 }}});
}

TEST_CASE_FIXTURE(ShapeIntFixture, "IntValidShapeTest")
{
    RunTest<1, int>({{"Input", { 0, 1, 2, 3, 4,
                                 4, 3, 2, 1, 0,
                                 0, 1, 2, 3, 4 }}}, {{"Output", { 1, 3, 1, 5 }}});
}

TEST_CASE_FIXTURE(Shape3DFixture, "Shape3DTest")
{
    RunTest<1, int>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
                                 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }}}, {{"Output", { 3, 2, 3 }}});
}

TEST_CASE_FIXTURE(Shape2DFixture, "Shape2DTest")
{
    RunTest<1, int>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }}}, {{"Output", { 2, 3 }}});
}

TEST_CASE_FIXTURE(Shape1DFixture, "Shape1DTest")
{
    RunTest<1, int>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }}}, {{"Output", { 5 }}});
}

}
