//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <boost/test/unit_test.hpp>
#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(OnnxParser)

struct ReshapeMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ReshapeMainFixture(const std::string& dataType)
    {
        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK"
                   producer_version:  "2.5.1"
                   domain:  "ai.cntk"
                   model_version: 1
                   graph {
                     name:  "CNTKGraph"
                     input {
                        name: "Input"
                        type {
                          tensor_type {
                            elem_type: )" + dataType + R"(
                            shape {
                              dim {
                                dim_value: 4
                              }
                            }
                          }
                        }
                      }
                      input {
                         name: "Shape"
                         type {
                           tensor_type {
                             elem_type: INT64
                             shape {
                               dim {
                                 dim_value: 2
                               }
                             }
                           }
                         }
                       }
                     node {
                         input: "Input"
                         input: "Shape"
                         output: "Output"
                         name: "reshape"
                         op_type: "Reshape"

                      }
                      initializer {
                        dims: 2
                        data_type: INT64
                        int64_data: 2
                        int64_data: 2
                        name: "Shape"
                     }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: FLOAT
                               shape {
                                   dim {
                                       dim_value: 2
                                   }
                                   dim {
                                       dim_value: 2
                                   }
                               }
                            }
                          }
                       }
                    }
                   opset_import {
                      version: 7
                    })";
    }
};

struct ReshapeValidFixture : ReshapeMainFixture
{
    ReshapeValidFixture() : ReshapeMainFixture("FLOAT") {
        Setup();
    }
};

struct ReshapeInvalidFixture : ReshapeMainFixture
{
    ReshapeInvalidFixture() : ReshapeMainFixture("FLOAT16") { }
};

BOOST_FIXTURE_TEST_CASE(ValidReshapeTest, ReshapeValidFixture)
{
    RunTest<2>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f }}}, {{"Output", { 0.0f, 1.0f, 2.0f, 3.0f }}});
}

BOOST_FIXTURE_TEST_CASE(IncorrectDataTypeReshape, ReshapeInvalidFixture)
{
   BOOST_CHECK_THROW(Setup(), armnn::ParseException);
}

BOOST_AUTO_TEST_SUITE_END()
