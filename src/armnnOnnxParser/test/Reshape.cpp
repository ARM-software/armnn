//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
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
                             elem_type: 7
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
                        data_type: 7
                        int64_data: 2
                        int64_data: 2
                        name: "Shape"
                     }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
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

struct ReshapeRank4Fixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ReshapeRank4Fixture(const std::string& dataType)
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
                                dim_value: 2
                              }
                              dim {
                                dim_value: 2
                              }
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 3
                              }
                            }
                          }
                        }
                      }
                      input {
                         name: "Shape"
                         type {
                           tensor_type {
                             elem_type: 7
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
                        data_type: 7
                        int64_data: 2
                        int64_data: 2
                        name: "Shape"
                     }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 6
                                   }
                                   dim {
                                       dim_value: 6
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
    ReshapeValidFixture() : ReshapeMainFixture("1") {
        Setup();
    }
};

struct ReshapeValidRank4Fixture : ReshapeRank4Fixture
{
    ReshapeValidRank4Fixture() : ReshapeRank4Fixture("1") {
        Setup();
    }
};

struct ReshapeInvalidFixture : ReshapeMainFixture
{
    ReshapeInvalidFixture() : ReshapeMainFixture("10") { }
};

BOOST_FIXTURE_TEST_CASE(ValidReshapeTest, ReshapeValidFixture)
{
    RunTest<2>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f }}}, {{"Output", { 0.0f, 1.0f, 2.0f, 3.0f }}});
}

BOOST_FIXTURE_TEST_CASE(ValidRank4ReshapeTest, ReshapeValidRank4Fixture)
{
    RunTest<2>(
        {{"Input",
                   {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                    1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                    1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f}}},
        {{"Output",
                    {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                     1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                     1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f}}});
}

BOOST_FIXTURE_TEST_CASE(IncorrectDataTypeReshape, ReshapeInvalidFixture)
{
   BOOST_CHECK_THROW(Setup(), armnn::ParseException);
}

BOOST_AUTO_TEST_SUITE_END()
