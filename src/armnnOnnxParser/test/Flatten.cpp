//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_Flatter")
{
struct FlattenMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    FlattenMainFixture(const std::string& dataType)
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "flatten"
                         op_type: "Flatten"
                         attribute {
                           name: "axis"
                           i: 2
                           type: INT
                         }
                      }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 4
                                   }
                                   dim {
                                       dim_value: 9
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

struct FlattenDefaultAxisFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    FlattenDefaultAxisFixture(const std::string& dataType)
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "flatten"
                         op_type: "Flatten"
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
                                       dim_value: 18
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

struct FlattenAxisZeroFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    FlattenAxisZeroFixture(const std::string& dataType)
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "flatten"
                         op_type: "Flatten"
                         attribute {
                           name: "axis"
                           i: 0
                           type: INT
                         }
                      }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 1
                                   }
                                   dim {
                                       dim_value: 36
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

struct FlattenNegativeAxisFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    FlattenNegativeAxisFixture(const std::string& dataType)
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "flatten"
                         op_type: "Flatten"
                         attribute {
                           name: "axis"
                           i: -1
                           type: INT
                         }
                      }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 12
                                   }
                                   dim {
                                       dim_value: 3
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

struct FlattenInvalidNegativeAxisFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    FlattenInvalidNegativeAxisFixture(const std::string& dataType)
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "flatten"
                         op_type: "Flatten"
                         attribute {
                           name: "axis"
                           i: -5
                           type: INT
                         }
                      }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 12
                                   }
                                   dim {
                                       dim_value: 3
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

struct FlattenValidFixture : FlattenMainFixture
{
    FlattenValidFixture() : FlattenMainFixture("1") {
        Setup();
    }
};

struct FlattenDefaultValidFixture : FlattenDefaultAxisFixture
{
    FlattenDefaultValidFixture() : FlattenDefaultAxisFixture("1") {
        Setup();
    }
};

struct FlattenAxisZeroValidFixture : FlattenAxisZeroFixture
{
    FlattenAxisZeroValidFixture() : FlattenAxisZeroFixture("1") {
        Setup();
    }
};

struct FlattenNegativeAxisValidFixture : FlattenNegativeAxisFixture
{
    FlattenNegativeAxisValidFixture() : FlattenNegativeAxisFixture("1") {
        Setup();
    }
};

struct FlattenInvalidFixture : FlattenMainFixture
{
    FlattenInvalidFixture() : FlattenMainFixture("10") { }
};

struct FlattenInvalidAxisFixture : FlattenInvalidNegativeAxisFixture
{
    FlattenInvalidAxisFixture() : FlattenInvalidNegativeAxisFixture("1") { }
};

TEST_CASE_FIXTURE(FlattenValidFixture, "ValidFlattenTest")
{
    RunTest<2>({{"Input",
                          { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}},
                {{"Output",
                          { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}});
}

TEST_CASE_FIXTURE(FlattenDefaultValidFixture, "ValidFlattenDefaultTest")
{
    RunTest<2>({{"Input",
                    { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}},
               {{"Output",
                    { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}});
}

TEST_CASE_FIXTURE(FlattenAxisZeroValidFixture, "ValidFlattenAxisZeroTest")
{
    RunTest<2>({{"Input",
                    { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}},
               {{"Output",
                    { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}});
}

TEST_CASE_FIXTURE(FlattenNegativeAxisValidFixture, "ValidFlattenNegativeAxisTest")
{
    RunTest<2>({{"Input",
                    { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}},
               {{"Output",
                    { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f }}});
}

TEST_CASE_FIXTURE(FlattenInvalidFixture, "IncorrectDataTypeFlatten")
{
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

TEST_CASE_FIXTURE(FlattenInvalidAxisFixture, "IncorrectAxisFlatten")
{
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

}
