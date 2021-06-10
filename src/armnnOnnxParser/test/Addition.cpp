//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_Addition")
{
struct AddMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    AddMainFixture(const std::string& dataType)
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
                        name: "Input0"
                        type {
                          tensor_type {
                            elem_type: )" + dataType + R"(
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
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
                      input {
                         name: "Input1"
                         type {
                           tensor_type {
                             elem_type: )" + dataType + R"(
                             shape {
                               dim {
                                 dim_value: 1
                               }
                               dim {
                                 dim_value: 1
                               }
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
                       node {
                            input: "Input0"
                            input: "Input1"
                            output: "Output"
                            name: "addition"
                            op_type: "Add"
                            doc_string: ""
                            domain: ""
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
                                           dim_value: 1
                                       }
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

struct AddValidFixture : AddMainFixture
{
    AddValidFixture() : AddMainFixture("1") {
        Setup();
    }
};

struct AddInvalidFixture : AddMainFixture
{
    AddInvalidFixture() : AddMainFixture("6") { }
};

struct AddValidBroadcastFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    AddValidBroadcastFixture() {

        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK"
                   producer_version:  "2.5.1"
                   domain:  "ai.cntk"
                   model_version: 1
                   graph {
                     name:  "CNTKGraph"
                     input {
                        name: "Input0"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 4
                              }
                            }
                          }
                        }
                      }
                      input {
                         name: "Input1"
                         type {
                           tensor_type {
                             elem_type: 1
                             shape {
                                 dim {
                                   dim_value: 4
                                 }
                             }
                           }
                         }
                       }
                       node {
                            input: "Input0"
                            input: "Input1"
                            output: "Output"
                            name: "addition"
                            op_type: "Add"
                            doc_string: ""
                            domain: ""
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
                                           dim_value: 1
                                       }
                                       dim {
                                           dim_value: 1
                                       }
                                       dim {
                                           dim_value: 4
                                       }
                                   }
                                }
                            }
                        }
                    }
                   opset_import {
                      version: 7
                    })";
        Setup();
    }
};

struct AddInvalidBroadcastFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    AddInvalidBroadcastFixture() {

        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK"
                   producer_version:  "2.5.1"
                   domain:  "ai.cntk"
                   model_version: 1
                   graph {
                     name:  "CNTKGraph"
                     input {
                        name: "Input0"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 3
                              }
                            }
                          }
                        }
                      }
                      input {
                         name: "Input1"
                         type {
                           tensor_type {
                             elem_type: 1
                             shape {
                                 dim {
                                   dim_value: 4
                                 }
                             }
                           }
                         }
                       }
                       node {
                            input: "Input0"
                            input: "Input1"
                            output: "Output"
                            name: "addition"
                            op_type: "Add"
                            doc_string: ""
                            domain: ""
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
                                           dim_value: 1
                                       }
                                       dim {
                                           dim_value: 1
                                       }
                                       dim {
                                           dim_value: 4
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

struct AddScalarFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    AddScalarFixture(const std::string& dataType)
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
                        name: "Input0"
                        type {
                          tensor_type {
                            elem_type: )" + dataType + R"(
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
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
                      input {
                         name: "Input1"
                         type {
                           tensor_type {
                             elem_type: )" + dataType + R"(
                             shape {
                               dim {
                                 dim_value: 1
                               }
                             }
                           }
                         }
                       }
                       node {
                            input: "Input0"
                            input: "Input1"
                            output: "Output"
                            name: "addition"
                            op_type: "Add"
                            doc_string: ""
                            domain: ""
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
                                           dim_value: 1
                                       }
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

struct AddValidScalarFixture : AddScalarFixture
{
    AddValidScalarFixture() : AddScalarFixture("1") {
        Setup();
    }
};

struct AddInvalidScalarFixture : AddScalarFixture
{
    AddInvalidScalarFixture() : AddScalarFixture("6") { }
};

TEST_CASE_FIXTURE(AddValidFixture, "ValidAddTest")
{
    RunTest<4>({{"Input0", {1.0f, 2.0f, -3.0f, -4.0f}},
                {"Input1", {1.0f, 2.0f, 3.0, 4.0f}}}, {{"Output", {2.0, 4.0, 0, 0.0}}});
}

TEST_CASE_FIXTURE(AddInvalidFixture, "IncorrectDataTypeAdd")
{
   CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

TEST_CASE_FIXTURE(AddInvalidBroadcastFixture, "InvalidBroadcastAdd")
{
   CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

TEST_CASE_FIXTURE(AddValidBroadcastFixture, "ValidBroadcastAdd")
{
    RunTest<4>({{"Input0", {1.0f, 2.0f, -3.0f, -4.0f}},
                {"Input1", {1.0f, 2.0f, 3.0, 4.0f}}}, {{"Output", {2.0, 4.0, 0, 0.0}}});
}

TEST_CASE_FIXTURE(AddValidScalarFixture, "ValidAddScalarTest")
{
    RunTest<4>({{"Input0", {1.0f, 2.0f, -3.0f, -4.0f}},
                {"Input1", {-8.0f}}}, {{"Output", {-7.0, -6.0, -11.0, -12.0}}});
}

TEST_CASE_FIXTURE(AddInvalidScalarFixture, "IncorrectDataTypeAddScalar")
{
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

}