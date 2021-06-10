//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_FullyConnected")
{
// A MatMul in isolation, not connected to an add. Should result in a non-biased FullyConnected layer.
struct MatMulFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    MatMulFixture()
    {
        m_Prototext = R"(
                    ir_version: 3
                    producer_name:  "CNTK "
                    producer_version:  "2.5.1 "
                    domain:  "ai.cntk "
                    model_version: 1
                    graph {
                      name:  "CNTKGraph "
                      input {
                         name: "Input"
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
                             }
                           }
                         }
                       }
                       input {
                          name: "Const"
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
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          dims: 1
                          data_type: 1
                          float_data: 17.0
                          name: "Const"
                       }
                       node {
                           input: "Input"
                           input: "Const"
                           output: "Output"
                           name: "SimpleMatmul"
                           op_type: "MatMul"
                       }
                      output {
                           name:  "Output"
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

TEST_CASE_FIXTURE(MatMulFixture, "MatMul")
{
    RunTest<1>({{"Input", { 2 }}}, {{"Output", { 34 }}});
}

// In Onnx fully connected layers are expressed as a MatMul followed by an Add.
// The OnnxParser must detect this case and convert them to a FullyConnected layer.
struct FullyConnectedFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    FullyConnectedFixture()
    {
        m_Prototext = R"(
                    ir_version: 3
                    producer_name:  "CNTK "
                    producer_version:  "2.5.1 "
                    domain:  "ai.cntk "
                    model_version: 1
                    graph {
                      name:  "CNTKGraph "
                      input {
                         name: "Input"
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
                             }
                           }
                         }
                       }
                       input {
                          name: "Weight"
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
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          dims: 1
                          data_type: 1
                          float_data: 2
                          name: "Weight"
                       }
                       input {
                          name: "Bias"
                          type {
                            tensor_type {
                              elem_type: 1
                              shape {
                                dim {
                                  dim_value: 1
                                }
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          data_type: 1
                          float_data: 1
                          name: "Bias"
                       }
                       node {
                           input: "Input"
                           input: "Weight"
                           output: "AddInput"
                           name: "FCMatmul"
                           op_type: "MatMul"
                       }
                       node {
                           input: "AddInput"
                           input: "Bias"
                           output: "Output"
                           name: "FCAdd"
                           op_type: "Add"
                       }
                       value_info {
                            name: "AddInput"
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
                                }
                              }
                            }
                          }
                      output {
                           name:  "Output"
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

TEST_CASE_FIXTURE(FullyConnectedFixture, "FullyConnected")
{
    RunTest<1>({{"Input", { 3 }}}, {{"Output", { 7 }}});
}


// Similar to FullyConnectedFixture, but this time the MatMul's output is used by two Adds. This should result
// in two FullyConnected layers being created.
//      I
//      |
//      M -- C
//     / \'
// C-- A  A -- C
//     \ /
//      A
struct MatMulUsedInTwoFcFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    MatMulUsedInTwoFcFixture()
    {
        m_Prototext = R"(
                    ir_version: 3
                    producer_name:  "CNTK "
                    producer_version:  "2.5.1 "
                    domain:  "ai.cntk "
                    model_version: 1
                    graph {
                      name:  "CNTKGraph "
                      input {
                         name: "Input"
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
                             }
                           }
                         }
                       }
                       input {
                          name: "Weight"
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
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          dims: 1
                          data_type: 1
                          float_data: 2
                          name: "Weight"
                       }
                       input {
                          name: "Bias"
                          type {
                            tensor_type {
                              elem_type: 1
                              shape {
                                dim {
                                  dim_value: 1
                                }
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          data_type: 1
                          float_data: 1
                          name: "Bias"
                       }
                       input {
                          name: "Bias_1"
                          type {
                            tensor_type {
                              elem_type: 1
                              shape {
                                dim {
                                  dim_value: 1
                                }
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          data_type: 1
                          float_data: 10.0
                          name: "Bias_1"
                       }
                       node {
                           input: "Input"
                           input: "Weight"
                           output: "AddInput"
                           name: "FCMatmul"
                           op_type: "MatMul"
                       }
                       node {
                           input: "AddInput"
                           input: "Bias"
                           output: "AddOutput"
                           name: "FCAdd"
                           op_type: "Add"
                       }
                       node {
                           input: "AddInput"
                           input: "Bias_1"
                           output: "AddOutput_1"
                           name: "FCAdd_1"
                           op_type: "Add"
                       }
                       node {
                           input: "AddOutput"
                           input: "AddOutput_1"
                           output: "Output"
                           name: "FinalAdd"
                           op_type: "Add"
                       }
                       value_info {
                            name: "AddInput"
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
                                }
                              }
                            }
                          }
                      value_info {
                           name:  "AddOutput"
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
                                }
                              }
                           }
                       }
                       value_info {
                            name:  "AddOutput_1"
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
                                 }
                               }
                            }
                        }
                        output {
                             name:  "Output"
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

TEST_CASE_FIXTURE(MatMulUsedInTwoFcFixture, "MatMulUsedInTwoFc")
{
    RunTest<1>({{"Input", { 3 }}}, {{"Output", { 23 }}});
}


// Similar to MatMulUsedInTwoFc, but this time the Adds are 'staggered' (see diagram), which means that only one
// FullyConnected layer can be created (the other should just be an Add).
//        I
//        |
//        M -- C1
//       / \'
// C2 -- A  |
//       \ /
//        A
struct MatMulUsedInTwoFcStaggeredFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    MatMulUsedInTwoFcStaggeredFixture()
    {
        m_Prototext = R"(
                    ir_version: 3
                    producer_name:  "CNTK "
                    producer_version:  "2.5.1 "
                    domain:  "ai.cntk "
                    model_version: 1
                    graph {
                      name:  "CNTKGraph "
                      input {
                         name: "Input"
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
                             }
                           }
                         }
                       }
                       input {
                          name: "Weight"
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
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          dims: 1
                          data_type: 1
                          float_data: 2
                          name: "Weight"
                       }
                       input {
                          name: "Bias"
                          type {
                            tensor_type {
                              elem_type: 1
                              shape {
                                dim {
                                  dim_value: 1
                                }
                              }
                            }
                          }
                        }
                        initializer {
                          dims: 1
                          data_type: 1
                          float_data: 1
                          name: "Bias"
                       }
                        node {
                           input: "Input"
                           input: "Weight"
                           output: "AddInput"
                           name: "MatmulFC&NFC"
                           op_type: "MatMul"
                       }
                       node {
                           input: "AddInput"
                           input: "Bias"
                           output: "AddOutput"
                           name: "FCAdd"
                           op_type: "Add"
                       }

                       node {
                           input: "AddInput"
                           input: "AddOutput"
                           output: "Output"
                           name: "FinalAdd"
                           op_type: "Add"
                       }
                       value_info {
                            name: "AddInput"
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
                                }
                              }
                            }
                          }
                      value_info {
                           name:  "AddOutput"
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
                                }
                              }
                           }
                       }
                       output {
                             name:  "Output"
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

TEST_CASE_FIXTURE(MatMulUsedInTwoFcStaggeredFixture, "MatMulUsedInTwoFcStaggered")
{
    RunTest<1>({{"Input", { 3 }}}, {{"Output", { 13 }}});
}

}
