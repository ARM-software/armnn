//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_Conv2D")
{
struct SimpleConv2DFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    SimpleConv2DFixture()
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
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
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
                      initializer {
                          dims: 1
                          dims: 1
                          dims: 3
                          dims: 3
                          data_type: 1
                          float_data: 2
                          float_data: 1
                          float_data: 0
                          float_data: 6
                          float_data: 2
                          float_data: 1
                          float_data: 4
                          float_data: 1
                          float_data: 2
                          name: "Weight"
                        }
                      node {
                         input: "Input"
                         input: "Weight"
                         output: "Output"
                         name: "Convolution"
                         op_type: "Conv"
                         attribute {
                           name: "kernel_shape"
                           ints: 3
                           ints: 3
                           type: INTS
                         }
                         attribute {
                           name: "strides"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "auto_pad"
                           s: "VALID"
                           type: STRING
                         }
                         attribute {
                           name: "group"
                           i: 1
                           type: INT
                         }
                         attribute {
                           name: "dilations"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
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

struct Conv2DWithBiasesFixture :  public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    Conv2DWithBiasesFixture() {
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
                      initializer {
                          dims: 1
                          dims: 1
                          dims: 2
                          dims: 2
                          data_type: 1
                          float_data: 2
                          float_data: 1
                          float_data: 0
                          float_data: 6
                          name: "Weight"
                        }
                        input {
                          name: "Bias"
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
                        initializer {
                            dims: 4
                            data_type: 1
                            float_data: 10
                            float_data: 0
                            float_data: 0
                            float_data: 0
                            name: "Bias"
                          }
                      node {
                         input: "Input"
                         input: "Weight"
                         input: "Bias"
                         output: "Output"
                         name: "Convolution"
                         op_type: "Conv"
                         attribute {
                           name: "kernel_shape"
                           ints: 2
                           ints: 2
                           type: INTS
                         }
                         attribute {
                           name: "strides"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "auto_pad"
                           s: "SAME_UPPER"
                           type: STRING
                         }
                         attribute {
                           name: "group"
                           i: 1
                           type: INT
                         }
                         attribute {
                           name: "dilations"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
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
        Setup();
    }
};


struct Conv2DDimReducingFixture :  public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    Conv2DDimReducingFixture() {
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
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 3
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
                        name: "Weight"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 2
                              }
                              dim {
                                dim_value: 3
                              }
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
                          dims: 2
                          dims: 3
                          dims: 1
                          dims: 1
                          data_type: 1
                          float_data: -1
                          float_data: 2
                          float_data: 0
                          float_data: 1
                          float_data: 0
                          float_data: 0
                          name: "Weight"
                        }
                      node {
                         input: "Input"
                         input: "Weight"
                         output: "Output"
                         name: "Convolution"
                         op_type: "Conv"
                         attribute {
                           name: "kernel_shape"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "strides"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "group"
                           i: 1
                           type: INT
                         }
                         attribute {
                           name: "dilations"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
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
                                       dim_value: 2
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
        Setup();
    }
};

struct Conv2DwithDilationFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    Conv2DwithDilationFixture()
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
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
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
                      initializer {
                          dims: 1
                          dims: 1
                          dims: 3
                          dims: 3
                          data_type: 1
                          float_data: 2
                          float_data: 1
                          float_data: 0
                          float_data: 6
                          float_data: 2
                          float_data: 1
                          float_data: 4
                          float_data: 1
                          float_data: 2
                          name: "Weight"
                        }
                      node {
                         input: "Input"
                         input: "Weight"
                         output: "Output"
                         name: "Convolution"
                         op_type: "Conv"
                         attribute {
                           name: "kernel_shape"
                           ints: 3
                           ints: 3
                           type: INTS
                         }
                         attribute {
                           name: "strides"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "auto_pad"
                           s: "VALID"
                           type: STRING
                         }
                         attribute {
                           name: "group"
                           i: 1
                           type: INT
                         }
                         attribute {
                           name: "dilations"
                           ints: 2
                           ints: 2
                           type: INTS
                         }
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
        Setup();
    }
};

TEST_CASE_FIXTURE(SimpleConv2DFixture, "ValidConvTest")
{
    RunTest<4>({{"Input", {1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0}}},
              {{"Output", {1.0 * 2 + 2.0 * 1 + 3.0 * 0 +
                           4.0 * 6 + 5.0 * 2 + 6.0 * 1 +
                           7.0 * 4 + 8.0 * 1 + 9.0 * 2}}});
}

TEST_CASE_FIXTURE(Conv2DWithBiasesFixture, "ValidConvWithBiasTest")
{
    RunTest<4>({{"Input", {1.0, 2.0,
                           3.0, 4.0}}},
              {{"Output", {1.0 * 2 + 2.0 * 1 + 3.0 * 0 + 4 * 6 + 10,
                           2.0 * 2 + 0 * 1 + 4.0 * 0 + 0 * 6 + 10,
                           3.0 * 2 + 4.0 * 1 + 0 * 0 + 0 * 6 + 10,
                           4.0 * 2 + 0 * 1 + 0 * 0 + 0 * 6 + 10}}});
}

TEST_CASE_FIXTURE(Conv2DDimReducingFixture, "ValidConvDimReducTest")
{
    RunTest<4>({{"Input", {1.0, 2.0, 3.0, 4.0, -1, -2, 3, 4, 1 , 1, 1, 1 }}},
              {{"Output", {-1 * 1 + 2 * -1, -1 * 2 + 2 * -2,
                           -1 * 3 + 2 * 3,  -1 * 4 + 2 * 4,
                           1, 2, 3, 4}}});
}

TEST_CASE_FIXTURE(Conv2DwithDilationFixture, "ValidConvWithDilationTest")
{
    RunTest<4>({{"Input", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                           1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0, 10.0, 11.0, 12.0}}},
               {{"Output", {39.0, 58.0, 153.0, 172.0 }}});
}

}
