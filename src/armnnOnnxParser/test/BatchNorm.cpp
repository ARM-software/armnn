//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_BatchNorm")
{
struct BatchNormalizationMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    BatchNormalizationMainFixture()
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
                         name: "mean"
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
                       input {
                          name: "var"
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
                        input {
                           name: "scale"
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
                         input {
                            name: "bias"
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
                     node {
                         input: "Input"
                         input: "scale"
                         input: "bias"
                         input: "mean"
                         input: "var"
                         output: "Output"
                         name: "batchNorm"
                         op_type: "BatchNormalization"
                         attribute {
                           name: "epsilon"
                           f:  0.0010000000475
                           type: 1
                         }
                      }
                      initializer {
                          dims: 1
                          data_type: 1
                          float_data: 5.0
                          name: "mean"
                        }
                      initializer {
                        dims: 1
                        data_type: 1
                        float_data: 2.0
                        name: "var"
                      }
                      initializer {
                        dims: 1
                        data_type: 1
                        float_data: 0.0
                        name: "bias"
                      }
                      initializer {
                        dims: 1
                        data_type: 1
                        float_data: 1.0
                        name: "scale"
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
                                       dim_value: 3
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
        Setup();
    }
};

TEST_CASE_FIXTURE(BatchNormalizationMainFixture, "ValidBatchNormalizationTest")
{
    RunTest<4>({{"Input", {1, 2, 3, 4, 5, 6, 7, 8, 9}}},             // Input data.
               {{"Output", {-2.8277204f, -2.12079024f, -1.4138602f,
                -0.7069301f, 0.0f, 0.7069301f,
                1.4138602f, 2.12079024f, 2.8277204f}}});  // Expected output data.
}


struct BatchNormalizationBisFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    BatchNormalizationBisFixture()
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
                                dim_value: 2
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
                         name: "mean"
                         type {
                           tensor_type {
                             elem_type: 1
                             shape {
                               dim {
                                 dim_value: 2
                               }
                             }
                           }
                         }
                       }
                       input {
                          name: "var"
                          type {
                            tensor_type {
                              elem_type: 1
                              shape {
                                dim {
                                  dim_value: 2
                                }
                              }
                            }
                          }
                        }
                        input {
                           name: "scale"
                           type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                 dim {
                                   dim_value: 2
                                 }
                               }
                             }
                           }
                         }
                         input {
                            name: "bias"
                            type {
                              tensor_type {
                                elem_type: 1
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
                         input: "scale"
                         input: "bias"
                         input: "mean"
                         input: "var"
                         output: "Output"
                         name: "batchNorm"
                         op_type: "BatchNormalization"
                         attribute {
                           name: "epsilon"
                           f:  0.00001
                           type: 1
                         }
                      }
                      initializer {
                          dims: 2
                          data_type: 1
                          float_data: 0.0
                          float_data: 3.0
                          name: "mean"
                        }
                      initializer {
                        dims: 2
                        data_type: 1
                        float_data: 1.0
                        float_data: 1.5
                        name: "var"
                      }
                      initializer {
                        dims: 2
                        data_type: 1
                        float_data: 0.0
                        float_data: 1.0
                        name: "bias"
                      }
                      initializer {
                        dims: 2
                        data_type: 1
                        float_data: 1.0
                        float_data: 1.5
                        name: "scale"
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
                                       dim_value: 1
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
        Setup();
    }
};

TEST_CASE_FIXTURE(BatchNormalizationBisFixture, "ValidBatchNormalizationBisTest")
{
    RunTest<4>({{"Input", {-1, 0.0, 1, 2, 3.0, 4.0}}},           // Input data.
               {{"Output", {-0.999995f, 0.0, 0.999995f,
                            -0.22474074f, 1.0f, 2.2247407f}}});  // Expected output data.
}

}
