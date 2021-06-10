//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_Relu")
{
struct ReluMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ReluMainFixture()
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
                                dim_value: 4
                              }
                            }
                          }
                        }
                      }
                     node {
                         input: "Input"
                         output: "Output"
                         name: "ActivationLayer"
                         op_type: "Relu"
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

TEST_CASE_FIXTURE(ReluMainFixture, "ValidReluTest")
{
    RunTest<1>({{"Input",  { -1.0f, -0.5f, 1.25f, -3.0f}}},
               {{ "Output", { 0.0f, 0.0f, 1.25f, 0.0f}}});
}

}
