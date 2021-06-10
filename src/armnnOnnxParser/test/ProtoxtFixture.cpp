//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include  "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_PrototxtFixture")
{
struct ProtoxtTestFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ProtoxtTestFixture()
    {
        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK "
                   producer_version:  "2.5.1 "
                   domain:  "ai.cntk "
                   model_version: 1
                   graph {
                     name:  "CNTKGraph "
                     node {
                        input:  "Input"
                        output:  "Output"
                        name:  "Plus112"
                        op_type:  "Add "
                      }
                      input {
                          name:  "Input"
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
                                    dim_value: 10
                                 }
                               }
                             }
                          }
                      }
                   }
                   opset_import {
                      version: 7
                    })";
       // Setup();
    }
};


TEST_CASE_FIXTURE(ProtoxtTestFixture, "ProtoxtTest")
{
    //TODO : add a test to check if the inputs and outputs are correctly inferred.
}

TEST_CASE_FIXTURE(ProtoxtTestFixture, "ProtoxtTestWithBadInputs")
{

   // CHECK_THROWS_AS(RunTest<4>({{ "InexistantInput" , {0.0, 1.0, 2.0, 3.0}}},
   //                              {{ "InexistantOutput" , {0.0, 1.0, 2.0, 3.0}}}),
   //                   armnn::InvalidArgumentException );
}

}
