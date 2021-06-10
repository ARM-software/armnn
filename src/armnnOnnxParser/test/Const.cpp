//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_Const")
{
struct ConstMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ConstMainFixture(const std::string& dataType)
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
                        output:  "Output"
                        attribute {
                          name: "value"
                          t {
                              dims: 7
                              data_type: )" + dataType + R"(
                              float_data: 0.0
                              float_data: 1.0
                              float_data: 2.0
                              float_data: 3.0
                              float_data: 4.0
                              float_data: 5.0
                              float_data: 6.0

                          }
                          type: 1
                        }
                        name:  "constantNode"
                        op_type:  "Constant"
                      }
                      output {
                          name:  "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                 dim {
                                    dim_value: 7
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

struct ConstValidFixture : ConstMainFixture
{
    ConstValidFixture() : ConstMainFixture("1") {
        Setup();
    }
};

struct ConstInvalidFixture : ConstMainFixture
{
    ConstInvalidFixture() : ConstMainFixture("10") { }
};

TEST_CASE_FIXTURE(ConstValidFixture, "ValidConstTest")
{
    RunTest<1>({ }, {{ "Output" , {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}}});
}

TEST_CASE_FIXTURE(ConstInvalidFixture, "IncorrectDataTypeConst")
{
   CHECK_THROWS_AS( Setup(), armnn::ParseException);
}

}
