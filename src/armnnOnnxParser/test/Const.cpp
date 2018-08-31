//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <boost/test/unit_test.hpp>
#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(OnnxParser)

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
                          type: FLOAT
                        }
                        name:  "constantNode"
                        op_type:  "Constant"
                      }
                      output {
                          name:  "Output"
                          type {
                             tensor_type {
                               elem_type: FLOAT
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
    ConstValidFixture() : ConstMainFixture("FLOAT") {
        Setup();
    }
};

struct ConstInvalidFixture : ConstMainFixture
{
    ConstInvalidFixture() : ConstMainFixture("FLOAT16") { }
};

BOOST_FIXTURE_TEST_CASE(ValidConstTest, ConstValidFixture)
{
    RunTest<1>({ }, {{ "Output" , {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}}});
}

BOOST_FIXTURE_TEST_CASE(IncorrectDataTypeConst, ConstInvalidFixture)
{
   BOOST_CHECK_THROW( Setup(), armnn::ParseException);
}

BOOST_AUTO_TEST_SUITE_END()
