//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include  "armnnOnnxParser/IOnnxParser.hpp"
#include "google/protobuf/stubs/logging.h"

BOOST_AUTO_TEST_SUITE(OnnxParser)

BOOST_AUTO_TEST_CASE(CreateNetworkFromString)
{
  std::string TestModel = R"(
                          ir_version: 3
                          producer_name:  "CNTK "
                          producer_version:  "2.5.1 "
                          domain:  "ai.cntk "
                          model_version: 1
                          graph {
                            name:  "CNTKGraph "
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

    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());

    armnn::INetworkPtr network = parser->CreateNetworkFromString(TestModel.c_str());
    BOOST_TEST(network.get());
}

BOOST_AUTO_TEST_CASE(CreateNetworkFromStringWithNullptr)
{
    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
    BOOST_CHECK_THROW(parser->CreateNetworkFromString(""), armnn::InvalidArgumentException );
}

BOOST_AUTO_TEST_CASE(CreateNetworkWithInvalidString)
{
    auto silencer = google::protobuf::LogSilencer(); //get rid of errors from protobuf
    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
    BOOST_CHECK_THROW(parser->CreateNetworkFromString( "I'm not a model so I should raise an error" ),
                      armnn::ParseException );
}

BOOST_AUTO_TEST_SUITE_END()
