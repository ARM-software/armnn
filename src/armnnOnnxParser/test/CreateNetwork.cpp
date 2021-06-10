//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include  "armnnOnnxParser/IOnnxParser.hpp"
#include <doctest/doctest.h>

#include "google/protobuf/stubs/logging.h"

TEST_SUITE("OnnxParser_CreateNetwork")
{
TEST_CASE("CreateNetworkFromString")
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
    CHECK(network.get());
}

TEST_CASE("CreateNetworkFromStringWithNullptr")
{
    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
    CHECK_THROWS_AS(parser->CreateNetworkFromString(""), armnn::InvalidArgumentException );
}

TEST_CASE("CreateNetworkWithInvalidString")
{
    auto silencer = google::protobuf::LogSilencer(); //get rid of errors from protobuf
    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
    CHECK_THROWS_AS(parser->CreateNetworkFromString( "I'm not a model so I should raise an error" ),
                      armnn::ParseException );
}

}
