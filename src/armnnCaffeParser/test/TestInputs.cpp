//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "armnn/IRuntime.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Exceptions.hpp"

#include "test/TensorHelpers.hpp"

#include <string>

#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(CaffeParser)


BOOST_AUTO_TEST_CASE(InputShapes)
{
    std::string explicitInput = "name: \"Minimal\"\n"
                                "layer {\n"
                                "  name: \"data\"\n"
                                "  type: \"Input\"\n"
                                "  top: \"data\"\n"
                                "  input_param { shape: { dim: 1 dim: 2 dim: 3 dim: 4 } }\n"
                                "}";
    std::string implicitInput = "name: \"Minimal\"\n"
                                "input: \"data\" \n"
                                "input_dim: 1 \n"
                                "input_dim: 2 \n"
                                "input_dim: 3 \n"
                                "input_dim: 4 \n";
    std::string implicitInputNoShape = "name: \"Minimal\"\n"
                                       "input: \"data\" \n";

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnnCaffeParser::ICaffeParserPtr parser(armnnCaffeParser::ICaffeParser::Create());
    armnn::INetworkPtr network(nullptr, nullptr);
    armnn::NetworkId netId;

    // Check everything works normally
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    {
        network = parser->CreateNetworkFromString(explicitInput.c_str(), {}, { "data" });
        BOOST_TEST(network.get());
        runtime->LoadNetwork(netId, Optimize(*network, backends, runtime->GetDeviceSpec()));

        armnnCaffeParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("data");
        armnn::TensorInfo inputTensorInfo = inputBindingInfo.second;
        BOOST_TEST((inputTensorInfo == runtime->GetInputTensorInfo(netId, inputBindingInfo.first)));

        BOOST_TEST(inputTensorInfo.GetShape()[0] == 1);
        BOOST_TEST(inputTensorInfo.GetShape()[1] == 2);
        BOOST_TEST(inputTensorInfo.GetShape()[2] == 3);
        BOOST_TEST(inputTensorInfo.GetShape()[3] == 4);
    }

    // Checks everything works with implicit input.
    {
        network = parser->CreateNetworkFromString(implicitInput.c_str(), {}, { "data" });
        BOOST_TEST(network.get());
        runtime->LoadNetwork(netId, Optimize(*network, backends, runtime->GetDeviceSpec()));

        armnnCaffeParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("data");
        armnn::TensorInfo inputTensorInfo = inputBindingInfo.second;
        BOOST_TEST((inputTensorInfo == runtime->GetInputTensorInfo(netId, inputBindingInfo.first)));

        BOOST_TEST(inputTensorInfo.GetShape()[0] == 1);
        BOOST_TEST(inputTensorInfo.GetShape()[1] == 2);
        BOOST_TEST(inputTensorInfo.GetShape()[2] == 3);
        BOOST_TEST(inputTensorInfo.GetShape()[3] == 4);
    }

    // Checks everything works with implicit and passing shape.
    {
        network = parser->CreateNetworkFromString(implicitInput.c_str(), { {"data", { 2, 2, 3, 4 } } }, { "data" });
        BOOST_TEST(network.get());
        runtime->LoadNetwork(netId, Optimize(*network, backends, runtime->GetDeviceSpec()));

        armnnCaffeParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("data");
        armnn::TensorInfo inputTensorInfo = inputBindingInfo.second;
        BOOST_TEST((inputTensorInfo == runtime->GetInputTensorInfo(netId, inputBindingInfo.first)));

        BOOST_TEST(inputTensorInfo.GetShape()[0] == 2);
        BOOST_TEST(inputTensorInfo.GetShape()[1] == 2);
        BOOST_TEST(inputTensorInfo.GetShape()[2] == 3);
        BOOST_TEST(inputTensorInfo.GetShape()[3] == 4);
    }

    // Checks everything works with implicit (no shape) and passing shape.
    {
        network = parser->CreateNetworkFromString(implicitInputNoShape.c_str(), {{"data", {2, 2, 3, 4} }}, { "data" });
        BOOST_TEST(network.get());
        runtime->LoadNetwork(netId, Optimize(*network, backends, runtime->GetDeviceSpec()));

        armnnCaffeParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("data");
        armnn::TensorInfo inputTensorInfo = inputBindingInfo.second;
        BOOST_TEST((inputTensorInfo == runtime->GetInputTensorInfo(netId, inputBindingInfo.first)));

        BOOST_TEST(inputTensorInfo.GetShape()[0] == 2);
        BOOST_TEST(inputTensorInfo.GetShape()[1] == 2);
        BOOST_TEST(inputTensorInfo.GetShape()[2] == 3);
        BOOST_TEST(inputTensorInfo.GetShape()[3] == 4);
    }

    // Checks exception on incompatible shapes.
    {
        BOOST_CHECK_THROW(parser->CreateNetworkFromString(implicitInput.c_str(), {{"data",{ 2, 2, 3, 2 }}}, {"data"}),
            armnn::ParseException);
    }

    // Checks exception when no shape available.
    {
        BOOST_CHECK_THROW(parser->CreateNetworkFromString(implicitInputNoShape.c_str(), {}, { "data" }),
            armnn::ParseException);
    }
}

BOOST_AUTO_TEST_SUITE_END()
