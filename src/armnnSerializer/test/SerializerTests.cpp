//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ArmNN.hpp>
#include <armnn/INetwork.hpp>
#include "../Serializer.hpp"
#include <sstream>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(SerializerTests)

BOOST_AUTO_TEST_CASE(SimpleNetworkSerialization)
{
    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    armnn::IConnectableLayer* const additionLayer0 = network->AddAdditionLayer();
    inputLayer0->GetOutputSlot(0).Connect(additionLayer0->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(additionLayer0->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer0 = network->AddOutputLayer(0);
    additionLayer0->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
}

BOOST_AUTO_TEST_SUITE_END()
