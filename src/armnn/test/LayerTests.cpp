//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Network.hpp>
#include <doctest/doctest.h>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace
{

TEST_SUITE("Layer")
{

TEST_CASE("InputSlotGetTensorInfo")
{
    armnn::NetworkImpl net;
    armnn::IConnectableLayer* add = net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Add);
    armnn::IConnectableLayer* out = net.AddOutputLayer(0);

    armnn::Layer* addlayer = armnn::PolymorphicDowncast<armnn::Layer*>(add);
    armnn::Layer* outlayer = armnn::PolymorphicDowncast<armnn::Layer*>(out);

    auto outTensorInfo = armnn::TensorInfo({1,2,2,1}, armnn::DataType::Float32);
    addlayer->GetOutputSlot(0).Connect(outlayer->GetInputSlot(0));
    CHECK_FALSE(outlayer->GetInputSlot(0).IsTensorInfoSet());

    addlayer->GetOutputSlot(0).SetTensorInfo(outTensorInfo);
    auto testTensorInfo = outlayer->GetInputSlot(0).GetTensorInfo();

    CHECK_EQ(outTensorInfo, testTensorInfo);
    CHECK(outlayer->GetInputSlot(0).IsTensorInfoSet());
    CHECK_FALSE(outlayer->GetInputSlot(0).IsTensorInfoOverridden());

    auto overRiddenTensorInfo = armnn::TensorInfo({2,2}, armnn::DataType::Float32);
    outlayer->GetInputSlot(0).SetTensorInfo(overRiddenTensorInfo);
    testTensorInfo = outlayer->GetInputSlot(0).GetTensorInfo();

    // Confirm that inputslot TensorInfo is changed
    CHECK_EQ(overRiddenTensorInfo, testTensorInfo);
    // Confirm that outputslot TensorInfo is unchanged
    CHECK_EQ(outTensorInfo, outlayer->GetInputSlot(0).GetConnection()->GetTensorInfo());

    CHECK(outlayer->GetInputSlot(0).IsTensorInfoOverridden());
}

}

}
