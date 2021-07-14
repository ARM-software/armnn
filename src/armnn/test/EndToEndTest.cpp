//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

#include <set>

TEST_SUITE("EndToEnd")
{
TEST_CASE("ErrorOnLoadNetwork")
{
    using namespace armnn;

    // Create runtime in which test will run
    // Note we don't allow falling back to CpuRef if an operation (excluding inputs, outputs, etc.) isn't supported
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc and isn't allowed to fall back, so Optimize will return null.
    NormalizationDescriptor descriptor;
    IConnectableLayer* pooling = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<BackendId> backends = {Compute::CpuAcc};
    std::vector<std::string> errMessages;

    try
    {
        Optimize(*net, backends, runtime->GetDeviceSpec(), OptimizerOptions(), errMessages);
        FAIL("Should have thrown an exception.");
    }
    catch (const InvalidArgumentException&)
    {
        // Different exceptions are thrown on different backends
    }
    CHECK(errMessages.size() > 0);
}

}
