//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/Filesystem.hpp>

#include <cl/test/ClContextControlFixture.hpp>

#include <doctest/doctest.h>

#include <fstream>

namespace
{

armnn::INetworkPtr CreateNetwork()
{
    // Builds up the structure of the network.
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0, "input");
    armnn::IConnectableLayer* softmax = net->AddSoftmaxLayer(armnn::SoftmaxDescriptor(), "softmax");
    armnn::IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the input and output tensors
    armnn::TensorInfo inputTensorInfo(armnn::TensorShape({1, 5}), armnn::DataType::QAsymmU8, 10000.0f, 1);
    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    armnn::TensorInfo outputTensorInfo(armnn::TensorShape({1, 5}), armnn::DataType::QAsymmU8, 1.0f/255.0f, 0);
    softmax->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return net;
}

void RunInference(armnn::NetworkId& netId, armnn::IRuntimePtr& runtime, std::vector<uint8_t>& outputData)
{
    // Creates structures for input & output.
    std::vector<uint8_t> inputData
    {
        1, 10, 3, 200, 5 // Some inputs - one of which is sufficiently larger than the others to saturate softmax.
    };

    armnn::TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    armnn::InputTensors inputTensors
    {
        {0, armnn::ConstTensor(inputTensorInfo, inputData.data())}
    };

    armnn::OutputTensors outputTensors
    {
        {0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Run inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);
}

std::vector<char> ReadBinaryFile(const std::string& binaryFileName)
{
    std::ifstream input(binaryFileName, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
}

} // anonymous namespace

TEST_CASE_FIXTURE(ClContextControlFixture, "ClContextSerializerTest")
{
    // Get tmp directory and create blank file.
    fs::path filePath = armnnUtils::Filesystem::NamedTempFile("Armnn-CachedNetworkFileTest-TempFile.bin");
    std::string const filePathString{filePath.string()};
    std::ofstream file { filePathString };

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};

    // Create two networks.
    // net1 will serialize and save context to file.
    // net2 will deserialize context saved from net1 and load.
    armnn::INetworkPtr net1 = CreateNetwork();
    armnn::INetworkPtr net2 = CreateNetwork();

    // Add specific optimizerOptions to each network.
    armnn::OptimizerOptions optimizerOptions1;
    armnn::OptimizerOptions optimizerOptions2;
    armnn::BackendOptions modelOptions1("GpuAcc",
                                       {{"SaveCachedNetwork", true}, {"CachedNetworkFilePath", filePathString}});
    armnn::BackendOptions modelOptions2("GpuAcc",
                                        {{"SaveCachedNetwork", false}, {"CachedNetworkFilePath", filePathString}});
    optimizerOptions1.m_ModelOptions.push_back(modelOptions1);
    optimizerOptions2.m_ModelOptions.push_back(modelOptions2);

    armnn::IOptimizedNetworkPtr optNet1 = armnn::Optimize(
            *net1, backends, runtime->GetDeviceSpec(), optimizerOptions1);
    armnn::IOptimizedNetworkPtr optNet2 = armnn::Optimize(
            *net2, backends, runtime->GetDeviceSpec(), optimizerOptions2);
    CHECK(optNet1);
    CHECK(optNet2);

    // Cached file should be empty until net1 is loaded into runtime.
    CHECK(fs::is_empty(filePathString));

    // Load net1 into the runtime.
    armnn::NetworkId netId1;
    CHECK(runtime->LoadNetwork(netId1, std::move(optNet1)) == armnn::Status::Success);

    // File should now exist and not be empty. It has been serialized.
    CHECK(fs::exists(filePathString));
    std::vector<char> dataSerialized = ReadBinaryFile(filePathString);
    CHECK(dataSerialized.size() != 0);

    // Load net2 into the runtime using file and deserialize.
    armnn::NetworkId netId2;
    CHECK(runtime->LoadNetwork(netId2, std::move(optNet2)) == armnn::Status::Success);

    // Run inference and get output data.
    std::vector<uint8_t> outputData1(5);
    RunInference(netId1, runtime, outputData1);

    std::vector<uint8_t> outputData2(5);
    RunInference(netId2, runtime, outputData2);

    // Compare outputs from both networks.
    CHECK(std::equal(outputData1.begin(), outputData1.end(), outputData2.begin(), outputData2.end()));

    // Remove temp file created.
    fs::remove(filePath);
}
