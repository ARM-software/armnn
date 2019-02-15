//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnnSerializer/ISerializer.hpp>

#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
    armnnQuantizer::CommandLineProcessor cmdline;
    if (!cmdline.ProcessCommandLine(argc, argv))
    {
        return -1;
    }
    armnnDeserializer::IDeserializerPtr parser = armnnDeserializer::IDeserializer::Create();
    std::ifstream inputFileStream(cmdline.GetInputFileName(), std::ios::binary);
    std::vector<std::uint8_t> binaryContent;
    while (inputFileStream)
    {
        char c;
        inputFileStream.get(c);
        if (inputFileStream)
        {
            binaryContent.push_back(static_cast<std::uint8_t>(c));
        }
    }
    inputFileStream.close();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinary(binaryContent);
    armnn::INetworkPtr quantizedNetwork = armnn::INetworkQuantizer::Create(network.get())->ExportNetwork();

    armnnSerializer::ISerializerPtr serializer = armnnSerializer::ISerializer::Create();
    serializer->Serialize(*quantizedNetwork);

    std::string output(cmdline.GetOutputDirectoryName());
    output.append(cmdline.GetOutputFileName());
    std::ofstream outputFileStream;
    outputFileStream.open(output);
    serializer->SaveSerializedToStream(outputFileStream);
    outputFileStream.flush();
    outputFileStream.close();

    return 0;
}