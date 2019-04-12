//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnnSerializer/ISerializer.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>

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
    armnn::INetworkQuantizerPtr quantizer = armnn::INetworkQuantizer::Create(network.get());

    std::string csvFileName = cmdline.GetCsvFileName();
    if (csvFileName != "")
    {
        // Call the Quantizer::Refine() function which will update the min/max ranges for the quantize constants
        std::ifstream csvFileStream(csvFileName);
        std::string line;
        std::string csvDirectory = cmdline.GetCsvFileDirectory();
        while(getline(csvFileStream, line))
        {
            std::istringstream s(line);
            std::vector<std::string> row;
            std::string entry;
            while(getline(s, entry, ','))
            {
                entry.erase(std::remove(entry.begin(), entry.end(), ' '), entry.end());
                entry.erase(std::remove(entry.begin(), entry.end(), '"'), entry.end());
                row.push_back(entry);
            }
            std::string rawFileName = cmdline.GetCsvFileDirectory() + "/" + row[2];
            // passId: row[0]
            // bindingId: row[1]
            // rawFileName: file contains the RAW input tensor data
            // LATER: Quantizer::Refine() function will be called with those arguments when it is implemented
        }
        csvFileStream.close();
    }

    armnn::INetworkPtr quantizedNetwork = quantizer->ExportNetwork();
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