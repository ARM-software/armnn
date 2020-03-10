//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>
#include <armnnSerializer/ISerializer.hpp>
#include "QuantizationDataSet.hpp"
#include "QuantizationInput.hpp"

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

    armnn::QuantizerOptions quantizerOptions;

    if (cmdline.GetQuantizationScheme() == "QAsymmS8")
    {
        quantizerOptions.m_ActivationFormat = armnn::DataType::QAsymmS8;
    }
    else if (cmdline.GetQuantizationScheme() == "QSymmS16")
    {
        quantizerOptions.m_ActivationFormat = armnn::DataType::QSymmS16;
    }
    else
    {
        quantizerOptions.m_ActivationFormat = armnn::DataType::QAsymmU8;
    }

    quantizerOptions.m_PreserveType = cmdline.HasPreservedDataType();

    armnn::INetworkPtr network = parser->CreateNetworkFromBinary(binaryContent);
    armnn::INetworkQuantizerPtr quantizer = armnn::INetworkQuantizer::Create(network.get(), quantizerOptions);

    if (cmdline.HasQuantizationData())
    {
        armnnQuantizer::QuantizationDataSet dataSet = cmdline.GetQuantizationDataSet();
        if (!dataSet.IsEmpty())
        {
            // Get the Input Tensor Infos
            armnnQuantizer::InputLayerVisitor inputLayerVisitor;
            network->Accept(inputLayerVisitor);

            for (armnnQuantizer::QuantizationInput quantizationInput : dataSet)
            {
                armnn::InputTensors inputTensors;
                std::vector<std::vector<float>> inputData(quantizationInput.GetNumberOfInputs());
                std::vector<armnn::LayerBindingId> layerBindingIds = quantizationInput.GetLayerBindingIds();
                unsigned int count = 0;
                for (armnn::LayerBindingId layerBindingId : quantizationInput.GetLayerBindingIds())
                {
                    armnn::TensorInfo tensorInfo = inputLayerVisitor.GetTensorInfo(layerBindingId);
                    inputData[count] = quantizationInput.GetDataForEntry(layerBindingId);
                    armnn::ConstTensor inputTensor(tensorInfo, inputData[count].data());
                    inputTensors.push_back(std::make_pair(layerBindingId, inputTensor));
                    count++;
                }
                quantizer->Refine(inputTensors);
            }
        }
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