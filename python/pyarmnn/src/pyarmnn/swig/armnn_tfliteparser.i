//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%module pyarmnn_tfliteparser
%{
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include "armnn/Types.hpp"
#include "armnn/INetwork.hpp"
%}

//typemap definitions and other common stuff
%include "standard_header.i"

namespace std {
   %template(BindingPointInfo)     pair<int, armnn::TensorInfo>;
   %template(MapStringTensorShape) map<std::string, armnn::TensorShape>;
   %template(StringVector)         vector<string>;
}

namespace armnnTfLiteParser
{
%feature("docstring",
"
Interface for creating a parser object using TfLite (https://www.tensorflow.org/lite) tflite files.

Parsers are used to automatically construct Arm NN graphs from model files.

") ITfLiteParser;
%nodefaultctor ITfLiteParser;
class ITfLiteParser
{
public:
    %feature("docstring",
        "
        Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name and subgraph id.
        Args:
            subgraphId (int): The subgraph id.
            name (str): Name of the input.

        Returns:
            tuple: (`int`, `TensorInfo`).
        ") GetNetworkInputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkInputBindingInfo(size_t subgraphId, const std::string& name);

    %feature("docstring",
        "
        Retrieve binding info (layer id and `TensorInfo`) for the network output identified by the given layer name and subgraph id.

        Args:
            subgraphId (int): The subgraph id.
            name (str): Name of the output.

        Returns:
            tuple: (`int`, `TensorInfo`).
        ") GetNetworkOutputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkOutputBindingInfo(size_t subgraphId, const std::string& name);

    %feature("docstring",
        "
        Return the number of subgraphs in the parsed model.
        Returns:
            int: The number of subgraphs.
        ") GetSubgraphCount;
    size_t GetSubgraphCount();

     %feature("docstring",
        "
        Return the input tensor names for a given subgraph.

        Args:
            subgraphId (int): The subgraph id.

        Returns:
            list: A list of the input tensor names for the given model.
        ") GetSubgraphInputTensorNames;
    std::vector<std::string> GetSubgraphInputTensorNames(size_t subgraphId);

    %feature("docstring",
        "
        Return the output tensor names for a given subgraph.

        Args:
            subgraphId (int): The subgraph id

        Returns:
            list: A list of the output tensor names for the given model.
        ") GetSubgraphOutputTensorNames;
    std::vector<std::string> GetSubgraphOutputTensorNames(size_t subgraphId);

    %feature("flatnested");
    %feature("docstring",
             "
    Options for TfLiteParser.

            Contains:
    m_StandInLayerForUnsupported (bool): Add StandInLayers as placeholders for unsupported operators.
            Default: False
    m_InferAndValidate (bool): Infer output shape of operations based on their input shape. Default: False
    ")TfLiteParserOptions;
    struct TfLiteParserOptions
    {
        TfLiteParserOptions();

        bool m_StandInLayerForUnsupported;
        bool m_InferAndValidate;
    };
};

%extend ITfLiteParser {
// This is not a substitution of the default constructor of the Armnn class. It tells swig to create custom __init__
// method for ITfLiteParser python object that will use static factory method to do the job.

    ITfLiteParser(const armnnTfLiteParser::ITfLiteParser::TfLiteParserOptions* options = nullptr) {
        if (options) {
            return armnnTfLiteParser::ITfLiteParser::CreateRaw(
                    armnn::Optional<armnnTfLiteParser::ITfLiteParser::TfLiteParserOptions>(*options));
        } else {
            return armnnTfLiteParser::ITfLiteParser::CreateRaw();
        }
    }

// The following does not replace a real destructor of the Armnn class.
// It creates a functions that will be called when swig object goes out of the scope to clean resources.
// so the user doesn't need to call ITfLiteParser::Destroy himself.
// $self` is a pointer to extracted ArmNN ITfLiteParser object.

    ~ITfLiteParser() {
        armnnTfLiteParser::ITfLiteParser::Destroy($self);
    }

    %feature("docstring",
    "
    Create the network from a flatbuffers binary file.

    Args:
        graphFile (str): Path to the tflite model to be parsed.

    Returns:
        INetwork: Parsed network.

    Raises:
        RuntimeError: If model file was not found.
    ") CreateNetworkFromBinaryFile;

    %newobject CreateNetworkFromBinaryFile;
    armnn::INetwork* CreateNetworkFromBinaryFile(const char* graphFile) {
        return $self->CreateNetworkFromBinaryFile(graphFile).release();
    }

}

} // end of namespace armnnTfLiteParser

// Clear exception typemap.
%exception;
