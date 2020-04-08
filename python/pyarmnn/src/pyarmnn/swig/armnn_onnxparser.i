//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%module pyarmnn_onnxparser
%{
#define SWIG_FILE_WITH_INIT
#include "armnnOnnxParser/IOnnxParser.hpp"
#include "armnn/INetwork.hpp"
%}

//typemap definitions and other common stuff
%include "standard_header.i"

namespace std {
   %template(BindingPointInfo)     pair<int, armnn::TensorInfo>;
   %template(MapStringTensorShape) map<std::string, armnn::TensorShape>;
   %template(StringVector)         vector<string>;
}

namespace armnnOnnxParser
{
%feature("docstring",
"
Interface for creating a parser object using ONNX (https://onnx.ai/) onnx files.

Parsers are used to automatically construct Arm NN graphs from model files.

") IOnnxParser;

%nodefaultctor IOnnxParser;
class IOnnxParser
{
public:
    %feature("docstring",
    "
    Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name.

    Args:
        name (string): Name of the input node.

    Returns:
        tuple: (`int`, `TensorInfo`)
    ") GetNetworkInputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkInputBindingInfo(const std::string& name);

    %feature("docstring",
        "
        Retrieve binding info (layer id and `TensorInfo`) for the network output identified by the given layer name.

        Args:
            name (string): Name of the output node.

        Returns:
            tuple: (`int`, `TensorInfo`)
    ") GetNetworkOutputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkOutputBindingInfo(const std::string& name);
};

%extend IOnnxParser {
    // This is not a substitution of the default constructor of the Armnn class. It tells swig to create custom __init__
    // method for IOnnxParser python object that will use static factory method to do the job.
    IOnnxParser() {
        return armnnOnnxParser::IOnnxParser::CreateRaw();
    }

    // The following does not replace a real destructor of the Armnn class.
    // It creates a functions that will be called when swig object goes out of the scope to clean resources.
    // so the user doesn't need to call IOnnxParser::Destroy himself.
    // $self` is a pointer to extracted ArmNN IOnnxParser object.
    ~IOnnxParser() {
        armnnOnnxParser::IOnnxParser::Destroy($self);
    }

    %feature("docstring",
        "
        Create the network from a binary file on disk.

        Args:
            graphFile (str): Path to the onnx model to be parsed.

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

}
// Clear exception typemap.
%exception;
