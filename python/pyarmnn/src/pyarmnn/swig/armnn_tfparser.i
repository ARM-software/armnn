//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%module pyarmnn_tfparser
%{
#define SWIG_FILE_WITH_INIT
#include "armnnTfParser/ITfParser.hpp"
#include "armnn/INetwork.hpp"
%}

//typemap definitions and other common stuff
%include "standard_header.i"

namespace std {
   %template(BindingPointInfo)     pair<int, armnn::TensorInfo>;
   %template(MapStringTensorShape) map<std::string, armnn::TensorShape>;
   %template(StringVector)         vector<string>;
}

namespace armnnTfParser
{
%feature("docstring",
"
Interface for creating a parser object using TensorFlow (https://www.tensorflow.org/) frozen pb files.

Parsers are used to automatically construct Arm NN graphs from model files.

") ITfParser;
%nodefaultctor ITfParser;
class ITfParser
{
public:
    %feature("docstring",
        "
        Retrieve binding info (layer id and `TensorInfo`) for the network input identified by the given layer name.

        Args:
            name (str): Name of the input.

        Returns:
            tuple: (`int`, `TensorInfo`).
        ") GetNetworkInputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkInputBindingInfo(const std::string& name);

    %feature("docstring",
        "
        Retrieve binding info (layer id and `TensorInfo`) for the network output identified by the given layer name.

        Args:
            name (str): Name of the output.

        Returns:
            tuple: (`int`, `TensorInfo`).
        ") GetNetworkOutputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkOutputBindingInfo(const std::string& name);
};

%extend ITfParser {
    // This is not a substitution of the default constructor of the Armnn class. It tells swig to create custom __init__
    // method for ITfParser python object that will use static factory method to do the job.

    ITfParser() {
        return armnnTfParser::ITfParser::CreateRaw();
    }

    // The following does not replace a real destructor of the Armnn class.
    // It creates a functions that will be called when swig object goes out of the scope to clean resources.
    // so the user doesn't need to call ITfParser::Destroy himself.
    // $self` is a pointer to extracted ArmNN ITfParser object.

    ~ITfParser() {
        armnnTfParser::ITfParser::Destroy($self);
    }

    %feature("docstring",
    "
    Create the network from a pb Protocol buffer file.

    Args:
        graphFile (str): Path to the tf model to be parsed.
        inputShapes (dict): A dict containing the input name as a key and `TensorShape` as a value.
        requestedOutputs (list of str): A list of the output tensor names.

    Returns:
        INetwork: Parsed network.

    Raises:
        RuntimeError: If model file was not found.
     ") CreateNetworkFromBinaryFile;
    %newobject CreateNetworkFromBinaryFile;
    armnn::INetwork* CreateNetworkFromBinaryFile(const char* graphFile,
                                                 const std::map<std::string, armnn::TensorShape>& inputShapes,
                                                 const std::vector<std::string>& requestedOutputs) {
        return $self->CreateNetworkFromBinaryFile(graphFile, inputShapes, requestedOutputs).release();
    }

}

}
// Clear exception typemap.
%exception;
