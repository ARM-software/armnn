//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%module pyarmnn_caffeparser
%{
#define SWIG_FILE_WITH_INIT
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "armnn/INetwork.hpp"
%}

//typemap definitions and other common stuff
%include "standard_header.i"

namespace std {
   %template(BindingPointInfo)     pair<int, armnn::TensorInfo>;
   %template(MapStringTensorShape) map<std::string, armnn::TensorShape>;
   %template(StringVector)         vector<string>;
}

namespace armnnCaffeParser
{

%feature("docstring",
"
Interface for creating a parser object using Caffe (http://caffe.berkeleyvision.org/) caffemodel files.

Parsers are used to automatically construct Arm NN graphs from model files.

") ICaffeParser;

%nodefaultctor ICaffeParser;
class ICaffeParser
{
public:
    // Documentation
    %feature("docstring",
    "
    Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name.

    Args:
        name (str): Name of the input.

    Returns:
        tuple: (`int`, `TensorInfo`)
    ") GetNetworkInputBindingInfo;

    %feature("docstring",
    "
    Retrieve binding info (layer id and `TensorInfo`) for the network output identified by the given layer name.

    Args:
        name (str): Name of the output.

    Returns:
        tuple: (`int`, `TensorInfo`)
    ") GetNetworkOutputBindingInfo;

    std::pair<int, armnn::TensorInfo> GetNetworkInputBindingInfo(const std::string& name);
    std::pair<int, armnn::TensorInfo> GetNetworkOutputBindingInfo(const std::string& name);
};

%extend ICaffeParser {
    // This is not a substitution of the default constructor of the Armnn class. It tells swig to create custom __init__
    // method for ICaffeParser python object that will use static factory method to do the job.

    ICaffeParser() {
        return armnnCaffeParser::ICaffeParser::CreateRaw();
    }

    // The following does not replace a real destructor of the Armnn class.
    // It creates a functions that will be called when swig object goes out of the scope to clean resources.
    // so the user doesn't need to call ICaffeParser::Destroy himself.
    // $self` is a pointer to extracted ArmNN ICaffeParser object.

    ~ICaffeParser() {
        armnnCaffeParser::ICaffeParser::Destroy($self);
    }

    %feature("docstring",
    "
    Create the network from a Caffe caffemodel binary file on disk.

    Args:
        graphFile: Path to the caffe model to be parsed.
        inputShapes (tuple): (`string`, `TensorShape`) A tuple containing the input name and TensorShape information for the network.
        requestedOutputs (list): A list of the output tensor names.

    Returns:
        INetwork: INetwork object for the parsed Caffe model.
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
