//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
%module pyarmnn_deserializer
%{
#include "armnnDeserializer/IDeserializer.hpp"
#include "armnn/Types.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Exceptions.hpp"
#include <string>
#include <fstream>
#include <sstream>
%}

//typemap definitions and other common stuff
%include "standard_header.i"

namespace std {
    %template(BindingPointInfo) pair<int, armnn::TensorInfo>;
    %template(MapStringTensorShape) map<std::string, armnn::TensorShape>;
    %template(StringVector)         vector<string>;
}

namespace armnnDeserializer
{
%feature("docstring",
"
Interface for creating a parser object using ArmNN files.

Parsers are used to automatically construct ArmNN graphs from model files.

") IDeserializer;
%nodefaultctor IDeserializer;
class IDeserializer
{
public:
};

%extend IDeserializer {
// This is not a substitution of the default constructor of the Armnn class. It tells swig to create custom __init__
// method for ArmNN python object that will use static factory method to do the job.

    IDeserializer() {
        return armnnDeserializer::IDeserializer::CreateRaw();
    }

// The following does not replace a real destructor of the Armnn class.
// It creates a functions that will be called when swig object goes out of the scope to clean resources.
// so the user doesn't need to call IDeserializer::Destroy himself.
// $self` is a pointer to extracted ArmNN IDeserializer object.

    ~IDeserializer() {
        armnnDeserializer::IDeserializer::Destroy($self);
    }

    %feature("docstring",
    "
    Create the network from a armnn binary file.

    Args:
        graphFile (str): Path to the armnn model to be parsed.

    Returns:
        INetwork: Parsed network.

    Raises:
        RuntimeError: If model file was not found.
    ") CreateNetworkFromBinaryFile;

    %newobject CreateNetworkFromBinary;
    armnn::INetwork* CreateNetworkFromBinary(const char *graphFile) {
        std::ifstream is(graphFile, std::ifstream::binary);
        if (!is.good()) {
            std::string locationString = CHECK_LOCATION().AsString();
            std::stringstream msg;
            msg << "Cannot read the file " << graphFile << locationString;
            throw armnn::FileNotFoundException(msg.str());
        }
        return $self->CreateNetworkFromBinary(is).release();
    }

// Make both GetNetworkInputBindingInfo and GetNetworkOutputBindingInfo return a std::pair like other parsers instead of struct.

    %feature("docstring",
        "
        Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name and subgraph id.
        Args:
            subgraphId (int): The layer id. Any value is acceptable since it is unused in the current implementation.
            name (str): Name of the input.

        Returns:
            tuple: (`int`, `TensorInfo`).
        ") GetNetworkInputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkInputBindingInfo(unsigned int layerId, const std::string& name){
        armnnDeserializer::BindingPointInfo info = $self->GetNetworkInputBindingInfo(layerId, name);
        return std::make_pair(info.m_BindingId, info.m_TensorInfo);
    }

    %feature("docstring",
        "
        Retrieve binding info (layer id and `TensorInfo`) for the network output identified by the given layer name and subgraph id.

        Args:
            layerId (int): The layer id. Any value is acceptable since it is unused in the current implementation.
            name (str): Name of the output.

        Returns:
            tuple: (`int`, `TensorInfo`).
        ") GetNetworkOutputBindingInfo;
    std::pair<int, armnn::TensorInfo> GetNetworkOutputBindingInfo(unsigned int layerId, const std::string& name){
        armnnDeserializer::BindingPointInfo info = $self->GetNetworkOutputBindingInfo(layerId, name);
        return std::make_pair(info.m_BindingId, info.m_TensorInfo);
    }
}

} // end of namespace armnnDeserializer

// Clear exception typemap.
%exception;
