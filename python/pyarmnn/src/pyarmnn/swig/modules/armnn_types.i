//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/Types.hpp"
%}

%include <typemaps/permutation_vector.i>

namespace armnn
{

%feature("docstring",
"
Vector used to permute a tensor.

For a 4-d tensor laid out in a memory with the format (Batch Element, Height, Width, Channels),
which is to be passed as an input to Arm NN, each source dimension is mapped to the corresponding
Arm NN dimension. The Batch dimension remains the same (0 -> 0). The source Height dimension is mapped
to the location of the Arm NN Height dimension (1 -> 2). Similar arguments are made for the Width and
Channels (2 -> 3 and 3 -> 1). This will lead to m_DimMappings pointing to the following array:
[ 0, 2, 3, 1 ].

Note that the mapping should be reversed if considering the case of Arm NN 4-d outputs (Batch Element,
Channels, Height, Width) being written to a destination with the format mentioned above. We now have
0 -> 0, 2 -> 1, 3 -> 2, 1 -> 3, which, when reordered, lead to the following m_DimMappings contents:
[ 0, 3, 1, 2 ].

Args:
    dimMappings (list): Indicates how to translate tensor elements from a given source into the target destination,
                        when source and target potentially have different memory layouts.
") PermutationVector;

class PermutationVector
{
public:
    using ValueType = unsigned int;
    using SizeType = unsigned int;

    %permutation_vector_typemap(const ValueType *dimMappings, SizeType numDimMappings);
    PermutationVector(const ValueType *dimMappings, SizeType numDimMappings);
    %clear_permutation_vector_typemap(const ValueType *dimMappings, SizeType numDimMappings);


    %feature("docstring",
    "
    Get the PermutationVector size.

    Return:
        SizeType: Current size of the PermutationVector.

    ") GetSize;
    SizeType GetSize();

    %feature("docstring",
    "
    Checks if a specified permutation vector is its inverse

    Return:
        bool: returns true if the specified Permutation vector is its inverse.

    ") IsInverse;
    bool IsInverse(const PermutationVector& other);
};

%extend PermutationVector {

    unsigned int __getitem__(unsigned int i) const {
        return $self->operator[](i);
    }

    bool __eq__(PermutationVector other) {
        int size = $self->GetSize();
        int otherSize = other.GetSize();
        if(size != otherSize)
        {
            return false;
        }
        for(int i = 0; i < size; ++i){
            if($self->operator[](i) != other[i])
            {
                return false;
            }
            return true;
        }
        return true;
    }
}

}
%feature("docstring",
"
Interface for device specifications. Main use is to get information relating to what compute capability the device being used has.
") IDeviceSpec;


%feature("docstring",
"
Returns the backends supported by this compute device.

Returns:
    set: This devices supported backends.

") GetSupportedBackends;

%ignore PermutationVector;
#define ARMNN_DEPRECATED_ENUM  // SWIG does not support C++ attributes, need this to help generate from Deprecated.hpp.
#define ARMNN_DEPRECATED_ENUM_MSG(message)  // SWIG does not support C++ attributes, need this to help generate from Deprecated.hpp.
%include "armnn/Types.hpp"



%extend armnn::IDeviceSpec {


    std::string __str__() {

        std::string deviceStr = "IDeviceSpec { supportedBackends: [";

        auto bends = $self->GetSupportedBackends();
        auto sizeBends = $self->GetSupportedBackends().size();
        for (std::unordered_set<armnn::BackendId>::const_iterator p = bends.begin(); p != bends.end(); ++p) {

            deviceStr += p->Get();

            if (sizeBends - 1 > 0) {
                deviceStr += ", ";
            }
            sizeBends--;

        }
        deviceStr = deviceStr + "]}";

        return deviceStr;
    }

}
