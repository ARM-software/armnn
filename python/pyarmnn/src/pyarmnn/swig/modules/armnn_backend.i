//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/BackendId.hpp"
%}

namespace std {
   %template(BackendIdVector) vector<armnn::BackendId>;
   %template(BackendIdSet) unordered_set<armnn::BackendId>;
}

namespace armnn
{

class BackendId
{
public:
    %feature("docstring",
        "
        Creates backend id instance.
        Supported backend ids: 'CpuRef', 'CpuAcc', 'GpuAcc', 'EthosNAcc'.

        Args:
            id (str): Computation backend identification.
        ") BackendId;

    BackendId(const std::string& id);

    %feature("docstring",
        "
        Checks if backend is cpu reference implementation.
        Returns:
            bool: True if backend supports cpu reference implementation, False otherwise.

        ") IsCpuRef;
    bool IsCpuRef();

    %feature("docstring",
        "
        Returns backend identification.

        >>> backendId = BackendId('CpuRef')
        >>> assert 'CpuRef' == str(backendId)
        >>> assert 'CpuRef' == backendId.Get()

        Returns:
            str: Backend identification.

        ") Get;
    const std::string& Get();
};

%extend BackendId {

    std::string __str__() {
        return $self->Get();
    }

}

using BackendIdVector = std::vector<armnn::BackendId>;
using BackendIdSet    = std::unordered_set<armnn::BackendId>;
}
