//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/BackendId.hpp"
#include "armnn/BackendOptions.hpp"
%}

#pragma SWIG nowarn=SWIGWARN_PARSE_NESTED_CLASS

%{
    typedef armnn::BackendOptions::BackendOption BackendOption;
%}

%feature("docstring",
"
Struct for the users to pass backend specific option.
") BackendOption;
%nodefaultctor BackendOption;
struct BackendOption
{
    BackendOption(std::string name, bool value);
    BackendOption(std::string name, int value);
    BackendOption(std::string name, unsigned int value);
    BackendOption(std::string name, float value);
    BackendOption(std::string name, std::string value);

    std::string GetName();
};

namespace armnn
{
%feature("docstring",
"
Struct for backend specific options, see `BackendOption`.
Options are assigned to a specific backend by providing a backend id.

") BackendOptions;
%nodefaultctor BackendOptions;
struct BackendOptions
{
    BackendOptions(BackendId backend);

    BackendOptions(const BackendOptions& other);
    
    %feature("docstring",
    "
    Add backend option.
    
    Args:
       option (`BackendOption`): backend option
    ") AddOption;
    void AddOption(const BackendOption& option);
    
    %feature("docstring",
    "
    Get a backend id.
    
    Returns:
        BackendId: assigned backend id.
    ") GetBackendId;
    const BackendId& GetBackendId();

    %feature("docstring",
    "
    Get backend options count.

    Returns:
        int: number of options for a backend.
    ") GetOptionCount;
    size_t GetOptionCount();

    %feature("docstring",
    "
    Get backend option by index.

    Args:
       idx (int): backend option index

    Returns:
        BackendOption: backend option.
    ") GetOption;
    const BackendOption& GetOption(size_t idx);

    %pythoncode %{
    def __iter__(self):
        for count in range(self.GetOptionCount()):
            yield self[count]
    %}
};

%extend BackendOptions {

    const BackendOption& __getitem__(size_t i) const {
        return $self->GetOption(i);
    }

    size_t __len__() const {
        return $self->GetOptionCount();
    }
}
}
