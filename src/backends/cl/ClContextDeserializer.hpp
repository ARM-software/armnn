//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <arm_compute/core/CL/CLCompileContext.h>

namespace armnn
{

class ClContextDeserializer
{
public:
    ClContextDeserializer()  = default;
    ~ClContextDeserializer() = default;

    /// Deserializes the CLCompileContext built-in programs from a binary file
    /// @param [in] clCompileContext The CLCompileContext to be serialized
    /// @param [in] context The CL Kernel context built-in program will be created from
    /// @param [in] device The CL Kernel device built-in program will be created from
    /// @param [in] filePath The serialized file
    void Deserialize(arm_compute::CLCompileContext& clCompileContext,
                     cl::Context& context,
                     cl::Device& device,
                     const std::string& filePath);

    /// Deserializes the CLCompileContext built-in programs from binary file contents
    /// @param [in] clCompileContext The CLCompileContext to be serialized
    /// @param [in] context The CL Kernel context built-in program will be created from
    /// @param [in] device The CL Kernel device built-in program will be created from
    /// @param [in] filePath The serialized file
    void DeserializeFromBinary(arm_compute::CLCompileContext& clCompileContext,
                               cl::Context& context,
                               cl::Device& device,
                               const std::vector<uint8_t>& binaryContent);

};

} // namespace armnn