//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <arm_compute/core/CL/CLCompileContext.h>

#include <flatbuffers/flatbuffers.h>

namespace armnn
{

class ClContextSerializer
{
public:
    ClContextSerializer()  = default;
    ~ClContextSerializer() = default;

    /// Serializes the CLCompileContext built-in programs
    /// @param [in] clCompileContext The CLCompileContext to be serialized.
    void Serialize(const arm_compute::CLCompileContext& clCompileContext);

    /// Serializes the ClContext to the stream.
    /// @param [stream] the stream to save to
    /// @return true if ClContext is Serialized to the Stream, false otherwise
    bool SaveSerializedToStream(std::ostream& stream);

private:
    /// FlatBufferBuilder to create the CLContext FlatBuffers.
    flatbuffers::FlatBufferBuilder m_FlatBufferBuilder;
};

} // namespace armnn