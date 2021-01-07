//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextSerializer.hpp"
#include "ClContextSchema_generated.h"

#include <armnn/Exceptions.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <fmt/format.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace armnn
{

void ClContextSerializer::Serialize(const arm_compute::CLCompileContext& clCompileContext)
{
    // Get map of built programs from clCompileContext
    std::map<std::string, cl::Program> builtProgramsMap = clCompileContext.get_built_programs();
    if (builtProgramsMap.empty())
    {
        ARMNN_LOG(warning) << "There are no built programs to be serialised.";
        return;
    }

    // Create Flatbuffer CL Programs
    std::vector<flatbuffers::Offset<armnn::Program>> clPrograms;
    for(const auto& program : builtProgramsMap)
    {
        std::vector<std::vector<uint8_t>> binaries = program.second.getInfo<CL_PROGRAM_BINARIES>();
        clPrograms.push_back(CreateProgram(m_FlatBufferBuilder,
                                           m_FlatBufferBuilder.CreateString(program.first),
                                           m_FlatBufferBuilder.CreateVector(binaries[0])));
    }

    // Create Flatbuffer CLContext
    auto clContext = CreateClContext(m_FlatBufferBuilder, m_FlatBufferBuilder.CreateVector(clPrograms));

    m_FlatBufferBuilder.Finish(clContext);
}

bool ClContextSerializer::SaveSerializedToStream(std::ostream& stream)
{
    // Write to a stream
    auto bytesToWrite = armnn::numeric_cast<std::streamsize>(m_FlatBufferBuilder.GetSize());
    stream.write(reinterpret_cast<const char*>(m_FlatBufferBuilder.GetBufferPointer()), bytesToWrite);
    return !stream.bad();
}

} // namespace armnn
