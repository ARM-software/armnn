//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextDeserializer.hpp"
#include "ClContextSchema_generated.h"

#include <armnn/Exceptions.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <flatbuffers/flexbuffers.h>

#include <fmt/format.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace armnn
{

void ClContextDeserializer::Deserialize(arm_compute::CLCompileContext& clCompileContext,
                                        cl::Context& context,
                                        cl::Device& device,
                                        const std::string& filePath)
{
    std::ifstream inputFileStream(filePath, std::ios::binary);
    std::vector<std::uint8_t> binaryContent;
    while (inputFileStream)
    {
        char input;
        inputFileStream.get(input);
        if (inputFileStream)
        {
            binaryContent.push_back(static_cast<std::uint8_t>(input));
        }
    }
    inputFileStream.close();
    DeserializeFromBinary(clCompileContext, context, device, binaryContent);
}

void ClContextDeserializer::DeserializeFromBinary(arm_compute::CLCompileContext& clCompileContext,
                                                  cl::Context& context,
                                                  cl::Device& device,
                                                  const std::vector<uint8_t>& binaryContent)
{
    if (binaryContent.data() == nullptr)
    {
        throw InvalidArgumentException(fmt::format("Invalid (null) binary content {}",
                                                   CHECK_LOCATION().AsString()));
    }

    size_t binaryContentSize = binaryContent.size();
    flatbuffers::Verifier verifier(binaryContent.data(), binaryContentSize);
    if (verifier.VerifyBuffer<ClContext>() == false)
    {
        throw ParseException(fmt::format("Buffer doesn't conform to the expected Armnn "
                                         "flatbuffers format. size:{0} {1}",
                                         binaryContentSize,
                                         CHECK_LOCATION().AsString()));
    }
    auto clContext = GetClContext(binaryContent.data());

    for (Program const* program : *clContext->programs())
    {
        const char* volatile programName = program->name()->c_str();
        auto programBinary = program->binary();
        std::vector<uint8_t> binary(programBinary->begin(), programBinary->begin() + programBinary->size());

        cl::Program::Binaries   binaries{ binary };
        std::vector<cl::Device> devices {device};
        cl::Program             theProgram(context, devices, binaries);
        theProgram.build();
        clCompileContext.add_built_program(programName, theProgram);
    }
}

} // namespace armnn
