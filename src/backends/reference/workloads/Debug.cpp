//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Debug.hpp"
#include <common/include/ProfilingGuid.hpp>
#include <armnnUtils/Filesystem.hpp>

#include <BFloat16.hpp>
#include <Half.hpp>

#include <algorithm>
#include <iostream>
#include <iosfwd>
#include <fstream>
#include <sys/stat.h>

namespace armnn
{

template<typename T>
void PrintOutput(const TensorInfo& inputInfo,
                 const T* inputData,
                 LayerGuid guid,
                 const std::string& layerName,
                 unsigned int slotIndex,
                 std::ostream& os)
{
    const unsigned int numDims = inputInfo.GetNumDimensions();
    const unsigned int numElements = inputInfo.GetNumElements();
    const TensorShape& inputShape = inputInfo.GetShape();

    std::vector<unsigned int> strides(numDims, 0);
    strides[numDims - 1] = inputShape[numDims - 1];

    for (unsigned int i = 2; i <= numDims; i++)
    {
        strides[numDims - i] = strides[numDims - i + 1] * inputShape[numDims - i];
    }

    os << "{ ";
    os << "\"layerGuid\": " << guid << ", ";
    os << "\"layerName\": \"" << layerName << "\", ";
    os << "\"outputSlot\": " << slotIndex << ", ";
    os << "\"shape\": ";

    os << "[";
    for (unsigned int i = 0; i < numDims; i++)
    {
        os << inputShape[i];
        if (i != numDims - 1)
        {
            os << ", ";
        }
    }
    os << "], ";

    os << "\"min\": "
              << static_cast<float>(*std::min_element(inputData, inputData + numElements)) << ", ";

    os << "\"max\": "
              << static_cast<float>(*std::max_element(inputData, inputData + numElements)) << ", ";

    os << "\"data\": ";

    for (unsigned int i = 0; i < numElements; i++)
    {
        for (unsigned int j = 0; j < numDims; j++)
        {
            if (i % strides[j] == 0)
            {
                os << "[";
            }
        }

        os << static_cast<float>(inputData[i]);

        for (unsigned int j = 0; j < numDims; j++)
        {
            if ((i + 1) % strides[j] == 0)
            {
                os << "]";
            }
        }

        if (i != numElements - 1)
        {
            os << ", ";
        }
    }

    os << " }" << std::endl;
}

template<typename T>
void Debug(const TensorInfo& inputInfo,
           const T* inputData,
           LayerGuid guid,
           const std::string& layerName,
           unsigned int slotIndex,
           bool outputsToFile)
{
    if (outputsToFile)
    {
        fs::path tmpDir = fs::temp_directory_path();
        std::ofstream out(tmpDir.generic_string() + "/ArmNNIntermediateLayerOutputs/" + layerName + ".numpy");
        PrintOutput<T>(inputInfo, inputData, guid, layerName, slotIndex, out);
        out.close();
    }
    else
    {
        PrintOutput<T>(inputInfo, inputData, guid, layerName, slotIndex, std::cout);
    }
}

template void Debug<BFloat16>(const TensorInfo& inputInfo,
                              const BFloat16* inputData,
                              LayerGuid guid,
                              const std::string& layerName,
                              unsigned int slotIndex,
                              bool outputsToFile);

template void Debug<Half>(const TensorInfo& inputInfo,
                          const Half* inputData,
                          LayerGuid guid,
                          const std::string& layerName,
                          unsigned int slotIndex,
                          bool outputsToFile);

template void Debug<float>(const TensorInfo& inputInfo,
                           const float* inputData,
                           LayerGuid guid,
                           const std::string& layerName,
                           unsigned int slotIndex,
                           bool outputsToFile);

template void Debug<uint8_t>(const TensorInfo& inputInfo,
                             const uint8_t* inputData,
                             LayerGuid guid,
                             const std::string& layerName,
                             unsigned int slotIndex,
                             bool outputsToFile);

template void Debug<int8_t>(const TensorInfo& inputInfo,
                            const int8_t* inputData,
                            LayerGuid guid,
                            const std::string& layerName,
                            unsigned int slotIndex,
                            bool outputsToFile);

template void Debug<int16_t>(const TensorInfo& inputInfo,
                             const int16_t* inputData,
                             LayerGuid guid,
                             const std::string& layerName,
                             unsigned int slotIndex,
                             bool outputsToFile);

template void Debug<int32_t>(const TensorInfo& inputInfo,
                             const int32_t* inputData,
                             LayerGuid guid,
                             const std::string& layerName,
                             unsigned int slotIndex,
                             bool outputsToFile);

} // namespace armnn
