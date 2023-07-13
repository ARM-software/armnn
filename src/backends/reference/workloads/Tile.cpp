//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Tile.hpp"
#include "Encoders.hpp"
#include <numeric>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/Logging.hpp>

namespace armnn
{

// Converts a flatten index into a multi-dimensional coordinate.
std::vector<uint32_t> IndexToCoordinates(std::vector<uint32_t>& shape, uint32_t index)
{
    std::vector<uint32_t> coordinates;
    // Iterating through dimensions starting from the last dimension to the first
    for (std::size_t i = shape.size() - 1; i < shape.size(); --i)
    {
        // Coordinate is found by getting the index and modulus it by the current dimension size
        // shape of dimension = dimension size
        coordinates.insert(coordinates.begin(), index % shape[i]);
        // Pass the index to next iteration making index = index / size of the current dimension
        index = index/shape[i];
    }
    return coordinates;
}

// Convert a multidimensional coordinate to a flattened index.
uint32_t CoordinatesToIndex(TensorShape& shape, std::vector<uint32_t>& coordinates)
{
    uint32_t index = 0;
    uint32_t base = 1;
    uint32_t rank = shape.GetNumDimensions();
    for (uint32_t i = rank; i > 0; --i)
    {
        index = index + coordinates[i - 1] * base;
        base = base * shape[i - 1];
    }
    return index;
}

void Tile(const TileDescriptor& params,
          const TensorInfo& inputInfo,
          Decoder<float>& inputDecoder,
          Encoder<float>& outputEncoder)
{
    // Input and output will always have same rank
    uint32_t rank = inputInfo.GetNumDimensions();

    TensorShape inputShape = inputInfo.GetShape();

    std::vector<uint32_t> outputShape(rank);
    for (uint32_t i = 0; i < rank; ++i)
    {
        outputShape[i] = inputShape[i] * params.m_Multiples[i];
    }

    // If all values of multiples are 1, then return the input
    if ( std::adjacent_find( params.m_Multiples.begin(), params.m_Multiples.end(),
                             std::not_equal_to<>() ) == params.m_Multiples.end() && params.m_Multiples[0] == 1)
    {
        for (uint32_t idx = 0; idx < inputInfo.GetNumElements(); ++idx)
        {
            float inputValue = inputDecoder.Get();
            ++inputDecoder;
            outputEncoder.Set(inputValue);
            ++outputEncoder;
        }
        return;
    }

    std::vector<float> inputData = inputDecoder.DecodeTensor(inputInfo.GetShape());
    std::vector<float> outputData;
    auto outputNumElements = inputData.size() * static_cast<uint32_t>(std::accumulate(begin(params.m_Multiples),
                                                                                      end(params.m_Multiples),
                                                                                      1,
                                                                                      std::multiplies<>()));
    outputData.reserve(outputNumElements);

    for (uint32_t outputIndex = 0; outputIndex < outputNumElements; ++outputIndex)
    {
        std::vector<uint32_t> outputCoords = IndexToCoordinates(outputShape, outputIndex);

        // Converting output coordinates to input coordinates using modulus
        std::vector<uint32_t> inputCoordinates;
        inputCoordinates.reserve(rank);
        for (uint32_t i = 0; i < rank; ++i)
        {
            inputCoordinates.push_back(outputCoords[i] % inputShape[i]);
        }

        uint32_t inputIndex = CoordinatesToIndex(inputShape, inputCoordinates);

        outputEncoder[outputIndex];
        outputEncoder.Set(inputData[inputIndex]);
    }
}

} // namespace armnn