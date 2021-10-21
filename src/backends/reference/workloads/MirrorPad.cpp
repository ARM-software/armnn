//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MirrorPad.hpp"

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace
{

// Convert a linear index into n-dimensional coordinates.
// E.g. index = 2 returns [0, 0, 2].
inline std::vector<unsigned int> IndexToCoord(const armnn::TensorShape& shape, unsigned int index)
{
    unsigned int numOfElements = shape.GetNumElements();

    ARMNN_ASSERT_MSG(index <= numOfElements, "Index has to be in [0, num_elements]");
    ARMNN_ASSERT_MSG(numOfElements != 0, "Cannot create coordinate from empty shape");

    std::vector<unsigned int> coord(shape.GetNumDimensions());
    for(unsigned int i = 0; i < shape.GetNumDimensions(); ++i)
    {
        numOfElements /= shape[i];
        coord[i] = index / numOfElements;
        index %= numOfElements;
    }

    return coord;
}

// Returns the index of a given coordinate.
// E.g. [0, 0, 2] returns 2.
inline unsigned int CoordToIndex(const armnn::TensorShape& shape, const std::vector<unsigned int>& coord)
{
    ARMNN_ASSERT_MSG(shape.GetNumDimensions() != 0, "Cannot get index from empty shape");
    ARMNN_ASSERT_MSG(coord.size() != 0, "Cannot get index of empty coordinate");

    unsigned int index    = 0;
    unsigned int dimSize  = 1;

    for (unsigned int i = shape.GetNumDimensions(); i > 0; --i)
    {
        index += coord[i - 1] * dimSize;
        dimSize *= shape[i - 1];
    }

    return index;
}

} // anonymous namespace

namespace armnn
{

void MirrorPad(const TensorInfo& inputInfo,
               const TensorInfo& outputInfo,
               const ITensorHandle* inputHandle,
               ITensorHandle* outputHandle,
               const PadQueueDescriptor& data)
{
    auto padList  = data.m_Parameters.m_PadList;
    PaddingMode paddingMode = data.m_Parameters.m_PaddingMode;

    TensorShape outputShape = outputInfo.GetShape();
    TensorShape inputShape  = inputInfo.GetShape();

    unsigned int numOutputElements = outputInfo.GetNumElements();
    unsigned int numInputDimensions = inputShape.GetNumDimensions();
    assert(numInputDimensions == outputShape.GetNumDimensions());

    // If padding mode is Reflect then both paddings must be no greater than inputShape(i) - 1.
    // If padding mode is Symmetric then both paddings must be no greater than inputShape(i).
    const unsigned int isReflect = static_cast<unsigned int>(paddingMode == PaddingMode::Reflect);
    for(unsigned int i = 0; i < padList.size(); ++i)
    {
        if(padList.at(i).first > (inputShape[i] - isReflect) ||
           padList.at(i).second > (inputShape[i] - isReflect))
        {
            throw armnn::InvalidArgumentException("Paddings must be less (Reflect) or "
                                                  "equal (Symmetric) to the dimension size.");
        }
    }

    auto inputData = MakeDecoder<float>(inputInfo, inputHandle->Map());
    auto outData   = MakeEncoder<float>(outputInfo, outputHandle->Map());

    Decoder<float>& input  = *inputData;
    Encoder<float>& output = *outData;

    for(unsigned int idx = 0; idx < numOutputElements; ++idx)
    {
        // Get the coordinates of the current index in vector form. E.g inx 1 = [0, 0, 0, 1 ]
        const std::vector<unsigned int> coord = IndexToCoord(outputShape, idx);

        std::vector<unsigned int> dimensions;
        std::vector<unsigned int> coords;

        for(unsigned int i = 0; i < numInputDimensions; ++i)
        {
            dimensions.emplace_back(i);
            coords.emplace_back(coord[i]);
        }

        auto isInPadding = [&](unsigned int i)
        {
            return (coords[i] < padList[i].first || coords[i] > inputShape[i] + padList[i].first - 1);
        };

        auto getReflectIndex = [&](unsigned int i) -> unsigned int
        {
            if(isInPadding(i))
            {
                if(coords[i] < padList[i].first)
                {
                    return padList[i].first - coords[i];
                }
                else
                {
                    return 2 * inputShape[i] + padList[i].first - 2 - coords[i];
                }
            }
            return coords[i] - padList[i].first;
        };

        auto getSymmetricIndex = [&](unsigned int i) -> unsigned int
        {
            if(isInPadding(i))
            {
                if(coords[i] < padList[i].first)
                {
                    return padList[i].first - coords[i] - 1;
                }
                else
                {
                    return 2 * inputShape[i] + padList[i].first - 1 - coords[i];
                }
            }
            return coords[i] - padList[i].first;
        };

        // Location of the value in the input tensor to use in the output.
        std::vector<unsigned int> coordOfInput;

        // any_of works as a loop here to check if any of the dimensions are in the padding.
        // If dimensions is in the padding area, then create the coordinates of the location in the
        // input tensor to use in the output.
        // E.g.
        // Input tensor = [ 1, 2, 3 ], Rank = 1.
        // Output tensor = [ 2, 1, 2, 3, 1 ] if Reflect or [ 1, 1, 2, 3, 3 ] if Symmetric with a padding of (1, 1).
        // So it will either return [ 1 ] or [ 0 ] which is used to set the first value in the output tensor and so on.
        if(std::any_of(dimensions.begin(), dimensions.end(), isInPadding))
        {
            switch(paddingMode)
            {
                case PaddingMode::Reflect:
                {
                    for(unsigned int i = 0; i < numInputDimensions; ++i)
                    {
                        coordOfInput.emplace_back(getReflectIndex(i));
                    }
                    break;
                }
                case PaddingMode::Symmetric:
                {
                    for(unsigned int i = 0; i < numInputDimensions; ++i)
                    {
                        coordOfInput.emplace_back(getSymmetricIndex(i));
                    }
                    break;
                }
                default:
                    throw InvalidArgumentException("Padding mode not supported.");
                    break;
            }
        }
        else
        {
            for(unsigned int i = 0; i < numInputDimensions; ++i)
            {
                coordOfInput.emplace_back(coord[i] - padList[i].first);
            }
        }

        // Set output value using the coordinate of the input value to use.
        const unsigned int indexOfInput = CoordToIndex(inputShape, coordOfInput);

        input[indexOfInput];
        auto inputValue = input.Get();

        output[idx];
        output.Set(inputValue);
    }
}

} //namespace armnn