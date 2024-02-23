//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ScatterNd.hpp"
#include "Encoders.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/Logging.hpp>

#include <fmt/format.h>

#include <numeric>

namespace armnn
{

float ScatterOperation(ScatterNdFunction operation,
                       float input,
                       float update)
{
    switch (operation)
    {
        case ScatterNdFunction::Update:
            return update;
        case ScatterNdFunction::Add:
            return input + update;
        case ScatterNdFunction::Sub:
            return input - update;
        case ScatterNdFunction::Max:
            return std::max(input, update);
        case ScatterNdFunction::Min:
            return std::min(input, update);
        case ScatterNdFunction::Mul:
            return input * update;
        default:
            throw InvalidArgumentException("ScatterNd: cannot execute this operation.");
    }
}

void ScatterNd(const TensorInfo& inputInfo,
               const TensorInfo& indicesInfo,
               const TensorInfo& updatesInfo,
               Decoder<float>& input,
               Decoder<int>& indices,
               Decoder<float>& updates,
               Encoder<float>& output,
               ScatterNdDescriptor descriptor)
{
    // Axis Unsupported
    if (descriptor.m_AxisEnabled)
    {
        throw InvalidArgumentException("ScatterNd: axis param not supported.");
    }

    // Get the shape for indices, updates, and input
    TensorShape indicesShape = indicesInfo.GetShape();
    TensorShape updatesShape = updatesInfo.GetShape();
    TensorShape inputShape = inputInfo.GetShape();

    // Get the dimensions for indices and updates
    unsigned int dimension = inputInfo.GetNumDimensions();
    unsigned int indicesDim = indicesInfo.GetNumDimensions();
    unsigned int updatesDim = updatesInfo.GetNumDimensions();

    // Calculate the outter and inner dimensions
    unsigned int outterDim = indicesShape[indicesDim - 1];
    unsigned int innerDim = dimension - outterDim;

    // Calculate the number of elements in each dimension
    unsigned int numElementsCount = 1;
    std::vector<unsigned int> elementInDim(dimension);
    for (unsigned int dimIndex = dimension; dimIndex > 0; --dimIndex)
    {
        elementInDim[dimIndex - 1] = numElementsCount;
        numElementsCount *= inputShape[dimIndex - 1];
    }

    // Number of updates per index
    unsigned int numUpdatesPerIndex = elementInDim[dimension - innerDim - 1];

    // Number of indices to update
    unsigned int numIndices = indicesShape[0];

    // Check Input Requirements
    // Requirement 1: Indices and Updates must have rank at least 1
    if (indicesDim < 1 || updatesDim < 1)
    {
        throw InvalidArgumentException("ScatterNd: indices and updates must have rank >= 1.");
    }

    // Requirement 2: Input, Indices and Updates must have values
    if (inputInfo.GetNumElements() == 0 ||
        indicesInfo.GetNumElements() == 0 ||
        updatesInfo.GetNumElements() == 0)
    {
        throw InvalidArgumentException("ScatterNd: input, indices and updates tensor must have values.");
    }

    // Requirement 3: Indices and Updates must match in shape
    // The updates dimension should equals to 1 + inner dimension
    if (updatesDim != 1 + innerDim)
    {
        throw InvalidArgumentException("ScatterNd: updates dimension should equal to 1 + inner dimension.");
    }
    // The inner dimension of updates has to match with shape of input
    for (unsigned int dimBackIndex = 0; dimBackIndex < innerDim; ++dimBackIndex)
    {
        if (updatesShape[updatesDim - dimBackIndex - 1] != inputShape[dimension - dimBackIndex - 1])
        {
            throw InvalidArgumentException(
                fmt::format("ScatterNd: input and updates shape not match on dimension {}",
                            dimension - dimBackIndex));
        }
    }

    // Requirement 4: Check duplicate indices and out of bound indices
    std::set<int> indicesSet;
    std::vector<int> flattenIndices(numIndices);
    for (unsigned int indicesIdx = 0; indicesIdx < numIndices; ++indicesIdx)
    {
        // Get the index
        int flattenIndex = 0;

        for (unsigned int outterIdx = 0; outterIdx < outterDim; ++outterIdx) {

            int outterIndexValue = indices.Get();

            // Check bounds
            if (outterIndexValue < 0 || outterIndexValue >= int(inputShape[outterIdx]))
            {
                throw InvalidArgumentException(
                    fmt::format("ScatterNd: indices {} out of bound [0, {})",
                                outterIndexValue, inputShape[outterIdx]));
            }

            flattenIndex += int(elementInDim[outterIdx]) * outterIndexValue;
            ++indices;
        }

        // Check duplicates when executing ScatterNd::Update
        if (descriptor.m_Function == ScatterNdFunction::Update &&
            indicesSet.find(flattenIndex) != indicesSet.end())
        {
            throw InvalidArgumentException(
                    fmt::format("ScatterNd: duplicate indices occurs {}", flattenIndex));
        }

        flattenIndices[indicesIdx] = flattenIndex;
        indicesSet.insert(flattenIndex);
    }

    // Set the input data to output
    for (unsigned int idx = 0; idx < inputInfo.GetNumElements(); ++idx)
    {
        float inputValue = input.Get();
        ++input;
        output.Set(inputValue);
        ++output;
    }

    // Iterate through all indices to scatter updates
    for (unsigned int indicesIdx = 0; indicesIdx < numIndices; ++indicesIdx)
    {
        // Get the index and calculate the flatten index
        int flattenIndex = flattenIndices[indicesIdx];

        // FlattenIndex is the place that we are going to update the elements
        unsigned int updatesStartIdx = indicesIdx * numUpdatesPerIndex;
        for (unsigned int updatesIdx = 0; updatesIdx < numUpdatesPerIndex; ++updatesIdx)
        {
            updates[updatesStartIdx + updatesIdx];
            input[static_cast<unsigned int>(flattenIndex) + updatesIdx];
            float updateValue = ScatterOperation(descriptor.m_Function, input.Get(), updates.Get());
            output[static_cast<unsigned int>(flattenIndex) + updatesIdx];
            output.Set(updateValue);
        }
    }
}

void ScatterNd(const TensorInfo& indicesInfo,
               const TensorInfo& updatesInfo,
               const TensorInfo& shapeInfo,
               Decoder<int>& indices,
               Decoder<float>& updates,
               Decoder<int>& shape,
               Encoder<float>& output,
               ScatterNdDescriptor descriptor)
{
    // Axis Unsupported
    if (descriptor.m_AxisEnabled)
    {
        throw InvalidArgumentException("ScatterNd: axis param not supported.");
    }

    // Get the shape for indices, updates, and input
    TensorShape indicesShape = indicesInfo.GetShape();
    TensorShape updatesShape = updatesInfo.GetShape();

    // Get the shape values
    std::vector<float> shapeValues = shape.DecodeTensor(shapeInfo.GetShape());
    // Check the shape
    if (shapeInfo.GetNumElements() == 0)
    {
        throw InvalidArgumentException("ScatterNd: shape must have values.");
    }
    for (auto shapeValue : shapeValues)
    {
        if (shapeValue <= 0)
        {
            throw InvalidArgumentException("ScatterNd: shape values must >= 0.");
        }
    }
    // Get the input shape
    std::vector<unsigned int> inputShape (shapeValues.begin(), shapeValues.end());
    unsigned int inputElementsNum = static_cast<unsigned int>(
                std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<unsigned int>()));

    // Get the dimensions for indices and updates
    unsigned int dimension = shapeInfo.GetNumElements();
    unsigned int indicesDim = indicesInfo.GetNumDimensions();
    unsigned int updatesDim = updatesInfo.GetNumDimensions();

    // Calculate the outter and inner dimensions
    unsigned int outterDim = indicesShape[indicesDim - 1];
    unsigned int innerDim = dimension - outterDim;

    // Calculate the number of elements in each dimension
    unsigned int numElementsCount = 1;
    std::vector<unsigned int> elementInDim(dimension);
    for (unsigned int dimIndex = dimension; dimIndex > 0; --dimIndex)
    {
        elementInDim[dimIndex - 1] = numElementsCount;
        numElementsCount *= inputShape[dimIndex - 1];
    }

    // Number of updates per index
    unsigned int numUpdatesPerIndex = elementInDim[dimension - innerDim - 1];

    // Number of indices to update
    unsigned int numIndices = indicesShape[0];

    // Check Input Requirements
    // Requirement 1: Indices and Updates must have rank at least 1
    if (indicesDim < 1 || updatesDim < 1)
    {
        throw InvalidArgumentException("ScatterNd: indices and updates must have rank >= 1.");
    }

    // Requirement 2: shape, Indices and Updates must have values
    if (indicesInfo.GetNumElements() == 0 ||
        updatesInfo.GetNumElements() == 0)
    {
        throw InvalidArgumentException("ScatterNd: indices and updates tensor must have values.");
    }

    // Requirement 3: Indices and Updates must match in shape
    // The updates dimension should equals to 1 + inner dimension
    if (updatesDim != 1 + innerDim)
    {
        throw InvalidArgumentException("ScatterNd: updates dimension should equal to 1 + inner dimension.");
    }
    // The inner dimension of updates has to match with shape of input
    for (unsigned int dimBackIndex = 0; dimBackIndex < innerDim; ++dimBackIndex)
    {
        if (updatesShape[updatesDim - dimBackIndex - 1] != inputShape[dimension - dimBackIndex - 1])
        {
            throw InvalidArgumentException(
                    fmt::format("ScatterNd: input and updates shape not match on dimension {}",
                                dimension - dimBackIndex));
        }
    }

    // Requirement 4: Check duplicate indices and out of bound indices
    std::set<int> indicesSet;
    std::vector<int> flattenIndices(numIndices);
    for (unsigned int indicesIdx = 0; indicesIdx < numIndices; ++indicesIdx)
    {
        // Get the index
        int flattenIndex = 0;

        for (unsigned int outterIdx = 0; outterIdx < outterDim; ++outterIdx) {

            int outterIndexValue = indices.Get();

            // Check bounds
            if (outterIndexValue < 0 || outterIndexValue >= int(inputShape[outterIdx]))
            {
                throw InvalidArgumentException(
                        fmt::format("ScatterNd: indices {} out of bound [0, {})",
                                    outterIndexValue, inputShape[outterIdx]));
            }

            flattenIndex += int(elementInDim[outterIdx]) * outterIndexValue;
            ++indices;
        }

        // Check duplicates when executing ScatterNd::Update
        if (descriptor.m_Function == ScatterNdFunction::Update &&
            indicesSet.find(flattenIndex) != indicesSet.end())
        {
            throw InvalidArgumentException(
                    fmt::format("ScatterNd: duplicate indices {} occurs when executing ScatterNd::Update.",
                                flattenIndex));
        }

        flattenIndices[indicesIdx] = flattenIndex;
        indicesSet.insert(flattenIndex);
    }

    // Set zeros to output
    for (unsigned int idx = 0; idx < inputElementsNum; ++idx)
    {
        output.Set(0.0f);
        ++output;
    }

    // Iterate through all indices to scatter updates
    for (unsigned int indicesIdx = 0; indicesIdx < numIndices; ++indicesIdx)
    {
        // Get the index and calculate the flatten index
        int flattenIndex = flattenIndices[indicesIdx];

        // FlattenIndex is the place that we are going to update the elements
        unsigned int updatesStartIdx = indicesIdx * numUpdatesPerIndex;
        for (unsigned int updatesIdx = 0; updatesIdx < numUpdatesPerIndex; ++updatesIdx)
        {
            updates[updatesStartIdx + updatesIdx];
            float updateValue = ScatterOperation(descriptor.m_Function, 0.0f, updates.Get());
            output[static_cast<unsigned int>(flattenIndex) + updatesIdx];
            output.Set(updateValue);
        }
    }
}

} // namespace armnn