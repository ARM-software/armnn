//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchMatMulImpl.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/Logging.hpp>

namespace armnn
{

void BatchMatMul::BatchMatMulImpl()
{
    inputXData = inputXDecoder.DecodeTensor(inputXInfo.GetShape());
    inputYData = inputYDecoder.DecodeTensor(inputYInfo.GetShape());
    // At this point, we don't touch the input decoders - just the resultant vectors

    // Pre-transpose and pre-adjoint if their vectors aren't empty
    // and also DataLayouts which may change with permutations/adjoints

    // Todo: Have you updated input validation and inferred output shapes to accommodate for these pre-permutes?

    auto idx = std::vector<unsigned int>(outputInfo.GetNumDimensions(), 0);
    RecurseBMM(idx, 0);
}

void BatchMatMul::RecurseBMM(std::vector<unsigned int>& curIdx, unsigned int curDim)
{
    // We're working off of the indexes of the output tensor (the max possible shape)

    if(!(curDim < outputInfo.GetNumDimensions()))
    {
        // We're at the leaf level of this call tree, so we operate here (each leaf is a data point)

        auto axesToMul = BatchMatMulDescriptor::GetAxesToMul(params,
                                                             inputXInfo.GetShape(),
                                                             inputYInfo.GetShape());
        AdjustAxesToMulForUnequalRanks(axesToMul);

        unsigned int inputXColDim = axesToMul.first.second;
        unsigned int inputYRowDim = axesToMul.second.first;

        unsigned int inputYRowSize = inputYInfo.GetShape()[inputYRowDim];

        float sum = 0.0f;

        // You could also use inputXColSize
        for (unsigned int inputYRowIdx = 0; inputYRowIdx < inputYRowSize; inputYRowIdx++) {
            auto xIdx = curIdx;
            xIdx[inputXColDim] = inputYRowIdx;

            auto yIdx = curIdx;
            yIdx[inputYRowDim] = inputYRowIdx;

            sum += (GetValueAt(DataSlot::InputX, xIdx)
                  * GetValueAt(DataSlot::InputY, yIdx));
        }

        SetValueAt(sum, DataSlot::Output, curIdx);

        return;
    }

    for (unsigned int i = 0; i < outputInfo.GetShape()[curDim]; i++)
    {
        curIdx[curDim] = i;
        RecurseBMM(curIdx, curDim+1);
    }
}

void BatchMatMul::AdjustAxesToMulForUnequalRanks(
    std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>>& axesToMul)
{
    long rankDiff = static_cast<long>(inputXInfo.GetNumDimensions()) - inputYInfo.GetNumDimensions();
    if(rankDiff == 0)
    {
        return;
    }
    else if(rankDiff < 0)
    {
        // Y is the larger one
        axesToMul.first.first += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
        axesToMul.first.second += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
    }
    else if(rankDiff > 0)
    {
        // X is the larger one
        axesToMul.second.first += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
        axesToMul.second.second += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
    }
}

float BatchMatMul::GetValueAt(DataSlot type, std::vector<unsigned int> idx)
{
    // This gets the data from the input vector that we have, Not the decoder
    // But for the output, it is operating on the encoder itself

    AdjustToSafeIdx(type, idx);
    unsigned int flatIdx = CalcFlatIdx(type, idx);
    float value = 0.0f;

    switch(type)
    {
        case DataSlot::InputX:
            value = inputXData[flatIdx];
            break;
        case DataSlot::InputY:
            value = inputYData[flatIdx];
            break;
        case DataSlot::Output:
            outputEncoder[flatIdx];
            value = outputEncoder.Get();
            break;
        default:
            break;
    }

    return value;
}

void BatchMatMul::SetValueAt(float value, DataSlot type, std::vector<unsigned int> idx)
{
    AdjustToSafeIdx(type, idx);

    unsigned int flatIdx = CalcFlatIdx(type, idx);

    switch(type)
    {
        case DataSlot::InputX:
            inputXData[flatIdx] = value;
            break;
        case DataSlot::InputY:
            inputYData[flatIdx] = value;
            break;
        case DataSlot::Output:
            outputEncoder[flatIdx];
            outputEncoder.Set(value);
            break;
        default:
            break;
    }
}

void BatchMatMul::AdjustToSafeIdx(DataSlot type, std::vector<unsigned int>& idx)
{
    for(unsigned int dim = 0; dim < idx.size(); dim++)
    {
        switch(type)
        {
            case DataSlot::InputX:
            {
                auto xRank = inputXInfo.GetNumDimensions();
                auto xDiff = outputInfo.GetNumDimensions() - xRank;
                if (dim < xDiff ||
                    idx[dim] > inputXInfo.GetShape()[dim-xDiff]-1)
                {
                    idx[dim] = 0; // Broadcasting
                }
                break;
            }
            case DataSlot::InputY:
            {
                auto yRank = inputYInfo.GetNumDimensions();
                auto yDiff = outputInfo.GetNumDimensions() - yRank;
                if (dim < yDiff ||
                    idx[dim] > inputYInfo.GetShape()[dim-yDiff]-1)
                {
                    idx[dim] = 0;
                }
                break;
            }
            case DataSlot::Output:
            {
                // Our indices are based off the output
                break;
            }
            default:
                break;
        }
    }
}

unsigned int BatchMatMul::CalcFlatIdx(DataSlot type, const std::vector<unsigned int>& idx)
{
    unsigned int result = idx[idx.size()-1];

    unsigned int dimMultiplier = 1;

    unsigned int offset;

    // -2 because final dim is already accounted for in the multiplier (last dim is just a multiplier of 1x)
    for(unsigned int i = static_cast<unsigned int>(idx.size()-2); static_cast<int>(i) >= 0; i--)
    {
        switch(type)
        {
            case DataSlot::InputX:
                offset = outputInfo.GetNumDimensions() - inputXInfo.GetNumDimensions();
                dimMultiplier *= inputXInfo.GetShape()[i + 1 - offset];
                break;
            case DataSlot::InputY:
                offset = outputInfo.GetNumDimensions() - inputYInfo.GetNumDimensions();
                dimMultiplier *= inputYInfo.GetShape()[i + 1 - offset];
                break;
            case DataSlot::Output:
                dimMultiplier *= outputInfo.GetShape()[i+1];
                break;
            default:
                break;
        }
        result += (idx[i] * dimMultiplier);
    }
    return result;
}

template <typename T>
std::string BatchMatMul::StringifyVec(const std::vector<T>& vec)
{
    std::string res = "{ ";
    for(auto x : vec)
    {
        res += std::to_string(x);
        res += " ";
    }
    res += "}";
    return res;
}

} // namespace armnn