//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchMatMulImpl.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/Logging.hpp>
#include <armnnUtils/Permute.hpp>

namespace armnn
{

BatchMatMul::BatchMatMul(const BatchMatMulDescriptor& params,
                         const TensorInfo& inputXInfo,
                         const TensorInfo& inputYInfo,
                         const TensorInfo& outputInfo,
                         Decoder<float>& inputXDecoder,
                         Decoder<float>& inputYDecoder,
                         Encoder<float>& outputEncoder)
    : params(params),
      inputXInfo(inputXInfo),
      inputYInfo(inputYInfo),
      outputInfo(outputInfo),
      inputXDecoder(inputXDecoder),
      inputYDecoder(inputYDecoder),
      outputEncoder(outputEncoder)
{
    inputXData = this->inputXDecoder.DecodeTensor(inputXInfo.GetShape());
    inputYData = this->inputYDecoder.DecodeTensor(inputYInfo.GetShape());
    // At this point, we don't touch the input decoders - just the resultant vectors

    ApplyParams();

    ApplyBatchMatMul();
}

void BatchMatMul::ApplyBatchMatMul()
{
    auto axesXToMul = BatchMatMulDescriptor::GetAxesToMul(params.m_DataLayoutX,
                                                          inputXInfo.GetShape());
    auto axesYToMul = BatchMatMulDescriptor::GetAxesToMul(params.m_DataLayoutY,
                                                          inputYInfo.GetShape());
    AdjustAxesToMulForUnequalRanks(axesXToMul, axesYToMul);

    unsigned int inputXColDim = axesXToMul.second;
    unsigned int inputYRowDim = axesYToMul.first;

    unsigned int inputYRowSize = inputYInfo.GetShape()[inputYRowDim];

    auto batchMatMulOperation = [&](const std::vector<unsigned int>& curIdx)
    {
        float sum = 0.0f;

        // InputYRowSize is synonymous with inputXColSize
        for (unsigned int inputYRowIdx = 0; inputYRowIdx < inputYRowSize; inputYRowIdx++) {
            auto xIdx = curIdx;
            xIdx[inputXColDim] = inputYRowIdx;

            auto yIdx = curIdx;
            yIdx[inputYRowDim] = inputYRowIdx;

            sum += (GetValueAt(DataSlot::InputX, xIdx) * GetValueAt(DataSlot::InputY, yIdx));
        }

        SetValueAt(sum, DataSlot::Output, curIdx);
    };

    auto startIdx = std::vector<unsigned int>(outputInfo.GetNumDimensions(), 0);
    RecurseTensor(outputInfo,
                  batchMatMulOperation,
                  startIdx,
                  0);
}

void BatchMatMul::ApplyParams()
{
    if(params.m_TransposeX)
    {
        Transpose(DataSlot::InputX);
    }
    else if(params.m_AdjointX)
    {
        Adjoint(DataSlot::InputX);
    }
    if(params.m_TransposeY)
    {
        Transpose(DataSlot::InputY);
    }
    else if(params.m_AdjointY)
    {
        Adjoint(DataSlot::InputY);
    }
}

void BatchMatMul::Transpose(DataSlot type)
{
    // AKA the permute of the tensor
    // This modifies the tensor's info.

    switch(type)
    {
        case DataSlot::InputX:
        {
            auto permuteVec = BatchMatMulDescriptor::GetPermuteVec(params.m_DataLayoutX,
                                                                   inputXInfo.GetShape());
            inputXInfo = armnnUtils::Permuted(inputXInfo, permuteVec);
            std::vector<float> temp(inputXData.size());
            armnnUtils::Permute(inputXInfo.GetShape(),
                                permuteVec,
                                inputXData.data(),
                                temp.data(),
                                sizeof(float));
            inputXData = temp;
            break;
        }
        case DataSlot::InputY:
        {
            auto permuteVec = BatchMatMulDescriptor::GetPermuteVec(params.m_DataLayoutY,
                                                                   inputYInfo.GetShape());
            inputYInfo = armnnUtils::Permuted(inputYInfo, permuteVec);
            std::vector<float> temp(inputYData.size());
            armnnUtils::Permute(inputYInfo.GetShape(),
                                permuteVec,
                                inputYData.data(),
                                temp.data(),
                                sizeof(float));
            inputYData = temp;
            break;
        }
        case DataSlot::Output: // We needn't transpose the output tensor
        default:
            break;
    }
}

void BatchMatMul::Adjoint(DataSlot type)
{
    // Finding the adjoint of a square matrix:
    // Calculate the cofactor of each element (using Gauss elimination here)
    // Apply a transpose to it (this also modifies the tensor's info)

    TensorInfo& inputInfo = (type == DataSlot::InputX) ? inputXInfo : inputYInfo;
    const auto& dataLayout = (type == DataSlot::InputX) ? params.m_DataLayoutX : params.m_DataLayoutY;
    const auto axesToAdjoint = BatchMatMulDescriptor::GetAxesToMul(dataLayout,inputInfo.GetShape());

    ARMNN_ASSERT(inputInfo.GetShape()[axesToAdjoint.first] == inputInfo.GetShape()[axesToAdjoint.second]);
    // We grab a copy of the tensor data to prevent overwriting
    std::vector<float> inputDataClone = (type == DataSlot::InputX) ? inputXData : inputYData;

    // The sub-matrix is the resultant matrix when the row and column of the current index is removed
    unsigned int subMatAxisSize = inputInfo.GetShape()[axesToAdjoint.first] - 1;
    std::vector<std::vector<float>> subMat(subMatAxisSize,
                                           std::vector<float>(subMatAxisSize));

    // Lambdas for each sub-step of the cofactor operation
    auto almostEquals = [&](const float& a, const float& b, float unitsInLastPlace = 2.0f)
    {
        float diff = std::fabs(a-b);
        float bound = diff * std::numeric_limits<float>::epsilon() * unitsInLastPlace;
        return (diff <= bound) || (diff < std::numeric_limits<float>::min());
    };

    float swapMultiplier = std::numeric_limits<float>::max();
    auto swapRows = [&](unsigned int rowIdxA, unsigned int rowIdxB)
    {
        // Every row swap flips this around by the negative (set to 1 at the beginning of each cofactor op run)
        for(unsigned int colIdx = 0; colIdx < subMatAxisSize; colIdx++)
        {
            float tmp = subMat[rowIdxA][colIdx];
            subMat[rowIdxA][colIdx] = subMat[rowIdxB][colIdx];
            subMat[rowIdxB][colIdx] = tmp;
        }
        swapMultiplier *= -1.0f;
    };

    auto findNextValidPivotRowIdx = [&](unsigned int colIdx)
    {
        unsigned int result = std::numeric_limits<unsigned int>::max();

        // The original diagonal has been checked and is invalid
        for(unsigned int rowIdx = colIdx+1; rowIdx < subMatAxisSize; rowIdx++)
        {
            if(!almostEquals(subMat[rowIdx][colIdx], 0.0f))
            {
                result = rowIdx;
                break;
            }
        }
        return result;
    };

    auto eliminate = [&](const float& pivot, unsigned int pivotPos)
    {
        for(unsigned int rowIdx = pivotPos+1; rowIdx < subMatAxisSize; rowIdx++)
        {
            float multiplierNumerator = subMat[rowIdx][pivotPos];
            if(almostEquals(multiplierNumerator, 0.0f))
            {
                continue;
            }
            float multiplier = multiplierNumerator / pivot; // Susceptible to floating point inaccuracies
                                                            // Hence the almostEquals usage to counteract this
            for(unsigned int colIdx = pivotPos; colIdx < subMatAxisSize; colIdx++)
            {
                // We start at col=pivotPos as we have assumed that all elements
                // to our left have been eliminated to zero already

                // We subtract based on the element directly above us in our pivot row
                subMat[rowIdx][colIdx] -= multiplier * subMat[pivotPos][colIdx];
            }
        }
    };

    auto cofactorOperation = [&](const std::vector<unsigned int>& curIdx)
    {
        auto row = curIdx[axesToAdjoint.first];
        auto col = curIdx[axesToAdjoint.second];

        float minorMultiplier = static_cast<float>(std::pow(-1, (row + 1 + col + 1)));

        for(unsigned int subRow = 0; subRow < subMatAxisSize; subRow++)
        {
            for(unsigned int subCol = 0; subCol < subMatAxisSize; subCol++)
            {
                unsigned int outerRow = (subRow >= row)?subRow + 1:subRow;
                unsigned int outerCol = (subCol >= col)?subCol + 1:subCol;
                auto cloneIdx = curIdx;
                cloneIdx[axesToAdjoint.first] = outerRow;
                cloneIdx[axesToAdjoint.second] = outerCol;
                subMat[subRow][subCol] = GetValueAt(type,cloneIdx,inputDataClone);
            }
        }

        float determinant = 1.0f;

        // Cover the edge cases and simple base cases before resorting to Gauss elimination for larger matrices
        switch(subMatAxisSize)
        {
            case 0:
            {
                determinant = GetValueAt(type, curIdx, inputDataClone);
                break;
            }
            case 1:
            {
                // If the resultant sub-matrix is just one element - that's the determinant
                determinant = subMat[0][0];
                break;
            }
            case 2:
            {
                // For a 2x2 sub-matrix, the determinant is just a*d-b*c
                determinant = subMat[0][0] * subMat[1][1] -
                              subMat[0][1] * subMat[1][0];
                break;
            }
            default:
            {
                // Gaussian elimination to find the determinant of this sub-matrix
                swapMultiplier = 1.0f;
                // March diagonally down the pivots and if it's invalid (a zero), swap the row with the
                // nearest non-zero down within the column
                for(unsigned int pivotRow = 0, pivotCol = 0;
                    pivotRow < subMatAxisSize;
                    pivotRow++, pivotCol++)
                {
                    float& pivot = subMat[pivotRow][pivotCol];

                    if(almostEquals(pivot, 0.0f))
                    {
                        unsigned int nextValidPivotRowIdx = findNextValidPivotRowIdx(pivotCol);
                        if(nextValidPivotRowIdx == std::numeric_limits<unsigned int>::max())
                        {
                            // No valid pivot down this column, which means that this pivot remains a zero.
                            // This results in the determinant for this entire sub-matrix to just be zero.
                            determinant = 0.0f;
                            break;
                        }
                        swapRows(pivotRow, nextValidPivotRowIdx);
                    }
                    determinant *= pivot;
                    // The actual elimination bit (which will update/propagate to the pivots down the line)
                    eliminate(pivot, pivotRow); // Synonymous with pivotCol
                }

                determinant *= swapMultiplier;
                break;
            }
        }
        float cofactor = minorMultiplier * determinant;
        SetValueAt(cofactor, type, curIdx);
    };

    auto startIdx = std::vector<unsigned int>(inputInfo.GetNumDimensions(), 0);
    RecurseTensor(inputInfo,
                  cofactorOperation,
                  startIdx,
                  0);

    Transpose(type);
}

void BatchMatMul::RecurseTensor(const TensorInfo& tensorInfo,
                                const std::function<void(const std::vector<unsigned int>&)>& operation,
                                std::vector<unsigned int>& curIdx,
                                unsigned int curDim)
{
    if(!(curDim < tensorInfo.GetNumDimensions()))
    {
        // We're at the leaf level of this call tree, so we operate here (each leaf is a data point)
        operation(curIdx);
        return;
    }

    for(unsigned int i = 0; i < tensorInfo.GetShape()[curDim]; i++)
    {
        curIdx[curDim] = i;
        RecurseTensor(tensorInfo,
                      operation,
                      curIdx,
                      curDim + 1);
    }
}

void BatchMatMul::AdjustAxesToMulForUnequalRanks(std::pair<unsigned int, unsigned int>& axesXToMul,
                                                 std::pair<unsigned int, unsigned int>& axesYToMul)
{
    int rankDiff = static_cast<int>(inputXInfo.GetNumDimensions()) -
                   static_cast<int>(inputYInfo.GetNumDimensions());
    if(rankDiff == 0)
    {
        return;
    }
    else if(rankDiff < 0)
    {
        // Y is the larger one
        axesXToMul.first += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
        axesXToMul.second += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
    }
    else if(rankDiff > 0)
    {
        // X is the larger one
        axesYToMul.first += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
        axesYToMul.second += static_cast<std::make_unsigned<unsigned int>::type>(std::abs(rankDiff));
    }
}

float BatchMatMul::GetValueAt(DataSlot type, std::vector<unsigned int> idx, const std::vector<float>& customData)
{
    // This gets the data from the input vector that we have, Not the decoder
    // But for the output, it is operating on the encoder itself

    AdjustToSafeIdx(type, idx);
    unsigned int flatIdx = CalcFlatIdx(type, idx);
    float value = 0.0f;
    switch(type)
    {
        case DataSlot::InputX:
            value = customData.empty() ? inputXData[flatIdx] : customData[flatIdx];
            break;
        case DataSlot::InputY:
            value = customData.empty() ? inputYData[flatIdx] : customData[flatIdx];
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

} // namespace armnn