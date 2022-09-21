//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ConversionUtils.hpp"

#include <nnapi/OperandTypes.h>
#include <nnapi/Result.h>
#include <nnapi/Types.h>

#include <armnn/Types.hpp>
using namespace armnn;

namespace armnn_driver
{

class Converter
{

public:
    using Model                     = ::android::nn::Model;
    using Operand                   = ::android::nn::Operand;
    using OperandLifeTime           = ::android::nn::Operand::LifeTime;
    using OperandType               = ::android::nn::OperandType;
    using Operation                 = ::android::nn::Operation;
    using OperationType             = ::android::nn::OperationType;
    using ErrorStatus               = ::android::nn::ErrorStatus;
    static bool ConvertOperation(const Operation& operation, const Model& model, ConversionData& data);

private:
    static bool ConvertAdd(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertArgMinMax(const Operation& operation,
                                 const Model& model,
                                 ConversionData& data,
                                 armnn::ArgMinMaxFunction argMinMaxFunction);

    static bool ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertBatchMatMul(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertCast(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertComparison(const Operation& operation,
                                  const Model& model,
                                  ConversionData& data,
                                  armnn::ComparisonOperation comparisonOperation);

    static bool ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertDiv(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertElementwiseUnary(const Operation& operation,
                                        const Model& model,
                                        ConversionData& data,
                                        armnn::UnaryOperation unaryOperation);

    static bool ConvertElu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertExpandDims(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFill(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFloor(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertGather(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertGroupedConv2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertHardSwish(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertInstanceNormalization(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data);

    static bool ConvertLogicalBinary(const Operation& operation,
                                     const Model& model,
                                     ConversionData& data,
                                     armnn::LogicalBinaryOperation logicalOperation);

    static bool ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLogSoftmax(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMaximum(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMean(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMinimum(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertMul(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPad(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantize(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantizedLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertQuantized16BitLstm(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertRank(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertReshape(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              armnn::ResizeMethod resizeMethod);

    static bool ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSqrt(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertSub(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTanH(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data);

    static bool ConvertTransposeConv2d(const Operation& operation, const Model& model, ConversionData& data);
};

} // namespace armnn_driver
