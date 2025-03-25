//
// Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TosaOperatorUtils.hpp"

using namespace armnn;
using namespace tosa;

TosaSerializationBasicBlock* ConvertElementwiseBinaryToTosaOperator(const Layer* layer,
                                                                    const LayerType type,
                                                                    const std::vector<const TensorInfo*>& inputs,
                                                                    const std::vector<const TensorInfo*>& outputs,
                                                                    const ElementwiseBinaryDescriptor*
                                                                        descriptor = nullptr);

TosaSerializationBasicBlock* ConvertSquaredDifferenceToTosaOperator(const Layer* layer,
                                                                    const LayerType type,
                                                                    const std::vector<const TensorInfo*>& inputs,
                                                                    const std::vector<const TensorInfo*>& outputs,
                                                                    const ElementwiseBinaryDescriptor* descriptor);

/// Function used to add the ADD operator to the operator vector.
void ConvertAddToTosaOperator(const std::vector<string>& inputs,
                              const std::vector<string>& outputs,
                              std::vector<TosaSerializationOperator*>& operators);

/// Function used to add the MUL operator to the operator vector.
void ConvertMulToTosaOperator(const std::vector<string>& inputs,
                              const std::vector<string>& outputs,
                              std::vector<TosaSerializationOperator*>& operators);

/// Function used to add the SUB operator to the operator vector.
void ConvertSubToTosaOperator(const std::vector<string>& inputs,
                              const std::vector<string>& outputs,
                              std::vector<TosaSerializationOperator*>& operators);

/// Function used to calculate correct scales for rescales for Int8 input to ADD, MUL and SUB operators.
void CalculateRescaleScales(float& input0Scale,
                            float& input1Scale,
                            float& outputScale,
                            const BinaryOperation& operation);
