//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

#include <tosa_serialization_handler.h>
#include "operatorMappings/AdditionOperator.hpp"

using namespace armnn;
using namespace tosa;

// From the input armnn::Layer, set the corresponding data field in the
// tosa::TosaSerializationTensor where constant tensor data exists in the armnn::Layer.
void SetBasicBlockConstantTensorData(Layer* layer, TosaSerializationBasicBlock* /*basicBlock*/);

// Populates a tosa::TosaSerializationBasicBlock from constructing
// tosa::TosaSerializationOperator(s) and tosa::TosaSerializationTensor(s)
// based on the input armnn::LayerType and associated armnn::TensorInfos and armnn::Descriptor.
//
// If an armnn::LayerType does not have a tosa mapping or the mapping is not implemented in ArmNN,
// an empty tosa::TosaSerializationBasicBlock() is returned with operator tosa::Op_UNKNOWN.
TosaSerializationBasicBlock* GetTosaMapping(const LayerType type,
                                            const std::vector<const TensorInfo*>& inputs,
                                            const std::vector<const TensorInfo*>& outputs,
                                            const BaseDescriptor& /*descriptor*/,
                                            bool isMain);

// Function called in armnn::OptimizeSubgraphView() when access to armnn::Layer is available
// and there is an option to set tosa basic block data from constant layer tenors available from the input layer.
TosaSerializationBasicBlock* GetTosaMappingFromLayer(Layer* layer, bool isMain);
