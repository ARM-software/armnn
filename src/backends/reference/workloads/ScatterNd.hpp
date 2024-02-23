//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include "Encoders.hpp"
#include "Decoders.hpp"
#include "armnn/Descriptors.hpp"

namespace armnn
{
// ScatterNd with input tensor
void ScatterNd(const TensorInfo& inputInfo,
               const TensorInfo& indicesInfo,
               const TensorInfo& updatesInfo,
               Decoder<float>& input,
               Decoder<int>& indices,
               Decoder<float>& updates,
               Encoder<float>& output,
               ScatterNdDescriptor descriptor);

// ScatterNd without input tensor, only shape provided
void ScatterNd(const TensorInfo& indicesInfo,
               const TensorInfo& updatesInfo,
               const TensorInfo& shapeInfo,
               Decoder<int>& indices,
               Decoder<float>& updates,
               Decoder<int>& shape,
               Encoder<float>& output,
               ScatterNdDescriptor descriptor);
} // namespace armnn