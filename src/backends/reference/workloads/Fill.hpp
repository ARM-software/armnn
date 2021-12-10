//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"
#include <armnn/Tensor.hpp>
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

/// Creates a tensor and fills it with a scalar value.
void Fill(Encoder<float>& output,
          const TensorShape& desiredOutputShape,
          const float value);

} //namespace armnn
