//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

#include "BaseIterator.hpp"

namespace armnn
{

void LogSoftmax(Decoder<float>& input,
                Encoder<float>& output,
                const TensorInfo& inputInfo,
                const LogSoftmaxDescriptor& descriptor);

} // namespace armnn
