//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include <armnn/Tensor.hpp>

namespace armnn
{

void LogSoftmax(Decoder<float>& input,
                Encoder<float>& output,
                const TensorInfo& inputInfo,
                const LogSoftmaxDescriptor& descriptor);

} // namespace armnn
