//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma  once

#include "armnn/Tensor.hpp"

namespace armnn
{

template <typename T>
void Gather(const TensorInfo& paramsInfo,
            const TensorInfo& indicesInfo,
            const TensorInfo& outputInfo,
            const T* params,
            const int32_t* indices,
            T* output);

} //namespace armnn
