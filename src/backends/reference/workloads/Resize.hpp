//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include <armnn/Tensor.hpp>

#include <DataLayoutIndexed.hpp>

namespace armnn
{

void Resize(Decoder<float>&               in,
            const TensorInfo&             inputInfo,
            Encoder<float>&               out,
            const TensorInfo&             outputInfo,
            armnnUtils::DataLayoutIndexed dataLayout = DataLayout::NCHW,
            ResizeMethod                  resizeMethod = ResizeMethod::NearestNeighbor);

} //namespace armnn
