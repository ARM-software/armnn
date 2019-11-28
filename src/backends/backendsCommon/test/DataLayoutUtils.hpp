//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <armnnUtils/Permute.hpp>

template<typename T>
void PermuteTensorNchwToNhwc(armnn::TensorInfo& tensorInfo, std::vector<T>& tensorData)
{
    const armnn::PermutationVector nchwToNhwc = { 0, 3, 1, 2 };

    tensorInfo = armnnUtils::Permuted(tensorInfo, nchwToNhwc);

    std::vector<T> tmp(tensorData.size());
    armnnUtils::Permute(tensorInfo.GetShape(), nchwToNhwc, tensorData.data(), tmp.data(), sizeof(T));
    tensorData = tmp;
}

template<typename T>
void PermuteTensorNhwcToNchw(armnn::TensorInfo& tensorInfo, std::vector<T>& tensorData)
{
    const armnn::PermutationVector nhwcToNchw = { 0, 2, 3, 1 };

    tensorInfo = armnnUtils::Permuted(tensorInfo, nhwcToNchw);

    std::vector<T> tmp(tensorData.size());
    armnnUtils::Permute(tensorInfo.GetShape(), nhwcToNchw, tensorData.data(), tmp.data(), sizeof(T));

    tensorData = tmp;
}
