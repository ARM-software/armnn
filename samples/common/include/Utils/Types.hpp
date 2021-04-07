//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <tuple>

#include <armnn/BackendId.hpp>

namespace common
{

struct Size
{

    uint32_t m_Width;
    uint32_t m_Height;

    Size() : Size(0, 0) {}

    Size(uint32_t width, uint32_t height) :
            m_Width{width}, m_Height{height} {}

    Size(const Size& other)
            : Size(other.m_Width, other.m_Height) {}

    ~Size() = default;

    Size &operator=(const Size& other) = default;
};

struct BBoxColor
{
    std::tuple<int, int, int> colorCode;
};

struct PipelineOptions
{
    std::string m_ModelName;
    std::string m_ModelFilePath;
    std::vector<armnn::BackendId> m_backends;
};

template<typename T>
using InferenceResult = std::vector<T>;

template<typename T>
using InferenceResults = std::vector<InferenceResult<T>>;
} // namespace common