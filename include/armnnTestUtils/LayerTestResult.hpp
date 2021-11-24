//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/utility/Assert.hpp>

#include <cstddef>
#include <vector>

template <typename T, std::size_t n>
struct LayerTestResult
{
    LayerTestResult(const armnn::TensorInfo& outputInfo)
        : m_Supported(true)
        , m_CompareBoolean(false)
    {
        m_ActualData.reserve(outputInfo.GetNumElements());
        m_ExpectedData.reserve(outputInfo.GetNumElements());
        m_ActualShape = outputInfo.GetShape();
        m_ExpectedShape = outputInfo.GetShape();
    }

    LayerTestResult(const std::vector<T>& actualData,
                    const std::vector<T>& expectedData,
                    const armnn::TensorShape& actualShape,
                    const armnn::TensorShape& expectedShape)
        : m_ActualData(actualData)
        , m_ExpectedData(expectedData)
        , m_ActualShape(actualShape)
        , m_ExpectedShape(expectedShape)
        , m_Supported(true)
        , m_CompareBoolean(false)
    {}

    LayerTestResult(const std::vector<T>& actualData,
                    const std::vector<T>& expectedData,
                    const armnn::TensorShape& actualShape,
                    const armnn::TensorShape& expectedShape,
                    const bool compareBoolean)
        : m_ActualData(actualData)
        , m_ExpectedData(expectedData)
        , m_ActualShape(actualShape)
        , m_ExpectedShape(expectedShape)
        , m_Supported(true)
        , m_CompareBoolean(compareBoolean)
    {}

    std::vector<T> m_ActualData;
    std::vector<T> m_ExpectedData;
    armnn::TensorShape m_ActualShape;
    armnn::TensorShape m_ExpectedShape;

    bool m_Supported;
    bool m_CompareBoolean;
};




