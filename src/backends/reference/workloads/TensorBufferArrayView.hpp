//
// Copyright © 2017, 2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

namespace armnn
{

// Utility class providing access to raw tensor memory based on indices along each dimension.
template <typename DataType>
class TensorBufferArrayView
{
public:
    TensorBufferArrayView(const TensorShape& shape, DataType* data,
                          armnnUtils::DataLayoutIndexed dataLayout = DataLayout::NCHW)
        : m_Shape(shape)
        , m_Data(data)
        , m_DataLayout(dataLayout)
    {
        ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(m_Shape.GetNumDimensions() == 4,
                                            "Only $d tensors are supported by TensorBufferArrayView.");
    }

    DataType& Get(unsigned int b, unsigned int c, unsigned int h, unsigned int w) const
    {
        return m_Data[m_DataLayout.GetIndex(m_Shape, b, c, h, w)];
    }

private:
    const TensorShape             m_Shape;
    DataType*                     m_Data;
    armnnUtils::DataLayoutIndexed m_DataLayout;
};

} //namespace armnn
