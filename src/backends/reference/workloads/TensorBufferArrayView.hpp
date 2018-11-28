//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>

#include <boost/assert.hpp>

#include <DataLayoutIndexed.hpp>

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
        BOOST_ASSERT(m_Shape.GetNumDimensions() == 4);
    }

    DataType& Get(unsigned int b, unsigned int c, unsigned int h, unsigned int w) const
    {
        BOOST_ASSERT( b < m_Shape[0] || ( m_Shape[0]   == 0 && b == 0 ) );
        BOOST_ASSERT( c < m_Shape[m_DataLayout.GetChannelsIndex()] ||
            ( m_Shape[m_DataLayout.GetChannelsIndex()] == 0 && c == 0) );
        BOOST_ASSERT( h < m_Shape[m_DataLayout.GetHeightIndex()] ||
            ( m_Shape[m_DataLayout.GetHeightIndex()]   == 0 && h == 0) );
        BOOST_ASSERT( w < m_Shape[m_DataLayout.GetWidthIndex()] ||
            ( m_Shape[m_DataLayout.GetWidthIndex()]    == 0 && w == 0) );

        // Offset the given indices appropriately depending on the data layout.
        switch (m_DataLayout.GetDataLayout())
        {
        case DataLayout::NHWC:
            b *= m_Shape[1] * m_Shape[2] * m_Shape[3]; // b *= height_index * width_index * channel_index;
            h *= m_Shape[m_DataLayout.GetWidthIndex()] * m_Shape[m_DataLayout.GetChannelsIndex()];
            w *= m_Shape[m_DataLayout.GetChannelsIndex()];
            // c stays unchanged
            break;
        case DataLayout::NCHW:
        default:
            b *= m_Shape[1] * m_Shape[2] * m_Shape[3]; // b *= height_index * width_index * channel_index;
            c *= m_Shape[m_DataLayout.GetHeightIndex()] * m_Shape[m_DataLayout.GetWidthIndex()];
            h *= m_Shape[m_DataLayout.GetWidthIndex()];
            // w stays unchanged
            break;
        }

        // Get the value using the correct offset.
        return m_Data[b + c + h + w];
    }

private:
    const TensorShape             m_Shape;
    DataType*                     m_Data;
    armnnUtils::DataLayoutIndexed m_DataLayout;
};

} //namespace armnn
