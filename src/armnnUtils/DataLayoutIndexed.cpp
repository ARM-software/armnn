//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DataLayoutIndexed.hpp"

#include <boost/assert.hpp>

using namespace armnn;

namespace armnnUtils
{

DataLayoutIndexed::DataLayoutIndexed(armnn::DataLayout dataLayout)
    : m_DataLayout(dataLayout)
{
    switch (dataLayout)
    {
        case armnn::DataLayout::NHWC:
            m_ChannelsIndex = 3;
            m_HeightIndex   = 1;
            m_WidthIndex    = 2;
            break;
        case armnn::DataLayout::NCHW:
            m_ChannelsIndex = 1;
            m_HeightIndex   = 2;
            m_WidthIndex    = 3;
            break;
        default:
            throw armnn::InvalidArgumentException("Unknown DataLayout value: " +
                                                  std::to_string(static_cast<int>(dataLayout)));
    }
}

unsigned int DataLayoutIndexed::GetIndex(const TensorShape& shape,
                                         unsigned int batchIndex, unsigned int channelIndex,
                                         unsigned int heightIndex, unsigned int widthIndex) const
{
    BOOST_ASSERT( batchIndex < shape[0] || ( shape[0] == 0 && batchIndex == 0 ) );
    BOOST_ASSERT( channelIndex < shape[m_ChannelsIndex] ||
                ( shape[m_ChannelsIndex] == 0 && channelIndex == 0) );
    BOOST_ASSERT( heightIndex < shape[m_HeightIndex] ||
                ( shape[m_HeightIndex] == 0 && heightIndex == 0) );
    BOOST_ASSERT( widthIndex < shape[m_WidthIndex] ||
                ( shape[m_WidthIndex] == 0 && widthIndex == 0) );

    // Offset the given indices appropriately depending on the data layout
    switch (m_DataLayout)
    {
    case DataLayout::NHWC:
        batchIndex  *= shape[1] * shape[2] * shape[3]; // batchIndex *= heightIndex * widthIndex * channelIndex
        heightIndex *= shape[m_WidthIndex] * shape[m_ChannelsIndex];
        widthIndex  *= shape[m_ChannelsIndex];
        // channelIndex stays unchanged
        break;
    case DataLayout::NCHW:
    default:
        batchIndex   *= shape[1] * shape[2] * shape[3]; // batchIndex *= heightIndex * widthIndex * channelIndex
        channelIndex *= shape[m_HeightIndex] * shape[m_WidthIndex];
        heightIndex  *= shape[m_WidthIndex];
        // widthIndex stays unchanged
        break;
    }

    // Get the value using the correct offset
    return batchIndex + channelIndex + heightIndex + widthIndex;
}

bool operator==(const DataLayout& dataLayout, const DataLayoutIndexed& indexed)
{
    return dataLayout == indexed.GetDataLayout();
}

bool operator==(const DataLayoutIndexed& indexed, const DataLayout& dataLayout)
{
    return indexed.GetDataLayout() == dataLayout;
}

} // namespace armnnUtils
