//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

#include <armnn/utility/Assert.hpp>

namespace armnnUtils
{

/// Provides access to the appropriate indexes for Channels, Height and Width based on DataLayout
class DataLayoutIndexed
{
public:
    DataLayoutIndexed(armnn::DataLayout dataLayout);

    armnn::DataLayout GetDataLayout()    const { return m_DataLayout; }
    unsigned int      GetChannelsIndex() const { return m_ChannelsIndex; }
    unsigned int      GetHeightIndex()   const { return m_HeightIndex; }
    unsigned int      GetWidthIndex()    const { return m_WidthIndex; }

    inline unsigned int GetIndex(const armnn::TensorShape& shape,
                                 unsigned int batchIndex, unsigned int channelIndex,
                                 unsigned int heightIndex, unsigned int widthIndex) const
    {
        ARMNN_ASSERT( batchIndex < shape[0] || ( shape[0] == 0 && batchIndex == 0 ) );
        ARMNN_ASSERT( channelIndex < shape[m_ChannelsIndex] ||
                    ( shape[m_ChannelsIndex] == 0 && channelIndex == 0) );
        ARMNN_ASSERT( heightIndex < shape[m_HeightIndex] ||
                    ( shape[m_HeightIndex] == 0 && heightIndex == 0) );
        ARMNN_ASSERT( widthIndex < shape[m_WidthIndex] ||
                    ( shape[m_WidthIndex] == 0 && widthIndex == 0) );

        /// Offset the given indices appropriately depending on the data layout
        switch (m_DataLayout)
        {
        case armnn::DataLayout::NHWC:
            batchIndex  *= shape[1] * shape[2] * shape[3]; // batchIndex *= heightIndex * widthIndex * channelIndex
            heightIndex *= shape[m_WidthIndex] * shape[m_ChannelsIndex];
            widthIndex  *= shape[m_ChannelsIndex];
            /// channelIndex stays unchanged
            break;
        case armnn::DataLayout::NCHW:
        default:
            batchIndex   *= shape[1] * shape[2] * shape[3]; // batchIndex *= heightIndex * widthIndex * channelIndex
            channelIndex *= shape[m_HeightIndex] * shape[m_WidthIndex];
            heightIndex  *= shape[m_WidthIndex];
            /// widthIndex stays unchanged
            break;
        }

        /// Get the value using the correct offset
        return batchIndex + channelIndex + heightIndex + widthIndex;
    }

private:
    armnn::DataLayout m_DataLayout;
    unsigned int      m_ChannelsIndex;
    unsigned int      m_HeightIndex;
    unsigned int      m_WidthIndex;
};

/// Equality methods
bool operator==(const armnn::DataLayout& dataLayout, const DataLayoutIndexed& indexed);
bool operator==(const DataLayoutIndexed& indexed, const armnn::DataLayout& dataLayout);

} // namespace armnnUtils
