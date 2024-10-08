//
// Copyright © 2018-2021,2023 Arm Ltd. All rights reserved.
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
    unsigned int      GetDepthIndex()    const { return m_DepthIndex; }

    inline unsigned int GetIndex(const armnn::TensorShape& shape,
                                 unsigned int batchIndex, unsigned int channelIndex,
                                 unsigned int heightIndex, unsigned int widthIndex) const
    {
        if (batchIndex >= shape[0] && !( shape[0] == 0 && batchIndex == 0))
        {
            throw armnn::Exception("Unable to get batch index", CHECK_LOCATION());
        }
        if (channelIndex >= shape[m_ChannelsIndex] &&
                    !(shape[m_ChannelsIndex] == 0 && channelIndex == 0))
        {
            throw armnn::Exception("Unable to get channel index", CHECK_LOCATION());

        }
        if (heightIndex >= shape[m_HeightIndex] &&
                    !( shape[m_HeightIndex] == 0 && heightIndex == 0))
        {
            throw armnn::Exception("Unable to get height index", CHECK_LOCATION());
        }
        if (widthIndex >= shape[m_WidthIndex] &&
                    ( shape[m_WidthIndex] == 0 && widthIndex == 0))
        {
            throw armnn::Exception("Unable to get width index", CHECK_LOCATION());
        }

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
    unsigned int      m_DepthIndex;
};

/// Equality methods
bool operator==(const armnn::DataLayout& dataLayout, const DataLayoutIndexed& indexed);
bool operator==(const DataLayoutIndexed& indexed, const armnn::DataLayout& dataLayout);

} // namespace armnnUtils
