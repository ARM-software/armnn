//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <memory>

namespace common
{
/**
 * @brief Frame source reader interface
 *
 * @tparam FrameDataT frame container data type
 */
template<typename FrameDataT> class IFrameReader
{

public:
    /**
     * @brief Reads the next frame from the source
     *
     * @return pointer to the frame container
     */
    virtual std::shared_ptr <FrameDataT> ReadFrame() = 0;

    /**
     * @brief Checks if the frame source has more frames to read.
     *
     * @param[in] frame the pointer to the last frame captured with the ReadFrame method could be used in
     *                  implementation specific logic to check frames source state.
     * @return True if frame source was exhausted, False otherwise
     */
    virtual bool IsExhausted(const std::shared_ptr <FrameDataT>& frame) const = 0;

    /**
     * @brief Default destructor
     */
    virtual ~IFrameReader() = default;

};

}// namespace common