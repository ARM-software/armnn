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
 * @brief Frames output interface
 *
 * @tparam FrameDataT frame container data type
 */
    template<typename FrameDataT> class IFrameOutput
    {

    public:
        /**
         * @brief Writes frame to the selected output
         *
         * @param frame container
         */
        virtual void WriteFrame(std::shared_ptr <FrameDataT>& frame) = 0;

        /**
         * @brief Closes the frame output
         */
        virtual void Close() = 0;

        /**
         * @brief Checks if the frame sink is ready to write.
         *
         * @return True if frame sink is ready, False otherwise
         */
        virtual bool IsReady() const = 0;

        /**
         * @brief Default destructor
         */
        virtual ~IFrameOutput() = default;

    };

}// namespace common
