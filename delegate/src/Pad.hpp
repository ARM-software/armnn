//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitPadOperator(DelegateData& delegateData,
                              TfLiteContext* tfLiteContext,
                              TfLiteNode* tfLiteNode,
                              int nodeIndex,
                              int32_t padOperatorCode)
{
    armnn::IgnoreUnused(delegateData,
                        tfLiteContext,
                        tfLiteNode,
                        nodeIndex,
                        padOperatorCode);

    return kTfLiteError;
}

} // namespace armnnDelegate
