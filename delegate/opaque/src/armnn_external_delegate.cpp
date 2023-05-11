//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>

namespace {

    TfLiteOpaqueDelegate* ArmNNDelegateCreateFunc(const void* tflite_settings)
    {
        auto delegate = armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(tflite_settings);
        return delegate;
    }

    void ArmNNDelegateDestroyFunc(TfLiteOpaqueDelegate* armnnDelegate)
    {
        armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete(
                armnnDelegate);
    }

    int ArmNNDelegateErrnoFunc(TfLiteOpaqueDelegate* sample_stable_delegate)
    {
        return 0;
    }

    const TfLiteOpaqueDelegatePlugin armnn_delegate_plugin = {
            ArmNNDelegateCreateFunc, ArmNNDelegateDestroyFunc,
            ArmNNDelegateErrnoFunc};

    const TfLiteStableDelegate armnn_delegate = {
            /*delegate_abi_version=*/ TFL_STABLE_DELEGATE_ABI_VERSION,
            /*delegate_name=*/        "armnn_delegate",
            /*delegate_version=*/     OPAQUE_DELEGATE_VERSION,
            /*delegate_plugin=*/      &armnn_delegate_plugin
    };

}  // namespace

/**
 * The ArmNN delegate to be loaded dynamically
 */
extern "C" const TfLiteStableDelegate TFL_TheStableDelegate = armnn_delegate;