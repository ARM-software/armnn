//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <DelegateTestInterpreter.hpp>

#include <armnn_delegate.hpp>

namespace delegateTestInterpreter
{

DelegateTestInterpreter::DelegateTestInterpreter(std::vector<char>& modelBuffer,
                                                 const std::vector<armnn::BackendId>& backends,
                                                 const std::string& customOp,
                                                 bool disableFallback)
{
    TfLiteModel* tfLiteModel = delegateTestInterpreter::CreateTfLiteModel(modelBuffer);

    TfLiteInterpreterOptions* options = delegateTestInterpreter::CreateTfLiteInterpreterOptions();
    if (!customOp.empty())
    {
        options->mutable_op_resolver = delegateTestInterpreter::GenerateCustomOpResolver(customOp);
    }

    // Disable fallback by default for unit tests unless specified.
    armnnDelegate::DelegateOptions delegateOptions(backends);
    delegateOptions.DisableTfLiteRuntimeFallback(disableFallback);

    auto armnnDelegate = armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions);
    TfLiteInterpreterOptionsAddDelegate(options, armnnDelegate);

    m_TfLiteDelegate = armnnDelegate;
    m_TfLiteInterpreter = TfLiteInterpreterCreate(tfLiteModel, options);

    // The options and model can be deleted after the interpreter is created.
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(tfLiteModel);
}

DelegateTestInterpreter::DelegateTestInterpreter(std::vector<char>& modelBuffer,
                                                 const armnnDelegate::DelegateOptions& delegateOptions,
                                                 const std::string& customOp)
{
    TfLiteModel* tfLiteModel = delegateTestInterpreter::CreateTfLiteModel(modelBuffer);

    TfLiteInterpreterOptions* options = delegateTestInterpreter::CreateTfLiteInterpreterOptions();
    if (!customOp.empty())
    {
        options->mutable_op_resolver = delegateTestInterpreter::GenerateCustomOpResolver(customOp);
    }

    auto armnnDelegate = armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions);
    TfLiteInterpreterOptionsAddDelegate(options, armnnDelegate);

    m_TfLiteDelegate = armnnDelegate;
    m_TfLiteInterpreter = TfLiteInterpreterCreate(tfLiteModel, options);

    // The options and model can be deleted after the interpreter is created.
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(tfLiteModel);
}

void DelegateTestInterpreter::Cleanup()
{
    TfLiteInterpreterDelete(m_TfLiteInterpreter);

    if (m_TfLiteDelegate)
    {
        armnnDelegate::TfLiteArmnnDelegateDelete(static_cast<TfLiteDelegate*>(m_TfLiteDelegate));
    }
}

} // anonymous namespace