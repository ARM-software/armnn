//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <DelegateTestInterpreter.hpp>

#include <armnn_delegate.hpp>

#include <armnn/utility/IgnoreUnused.hpp>

namespace delegateTestInterpreter
{

DelegateTestInterpreter::DelegateTestInterpreter(std::vector<char>& modelBuffer,
                                                 const std::vector<armnn::BackendId>& backends,
                                                 const std::string& customOp,
                                                 bool disableFallback)
{
    armnn::IgnoreUnused(backends);
    armnn::IgnoreUnused(disableFallback);

    TfLiteModel* tfLiteModel = delegateTestInterpreter::CreateTfLiteModel(modelBuffer);

    TfLiteInterpreterOptions* options = delegateTestInterpreter::CreateTfLiteInterpreterOptions();
    if (!customOp.empty())
    {
        options->mutable_op_resolver = delegateTestInterpreter::GenerateCustomOpResolver(customOp);
    }

    // Use default settings until options have been enabled.
    auto armnnDelegate = armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(nullptr);
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
    armnn::IgnoreUnused(delegateOptions);

    TfLiteModel* tfLiteModel = delegateTestInterpreter::CreateTfLiteModel(modelBuffer);

    TfLiteInterpreterOptions* options = delegateTestInterpreter::CreateTfLiteInterpreterOptions();
    if (!customOp.empty())
    {
        options->mutable_op_resolver = delegateTestInterpreter::GenerateCustomOpResolver(customOp);
    }

    // Use default settings until options have been enabled.
    auto armnnDelegate = armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(nullptr);
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
        armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete(static_cast<TfLiteOpaqueDelegate*>(m_TfLiteDelegate));
    }
}

} // anonymous namespace