//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IExecutor.hpp"
#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"
#include "ExecuteNetworkProgramOptions.hpp"
#include "armnn/utility/NumericCast.hpp"
#include "armnn/utility/Timer.hpp"

#include <armnn_delegate.hpp>
#include <DelegateOptions.hpp>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

using namespace tflite;
class  TfLiteExecutor : public IExecutor
{
public:
    TfLiteExecutor(const ExecuteNetworkParams& m_Params, armnn::IRuntime::CreationOptions runtimeOptions);

    std::vector<const void*> Execute() override;
    void PrintNetworkInfo() override{};
    void CompareAndPrintResult(std::vector<const void*> otherOutput) override;

private:
    std::unique_ptr<tflite::FlatBufferModel> m_Model;
    const ExecuteNetworkParams& m_Params;
    std::unique_ptr<Interpreter> m_TfLiteInterpreter;
};

