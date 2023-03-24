//
// Copyright © 2022, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Types.hpp"

#include "armnn/ArmNN.hpp"
#include <armnn/Logging.hpp>
#include <armnn_delegate.hpp>
#include <DelegateOptions.hpp>
#include <DelegateUtils.hpp>
#include <Profiling.hpp>
#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/kernels/builtin_op_kernels.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include <string>
#include <vector>

namespace common
{
/**
* @brief Used to load in a network through Tflite Interpreter,
*        register Armnn Delegate file to it, and run inference
*        on it against a given backend.
*        currently it is assumed that the input data will be
*        cv:MAT (Frame), the assumption is implemented in
*        PrepareTensors method, it can be generalized later
*
*/
template <typename Tout>
class ArmnnNetworkExecutor
{
private:
    std::unique_ptr<tflite::Interpreter> m_interpreter;
    std::unique_ptr<tflite::FlatBufferModel> m_model;
    Profiling m_profiling;

    void PrepareTensors(const void* inputData, const size_t dataBytes);

    template <typename Enumeration>
    auto log_as_int(Enumeration value)
    -> typename std::underlying_type<Enumeration>::type
    {
        return static_cast<typename std::underlying_type<Enumeration>::type>(value);
    }

public:
    ArmnnNetworkExecutor() = delete;

    /**
    * @brief Initializes the network with the given input data.
    *
    *
    *       * @param[in] modelPath - Relative path to the model file
    *       * @param[in] backends - The list of preferred backends to run inference on
    */
    ArmnnNetworkExecutor(std::string& modelPath,
                         std::vector<armnn::BackendId>& backends,
                         bool isProfilingEnabled = false);

    /**
    * @brief Returns the aspect ratio of the associated model in the order of width, height.
    */
    Size GetImageAspectRatio();

    /**
    * @brief Returns the data type of the associated model.
    */
    armnn::DataType GetInputDataType() const;

    float GetQuantizationScale();

    int GetQuantizationOffset();

    float GetOutputQuantizationScale(int tensorIndex);

    int GetOutputQuantizationOffset(int tensorIndex);


    /**
    * @brief Runs inference on the provided input data, and stores the results
    * in the provided InferenceResults object.
    *
    * @param[in] inputData - input frame data
    * @param[in] dataBytes - input data size in bytes
    * @param[out] outResults - Vector of DetectionResult objects used to store the output result.
    */
    bool Run(const void *inputData, const size_t dataBytes,
             InferenceResults<Tout> &outResults);
};

template <typename Tout>
ArmnnNetworkExecutor<Tout>::ArmnnNetworkExecutor(std::string& modelPath,
                                           std::vector<armnn::BackendId>& preferredBackends,
                                           bool isProfilingEnabled):
                                           m_profiling(isProfilingEnabled)
{
    m_profiling.ProfilingStart();
    armnn::OptimizerOptionsOpaque optimizerOptions;
    m_model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (m_model == nullptr)
    {
        const std::string errorMessage{"ArmnnNetworkExecutor: Failed to build the model"};
        ARMNN_LOG(error) << errorMessage;
        throw armnn::Exception(errorMessage);
    }
    m_profiling.ProfilingStopAndPrintUs("Loading the model took");

    m_profiling.ProfilingStart();
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter);
    if (m_interpreter->AllocateTensors() != kTfLiteOk)
    {
        const std::string errorMessage{"ArmnnNetworkExecutor: Failed to alloc tensors"};
        ARMNN_LOG(error) << errorMessage;
        throw armnn::Exception(errorMessage);
    }
    m_profiling.ProfilingStopAndPrintUs("Create the tflite interpreter");

    /* create delegate options */
    m_profiling.ProfilingStart();

    /* enable fast math optimization */
    armnn::BackendOptions modelOptionGpu("GpuAcc", {{"FastMathEnabled", true}});
    optimizerOptions.AddModelOption(modelOptionGpu);

    armnn::BackendOptions modelOptionCpu("CpuAcc", {{"FastMathEnabled", true}});
    optimizerOptions.AddModelOption(modelOptionCpu);
    /* enable reduce float32 to float16 optimization */
    optimizerOptions.SetReduceFp32ToFp16(true);

    armnnDelegate::DelegateOptions delegateOptions(preferredBackends, optimizerOptions);

    /* create delegate object */
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                 armnnDelegate::TfLiteArmnnDelegateDelete);

    /* Register the delegate file */
    m_interpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
    m_profiling.ProfilingStopAndPrintUs("Create and load ArmNN Delegate");

}

template<typename Tout>
void ArmnnNetworkExecutor<Tout>::PrepareTensors(const void *inputData, const size_t dataBytes)
{
    size_t inputTensorSize = m_interpreter->input_tensor(0)->bytes;
    auto * inputTensorPtr = m_interpreter->input_tensor(0)->data.raw;
    assert(inputTensorSize >= dataBytes);
    if (inputTensorPtr != nullptr)
    {
       memcpy(inputTensorPtr, inputData, inputTensorSize);
    }
    else
    {
        const std::string errorMessage{"ArmnnNetworkExecutor: input tensor is null"};
        ARMNN_LOG(error) << errorMessage;
        throw armnn::Exception(errorMessage);
    }

}

template <typename Tout>
bool ArmnnNetworkExecutor<Tout>::Run(const void *inputData, const size_t dataBytes,
                                             InferenceResults<Tout>& outResults)
{
    bool ret = false;
    m_profiling.ProfilingStart();
    PrepareTensors(inputData, dataBytes);

    if (m_interpreter->Invoke() == kTfLiteOk)
    {


        ret = true;
        // Extract the output tensor data.
        outResults.clear();
        outResults.reserve(m_interpreter->outputs().size());
        for (int index = 0; index < m_interpreter->outputs().size(); index++)
        {
            size_t size = m_interpreter->output_tensor(index)->bytes / sizeof(Tout);
            const Tout *p_Output = m_interpreter->typed_output_tensor<Tout>(index);
            if (p_Output != nullptr) {
                InferenceResult<float> outRes(p_Output, p_Output + size);
                outResults.emplace_back(outRes);
            }
            else
            {
                const std::string errorMessage{"ArmnnNetworkExecutor: p_Output tensor is null"};
                ARMNN_LOG(error) << errorMessage;
                ret = false;
            }
        }
    }
    else
    {
        const std::string errorMessage{"ArmnnNetworkExecutor: Invoke has failed"};
        ARMNN_LOG(error) << errorMessage;
    }
    m_profiling.ProfilingStopAndPrintUs("Perform inference");
    return ret;
}

template <typename Tout>
Size ArmnnNetworkExecutor<Tout>::GetImageAspectRatio()
{
    assert(m_interpreter->tensor(m_interpreter->inputs()[0])->dims->size == 4);
    return Size(m_interpreter->tensor(m_interpreter->inputs()[0])->dims->data[2],
                m_interpreter->tensor(m_interpreter->inputs()[0])->dims->data[1]);
}

template <typename Tout>
armnn::DataType ArmnnNetworkExecutor<Tout>::GetInputDataType() const
{
    return GetDataType(*(m_interpreter->tensor(m_interpreter->inputs()[0])));
}

template <typename Tout>
float ArmnnNetworkExecutor<Tout>::GetQuantizationScale()
{
    return m_interpreter->tensor(m_interpreter->inputs()[0])->params.scale;
}

template <typename Tout>
int ArmnnNetworkExecutor<Tout>::GetQuantizationOffset()
{
    return m_interpreter->tensor(m_interpreter->inputs()[0])->params.zero_point;
}

template <typename Tout>
float ArmnnNetworkExecutor<Tout>::GetOutputQuantizationScale(int tensorIndex)
{
    assert(m_interpreter->outputs().size() > tensorIndex);
    return m_interpreter->tensor(m_interpreter->outputs()[tensorIndex])->params.scale;
}

template <typename Tout>
int ArmnnNetworkExecutor<Tout>::GetOutputQuantizationOffset(int tensorIndex)
{
    assert(m_interpreter->outputs().size() > tensorIndex);
    return m_interpreter->tensor(m_interpreter->outputs()[tensorIndex])->params.zero_point;
}

}// namespace common