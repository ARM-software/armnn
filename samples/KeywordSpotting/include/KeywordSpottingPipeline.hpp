//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ArmnnNetworkExecutor.hpp"
#include "Decoder.hpp"
#include "MFCC.hpp"
#include "DsCNNPreprocessor.hpp"

namespace kws
{
/**
 * Generic Keyword Spotting pipeline with 3 steps: data pre-processing, inference execution and inference
 * result post-processing.
 *
 */
class KWSPipeline
{
public:

    /**
     * Creates speech recognition pipeline with given network executor and decoder.
     * @param executor - unique pointer to inference runner
     * @param decoder - unique pointer to inference results decoder
     */
    KWSPipeline(std::unique_ptr<common::ArmnnNetworkExecutor<int8_t>> executor,
                std::unique_ptr<Decoder> decoder,
                std::unique_ptr<DsCNNPreprocessor> preProcessor);

    /**
     * @brief Standard audio pre-processing implementation.
     *
     * Preprocesses and prepares the data for inference by
     * extracting the MFCC features.

     * @param[in] audio - the raw audio data
     */

    std::vector<int8_t> PreProcessing(std::vector<float>& audio);

    /**
     * @brief Executes inference
     *
     * Calls inference runner provided during instance construction.
     *
     * @param[in] preprocessedData - input inference data. Data type should be aligned with input tensor.
     * @param[out] result - raw inference results.
     */
    void Inference(const std::vector<int8_t>& preprocessedData, common::InferenceResults<int8_t>& result);

    /**
     * @brief Standard inference results post-processing implementation.
     *
     * Decodes inference results using decoder provided during construction.
     *
     * @param[in] inferenceResult - inference results to be decoded.
     * @param[in] labels - the words we use for the model
     */
    void PostProcessing(common::InferenceResults<int8_t>& inferenceResults,
                        std::map<int, std::string>& labels,
                        const std::function<void (int, std::string&, float)>& callback);

    /**
     * @brief Get the number of samples for the pipeline input

     * @return - number of samples for the pipeline
     */
    int getInputSamplesSize();

protected:
    std::unique_ptr<common::ArmnnNetworkExecutor<int8_t>> m_executor;
    std::unique_ptr<Decoder> m_decoder;
    std::unique_ptr<DsCNNPreprocessor> m_preProcessor;
};

using IPipelinePtr = std::unique_ptr<kws::KWSPipeline>;

/**
 * Constructs speech recognition pipeline based on configuration provided.
 *
 * @param[in] config - speech recognition pipeline configuration.
 * @param[in] labels - asr labels
 *
 * @return unique pointer to asr pipeline.
 */
IPipelinePtr CreatePipeline(common::PipelineOptions& config);

};// namespace kws