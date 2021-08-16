//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpeechRecognitionPipeline.hpp"
#include "ArmnnNetworkExecutor.hpp"

namespace asr 
{

ASRPipeline::ASRPipeline(std::unique_ptr<common::ArmnnNetworkExecutor<int8_t>> executor,
                         std::unique_ptr<Decoder> decoder, std::unique_ptr<Wav2LetterPreprocessor> preProcessor) :
        m_executor(std::move(executor)),
        m_decoder(std::move(decoder)), m_preProcessor(std::move(preProcessor)) {}

int ASRPipeline::getInputSamplesSize() 
{
    return this->m_preProcessor->m_windowLen +
           ((this->m_preProcessor->m_mfcc->m_params.m_numMfccVectors - 1) * this->m_preProcessor->m_windowStride);
}

int ASRPipeline::getSlidingWindowOffset()
{
    // Hardcoded for now until refactor
    return ASRPipeline::SLIDING_WINDOW_OFFSET;
}

std::vector<int8_t> ASRPipeline::PreProcessing(std::vector<float>& audio) 
{
    int audioDataToPreProcess = m_preProcessor->m_windowLen +
                                ((m_preProcessor->m_mfcc->m_params.m_numMfccVectors - 1) *
                                 m_preProcessor->m_windowStride);
    int outputBufferSize = m_preProcessor->m_mfcc->m_params.m_numMfccVectors
                           * m_preProcessor->m_mfcc->m_params.m_numMfccFeatures * 3;
    std::vector<int8_t> outputBuffer(outputBufferSize);
    m_preProcessor->Invoke(audio.data(), audioDataToPreProcess, outputBuffer, m_executor->GetQuantizationOffset(),
                           m_executor->GetQuantizationScale());
    return outputBuffer;
}

IPipelinePtr CreatePipeline(common::PipelineOptions& config, std::map<int, std::string>& labels) 
{
    if (config.m_ModelName == "Wav2Letter") 
    {
        // Wav2Letter ASR SETTINGS
        int SAMP_FREQ = 16000;
        int FRAME_LEN_MS = 32;
        int FRAME_LEN_SAMPLES = SAMP_FREQ * FRAME_LEN_MS * 0.001;
        int NUM_MFCC_FEATS = 13;
        int MFCC_WINDOW_LEN = 512;
        int MFCC_WINDOW_STRIDE = 160;
        const int NUM_MFCC_VECTORS = 296;
        int SAMPLES_PER_INFERENCE = MFCC_WINDOW_LEN + ((NUM_MFCC_VECTORS - 1) * MFCC_WINDOW_STRIDE);
        int MEL_LO_FREQ = 0;
        int MEL_HI_FREQ = 8000;
        int NUM_FBANK_BIN = 128;
        int INPUT_WINDOW_LEFT_CONTEXT = 98;
        int INPUT_WINDOW_RIGHT_CONTEXT = 98;
        int INPUT_WINDOW_INNER_CONTEXT = NUM_MFCC_VECTORS -
                                         (INPUT_WINDOW_LEFT_CONTEXT + INPUT_WINDOW_RIGHT_CONTEXT);
        int SLIDING_WINDOW_OFFSET = INPUT_WINDOW_INNER_CONTEXT * MFCC_WINDOW_STRIDE;


        MfccParams mfccParams(SAMP_FREQ, NUM_FBANK_BIN,
                              MEL_LO_FREQ, MEL_HI_FREQ, NUM_MFCC_FEATS, FRAME_LEN_SAMPLES, false, NUM_MFCC_VECTORS);

        std::unique_ptr<Wav2LetterMFCC> mfccInst = std::make_unique<Wav2LetterMFCC>(mfccParams);

        auto executor = std::make_unique<common::ArmnnNetworkExecutor<int8_t>>(config.m_ModelFilePath,
                                                                               config.m_backends);

        auto decoder = std::make_unique<asr::Decoder>(labels);

        auto preprocessor = std::make_unique<Wav2LetterPreprocessor>(MFCC_WINDOW_LEN, MFCC_WINDOW_STRIDE,
                                                                     std::move(mfccInst));

        auto ptr = std::make_unique<asr::ASRPipeline>(
                std::move(executor), std::move(decoder), std::move(preprocessor));

        ptr->SLIDING_WINDOW_OFFSET = SLIDING_WINDOW_OFFSET;

        return ptr;
    } 
    else
    {
        throw std::invalid_argument("Unknown Model name: " + config.m_ModelName + " .");
    }
}

}// namespace asr