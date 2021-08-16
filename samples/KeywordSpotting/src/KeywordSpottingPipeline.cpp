//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "KeywordSpottingPipeline.hpp"
#include "ArmnnNetworkExecutor.hpp"
#include "DsCNNPreprocessor.hpp"

namespace kws
{
KWSPipeline::KWSPipeline(std::unique_ptr<common::ArmnnNetworkExecutor<int8_t>> executor,
                         std::unique_ptr<Decoder> decoder,
                         std::unique_ptr<DsCNNPreprocessor> preProcessor
                         ) :
        m_executor(std::move(executor)),
        m_decoder(std::move(decoder)),
        m_preProcessor(std::move(preProcessor)) {}


std::vector<int8_t> KWSPipeline::PreProcessing(std::vector<float>& audio)
{
    return m_preProcessor->Invoke(audio.data(), audio.size(), m_executor->GetQuantizationOffset(),
                                  m_executor->GetQuantizationScale());
}

void KWSPipeline::Inference(const std::vector<int8_t>& preprocessedData, 
                            common::InferenceResults<int8_t>& result)
{
    m_executor->Run(preprocessedData.data(), preprocessedData.size(), result);
}

void KWSPipeline::PostProcessing(common::InferenceResults<int8_t>& inferenceResults,
                    std::map<int, std::string>& labels,
                    const std::function<void (int, std::string&, float)>& callback)
{
    std::pair<int,float> outputDecoder = this->m_decoder->decodeOutput(inferenceResults[0]);
    int keywordIndex = std::get<0>(outputDecoder);
    std::string output = labels[keywordIndex];
    callback(keywordIndex, output, std::get<1>(outputDecoder));
}

int KWSPipeline::getInputSamplesSize()
{
    return this->m_preProcessor->m_windowLen +
            ((this->m_preProcessor->m_mfcc->m_params.m_numMfccVectors - 1) * 
              this->m_preProcessor->m_windowStride);
}

IPipelinePtr CreatePipeline(common::PipelineOptions& config)
{
    if (config.m_ModelName == "DS_CNN_CLUSTERED_INT8") 
    {
        //DS-CNN model settings
        float SAMP_FREQ = 16000;
        int MFCC_WINDOW_LEN = 640;
        int MFCC_WINDOW_STRIDE = 320;
        int NUM_MFCC_FEATS = 10;
        int NUM_MFCC_VECTORS = 49;
        //todo: calc in pipeline and use in main
        int SAMPLES_PER_INFERENCE = NUM_MFCC_VECTORS * MFCC_WINDOW_STRIDE + 
                                    MFCC_WINDOW_LEN - MFCC_WINDOW_STRIDE; //16000
        float MEL_LO_FREQ = 20;
        float MEL_HI_FREQ = 4000;
        int NUM_FBANK_BIN = 40;

        MfccParams mfccParams(SAMP_FREQ,
                              NUM_FBANK_BIN,
                              MEL_LO_FREQ,
                              MEL_HI_FREQ,
                              NUM_MFCC_FEATS,
                              MFCC_WINDOW_LEN, false,
                              NUM_MFCC_VECTORS);

        std::unique_ptr<DsCnnMFCC> mfccInst = std::make_unique<DsCnnMFCC>(mfccParams);
        auto preprocessor = std::make_unique<kws::DsCNNPreprocessor>(
            MFCC_WINDOW_LEN, MFCC_WINDOW_STRIDE, std::move(mfccInst));

        auto executor = std::make_unique<common::ArmnnNetworkExecutor<int8_t>>(
            config.m_ModelFilePath, config.m_backends);

        auto decoder = std::make_unique<kws::Decoder>(executor->GetOutputQuantizationOffset(0),
                                                      executor->GetOutputQuantizationScale(0));

        return std::make_unique<kws::KWSPipeline>(std::move(executor), 
                                                  std::move(decoder), std::move(preprocessor));
    }  
    else 
    {
        throw std::invalid_argument("Unknown Model name: " + config.m_ModelName + " .");
    }
}

};// namespace kws