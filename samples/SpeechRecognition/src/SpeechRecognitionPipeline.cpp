//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpeechRecognitionPipeline.hpp"
#include "ArmnnNetworkExecutor.hpp"

namespace asr
{
ASRPipeline::ASRPipeline(std::unique_ptr<common::ArmnnNetworkExecutor<int8_t>> executor,
                         std::unique_ptr<Decoder> decoder
                         ) :
        m_executor(std::move(executor)),
        m_decoder(std::move(decoder)){}

IPipelinePtr CreatePipeline(common::PipelineOptions& config, std::map<int, std::string>& labels)
{
    auto executor = std::make_unique<common::ArmnnNetworkExecutor<int8_t>>(config.m_ModelFilePath, config.m_backends);

    auto decoder = std::make_unique<asr::Decoder>(labels);

    return std::make_unique<asr::ASRPipeline>(std::move(executor), std::move(decoder));
}

}// namespace asr