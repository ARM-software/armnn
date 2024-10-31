//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "../MobileNetSsdInferenceTest.hpp"

#include "armnnTfLiteParser/ITfLiteParser.hpp"

#include <algorithm>
#include <iterator>

using namespace armnnTfLiteParser;

int main(int argc, char* argv[])
{
    int retVal = EXIT_FAILURE;
    try
    {
        using DataType = float;
        using Parser   = armnnTfLiteParser::ITfLiteParser;
        using Model    = InferenceModel<Parser, DataType>;

        armnn::TensorShape inputTensorShape({ 1, 300, 300, 3  });

        std::vector<const char*> inputLayerNames  =
        {
            "normalized_input_image_tensor"
        };

        std::vector<const char*> outputLayerNames =
        {
            "TFLite_Detection_PostProcess",
            "TFLite_Detection_PostProcess:1",
            "TFLite_Detection_PostProcess:2",
            "TFLite_Detection_PostProcess:3"
        };

        retVal = InferenceTestMain(argc, argv, { 0 },
            [&inputTensorShape, inputLayerNames, outputLayerNames]()
            {
                return make_unique<MobileNetSsdTestCaseProvider<Model>>(
                    [&]
                    (const InferenceTestOptions& commonOptions,
                     typename Model::CommandLineOptions modelOptions)
                    {
                        if (!ValidateDirectory(modelOptions.m_ModelDir))
                        {
                            return std::unique_ptr<Model>();
                        }

                        typename Model::Params modelParams;
                        modelParams.m_ModelPath =
                            modelOptions.m_ModelDir + "ssd_mobilenet_v1.tflite";

                        std::copy(inputLayerNames.begin(), inputLayerNames.end(),
                                  std::back_inserter(modelParams.m_InputBindings));

                        std::copy(outputLayerNames.begin(), outputLayerNames.end(),
                                  std::back_inserter(modelParams.m_OutputBindings));

                        modelParams.m_InputShapes                    = { inputTensorShape };
                        modelParams.m_IsModelBinary                  = true;
                        modelParams.m_ComputeDevices                 = modelOptions.GetComputeDevicesAsBackendIds();
                        modelParams.m_VisualizePostOptimizationModel = modelOptions.m_VisualizePostOptimizationModel;
                        modelParams.m_EnableFp16TurboMode            = modelOptions.m_EnableFp16TurboMode;

                        return std::make_unique<Model>(modelParams,
                                                       commonOptions.m_EnableProfiling,
                                                       commonOptions.m_DynamicBackendsPath);
                });
            });
    }
    catch (const std::exception& e)
    {
        std::cerr << "WARNING: " << *argv << ": An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
