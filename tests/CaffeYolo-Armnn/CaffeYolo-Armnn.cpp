//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "../YoloInferenceTest.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "armnn/TypesUtils.hpp"

int main(int argc, char* argv[])
{
    armnn::TensorShape inputTensorShape{ { 1, 3, YoloImageHeight, YoloImageWidth } };

    using YoloInferenceModel = InferenceModel<armnnCaffeParser::ICaffeParser,
        float>;

    int retVal = EXIT_FAILURE;
    try
    {
        // Coverity fix: InferenceTestMain() may throw uncaught exceptions.
        retVal = InferenceTestMain(argc, argv, { 0 },
            [&inputTensorShape]()
            {
                return make_unique<YoloTestCaseProvider<YoloInferenceModel>>(
                    [&]
                    (const InferenceTestOptions &commonOptions,
                     typename YoloInferenceModel::CommandLineOptions modelOptions)
                    {
                        if (!ValidateDirectory(modelOptions.m_ModelDir))
                        {
                            return std::unique_ptr<YoloInferenceModel>();
                        }

                        typename YoloInferenceModel::Params modelParams;
                        modelParams.m_ModelPath = modelOptions.m_ModelDir + "yolov1_tiny_voc2007_model.caffemodel";
                        modelParams.m_InputBindings = { "data" };
                        modelParams.m_OutputBindings = { "fc12" };
                        modelParams.m_InputShapes = { inputTensorShape };
                        modelParams.m_IsModelBinary = true;
                        modelParams.m_ComputeDevices = modelOptions.GetComputeDevicesAsBackendIds();
                        modelParams.m_VisualizePostOptimizationModel = modelOptions.m_VisualizePostOptimizationModel;
                        modelParams.m_EnableFp16TurboMode = modelOptions.m_EnableFp16TurboMode;

                        return std::make_unique<YoloInferenceModel>(modelParams,
                                                                    commonOptions.m_EnableProfiling,
                                                                    commonOptions.m_DynamicBackendsPath);
                });
            });
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: CaffeYolo-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
