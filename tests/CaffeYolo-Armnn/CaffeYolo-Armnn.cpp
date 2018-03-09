//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../YoloInferenceTest.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "armnn/TypesUtils.hpp"

int main(int argc, char* argv[])
{
    armnn::TensorShape inputTensorShape{ { 1, 3, YoloImageHeight, YoloImageWidth } };

    using YoloInferenceModel = InferenceModel<armnnCaffeParser::ICaffeParser,
        float>;

    return InferenceTestMain(argc, argv, { 0 },
        [&inputTensorShape]()
        {
            return make_unique<YoloTestCaseProvider<YoloInferenceModel>>(
                [&]
                (typename YoloInferenceModel::CommandLineOptions modelOptions)
                {
                    if (!ValidateDirectory(modelOptions.m_ModelDir))
                    {
                        return std::unique_ptr<YoloInferenceModel>();
                    }

                    typename YoloInferenceModel::Params modelParams;
                    modelParams.m_ModelPath = modelOptions.m_ModelDir + "yolov1_tiny_voc2007_model.caffemodel";
                    modelParams.m_InputBinding = "data";
                    modelParams.m_OutputBinding = "fc12";
                    modelParams.m_InputTensorShape = &inputTensorShape;
                    modelParams.m_IsModelBinary = true;
                    modelParams.m_ComputeDevice = modelOptions.m_ComputeDevice;

                    return std::make_unique<YoloInferenceModel>(modelParams);
            });
        });
}
