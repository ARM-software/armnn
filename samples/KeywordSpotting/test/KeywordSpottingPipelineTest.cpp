//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <catch.hpp>
#include <map>
#include <cinttypes>
#include "KeywordSpottingPipeline.hpp"
#include "DsCNNPreprocessor.hpp"

static std::string GetResourceFilePath(const std::string& filename)
{
    std::string testResources = TEST_RESOURCE_DIR;
    if (testResources.empty())
    {
        throw std::invalid_argument("Invalid test resources directory provided");
    }
    else
    {
        if(testResources.back() != '/')
        {
            return testResources + "/" + filename;
        }
        else
        {
            return testResources + filename;
        }
    }
}

TEST_CASE("Test Keyword spotting pipeline")
{
    const int8_t ifm0_kws [] = 
    {
    -0x1b, 0x4f, 0x7a, -0x55, 0x6, -0x11, 0x6e, -0x6, 0x67, -0x7e, -0xd, 0x6, 0x49, 0x79, -0x1e, 0xe, 
     0x1d, 0x6e, 0x6f, 0x6f, -0x2e, -0x4b, 0x2, -0x3e, 0x40, -0x4b, -0x7, 0x31, -0x38, -0x64, -0x28, 
     0xc, -0x1d, 0xf, 0x1c, 0x5a, -0x4b, 0x56, 0x7e, 0x9, -0x29, 0x13, -0x65, -0xa, 0x34, -0x59, 0x41, 
    -0x6f, 0x75, 0x67, -0x5f, 0x17, 0x4a, -0x76, -0x7a, 0x49, -0x19, -0x41, 0x78, 0x40, 0x44, 0xe, 
    -0x51, -0x5c, 0x3d, 0x24, 0x76, -0x66, -0x11, 0x5e, 0x7b, -0x4, 0x7a, 0x9, 0x13, 0x8, -0x21, -0x11, 
     0x13, 0x7a, 0x25, 0x6, -0x68, 0x6a, -0x30, -0x16, -0x43, -0x27, 0x4c, 0x6b, -0x14, -0x12, -0x5f, 
     0x49, -0x2a, 0x44, 0x57, -0x78, -0x72, 0x62, -0x8, -0x38, -0x73, -0x2, -0x80, 0x79, -0x3f, 0x57, 
     0x9, -0x7e, -0x34, -0x59, 0x19, -0x66, 0x58, -0x3b, -0x69, -0x1a, 0x13, -0x2f, -0x2f, 0x13, 0x35, 
    -0x30, 0x1e, 0x3b, -0x71, 0x67, 0x7d, -0x5d, 0x1a, 0x69, -0x53, -0x38, -0xf, 0x76, 0x2, 0x7e, 0x45, 
    -0xa, 0x59, -0x6b, -0x28, -0x5d, -0x63, -0x7d, -0x3, 0x48, 0x74, -0x75, -0x7a, 0x1f, -0x53, 0x5b, 
     0x4d, -0x18, -0x4a, 0x39, -0x52, 0x5a, -0x6b, -0x41, -0x3e, -0x61, -0x80, -0x52, 0x67, 0x71, -0x47, 
     0x79, -0x41, 0x3a, -0x8, -0x1f, 0x4d, -0x7, 0x5b, 0x6b, -0x1b, -0x8, -0x20, -0x21, 0x7c, -0x74, 
     0x25, -0x68, -0xe, -0x7e, -0x45, -0x28, 0x45, -0x1a, -0x39, 0x78, 0x11, 0x48, -0x6b, -0x7b, -0x43, 
    -0x21, 0x38, 0x46, 0x7c, -0x5d, 0x59, 0x53, -0x3f, -0x15, 0x59, -0x17, 0x75, 0x2f, 0x7c, 0x68, 0x6a, 
     0x0, -0x10, 0x5b, 0x61, 0x36, -0x41, 0x33, 0x23, -0x80, -0x1d, -0xb, -0x56, 0x2d, 0x68, -0x68, 
     0x2f, 0x48, -0x5d, -0x44, 0x64, -0x27, 0x68, -0x13, 0x39, -0x3f, 0x18, 0x31, 0x15, -0x78, -0x2, 
     0x72, 0x60, 0x59, -0x30, -0x22, 0x73, 0x61, 0x76, -0x4, -0x62, -0x64, -0x80, -0x32, -0x16, 0x51,
    -0x2, -0x70, 0x71, 0x3f, -0x5f, -0x35, -0x3c, 0x79, 0x48, 0x61, 0x5b, -0x20, -0x1e, -0x68, -0x1c, 
     0x6c, 0x3a, 0x28, -0x36, -0x3e, 0x5f, -0x75, -0x73, 0x1e, 0x75, -0x66, -0x22, 0x20, -0x64, 0x67, 
     0x36, 0x14, 0x37, -0xa, -0xe, 0x8, -0x37, -0x43, 0x21, -0x8, 0x54, 0x1, 0x34, -0x2c, -0x73, -0x11, 
    -0x48, -0x1c, -0x40, 0x14, 0x4e, -0x53, 0x25, 0x5e, 0x14, 0x4f, 0x7c, 0x6d, -0x61, -0x38, 0x35, 
    -0x5a, -0x44, 0x12, 0x52, -0x60, 0x22, -0x1c, -0x8, -0x4, -0x6b, -0x71, 0x43, 0xb, 0x7b, -0x7, 
    -0x3c, -0x3b, -0x40, -0xd, 0x44, 0x6, 0x30, 0x38, 0x57, 0x1f, -0x7, 0x2, 0x4f, 0x64, 0x7c, -0x3,
    -0x13, -0x71, -0x45, -0x53, -0x52, 0x2b, -0x11, -0x1d, -0x2, -0x29, -0x37, 0x3d, 0x19, 0x76, 0x18,
     0x1d, 0x12, -0x29, -0x5e, -0x54, -0x48, 0x5d, -0x41, -0x3f, 0x7e, -0x2a, 0x41, 0x57, -0x65, -0x15, 
     0x12, 0x1f, -0x57, 0x79, -0x64, 0x3a, -0x2f, 0x7f, -0x6c, 0xa, 0x52, -0x1f, -0x41, 0x6e, -0x4b, 
     0x3d, -0x1b, -0x42, 0x22, -0x3c, -0x35, -0xf, 0xc, 0x32, -0x15, -0x68, -0x21, 0x0, -0x16, 0x14,
    -0x10, -0x5b, 0x2f, 0x21, 0x41, -0x8, -0x12, -0xa, 0x10, 0xf, 0x7e, -0x76, -0x1d, 0x2b, -0x49, 
     0x42, -0x25, -0x78, -0x69, -0x2c, 0x3f, 0xc, 0x52, 0x6d, 0x2e, -0x13, 0x76, 0x37, -0x36, -0x51,
    -0x5, -0x63, -0x4f, 0x1c, 0x6b, -0x4b, 0x71, -0x12, 0x72, -0x3f,-0x4a, 0xf, 0x3a, -0xd, 0x38, 0x3b,
    -0x5d, 0x75, -0x43, -0x10, -0xa, -0x7a, 0x1a, -0x44, 0x1c, 0x6a, 0x43, -0x1b, -0x35, 0x7d, -0x2c,
    -0x10, 0x5b, -0x42, -0x4f, 0x69, 0x1f, 0x1b, -0x64, -0x21, 0x19, -0x5d, 0x2e, -0x2a, -0x65, -0x13,
    -0x70, -0x6e
    };

    const int8_t ofm0_kws [] = 
    {
    -0x80, 0x7f, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80
    };

    // First 640 samples from yes.wav.
    std::vector<int16_t> testWav = std::vector<int16_t>
    {
    139, 143, 164, 163, 157, 156, 151, 148, 172, 171,
    165, 169, 149, 142, 145, 147, 166, 146, 112, 132,
    132, 136, 165, 176, 176, 152, 138, 158, 179, 185,
    183, 148, 121, 130, 167, 204, 163, 132, 165, 184,
    193, 205, 210, 204, 195, 178, 168, 197, 207, 201,
    197, 177, 185, 196, 191, 198, 196, 183, 193, 181,
    157, 170, 167, 159, 164, 152, 146, 167, 180, 171,
    194, 232, 204, 173, 171, 172, 184, 169, 175, 199,
    200, 195, 185, 214, 214, 193, 196, 191, 204, 191,
    172, 187, 183, 192, 203, 172, 182, 228, 232, 205,
    177, 174, 191, 210, 210, 211, 197, 177, 198, 217,
    233, 236, 203, 191, 169, 145, 149, 161, 198, 206,
    176, 137, 142, 181, 200, 215, 201, 188, 166, 162,
    184, 155, 135, 132, 126, 142, 169, 184, 172, 156,
    132, 119, 150, 147, 154, 160, 125, 130, 137, 154,
    161, 168, 195, 182, 160, 134, 138, 146, 130, 120,
    101, 122, 137, 118, 117, 131, 145, 140, 146, 148,
    148, 168, 159, 134, 114, 114, 130, 147, 147, 134,
    125, 98, 107, 127, 99, 79, 84, 107, 117, 114,
    93, 92, 127, 112, 109, 110, 96, 118, 97, 87,
    110, 95, 128, 153, 147, 165, 146, 106, 101, 137,
    139, 96, 73, 90, 91, 51, 69, 102, 100, 103,
    96, 101, 123, 107, 82, 89, 118, 127, 99, 100,
    111, 97, 111, 123, 106, 121, 133, 103, 100, 88,
    85, 111, 114, 125, 102, 91, 97, 84, 139, 157,
    109, 66, 72, 129, 111, 90, 127, 126, 101, 109,
    142, 138, 129, 159, 140, 80, 74, 78, 76, 98,
    68, 42, 106, 143, 112, 102, 115, 114, 82, 75,
    92, 80, 110, 114, 66, 86, 119, 101, 101, 103,
    118, 145, 85, 40, 62, 88, 95, 87, 73, 64,
    86, 71, 71, 105, 80, 73, 96, 92, 85, 90,
    81, 86, 105, 100, 89, 78, 102, 114, 95, 98,
    69, 70, 108, 112, 111, 90, 104, 137, 143, 160,
    145, 121, 98, 86, 91, 87, 115, 123, 109, 99,
    85, 120, 131, 116, 125, 144, 153, 111, 98, 110,
    93, 89, 101, 137, 155, 142, 108, 94, 136, 145,
    129, 129, 122, 109, 90, 76, 81, 110, 119, 96,
    95, 102, 105, 111, 90, 89, 111, 115, 86, 51,
    107, 140, 105, 105, 110, 142, 125, 76, 75, 69,
    65, 52, 61, 69, 55, 42, 47, 58, 37, 35,
    24, 20, 44, 22, 16, 26, 6, 3, 4, 23,
    60, 51, 30, 12, 24, 31, -9, -16, -13, 13,
    19, 9, 37, 55, 70, 36, 23, 57, 45, 33,
    50, 59, 18, 11, 62, 74, 52, 8, -3, 26,
    51, 48, -5, -9, 12, -7, -12, -5, 28, 41,
    -2, -30, -13, 31, 33, -12, -22, -8, -15, -17,
    2, -6, -25, -27, -24, -8, 4, -9, -52, -47,
    -9, -32, -45, -5, 41, 15, -32, -14, 2, -1,
    -10, -30, -32, -25, -21, -17, -14, 8, -4, -13,
    34, 18, -36, -38, -18, -19, -28, -17, -14, -16,
    -2, -20, -27, 12, 11, -17, -33, -12, -22, -64,
    -42, -26, -23, -22, -37, -51, -53, -30, -18, -48,
    -69, -38, -54, -96, -72, -49, -50, -57, -41, -22,
    -43, -64, -54, -23, -49, -69, -41, -44, -42, -49,
    -40, -26, -54, -50, -38, -49, -70, -94, -89, -69,
    -56, -65, -71, -47, -39, -49, -79, -91, -56, -46,
    -62, -86, -64, -32, -47, -50, -71, -77, -65, -68,
    -52, -51, -61, -67, -61, -81, -93, -52, -59, -62,
    -51, -75, -76, -50, -32, -54, -68, -70, -43, 1,
    -42, -92, -80, -41, -38, -79, -69, -49, -82, -122,
    -93, -21, -24, -61, -70, -73, -62, -74, -69, -43,
    -25, -15, -43, -23, -26, -69, -44, -12, 1, -51,
    -78, -13, 3, -53, -105, -72, -24, -62, -66, -31,
    -40, -65, -86, -64, -44, -55, -63, -61, -37, -41,
    };

    // Golden audio ops mfcc output for the above wav.
    const std::vector<float> testWavMfcc 
    {
    -22.67135, -0.61615, 2.07233, 0.58137, 1.01655, 0.85816, 0.46039, 0.03393, 1.16511, 0.0072,
    };

    std::vector<float> testWavFloat(640);
    constexpr float normaliser = 1.0/(1u<<15u);
    std::transform(testWav.begin(), testWav.end(), testWavFloat.begin(),
                   std::bind1st(std::multiplies<float>(), normaliser));

    const float DsCNNInputQuantizationScale = 1.107164;
    const int DsCNNInputQuantizationOffset = 95;

    std::map<int,std::string> labels = 
    {
        {0,"silence"},
        {1, "unknown"},
        { 2, "yes"},
        { 3,"no"},
        { 4, "up"},
        { 5, "down"},
        { 6, "left"},
        { 7, "right"},
        { 8, "on"},
        { 9, "off"},
        { 10, "stop"},
        {11, "go"}
    };
    common::PipelineOptions options;
    options.m_ModelFilePath = GetResourceFilePath("ds_cnn_clustered_int8.tflite");
    options.m_ModelName = "DS_CNN_CLUSTERED_INT8";
    options.m_backends = {"CpuAcc", "CpuRef"};
    kws::IPipelinePtr kwsPipeline = kws::CreatePipeline(options);

    CHECK(kwsPipeline->getInputSamplesSize() == 16000);
    std::vector<int8_t> expectedWavMfcc;
    for(auto& i : testWavMfcc)
    {
        expectedWavMfcc.push_back( 
            (i + DsCNNInputQuantizationScale * DsCNNInputQuantizationOffset) / DsCNNInputQuantizationScale);
    }

    SECTION("Pre-processing")
    {
        testWavFloat.resize(16000);
        expectedWavMfcc.resize(49 * 10);
        std::vector<int8_t> preprocessedData = kwsPipeline->PreProcessing(testWavFloat);
        CHECK(preprocessedData.size() == expectedWavMfcc.size());
        for(int i = 0; i < 10; ++i)
        {
            CHECK(expectedWavMfcc[i] == Approx(preprocessedData[i]).margin(1));
        }
    }

    SECTION("Execute inference")
    {
        common::InferenceResults<int8_t> result;
        std::vector<int8_t> IFM(std::begin(ifm0_kws), std::end(ifm0_kws));
        kwsPipeline->Inference(IFM, result);
        std::vector<int8_t> OFM(std::begin(ofm0_kws), std::end(ofm0_kws));

        CHECK(1 == result.size());
        CHECK(OFM.size() == result[0].size());

        int count = 0;
        for (auto& i : result)
        {
            for (signed char& j : i)
            {
                CHECK(j == OFM[count++]);

            }
        }
    }

    SECTION("Convert inference result to keyword")
    {
        std::vector< std::vector< int8_t >> modelOutput = {{1, 4, 2, 3, 1, 1, 3, 1, 43, 1, 6, 1}};
        kwsPipeline->PostProcessing(modelOutput, labels,
                                    [](int index, std::string& label, float prob) -> void {
                                        CHECK(index == 8);
                                        CHECK(label == "on");
                                    });
    }
}
