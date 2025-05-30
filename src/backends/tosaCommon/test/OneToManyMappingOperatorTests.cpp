//
// Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

// This file contains tests for validating the creation and configuration of
// TOSA rescale operators. It includes tests for error scenarios such as null
// operator pointers, empty vectors, and mismatched vector sizes, as well as
// tests for successful operator creation and attribute validation.


#include "CommonTestUtils.hpp"
#include "TosaTestUtils.hpp"


using namespace tosa;
using namespace armnn;


TEST_SUITE("TosaOperatorMappingOneToManyTests")
{
    // Test case for CreateRawRescaleTosaOperator function
    // This test checks various scenarios including null pointers, empty vectors,
    // mismatched sizes, and valid operator creation.
    TEST_CASE("TosaOperatorMappingOneToManyTests_CreateRawRescaleTosaOperator")
    {
        // Define input parameters for the operator creation
        // These parameters will be used in the test cases below.
        // Note: The values here are arbitrary and can be adjusted as needed for the tests.

        const std::string inputName  = "input";
        const std::string outputName = "output";
        bool input_unsigned          = true;
        bool output_unsigned         = true;
        bool double_round            = false;
        bool scale32                 = true;

        SUBCASE("Null op pointer")
        {
            // This test case checks the behavior when the operator pointer is null.
            // It should throw an exception indicating that the operator pointer is null.

            std::vector<int32_t> multipliers = { 1 };
            std::vector<int32_t> shifts      = { 1 };
            TosaSerializationOperator** opPtr = nullptr;

            // Expect an exception when opPtr is null.
            CHECK_THROWS_WITH(CreateRawRescaleTosaOperator(inputName, outputName, multipliers,
                                                           shifts, 0, 0,
                                                           input_unsigned, output_unsigned, double_round,
                                                           scale32, false, opPtr),
                              "CreateRawRescaleTosaOperator: nullptr op.");
        }
        SUBCASE("Empty multipliers vector")
        {
            // This test case checks the behavior when the multipliers vector is empty.
            // It should throw an exception indicating that the multipliers vector is empty.

            TosaSerializationOperator* op = nullptr;
            std::vector<int32_t> multipliers;
            std::vector<int32_t> shifts;

            // Expect an exception when multipliers vector is empty.
            CHECK_THROWS_WITH(CreateRawRescaleTosaOperator(inputName, outputName, multipliers,
                                                           shifts, 0, 0,
                                                           input_unsigned, output_unsigned, double_round,
                                                           scale32, false, &op),
                              "CreateRawRescaleTosaOperator: multipliers is empty.");
        }
        SUBCASE("Mismatched vector sizes")
        {
            // This test case checks the behavior when the multipliers and shifts vectors have different sizes.
            // It should throw an exception indicating that the sizes do not match.

            TosaSerializationOperator* op = nullptr;
            std::vector<int32_t> multipliers = { 1, 2 };
            std::vector<int32_t> shifts = { 1 };

            // Expect an exception when multipliers and shifts vectors have different sizes.
            CHECK_THROWS_WITH(CreateRawRescaleTosaOperator(inputName, outputName, multipliers,
                                                           shifts, 0, 0,
                                                           input_unsigned, output_unsigned, double_round,
                                                           scale32, false, &op),
                              "CreateRawRescaleTosaOperator: multipliers and shift not same size.");
        }
        SUBCASE("Multipliers size 1 with per_channel true")
        {
            // This test case checks the behavior when the multipliers vector has a single element
            // and per_channel is set to true. It should throw an exception indicating that multipliers must be greater
            // than 1.

            TosaSerializationOperator* op = nullptr;
            std::vector<int32_t> multipliers = { 1 };
            std::vector<int32_t> shifts = { 1 };

            // per_channel true should trigger exception for a single multiplier.
            CHECK_THROWS_WITH(CreateRawRescaleTosaOperator(inputName, outputName, multipliers,
                                                           shifts, 0, 0,
                                                           input_unsigned, output_unsigned, double_round,
                                                           scale32, true, &op),
                                "CreateRawRescaleTosaOperator: \
                                multipliers must be greater than 1 if per_channel is true.");
        }

        SUBCASE("Multipliers size > 1 with per_channel false")
        {
            // This test case checks the behavior when the multipliers vector has more than one element
            // and per_channel is set to false. It should throw an exception indicating that multipliers size must be 1.
            // This is a common mistake that should be caught by the function.

            TosaSerializationOperator* op = nullptr;
            std::vector<int32_t> multipliers = { 1, 2 };
            std::vector<int32_t> shifts = { 1, 2 };

            // per_channel false should allow only a single multiplier.
            CHECK_THROWS_WITH(CreateRawRescaleTosaOperator(inputName, outputName, multipliers,
                                                           shifts, 0, 0,
                                                           input_unsigned, output_unsigned, double_round,
                                                           scale32, false, &op),
                                "CreateRawRescaleTosaOperator: \
                                multipliers size must be 1 if per_channel is false.");
        }

        SUBCASE("Valid operator creation")
        {
            // This test case checks the behavior when all parameters are valid.
            // It should create a TosaSerializationOperator without throwing any exceptions.
            // The operator should be created successfully and not be null.

            TosaSerializationOperator* op = nullptr;
            std::vector<int32_t> multipliers = { 2 };
            std::vector<int32_t> shifts = { 3 };

            // Valid call should not throw.
            CreateRawRescaleTosaOperator(inputName, outputName, multipliers,
                                         shifts, 1, 2,
                                         false, false, true,
                                         false, false, &op);

            // Ensure the operator is created successfully
            REQUIRE(op != nullptr);

            // Clean up
            delete op;
        }
        SUBCASE("Validate attributes")
        {
            // This test case checks the attributes of the created operator.
            // It should validate that the attributes match the expected values.

            TosaSerializationOperator* op = nullptr;
            std::vector<int32_t> multipliers = { 2 };
            std::vector<int32_t> shifts = { 3 };

            // Valid call should not throw.
            CreateRawRescaleTosaOperator(inputName, outputName, multipliers,
                                         shifts, 1, 2,
                                         false, false, true,
                                         false, false, &op);

            // Ensure the operator is created successfully
            const TosaAttributeBase* attribute = op->GetAttribute();
            REQUIRE(attribute != nullptr);

            // Check if the attribute is of type TosaRescaleAttribute
            auto rescaleAttr = dynamic_cast<const TosaRescaleAttribute*>(attribute);
            REQUIRE(rescaleAttr != nullptr);

            // Validate the attributes of the rescale operator
            CHECK(rescaleAttr->input_zp()        == 1);
            CHECK(rescaleAttr->output_zp()       == 2);
            CHECK(rescaleAttr->multiplier()      == multipliers);
            CHECK(rescaleAttr->shift()           == shifts);
            CHECK(rescaleAttr->input_unsigned()  == false);
            CHECK(rescaleAttr->output_unsigned() == false);
            CHECK(rescaleAttr->double_round()    == true);
            CHECK(rescaleAttr->scale32()         == false);
            CHECK(rescaleAttr->per_channel()     == false);

            // Clean up
            delete op;

        }
    }

    TEST_CASE("ComputeMultiplierAndShiftTosaScale32")
    {
        // This test case checks the behavior of the ComputeMultiplierAndShiftTosaScale32 function
        SUBCASE("Mismatched vector sizes")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with a scale of 1.0

            int32_t multiplier = 0, shift = 0;
            double scale = 1.0;

            // Compute the multiplier and shift for the given scale
            ComputeMultiplierAndShiftTosaScale32(scale, multiplier, shift);

            // For scale 1.0, frexp returns mantissa = 0.5 and exponent = 1.
            // Then shiftedM = round(0.5 * (1 << 31)) = 1073741824.
            // After computing shift = (-1) + 31, we expect shift = 30.
            CHECK(multiplier == 1073741824);
            CHECK(shift == 30);
        }
        SUBCASE("Zero scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with a scale of 0.0

            int32_t multiplier = 0, shift = 0;
            double scale = 0.0;

            // Compute the multiplier and shift for the given scale
            ComputeMultiplierAndShiftTosaScale32(scale, multiplier, shift);

            // For scale 0.0, frexp returns mantissa = 0.0 and exponent = 0.
            // Then shiftedM = round(0.0 * (1 << 31)) = 0.
            // After computing shift = (-0) + 31, we expect shift = 31.
            CHECK(multiplier == 0);
            CHECK(shift == 31);
        }
        SUBCASE("Negative scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with a negative scale

            int32_t multiplier = 0, shift = 0;
            double scale = -1.0;

            // Compute the multiplier and shift for the given scale
            ComputeMultiplierAndShiftTosaScale32(scale, multiplier, shift);

            // For scale -1.0, frexp returns mantissa = -0.5 and exponent = 1.
            // Then shiftedM = round(-0.5 * (1 << 31)) = -1073741824.
            // After computing shift = (-1) + 31, we expect shift = 30.
            CHECK(multiplier == -1073741824);
            CHECK(shift == 30);
        }
        SUBCASE("Subnormal scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with a subnormal scale

            int32_t multiplier = 0, shift = 0;
            double scale = 1e-10;

            // Compute the multiplier and shift for the given scale
            ComputeMultiplierAndShiftTosaScale32(scale, multiplier, shift);

            // For scale 1e-10, frexp returns mantissa = 0.5 and exponent = -33.
            // Then shiftedM = round(0.5 * (1 << 31)) = 1073741824.
            // After computing shift = (-(-33)) + 31, then we clamp to 47, so we expect shift = 47.
            CHECK(multiplier == 14073);
            CHECK(shift == 47);
        }
        SUBCASE("Large scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with a large scale

            int32_t multiplier = 0, shift = 0;
            double scale = 1e10;

            // Compute the multiplier and shift for the given scale
            ComputeMultiplierAndShiftTosaScale32(scale, multiplier, shift);

            // For scale 1e10, frexp returns mantissa = 0.5 and exponent = 34.
            // Then shiftedM = round(0.5 * (1 << 31)) = 1250000000.
            // After computing shift = (-34) + 31, we expect shift = -3.
            CHECK(multiplier == 1250000000);
            CHECK(shift == -3);
        }
        SUBCASE("Scale with fractional part")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with a scale of 0.75
            int32_t multiplier = 0, shift = 0;
            double scale = 0.75;

            // Compute the multiplier and shift for the given scale
            ComputeMultiplierAndShiftTosaScale32(scale, multiplier, shift);

            // For scale 0.75, frexp returns mantissa = 0.75 and exponent = 0.
            // Then shiftedM = round(0.75 * (1 << 31)) = 1610612736.
            // After computing shift = (-(0)) + 31, we expect shift = 31.
            CHECK(multiplier == 1610612736);
            CHECK(shift == 31);
        }
        SUBCASE("Exception for NaN scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with NaN scale

            double scale = std::numeric_limits<double>::quiet_NaN();
            int32_t multiplier = 0;
            int32_t shift = 0;

            // Expect an exception when scale is NaN
            CHECK_THROWS_WITH(ComputeMultiplierAndShiftTosaScale32(scale,
                                                                   multiplier,
                                                                   shift),
                                                                   "Shifted mantissa exceeds 32 signed bits");
        }
        SUBCASE("Exception for shifted mantissa exceeding 32 signed bits")
        {
            // Test the ComputeMultiplierAndShiftTosaScale32 function with a scale that causes
            // the shifted mantissa to exceed 32 signed bits.

            // Pick a scale value that causes the computed shifted mantissa to be greater than
            // std::numeric_limits<double>::infinity(). The value below is chosen experimentally.
            // It is expected that the function will throw an exception in this case.
            double scale = std::numeric_limits<double>::infinity();
            int32_t multiplier = 0;
            int32_t shift = 0;

            // Expect an exception when scale is infinity
            CHECK_THROWS_WITH(ComputeMultiplierAndShiftTosaScale32(scale,
                                                                   multiplier,
                                                                   shift),
                                                                   "Shifted mantissa exceeds 32 signed bits");
        }

    }

    TEST_CASE("ComputeMultiplierAndShiftTosaScale16")
    {
        SUBCASE("Baseline")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with a scale of 1.0

            int32_t multiplier = 0;
            int32_t shift = 0;
            double scale = 0.5;

            // For 0.5, std::frexp returns mantissa = 0.5 and exponent = 0,
            // then shiftedM = round(0.5 * (1 << 16)) = round(0.5 * 32768) = 16384
            // and shift becomes (-0) + 15 = 15.
            ComputeMultiplierAndShiftTosaScale16(scale, multiplier, shift);

            // Check the computed multiplier and shift values
            CHECK(multiplier == 16384);
            CHECK(shift == 15);
        }
        SUBCASE("Zero scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with a scale of 0.0

            int32_t multiplier = 0;
            int32_t shift = 0;
            double scale = 0.0;

            // For 0.0, std::frexp returns mantissa = 0.0 and exponent = 0,
            // then shiftedM = round(0.0 * (1 << 16)) = 0
            // and shift becomes (-0) + 15 = 15.
            ComputeMultiplierAndShiftTosaScale16(scale, multiplier, shift);

            // Check the computed multiplier and shift values
            CHECK(multiplier == 0);
            CHECK(shift == 15);
        }
        SUBCASE("Negative scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with a negative scale of -0.5

            int32_t multiplier = 0;
            int32_t shift = 0;
            double scale = -0.5;

            // For -0.5, std::frexp returns mantissa = -0.5 and exponent = 0,
            // then shiftedM = round(-0.5 * (1 << 15)) = -16384
            // and shift becomes (-0) + 15 = 15.
            ComputeMultiplierAndShiftTosaScale16(scale, multiplier, shift);

            // Check the computed multiplier and shift values
            CHECK(multiplier == -16384);
            CHECK(shift == 15);
        }
        SUBCASE("Subnormal scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with a subnormal scale of 1e-10

            int32_t multiplier = 0;
            int32_t shift = 0;
            double scale = 1e-10;

            // For 1e-10, std::frexp returns mantissa ≈ 0.859 and exponent = -33,
            // then shiftedM = round(0.859 * (1 << 15)) = 28147
            // and shift becomes (-(-33)) + 15 = 48.
            ComputeMultiplierAndShiftTosaScale16(scale, multiplier, shift);

            // Check the computed multiplier and shift values
            CHECK(multiplier == 28147);
            CHECK(shift == 48);
        }
        SUBCASE("Large scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with a large scale of 1e10

            int32_t multiplier = 0;
            int32_t shift = 0;
            double scale = 1e10;

            // For 1e10, std::frexp returns mantissa = 0.5 and exponent = 34,
            // then shiftedM = round(0.5 * (1 << 15)) = 16384
            // and shift becomes (-34) + 15 = -19.
            ComputeMultiplierAndShiftTosaScale16(scale, multiplier, shift);

            // Check the computed multiplier and shift values
            CHECK(multiplier == 19073);
            CHECK(shift == -19);
        }
        SUBCASE("Scale with fractional part")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with a scale of 0.75

            int32_t multiplier = 0;
            int32_t shift = 0;
            double scale = 0.75;

            // For 0.75, std::frexp returns mantissa = 0.75 and exponent = 0,
            // then shiftedM = round(0.75 * (1 << 15)) = round(0.75 * 32768) = 24576
            // and shift becomes (-(0)) + 15 = 15.
            ComputeMultiplierAndShiftTosaScale16(scale, multiplier, shift);

            // Check the computed multiplier and shift values
            CHECK(multiplier == 24576);
            CHECK(shift == 15);
        }
        SUBCASE("Exception for NaN scale")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with NaN scale

            double scale = std::numeric_limits<double>::quiet_NaN();
            int32_t multiplier = 0;
            int32_t shift = 0;

            // Expect an exception when scale is NaN
            CHECK_THROWS_WITH(ComputeMultiplierAndShiftTosaScale16(scale,
                                                                   multiplier,
                                                                   shift),
                                                                   "Shifted mantissa exceeds 16 signed bits");
        }
        SUBCASE("Exception for shifted mantissa exceeding 16 signed bits")
        {
            // Test the ComputeMultiplierAndShiftTosaScale16 function with a scale that causes
            // the shifted mantissa to exceed 16 signed bits.

            // Pick a scale value that causes the computed shifted mantissa to be greater than
            // std::numeric_limits<double>::infinity(). The value below is chosen experimentally.
            // It is expected that the function will throw an exception in this case.
            double scale = std::numeric_limits<double>::infinity();
            int32_t multiplier = 0;
            int32_t shift = 0;

            // Expect an exception when scale is infinity
            CHECK_THROWS_WITH(ComputeMultiplierAndShiftTosaScale16(scale,
                                                                   multiplier,
                                                                   shift),
                                                                   "Shifted mantissa exceeds 16 signed bits");

        }
    }

}
