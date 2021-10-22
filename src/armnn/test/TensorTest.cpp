//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Tensor.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("Tensor")
{
struct TensorInfoFixture
{
    TensorInfoFixture()
    {
        unsigned int sizes[] = {6,7,8,9};
        m_TensorInfo = TensorInfo(4, sizes, DataType::Float32);
    }
    ~TensorInfoFixture() {};

    TensorInfo m_TensorInfo;
};

TEST_CASE_FIXTURE(TensorInfoFixture, "ConstructShapeUsingListInitialization")
{
    TensorShape listInitializedShape{ 6, 7, 8, 9 };
    CHECK(listInitializedShape == m_TensorInfo.GetShape());
}

TEST_CASE_FIXTURE(TensorInfoFixture, "ConstructTensorInfo")
{
    CHECK(m_TensorInfo.GetNumDimensions() == 4);
    CHECK(m_TensorInfo.GetShape()[0] == 6); // <= Outer most
    CHECK(m_TensorInfo.GetShape()[1] == 7);
    CHECK(m_TensorInfo.GetShape()[2] == 8);
    CHECK(m_TensorInfo.GetShape()[3] == 9);     // <= Inner most
}

TEST_CASE_FIXTURE(TensorInfoFixture, "CopyConstructTensorInfo")
{
    TensorInfo copyConstructed(m_TensorInfo);
    CHECK(copyConstructed.GetNumDimensions() == 4);
    CHECK(copyConstructed.GetShape()[0] == 6);
    CHECK(copyConstructed.GetShape()[1] == 7);
    CHECK(copyConstructed.GetShape()[2] == 8);
    CHECK(copyConstructed.GetShape()[3] == 9);
}

TEST_CASE_FIXTURE(TensorInfoFixture, "TensorInfoEquality")
{
    TensorInfo copyConstructed(m_TensorInfo);
    CHECK(copyConstructed == m_TensorInfo);
}

TEST_CASE_FIXTURE(TensorInfoFixture, "TensorInfoInequality")
{
    TensorInfo other;
    unsigned int sizes[] = {2,3,4,5};
    other = TensorInfo(4, sizes, DataType::Float32);

    CHECK(other != m_TensorInfo);
}

TEST_CASE_FIXTURE(TensorInfoFixture, "TensorInfoAssignmentOperator")
{
    TensorInfo copy;
    copy = m_TensorInfo;
    CHECK(copy == m_TensorInfo);
}

TEST_CASE("CopyNoQuantizationTensorInfo")
{
    TensorInfo infoA;
    infoA.SetShape({ 5, 6, 7, 8 });
    infoA.SetDataType(DataType::QAsymmU8);

    TensorInfo infoB;
    infoB.SetShape({ 5, 6, 7, 8 });
    infoB.SetDataType(DataType::QAsymmU8);
    infoB.SetQuantizationScale(10.0f);
    infoB.SetQuantizationOffset(5);
    infoB.SetQuantizationDim(Optional<unsigned int>(1));

    CHECK((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    CHECK((infoA.GetDataType() == DataType::QAsymmU8));
    CHECK(infoA.GetQuantizationScale() == 1);
    CHECK(infoA.GetQuantizationOffset() == 0);
    CHECK(!infoA.GetQuantizationDim().has_value());

    CHECK(infoA != infoB);
    infoA = infoB;
    CHECK(infoA == infoB);

    CHECK((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    CHECK((infoA.GetDataType() == DataType::QAsymmU8));
    CHECK(infoA.GetQuantizationScale() == 10.0f);
    CHECK(infoA.GetQuantizationOffset() == 5);
    CHECK(infoA.GetQuantizationDim().value() == 1);
}

TEST_CASE("CopyDifferentQuantizationTensorInfo")
{
    TensorInfo infoA;
    infoA.SetShape({ 5, 6, 7, 8 });
    infoA.SetDataType(DataType::QAsymmU8);
    infoA.SetQuantizationScale(10.0f);
    infoA.SetQuantizationOffset(5);
    infoA.SetQuantizationDim(Optional<unsigned int>(1));

    TensorInfo infoB;
    infoB.SetShape({ 5, 6, 7, 8 });
    infoB.SetDataType(DataType::QAsymmU8);
    infoB.SetQuantizationScale(11.0f);
    infoB.SetQuantizationOffset(6);
    infoB.SetQuantizationDim(Optional<unsigned int>(2));

    CHECK((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    CHECK((infoA.GetDataType() == DataType::QAsymmU8));
    CHECK(infoA.GetQuantizationScale() == 10.0f);
    CHECK(infoA.GetQuantizationOffset() == 5);
    CHECK(infoA.GetQuantizationDim().value() == 1);

    CHECK(infoA != infoB);
    infoA = infoB;
    CHECK(infoA == infoB);

    CHECK((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    CHECK((infoA.GetDataType() == DataType::QAsymmU8));
    CHECK(infoA.GetQuantizationScale() == 11.0f);
    CHECK(infoA.GetQuantizationOffset() == 6);
    CHECK(infoA.GetQuantizationDim().value() == 2);
}

void CheckTensor(const ConstTensor& t)
{
    t.GetInfo();
}

TEST_CASE("TensorVsConstTensor")
{
    int mutableDatum = 2;
    const int immutableDatum = 3;

    armnn::Tensor uninitializedTensor;
    uninitializedTensor.GetInfo().SetConstant(true);
    armnn::ConstTensor uninitializedTensor2;

    uninitializedTensor2 = uninitializedTensor;

    armnn::TensorInfo emptyTensorInfo;
    emptyTensorInfo.SetConstant(true);
    armnn::Tensor t(emptyTensorInfo, &mutableDatum);
    armnn::ConstTensor ct(emptyTensorInfo, &immutableDatum);

    // Checks that both Tensor and ConstTensor can be passed as a ConstTensor.
    CheckTensor(t);
    CheckTensor(ct);
}

TEST_CASE("ConstTensor_EmptyConstructorTensorInfoSet")
{
    armnn::ConstTensor t;
    CHECK(t.GetInfo().IsConstant() == true);
}

TEST_CASE("ConstTensor_TensorInfoNotConstantError")
{
    armnn::TensorInfo tensorInfo ({ 1 }, armnn::DataType::Float32);
    std::vector<float> tensorData =  { 1.0f };
    try
    {
        armnn::ConstTensor ct(tensorInfo, tensorData);
        FAIL("InvalidArgumentException should have been thrown");
    }
    catch(const InvalidArgumentException& exc)
    {
        CHECK(strcmp(exc.what(), "Invalid attempt to construct ConstTensor from non-constant TensorInfo.") == 0);
    }
}

TEST_CASE("PassTensorToConstTensor_TensorInfoNotConstantError")
{
    try
    {
        armnn::ConstTensor t = ConstTensor(Tensor());
        FAIL("InvalidArgumentException should have been thrown");
    }
    catch(const InvalidArgumentException& exc)
    {
        CHECK(strcmp(exc.what(), "Invalid attempt to construct ConstTensor from "
                                 "Tensor due to non-constant TensorInfo") == 0);
    }
}

TEST_CASE("ModifyTensorInfo")
{
    TensorInfo info;
    info.SetShape({ 5, 6, 7, 8 });
    CHECK((info.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    info.SetDataType(DataType::QAsymmU8);
    CHECK((info.GetDataType() == DataType::QAsymmU8));
    info.SetQuantizationScale(10.0f);
    CHECK(info.GetQuantizationScale() == 10.0f);
    info.SetQuantizationOffset(5);
    CHECK(info.GetQuantizationOffset() == 5);
}

TEST_CASE("TensorShapeOperatorBrackets")
{
    const TensorShape constShape({0,1,2,3});
    TensorShape shape({0,1,2,3});

    // Checks version of operator[] which returns an unsigned int.
    CHECK(shape[2] == 2);
    shape[2] = 20;
    CHECK(shape[2] == 20);

    // Checks the version of operator[] which returns a reference.
    CHECK(constShape[2] == 2);
}

TEST_CASE("TensorInfoPerAxisQuantization")
{
    // Old constructor
    TensorInfo tensorInfo0({ 1, 1 }, DataType::Float32, 2.0f, 1);
    CHECK(!tensorInfo0.HasMultipleQuantizationScales());
    CHECK(tensorInfo0.GetQuantizationScale() == 2.0f);
    CHECK(tensorInfo0.GetQuantizationOffset() == 1);
    CHECK(tensorInfo0.GetQuantizationScales()[0] == 2.0f);
    CHECK(!tensorInfo0.GetQuantizationDim().has_value());

    // Set per-axis quantization scales
    std::vector<float> perAxisScales{ 3.0f, 4.0f };
    tensorInfo0.SetQuantizationScales(perAxisScales);
    CHECK(tensorInfo0.HasMultipleQuantizationScales());
    CHECK(tensorInfo0.GetQuantizationScales() == perAxisScales);

    // Set per-tensor quantization scale
    tensorInfo0.SetQuantizationScale(5.0f);
    CHECK(!tensorInfo0.HasMultipleQuantizationScales());
    CHECK(tensorInfo0.GetQuantizationScales()[0] == 5.0f);

    // Set quantization offset
    tensorInfo0.SetQuantizationDim(Optional<unsigned int>(1));
    CHECK(tensorInfo0.GetQuantizationDim().value() == 1);

    // New constructor
    perAxisScales = { 6.0f, 7.0f };
    TensorInfo tensorInfo1({ 1, 1 }, DataType::Float32, perAxisScales, 1);
    CHECK(tensorInfo1.HasMultipleQuantizationScales());
    CHECK(tensorInfo1.GetQuantizationOffset() == 0);
    CHECK(tensorInfo1.GetQuantizationScales() == perAxisScales);
    CHECK(tensorInfo1.GetQuantizationDim().value() == 1);
}

TEST_CASE("TensorShape_scalar")
{
    float mutableDatum = 3.1416f;

    const armnn::TensorShape shape  (armnn::Dimensionality::Scalar );
    armnn::TensorInfo        info   ( shape, DataType::Float32 );
    const armnn::Tensor      tensor ( info, &mutableDatum );

    CHECK(armnn::Dimensionality::Scalar == shape.GetDimensionality());
    float scalarValue = *reinterpret_cast<float*>(tensor.GetMemoryArea());
    CHECK_MESSAGE(mutableDatum == scalarValue, "Scalar value is " << scalarValue);

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    CHECK(shape_equal == shape);
    CHECK(shape_different != shape);
    CHECK_MESSAGE(1 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    CHECK_MESSAGE(1 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    CHECK(true == shape.GetDimensionSpecificity(0));
    CHECK(shape.AreAllDimensionsSpecified());
    CHECK(shape.IsAtLeastOneDimensionSpecified());

    CHECK(1 == shape[0]);
    CHECK(1 == tensor.GetShape()[0]);
    CHECK(1 == tensor.GetInfo().GetShape()[0]);
    CHECK_THROWS_AS( shape[1], InvalidArgumentException );

    float newMutableDatum  = 42.f;
    std::memcpy(tensor.GetMemoryArea(), &newMutableDatum, sizeof(float));
    scalarValue = *reinterpret_cast<float*>(tensor.GetMemoryArea());
    CHECK_MESSAGE(newMutableDatum == scalarValue, "Scalar value is " << scalarValue);
}

TEST_CASE("TensorShape_DynamicTensorType1_unknownNumberDimensions")
{
    float       mutableDatum  = 3.1416f;

    armnn::TensorShape shape  (armnn::Dimensionality::NotSpecified );
    armnn::TensorInfo  info   ( shape, DataType::Float32 );
    armnn::Tensor      tensor ( info, &mutableDatum );

    CHECK(armnn::Dimensionality::NotSpecified == shape.GetDimensionality());
    CHECK_THROWS_AS( shape[0], InvalidArgumentException );
    CHECK_THROWS_AS( shape.GetNumElements(), InvalidArgumentException );
    CHECK_THROWS_AS( shape.GetNumDimensions(), InvalidArgumentException );

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    CHECK(shape_equal == shape);
    CHECK(shape_different != shape);
}

TEST_CASE("TensorShape_DynamicTensorType1_unknownAllDimensionsSizes")
{
    float       mutableDatum  = 3.1416f;

    armnn::TensorShape shape  ( 3, false );
    armnn::TensorInfo  info   ( shape, DataType::Float32 );
    armnn::Tensor      tensor ( info, &mutableDatum );

    CHECK(armnn::Dimensionality::Specified == shape.GetDimensionality());
    CHECK_MESSAGE(0 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    CHECK(false == shape.GetDimensionSpecificity(0));
    CHECK(false == shape.GetDimensionSpecificity(1));
    CHECK(false == shape.GetDimensionSpecificity(2));
    CHECK(!shape.AreAllDimensionsSpecified());
    CHECK(!shape.IsAtLeastOneDimensionSpecified());

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    CHECK(shape_equal == shape);
    CHECK(shape_different != shape);
}

TEST_CASE("TensorShape_DynamicTensorType1_unknownSomeDimensionsSizes")
{
    std::vector<float> mutableDatum  { 42.f, 42.f, 42.f,
                                       0.0f, 0.1f, 0.2f };

    armnn::TensorShape shape         ( {2, 0, 3}, {true, false, true} );
    armnn::TensorInfo  info          ( shape, DataType::Float32 );
    armnn::Tensor      tensor        ( info, &mutableDatum );

    CHECK(armnn::Dimensionality::Specified == shape.GetDimensionality());
    CHECK_MESSAGE(6 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    CHECK(true  == shape.GetDimensionSpecificity(0));
    CHECK(false == shape.GetDimensionSpecificity(1));
    CHECK(true  == shape.GetDimensionSpecificity(2));
    CHECK(!shape.AreAllDimensionsSpecified());
    CHECK(shape.IsAtLeastOneDimensionSpecified());

    CHECK_THROWS_AS(shape[1], InvalidArgumentException);
    CHECK_THROWS_AS(tensor.GetShape()[1], InvalidArgumentException);
    CHECK_THROWS_AS(tensor.GetInfo().GetShape()[1], InvalidArgumentException);

    CHECK(2 == shape[0]);
    CHECK(2 == tensor.GetShape()[0]);
    CHECK(2 == tensor.GetInfo().GetShape()[0]);
    CHECK_THROWS_AS( shape[1], InvalidArgumentException );

    CHECK(3 == shape[2]);
    CHECK(3 == tensor.GetShape()[2]);
    CHECK(3 == tensor.GetInfo().GetShape()[2]);

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    CHECK(shape_equal == shape);
    CHECK(shape_different != shape);
}

TEST_CASE("TensorShape_DynamicTensorType1_transitionFromUnknownToKnownDimensionsSizes")
{
    std::vector<float> mutableDatum  { 42.f, 42.f, 42.f,
                                       0.0f, 0.1f, 0.2f };

    armnn::TensorShape shape         (armnn::Dimensionality::NotSpecified );
    armnn::TensorInfo  info          ( shape, DataType::Float32 );
    armnn::Tensor      tensor        ( info, &mutableDatum );

    // Specify the number of dimensions
    shape.SetNumDimensions(3);
    CHECK(armnn::Dimensionality::Specified == shape.GetDimensionality());
    CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    CHECK(false == shape.GetDimensionSpecificity(0));
    CHECK(false == shape.GetDimensionSpecificity(1));
    CHECK(false == shape.GetDimensionSpecificity(2));
    CHECK(!shape.AreAllDimensionsSpecified());
    CHECK(!shape.IsAtLeastOneDimensionSpecified());

    // Specify dimension 0 and 2.
    shape.SetDimensionSize(0, 2);
    shape.SetDimensionSize(2, 3);
    CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    CHECK_MESSAGE(6 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    CHECK(true  == shape.GetDimensionSpecificity(0));
    CHECK(false == shape.GetDimensionSpecificity(1));
    CHECK(true  == shape.GetDimensionSpecificity(2));
    CHECK(!shape.AreAllDimensionsSpecified());
    CHECK(shape.IsAtLeastOneDimensionSpecified());

    info.SetShape(shape);
    armnn::Tensor tensor2( info, &mutableDatum );
    CHECK(2 == shape[0]);
    CHECK(2 == tensor2.GetShape()[0]);
    CHECK(2 == tensor2.GetInfo().GetShape()[0]);

    CHECK_THROWS_AS(shape[1], InvalidArgumentException);
    CHECK_THROWS_AS(tensor.GetShape()[1], InvalidArgumentException);
    CHECK_THROWS_AS(tensor.GetInfo().GetShape()[1], InvalidArgumentException);

    CHECK(3 == shape[2]);
    CHECK(3 == tensor2.GetShape()[2]);
    CHECK(3 == tensor2.GetInfo().GetShape()[2]);

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    CHECK(shape_equal == shape);
    CHECK(shape_different != shape);

    // Specify dimension 1.
    shape.SetDimensionSize(1, 5);
    CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    CHECK_MESSAGE(30 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    CHECK(true  == shape.GetDimensionSpecificity(0));
    CHECK(true  == shape.GetDimensionSpecificity(1));
    CHECK(true  == shape.GetDimensionSpecificity(2));
    CHECK(shape.AreAllDimensionsSpecified());
    CHECK(shape.IsAtLeastOneDimensionSpecified());
}

TEST_CASE("Tensor_emptyConstructors")
{
    auto shape = armnn::TensorShape();
    CHECK_MESSAGE( 0 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    CHECK_MESSAGE( 0 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    CHECK( armnn::Dimensionality::Specified == shape.GetDimensionality());
    CHECK( shape.AreAllDimensionsSpecified());
    CHECK_THROWS_AS( shape[0], InvalidArgumentException );

    auto tensor = armnn::Tensor();
    CHECK_MESSAGE( 0 == tensor.GetNumDimensions(), "Number of dimensions is " << tensor.GetNumDimensions());
    CHECK_MESSAGE( 0 == tensor.GetNumElements(), "Number of elements is " << tensor.GetNumElements());
    CHECK_MESSAGE( 0 == tensor.GetShape().GetNumDimensions(), "Number of dimensions is " <<
                        tensor.GetShape().GetNumDimensions());
    CHECK_MESSAGE( 0 == tensor.GetShape().GetNumElements(), "Number of dimensions is " <<
                        tensor.GetShape().GetNumElements());
    CHECK( armnn::Dimensionality::Specified == tensor.GetShape().GetDimensionality());
    CHECK( tensor.GetShape().AreAllDimensionsSpecified());
    CHECK_THROWS_AS( tensor.GetShape()[0], InvalidArgumentException );
}
}
