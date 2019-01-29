//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TestLayerVisitor.hpp"

namespace armnn
{

// Concrete TestLayerVisitor subclasses for layers taking Name argument with overridden VisitLayer methods
class TestAdditionLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestAdditionLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitAdditionLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestMultiplicationLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestMultiplicationLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitMultiplicationLayer(const IConnectableLayer* layer,
                                  const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestFloorLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestFloorLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitFloorLayer(const IConnectableLayer* layer,
                         const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestDivisionLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestDivisionLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitDivisionLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestSubtractionLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestSubtractionLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitSubtractionLayer(const IConnectableLayer* layer,
                               const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestMaximumLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestMaximumLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitMaximumLayer(const IConnectableLayer* layer,
                           const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestMinimumLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestMinimumLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitMinimumLayer(const IConnectableLayer* layer,
                           const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestGreaterLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestGreaterLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitGreaterLayer(const IConnectableLayer* layer,
                           const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestEqualLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestEqualLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitEqualLayer(const IConnectableLayer* layer,
                         const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestRsqrtLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestRsqrtLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitRsqrtLayer(const IConnectableLayer* layer,
                         const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

class TestGatherLayerVisitor : public TestLayerVisitor
{
public:
    explicit TestGatherLayerVisitor(const char* name = nullptr) : TestLayerVisitor(name) {};

    void VisitGatherLayer(const IConnectableLayer* layer,
                          const char* name = nullptr) override {
        CheckLayerPointer(layer);
        CheckLayerName(name);
    };
};

} //namespace armnn