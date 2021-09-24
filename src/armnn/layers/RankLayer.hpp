//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

namespace armnn
{

class RankLayer : public Layer
{
    public:
        /// Makes a workload for the Rank type.
        /// @param [in] factory The workload factory which will create the workload.
        /// @return A pointer to the created workload, or nullptr if not created.
        virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

        Layer* Clone(Graph& graph) const override;

        void ValidateTensorShapesFromInputs() override;

        ARMNN_NO_DEPRECATE_WARN_BEGIN
        void Accept(ILayerVisitor& visitor) const override;
        ARMNN_NO_DEPRECATE_WARN_END


        void ExecuteStrategy(IStrategy& strategy) const override;

protected:
        RankLayer(const char* name);
        ~RankLayer() = default;
};

} //namespace armnn


