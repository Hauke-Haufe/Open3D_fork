// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace image {

void To(const core::Tensor &src,
        core::Tensor &dst,
        double scale,
        double offset);

void ClipTransform(const core::Tensor &src,
                   core::Tensor &dst,
                   float scale,
                   float min_value,
                   float max_value,
                   float clip_fill = 0.0f);

void PyrDownDepth(const core::Tensor &src,
                  core::Tensor &dst,
                  float diff_threshold,
                  float invalid_fill);

void PyrDownMajority(const core::Tensor &src, 
                     core::Tensor &dst);

void CreateVertexMap(const core::Tensor &src,
                     core::Tensor &dst,
                     const core::Tensor &intrinsics,
                     float invalid_fill);

void CreateNormalMap(const core::Tensor &src,
                     core::Tensor &dst,
                     float invalid_fill);

void CreateNormalMapMaskout(const core::Tensor &src,
                            core::Tensor &dst,
                            const core::Tensor &mask, 
                            float invalid_fill);

void FilterSobelMaskout(const core::Tensor &src, 
                        const core::Tensor& mask,
                        core::Tensor& dx, 
                        core::Tensor& dy);

void ColorizeDepth(const core::Tensor &src,
                   core::Tensor &dst,
                   float scale,
                   float min_value,
                   float max_value);

void ToCPU(const core::Tensor &src,
           core::Tensor &dst,
           double scale,
           double offset);

void ClipTransformCPU(const core::Tensor &src,
                      core::Tensor &dst,
                      float scale,
                      float min_value,
                      float max_value,
                      float clip_fill = 0.0f);

void PyrDownDepthCPU(const core::Tensor &src,
                     core::Tensor &dst,
                     float diff_threshold,
                     float invalid_fill);

void PyrDownMajorityCPU(const core::Tensor &src, 
                    core::Tensor &dst);

void CreateVertexMapCPU(const core::Tensor &src,
                        core::Tensor &dst,
                        const core::Tensor &intrinsics,
                        float invalid_fill);

void CreateNormalMapCPU(const core::Tensor &src,
                        core::Tensor &dst,
                        float invalid_fill);

void CreateNormalMapMaskoutCPU(const core::Tensor& src, 
                                core::Tensor& dst, 
                                const core::Tensor& mask, 
                                float invalid_fill);

void FilterSobelMaskoutCPU(const core::Tensor &src, 
                        const core::Tensor& mask,
                        core::Tensor& dx, 
                        core::Tensor& dy);

void ColorizeDepthCPU(const core::Tensor &src,
                      core::Tensor &dst,
                      float scale,
                      float min_value,
                      float max_value);

#ifdef BUILD_CUDA_MODULE
void ToCUDA(const core::Tensor &src,
            core::Tensor &dst,
            double scale,
            double offset);

void ClipTransformCUDA(const core::Tensor &src,
                       core::Tensor &dst,
                       float scale,
                       float min_value,
                       float max_value,
                       float clip_fill = 0.0f);

void PyrDownDepthCUDA(const core::Tensor &src,
                      core::Tensor &dst,
                      float diff_threshold,
                      float invalid_fill);

void PyrDownMajorityCUDA(const core::Tensor &src, 
                    core::Tensor &dst);

void CreateVertexMapCUDA(const core::Tensor &src,
                         core::Tensor &dst,
                         const core::Tensor &intrinsics,
                         float invalid_fill);

void CreateNormalMapMaskoutCUDA(const core::Tensor& src, 
                                core::Tensor& dst, 
                                const core::Tensor& mask, 
                                float invalid_fill);

void FilterSobelMaskoutCUDA(const core::Tensor &src, 
                        const core::Tensor& mask,
                        core::Tensor& dx, 
                        core::Tensor& dy);

void CreateNormalMapCUDA(const core::Tensor &src,
                         core::Tensor &dst,
                         float invalid_fill);

void ColorizeDepthCUDA(const core::Tensor &src,
                       core::Tensor &dst,
                       float scale,
                       float min_value,
                       float max_value);

#endif
}  // namespace image
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
