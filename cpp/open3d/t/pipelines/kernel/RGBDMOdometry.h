#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ComputeSourceMaskoutOdometryResultIntensity(const core::Tensor &source_depth,
                                    const core::Tensor &target_depth,
                                    const core::Tensor &source_intensity,
                                    const core::Tensor &target_intensity,
                                    const core::Tensor &target_intensity_dx,
                                    const core::Tensor &target_intensity_dy,
                                    const core::Tensor &source_vertex_map,
                                    const core::Tensor &source_mask, 
                                    const core::Tensor &intrinsics,
                                    const core::Tensor &init_source_to_target,
                                    core::Tensor &delta,
                                    float &inlier_residual,
                                    int &inlier_count,
                                    const float depth_outlier_trunc,
                                    const float intensity_huber_delta);

void ComputeSourceMaskoutOdometryResultHybrid(const core::Tensor &source_depth,
                                 const core::Tensor &target_depth,
                                 const core::Tensor &source_intensity,
                                 const core::Tensor &target_intensity,
                                 const core::Tensor &target_depth_dx,
                                 const core::Tensor &target_depth_dy,
                                 const core::Tensor &target_intensity_dx,
                                 const core::Tensor &target_intensity_dy,
                                 const core::Tensor &source_vertex_map,
                                 const core::Tensor &source_mask,
                                 const core::Tensor &intrinsics,
                                 const core::Tensor &init_source_to_target,
                                 core::Tensor &delta,
                                 float &inlier_residual,
                                 int &inlier_count,
                                 const float depth_outlier_trunc,
                                 const float depth_huber_delta,
                                 const float intensity_huber_delta);

void ComputeSourceMaskoutOdometryResultPointToPlane(
        const core::Tensor &source_vertex_map,
        const core::Tensor &target_vertex_map,
        const core::Tensor &target_normal_map,
        const core::Tensor &source_mask, 
        const core::Tensor &intrinsics,
        const core::Tensor &init_source_to_target,
        core::Tensor &delta,
        float &inlier_residual,
        int &inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta);

void ComputeMaskoutOdometryResultIntensity(const core::Tensor &source_depth,
                                    const core::Tensor &target_depth,
                                    const core::Tensor &source_intensity,
                                    const core::Tensor &target_intensity,
                                    const core::Tensor &target_intensity_dx,
                                    const core::Tensor &target_intensity_dy,
                                    const core::Tensor &source_vertex_map,
                                    const core::Tensor &source_mask, 
                                    const core::Tensor &target_mask, 
                                    const core::Tensor &intrinsics,
                                    const core::Tensor &init_source_to_target,
                                    core::Tensor &delta,
                                    float &inlier_residual,
                                    int &inlier_count,
                                    const float depth_outlier_trunc,
                                    const float intensity_huber_delta);

void ComputeMaskoutOdometryResultPointToPlane(
        const core::Tensor &source_vertex_map,
        const core::Tensor &target_vertex_map,
        const core::Tensor &target_normal_map,
        const core::Tensor &source_mask, 
        const core::Tensor &target_mask,
        const core::Tensor &intrinsics,
        const core::Tensor &init_source_to_target,
        core::Tensor &delta,
        float &inlier_residual,
        int &inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta);



void ComputeMaskoutOdometryResultHybrid(const core::Tensor &source_depth,
                                 const core::Tensor &target_depth,
                                 const core::Tensor &source_intensity,
                                 const core::Tensor &target_intensity,
                                 const core::Tensor &target_depth_dx,
                                 const core::Tensor &target_depth_dy,
                                 const core::Tensor &target_intensity_dx,
                                 const core::Tensor &target_intensity_dy,
                                 const core::Tensor &source_vertex_map,
                                 const core::Tensor &source_mask, 
                                 const core::Tensor &target_mask,
                                 const core::Tensor &intrinsics,
                                 const core::Tensor &init_source_to_target,
                                 core::Tensor &delta,
                                 float &inlier_residual,
                                 int &inlier_count,
                                 const float depth_outlier_trunc,
                                 const float depth_huber_delta,
                                 const float intensity_huber_delta);



}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
