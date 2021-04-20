from dataclasses import dataclass
from typing import Any
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
from plot_utils import plot_result

"""Based on the example https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/PlanarSLAMExample.py"""


@dataclass
class MultivariateNormalParameters:
    mean: Any
    covariance: np.ndarray


def factor_graph_experiment():
    # Create an empty nonlinear factor graph.
    graph = gtsam.NonlinearFactorGraph()

    # Create the keys for the poses.
    X1 = X(1)
    X2 = X(2)
    X3 = X(3)
    pose_variables = [X1, X2, X3]

    # Create keys for the landmarks.
    L1 = L(1)
    L2 = L(2)
    landmark_variables = [L1, L2]

    # Add a prior on pose X1 at the origin.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    graph.add(gtsam.PriorFactorPose2(X1, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))

    # Add odometry factors between X1,X2 and X2,X3, respectively
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    graph.add(gtsam.BetweenFactorPose2(
        X1, X2, gtsam.Pose2(2.0, 0.0, 0.0), odometry_noise))
    graph.add(gtsam.BetweenFactorPose2(
        X2, X3, gtsam.Pose2(2.0, 0.0, 0.0), odometry_noise))

    # Add Range-Bearing measurements to two different landmarks L1 and L2
    measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))
    graph.add(gtsam.BearingRangeFactor2D(
        X1, L1, gtsam.Rot2.fromDegrees(45), np.sqrt(4.0 + 4.0), measurement_noise))
    graph.add(gtsam.BearingRangeFactor2D(
        X2, L1, gtsam.Rot2.fromDegrees(90), 2.0, measurement_noise))
    graph.add(gtsam.BearingRangeFactor2D(
        X3, L2, gtsam.Rot2.fromDegrees(90), 2.0, measurement_noise))

    # Create (deliberately inaccurate) initial estimate
    initial_estimate = gtsam.Values()
    initial_estimate.insert(X1, gtsam.Pose2(-0.25, 0.20, 0.15))
    initial_estimate.insert(X2, gtsam.Pose2(2.30, 0.10, -0.20))
    initial_estimate.insert(X3, gtsam.Pose2(4.10, 0.10, 0.10))
    initial_estimate.insert(L1, gtsam.Point2(1.80, 2.10))
    initial_estimate.insert(L2, gtsam.Point2(4.10, 1.80))

    # Create an optimizer.
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Solve the MAP problem.
    result = optimizer.optimize()

    # Calculate marginal covariances for all variables.
    marginals = gtsam.Marginals(graph, result)

    # Extract marginals
    pose_marginals = []
    for var in pose_variables:
        pose_marginals.append(MultivariateNormalParameters(result.atPose2(var), marginals.marginalCovariance(var)))

    landmark_marginals = []
    for var in landmark_variables:
        landmark_marginals.append(MultivariateNormalParameters(result.atPoint2(var), marginals.marginalCovariance(var)))

    # You can extract the joint marginals like this (not used further here).
    joint_poses = marginals.jointMarginalCovariance(gtsam.KeyVector(pose_variables))
    joint_full = marginals.jointMarginalCovariance(gtsam.KeyVector(pose_variables + landmark_variables))

    plot_result(pose_marginals, landmark_marginals)


if __name__ == "__main__":
    factor_graph_experiment()
