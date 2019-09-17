import argparse
import os
import tqdm
import numpy as np


def rotate_to_center(points):
    """Rotates the points in a frustum so that the frustum would line up with the depth axis
    This means that the mean value for points on the left-to-right axis is 0

    Arguments:
        points {np.ndarray} -- Points with labels, Nx4

    Returns:
        np.ndarray -- Rotated points with labels, Nx4
    """
    points_center_topdown = points.mean(axis=0)[:2]
    angle = np.arccos(points_center_topdown[0] / np.linalg.norm(points_center_topdown))

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.array([[cos_angle, -1.0 * sin_angle],
                                [sin_angle, cos_angle]])

    rotated_xy = points[:, :2].dot(rotation_matrix)
    return np.c_[rotated_xy, points[:, 2:]]


def scale_standard(points):
    """Scale points to follow the mean and standard deviation of the normal gaussian distribution

    Arguments:
        points {np.ndarray} -- Points with labels, Nx4

    Returns:
        np.ndarray -- Scaled points with labels, Nx4
    """
    scale_factors = points.std(axis=0)
    scaled_points = (points - points.mean(axis=0)) / scale_factors

    return np.c_[scaled_points[:, :3], points[:, 3]]


def get_arguments():
    parser = argparse.ArgumentParser(description='The normalization script for training and validation data used for training the model')

    parser.add_argument(
        'input', type=str,
        help='Path to directory containing points of each frustum as a separate file'
    )

    parser.add_argument(
        'output', type=str,
        help='Path to target directory where preprocessed frustum data should be saved'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    input_dir = args.input
    output_dir = args.output


    frustum_files = sorted(os.listdir(input_dir))
    frustum_files = [frustum_file for frustum_file in frustum_files if frustum_file.endswith('.npz')]

    for frustum_file in tqdm.tqdm(frustum_files):
        frustum_file_path = os.path.join(input_dir, frustum_file)
        with np.load(frustum_file_path) as data:
            labelled_points = data['points']
            class_name = data['class_name']

        labelled_points = rotate_to_center(labelled_points)
        labelled_points = scale_standard(labelled_points)

        output_file_path = os.path.join(output_dir, frustum_file)
        np.savez(output_file_path, points=labelled_points, class_name=class_name)
