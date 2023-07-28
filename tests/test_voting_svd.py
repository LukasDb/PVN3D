from ..models.geometry import (
    batch_rt_svd_transform,
    batch_pts_clustering_with_std,
    batch_get_pt_candidates_tf,
)
import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_equal, assert_array_almost_equal

# TODO test svd transform as well


def test_batch_pts_clustering_with_std():
    """
    batch_pts_clustering takes the top k shortest offsets and clusters them
    obj_kpts: [b, n_keypoints, n_points, 3]
    """
    # generate top keypoints
    obj_kpts = np.array(
        [  # batch
            [  # n_keypoints
                [  # for keypoint 1
                    [-0.1, -0.1, 1.0],  # voted keypoint location of point 1
                    [-0.1, -0.1, 1.0],  # voted keypoint location of point 2
                    [-0.1, -0.1, 1.0],  # voted keypoint location of point 3
                    [-0.1, -0.1, 1.0],  # voted keypoint location of point 4
                    [-0.11, -0.097, 1.01],  # voted keypoint location of point 5
                    [-0.105, -0.95, 1.06],  # voted keypoint location of point 6
                    [-0.095, 0.95, 0.96],  # outlier
                ],
                [  # for keypoint 2
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.11, 0.95, 1.05],
                    [-0.15, 0.99, 1.06],
                    [0.93, -0.16, 0.95],  # outlier
                ],
                [  # for center point
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [-0.5, 0.01, 1.04],  # outlier
                    [0.01, 0.08, 0.956],
                    [-0.01, -0.04, 1.01],
                ],
            ]
        ]
    ).astype(np.float32)

    kpts_clustered = batch_pts_clustering_with_std(obj_kpts)

    expected_kpts_clustered = np.array(
        [  # batch
            [  # n_keypoints
                [-0.1, -0.1, 1.0],  # keypoint 1
                [0.1, 0.1, 1.0],  # keypoint 2
                [0.0, 0.0, 1.0],  # center point
            ]
        ]
    ).astype(np.float32)
    # because of clustering its not perfectly the same
    assert_array_almost_equal(kpts_clustered, expected_kpts_clustered, decimal=2)


# Test case 1
def test_batch_get_pt_candidates_tf():
    """
    this batch_get_pt_candidates_tf is used in models/pprocessnet.py
    1) offsets, that are segmented as !object get an additonal length of 1000.
    2) it takes the offsets and calculats the top k shortest offsets
    3) clustering?
    4) the offsets are added to the point cloud and returned

    """

    # generate a point cloud with 9 points at z=1m in a grid with 0.1
    pcld_xyz = np.array(
        [  # batch
            [  # n_points
                [-0.1, -0.1, 1.0],
                [-0.1, 0.0, 1.0],
                [-0.1, 0.1, 1.0],
                [0.0, -0.1, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, -0.1, 1.0],
                [0.1, 0.0, 1.0],
                [0.1, 0.1, 1.0],
            ]
        ]
        * 2
    )  # [b, num_points, 3]

    # one keypoint is at first point (-0.1, -0.1, 1.)
    # the second at (0.1, 0.1, 1.)

    # for each point we generate a offset pointing towards the center
    kpts_ofst_pre = np.array(
        [  # batch
            [  # n_points
                [  # point 1
                    [0.0, 0.0, 0.0],  # offset 1st keypoint
                    [0.2, 0.2, 0.0],  # offset to 2nd keypoint
                ],
                [  # point 2
                    [0.0, -0.1, 0.0],
                    [0.2, 0.1, 0.0],
                ],
                [  # point 3
                    [0.0, -0.2, 0.0],
                    [0.2, 0.0, 0.0],
                ],
                [  # point 4
                    [-0.1, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                ],
                [  # point 5
                    [-0.1, -0.1, 0.0],
                    [0.1, 0.1, 0.0],
                ],
                [  # point 6
                    [-0.1, -0.2, 0.0],
                    [0.1, 0.0, 0.0],
                ],
                [  # point 7
                    [-0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                ],
                [  # point 8
                    [-0.2, -0.1, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                [  # point 9
                    [-0.2, -0.2, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]
        ]
        * 2
    )  # [b, num_points, n_kpts, 3] n_pkts = 8

    # generate the same for the centerpoint
    ctr_ofst_pre = np.array(
        [  # batch
            [  # n_points
                [  # point 1
                    [0.1, 0.1, 0.0],  # offset center
                ],
                [  # point 2
                    [0.1, 0.0, 0.0],
                ],
                [  # point 3
                    [0.1, -0.1, 0.0],
                ],
                [  # point 4
                    [0.0, 0.1, 0.0],
                ],
                [  # point 5
                    [0.0, 0.0, 0.0],
                ],
                [  # point 6
                    [0.0, -0.1, 0.0],
                ],
                [  # point 7
                    [-0.1, 0.1, 0.0],
                ],
                [  # point 8
                    [-0.1, 0.0, 0.0],
                ],
                [  # point 9
                    [-0.1, -0.1, 0.0],
                ],
            ]
        ]
        * 2
    )

    # [b, num_points, 1, 3]
    seg_pre = np.ones([2, 9, 1])  # [b, num_points, 1]
    k = 5  # must be less than num_points, which is 9
    obj_kpts = batch_get_pt_candidates_tf(
        pcld_xyz.astype(np.float32),
        kpts_ofst_pre.astype(np.float32),
        seg_pre.astype(np.float32),
        ctr_ofst_pre.astype(np.float32),
        k,
    )

    # we expect [2, 3, 9, 3] voted keypoints
    expected_kpts = np.array(
        [  # batch
            [
                [[-0.1, -0.1, 1.0]] * k,  # 1st keypoint for each point
                [[0.1, 0.1, 1.0]] * k,  # 2nd keypoint for each point
                [[0.0, 0.0, 1.0]] * k,  # center for each point
            ]
        ]
        * 2
    )
    assert_array_almost_equal(obj_kpts, expected_kpts)


# Test case 1
def test_batch_get_pt_candidates_tf_with_segmentation():
    """
    this batch_get_pt_candidates_tf is used in models/pprocessnet.py
    1) offsets, that are segmented as !object get an additonal length of 1000.
    2) it takes the offsets and calculats the top k shortest offsets
    3) clustering?
    4) the offsets are added to the point cloud and returned

    # some points with false offsets are segmented as background and should be
    discarded

    """

    # generate a point cloud with 9 points at z=1m in a grid with 0.1
    pcld_xyz = np.array(
        [  # batch
            [  # n_points
                [-0.1, -0.1, 1.0],
                [-0.1, 0.0, 1.0],
                [-0.1, 0.1, 1.0],
                [0.0, -0.1, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, -0.1, 1.0],
                [0.1, 0.0, 1.0],
                [0.1, 0.1, 1.0],
            ]
        ]
    )  # [b, num_points, 3]

    # one keypoint is at first point (-0.1, -0.1, 1.)
    # the second at (0.1, 0.1, 1.)

    # for each point we generate a offset pointing towards the center
    kpts_ofst_pre = np.array(
        [  # batch
            [  # n_points
                [  # point 1
                    [-0.01, -0.01, 0.0],  # FALSE OFFSET
                    [0.0, 0.0, 0.0],  # FALSE OFFSET
                ],
                [  # point 2
                    [0.01, -0.01, 0.0],
                    [0.02, 0.01, 0.0],
                ],
                [  # point 3
                    [0.0, -0.2, 0.0],
                    [0.2, 0.0, 0.0],
                ],
                [  # point 4
                    [-0.1, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                ],
                [  # point 5
                    [-0.1, -0.1, 0.0],
                    [0.1, 0.1, 0.0],
                ],
                [  # point 6
                    [-0.1, -0.2, 0.0],
                    [0.1, 0.0, 0.0],
                ],
                [  # point 7
                    [-0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                ],
                [  # point 8
                    [-0.2, -0.1, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                [  # point 9
                    [-0.2, -0.2, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]
        ]
    )  # [b, num_points, n_kpts, 3] n_pkts = 8

    # generate the same for the centerpoint
    ctr_ofst_pre = np.array(
        [  # batch
            [  # n_points
                [  # point 1
                    [-0.1, -0.1, 0.0],  # FALSE OFFSET
                ],
                [  # point 2
                    [0.1, 0.0, 0.0],
                ],
                [  # point 3
                    [0.1, -0.1, 0.0],
                ],
                [  # point 4
                    [0.0, 0.1, 0.0],
                ],
                [  # point 5
                    [0.0, 0.0, 0.0],
                ],
                [  # point 6
                    [0.0, -0.1, 0.0],
                ],
                [  # point 7
                    [-0.1, 0.1, 0.0],
                ],
                [  # point 8
                    [-0.1, 0.0, 0.0],
                ],
                [  # point 9
                    [-0.1, -0.1, 0.0],
                ],
            ]
        ]
    )

    # [b, num_points, 1, 3]
    seg_pre = np.ones([1, 9, 1])  # [b, num_points, 1]
    seg_pre[0, :2, 0] = 0  # set the first two points to not be an object
    # they should be consequently ignored
    k = 5

    obj_kpts = batch_get_pt_candidates_tf(
        pcld_xyz.astype(np.float32),
        kpts_ofst_pre.astype(np.float32),
        seg_pre.astype(np.float32),
        ctr_ofst_pre.astype(np.float32),
        k,
    )

    # we expect [1, 3, 9, 3] voted keypoints
    expected_kpts = np.array(
        [  # batch
            [
                [[-0.1, -0.1, 1.0]] * k,  # 1st keypoint for each point
                [[0.1, 0.1, 1.0]] * k,  # 2nd keypoint for each point
                [[0.0, 0.0, 1.0]] * k,  # center for each point
            ]
        ]
    )
    assert_array_almost_equal(obj_kpts, expected_kpts)


# Test case 1
def test_batch_get_pt_candidates_tf_large_offset():
    """
    this batch_get_pt_candidates_tf is used in models/pprocessnet.py
    1) offsets, that are segmented as !object get an additonal length of 1000.
    2) it takes the offsets and calculats the top k shortest offsets
    3) clustering?
    4) the offsets are added to the point cloud and returned

    some points now have a large offset assigned
    but should get disregarded

    """

    # generate a point cloud with 9 points at z=1m in a grid with 0.1
    pcld_xyz = np.array(
        [  # batch
            [  # n_points
                [-0.1, -0.1, 1.0],
                [-0.1, 0.0, 1.0],
                [-0.1, 0.1, 1.0],
                [0.0, -0.1, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, -0.1, 1.0],
                [0.1, 0.0, 1.0],
                [0.1, 0.1, 1.0],
            ]
        ]
    )  # [b, num_points, 3]

    # one keypoint is at first point (-0.1, -0.1, 1.)
    # the second at (0.1, 0.1, 1.)

    # for each point we generate a offset pointing towards the center
    kpts_ofst_pre = np.array(
        [  # batch
            [  # n_points
                [  # point 1
                    [1.0, 1.0, 0.0],  # LARGE OFFSET
                    [0.2, 0.2, 0.0],  # offset to 2nd keypoint
                ],
                [  # point 2
                    [0.0, -0.1, 0.0],
                    [0.2, 0.1, 0.0],
                ],
                [  # point 3
                    [0.0, -0.2, 0.0],
                    [0.2, 0.0, 0.0],
                ],
                [  # point 4
                    [-2.1, 0.0, 0.0],  # LARGE OFFSET
                    [0.1, 3.2, 0.0],  # LARGE OFFSET
                ],
                [  # point 5
                    [-0.1, -0.1, 0.0],
                    [0.1, 0.1, 0.0],
                ],
                [  # point 6
                    [-0.1, -3.2, 0.0],  # LARGE OFFSET
                    [0.1, 0.0, 0.0],
                ],
                [  # point 7
                    [-0.2, 0.0, 0.0],
                    [2.0, 0.2, 0.0],  # LARGE OFFSET
                ],
                [  # point 8
                    [-0.2, -0.1, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                [  # point 9
                    [-0.2, -0.2, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]
        ]
    )  # [b, num_points, n_kpts, 3] n_pkts = 8

    # generate the same for the centerpoint
    ctr_ofst_pre = np.array(
        [  # batch
            [  # n_points
                [  # point 1
                    [0.1, 0.1, 0.0],  # offset center
                ],
                [  # point 2
                    [0.1, 0.0, 0.0],
                ],
                [  # point 3
                    [2.1, -0.1, 0.0],  # LARGE OFFSET
                ],
                [  # point 4
                    [0.0, 0.1, 0.0],
                ],
                [  # point 5
                    [0.0, 0.0, 0.0],
                ],
                [  # point 6
                    [0.0, -0.1, 0.0],
                ],
                [  # point 7
                    [-0.1, 3.1, 0.0],  # LARGE OFFSET
                ],
                [  # point 8
                    [-0.1, 0.0, 0.0],
                ],
                [  # point 9
                    [-0.1, -0.1, 0.0],
                ],
            ]
        ]
    )

    # [b, num_points, 1, 3]
    seg_pre = np.ones([1, 9, 1])  # [b, num_points, 1]
    k = 5

    obj_kpts = batch_get_pt_candidates_tf(
        pcld_xyz.astype(np.float32),
        kpts_ofst_pre.astype(np.float32),
        seg_pre.astype(np.float32),
        ctr_ofst_pre.astype(np.float32),
        k,
    )

    # we expect [1, 3, 9, 3] voted keypoints
    expected_kpts = np.array(
        [  # batch
            [
                [[-0.1, -0.1, 1.0]] * k,  # 1st keypoint for each point
                [[0.1, 0.1, 1.0]] * k,  # 2nd keypoint for each point
                [[0.0, 0.0, 1.0]] * k,  # center for each point
            ]
        ]
    )
    assert_array_almost_equal(obj_kpts, expected_kpts)
