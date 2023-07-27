from PVN3D.models.pvn3d_e2e import PVN3D_E2E
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import cv2
import open3d as o3d


def test_crop_index():
    """
    crop index gets a integer crop factor and a bounding box with the same aspect ratio
    with batches
    """
    resnet_input_shape = [160, 160]  # hxw
    img_h = np.array(1080)
    img_w = np.array(1920)
    roi = np.array(
        [
            [0, 0, 160, 160],  # simplest case
            [0, 0, 320, 320],
            [0, 0, 80, 80],
            [200, 200, 680, 840],  # box with 480x640
            [0, 0, 480, 640],  # same box in top left corner
            [600, 1280, 1080, 1920],  # same box in bottom right corner
        ]
    )
    expected_bbox = np.array(
        [
            [0, 0, 160, 160],
            [0, 0, 320, 320],
            [0, 0, 160, 160],
            [120, 200, 760, 840],  # should be size 640x640
            [0, 0, 640, 640],  # should be size 640x640, but in corner
            [440, 1280, 1080, 1920],  # other corner
        ]
    )
    expected_crop_factor = np.array(
        [
            1,
            2,
            1,
            4,
            4,
            4,
        ]
    )  # 640 / 160 == 4

    for roi_, exp_bbox_, exp_crop_factor_ in zip(
        roi, expected_bbox, expected_crop_factor
    ):
        roi_ = roi_[None, :]
        exp_bbox_ = exp_bbox_[None, :]
        exp_crop_factor_ = exp_crop_factor_[None]
        bbox, crop_factor = PVN3D_E2E.get_crop_index(
            roi_, img_h, img_w, resnet_input_shape[0], resnet_input_shape[1]
        )
        assert_array_equal(
            bbox,
            exp_bbox_,
            err_msg=f"bbox size: ({bbox[:,2]-bbox[:,0]}, {bbox[:,3]-bbox[:,1]})",
        )
        assert_array_equal(crop_factor, exp_crop_factor_)


def test_index_transform():
    img_h = 1080
    img_w = 1920
    # theses boxes are from crop_index, so they are in the same aspect ratio with integer crop factor
    bbox = np.array(
        [
            [0, 0, 160, 160],
            [0, 0, 320, 320],
            [0, 0, 320, 320]
        ]
    )
    crop_factor = np.array(
        [
            1,
            2,
            2,
        ]
    )

    #  [b, num_points, 3], last dimensions contains inds in [batch, h, w]
    sampled_inds_in_original_image = np.array(
        [  # batch
            [  # list of points
                [0, 0, 0],  # top left corner
                [0, 79, 79],
                [0, 159, 159],  # bottom right corner
            ],
            [
                [1, 0, 0],  # top left corner
                [1, 159, 159],  # center point
                [1, 319, 319],  # bottom right corner
            ],
            [
                [2, 0, 0],  # top left corner
                [2, 159, 159],  # center point
                [2, 319, 319],  # bottom right corner
            ],
        ]
    )

    sampled_inds_in_roi = PVN3D_E2E.transform_indices_from_full_image_cropped(
        sampled_inds_in_original_image, bbox, crop_factor
    )

    expected_sampled_inds_in_roi = np.array(
        [  # batch
            [  # list of sampled indices
                [0, 0, 0],
                [0, 79, 79],
                [0, 159, 159],
            ],
            [
                [1, 0, 0],
                [1, 79, 79],  # center point
                [1, 159, 159],  # still bottom right corner of smaller bounding box
            ],
            [
                [2, 0, 0],
                [2, 79, 79],  # center point
                [2, 159, 159],  # still bottom right corner of smaller bounding box
            ],
        ]
    )

    assert_array_equal(sampled_inds_in_roi, expected_sampled_inds_in_roi)


def test_pcld_processor_tf():
    np.random.seed(0)
    rgb = np.random.uniform(size=(1, 1080, 1920, 3), low=0, high=255).astype(np.uint8)
    depth = np.random.uniform(size=(1, 1080, 1920, 1), low=0, high=5.0).astype(
        np.float32
    )
    # intrinsics of realsense d415 for 1920x1080
    camera_matrix = np.array(
        [
            [
                [1.0788e03, 0.0, 9.6000e02],
                [0.0, 1.0788e03, 5.4000e02],
                [0.0, 0.0, 1.0],
            ],
        ]
    )
    roi = np.array(
        [
            [0, 0, 160, 160],  # simplest case
        ]
    )
    n_sample = 10
    depth_trunc = 2.0
    xyz, feats, inds = PVN3D_E2E.pcld_processor_tf(
        (rgb / 255).astype(np.float32),
        depth.astype(np.float32),
        camera_matrix.astype(np.float32),
        roi,
        n_sample,
        depth_trunc,
    )

    # feats is [..., :3] == rgb # normalized from 0 to 1
    #          [..., 3:] == normals

    # use open3d as reference
    img_depth = o3d.geometry.Image(depth[0])
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1920, height=1080, fx=1.0788e03, fy=1.0788e03, cx=9.6000e02, cy=5.4000e02
    )
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        img_depth,
        pinhole_camera_intrinsic,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        project_valid_depth_only=False,
    )

    # convert sampled_indices to flattened indices
    sampled_indices = inds[0]
    sampled_indices = sampled_indices[:, 1] * 1920 + sampled_indices[:, 2]

    # sample points from pcd
    pcd_points = np.asarray(pcd.points)[sampled_indices]
    pcd_colors = (rgb / 255).astype(np.float32).reshape(-1, 3)[sampled_indices]

    assert_array_almost_equal(xyz[0], pcd_points)
    assert_array_almost_equal(feats[0, :, :3], pcd_colors)


def test_pvn3d_ee_normals():
    np.random.seed(0)
    # depth = np.random.uniform(size=(1, 1080, 1920, 1), low=0, high=5.0).astype(
    #     np.float32
    # )
    depth = np.ones(shape=(1, 1080, 1920, 1), dtype=np.float32)
    depth += 0.1 * np.sin(np.arange(1920)[None, None, :, None] / 1920.0 * 2 * np.pi)
    depth += 0.05 * np.sin(
        0.5 * np.arange(1080)[None, :, None, None] / 1080.0 * 2 * np.pi
    )

    # intrinsics of realsense d415 for 1920x1080
    camera_matrix = np.array(
        [
            [
                [1.0788e03, 0.0, 9.6000e02],
                [0.0, 1.0788e03, 5.4000e02],
                [0.0, 0.0, 1.0],
            ],
        ]
    ).astype(np.float32)

    normal_map = (
        PVN3D_E2E.compute_normal_map(depth, camera_matrix).numpy()[0].astype(np.float32)
    )

    img_depth = o3d.geometry.Image(depth[0])
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1920, height=1080, fx=1.0788e03, fy=1.0788e03, cx=9.6000e02, cy=5.4000e02
    )

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        img_depth,
        pinhole_camera_intrinsic,
        depth_scale=1.0,
        depth_trunc=20000.0,
        project_valid_depth_only=False,
    )
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=4))
    pcd.orient_normals_towards_camera_location()

    o3d_normal_map = np.asarray(pcd.normals).reshape(1080, 1920, 3).astype(np.float32)

    def vis_normals(nm_map):
        nm_map = (nm_map + 1) / 2
        nm_map = (nm_map * 255).astype(np.uint8)
        return nm_map

    cv2.imwrite("test.png", vis_normals(normal_map))
    cv2.imwrite("o3d.png", vis_normals(o3d_normal_map))

    # using very low precision, since it is a different algorithm
    # our algorithm produces 0 at corners
    assert_array_almost_equal(
        normal_map[50:-50:100, 50:-50:100],
        o3d_normal_map[50:-50:100, 50:-50:100],
        decimal=1,  # equal +- 1.5*10**(-decimal)
    )
