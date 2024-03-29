from cvde.tf import Dataset as _Dataset

import os
import tensorflow as tf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import pathlib
import streamlit as st
import itertools as it
import open3d as o3d
import itertools as it
import simpose

"""
data in the 6IMPOSE datasets:
for each datapoint:
    - [File] rgb
    - [File] depth
    - [File] mask_visib (using instance id and visible pixels)
    - for each object:
        - [FILE] masks without occlusion mask/mask_{obj.object_id:04}_{dataset_index:04}.exr (only for labelled objects)
    - [in gt.json]:
        - "cam_rotation": [x,y,z,w] (like scipy)
        - "cam_location": [x,y,z]
        - "cam_matrix": [3,3] intrinsic matrix (openCV)
        for each object:
            - "class": str
            - "object id": int (instance id)
            - "pos": [x,y,z]
            - "rotation": [x,y,z,w] (like scipy)
            - "bbox_visib": [x1,y1,x2,y2] in pixels, bbox of the object in the image
            - "bbox_obj": [x1,y1,x2,y2] in pixels, bbox of the object in the image without occlusion
            - "px_count_visib": int, number of visible pixels for this object,
            - "px_count_valid": int, number of visible pixels with valid depth for this object,
            - "px_count_all": int, number of visible pixels without occlusions for this object
            - "visib_fract": px_count_visib / px_count_all (or 0.)
        
"""


class _SPTFRecord(_Dataset):
    def __init__(
        self,
        *,
        if_augment,
        is_train,
        cls_type,
        data_name,
        batch_size,
        root,
        add_bbox_noise: bool,
        bbox_noise: int,
    ):
        super().__init__()
        self.colormap = [
            [0, 0, 0],
            [255, 255, 0],
            [0, 0, 255],
            [240, 240, 240],
            [0, 255, 0],
            [255, 0, 50],
            [0, 255, 255],
        ]
        self.cls_type = cls_type
        self.if_augment = if_augment
        self.batch_size = batch_size
        self.is_train = is_train
        self.add_bbox_noise = add_bbox_noise
        self.bbox_noise = bbox_noise

        self.data_root = pathlib.Path(root).joinpath(data_name)

        self.spds = ds = simpose.data.Dataset
        keys = [
            ds.RGB,
            ds.DEPTH,
            ds.MASK,
            ds.CAM_LOCATION,
            ds.CAM_ROTATION,
            ds.CAM_MATRIX,
            ds.OBJ_BBOX_VISIB,
            ds.OBJ_CLASSES,
            ds.OBJ_IDS,
            ds.OBJ_POS,
            ds.OBJ_ROT,
            ds.OBJ_VISIB_FRACT,
        ]

        # if cutoff is not None:
        #     self.file_ids = self.file_ids[:cutoff]

        mesh_path = self.data_root.joinpath(f"meshes/{self.cls_type}.obj")
        if not mesh_path.exists():
            mesh_path = mesh_path.with_suffix(".ply")
        if not mesh_path.exists():
            raise ValueError(f"Mesh file {mesh_path} does not exist.")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.mesh_vertices = np.asarray(mesh.sample_points_poisson_disk(1000).points)

        mesh_kpts_path = self.data_root.parent.joinpath(f"0_kpts/{self.cls_type}")
        if not mesh_kpts_path.exists():
            mesh_kpts_path.mkdir(parents=True)
            print("Generating mesh keypoints...")
            print("Make sure to use the correct keypoints!")

            center_point = mesh.get_center()
            if mesh_kpts_path.joinpath("center.txt").exists():
                print("WARNING: WOULD OVERWRITE KEYPOINTS")
            else:
                np.savetxt(mesh_kpts_path / "center.txt", center_point)

            mesh_kpts = np.asarray(mesh.sample_points_poisson_disk(8).points)
            if mesh_kpts_path.joinpath("farthest.txt").exists():
                print("WARNING: WOULD OVERWRITE KEYPOINTS")
            else:
                np.savetxt(mesh_kpts_path / "farthest.txt", mesh_kpts)

        kpts = np.loadtxt(mesh_kpts_path / "farthest.txt")
        center = [np.loadtxt(mesh_kpts_path / "center.txt")]

        self.mesh_kpts = np.concatenate([kpts, center], axis=0)
        self.mesh_kpts_tf = tf.constant(self.mesh_kpts, dtype=tf.float32)

        # num parallel files: restrict open files
        self._tfds = simpose.data.TFRecordDataset.get(
            self.data_root, get_keys=keys, num_parallel_files=1
        )

        # TODO augment data for sim2real transfer

        if self.add_bbox_noise:

            def add_noise(*, obj_bbox_visib, **data):
                obj_bbox_visib = obj_bbox_visib + tf.random.uniform(
                    tf.shape(obj_bbox_visib), -self.bbox_noise, self.bbox_noise, dtype=tf.int32
                )
                return {**data, "obj_bbox_visib": obj_bbox_visib}

            self._tfds = self._tfds.map(
                lambda data: add_noise(**data),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )

        self._tfds = self._tfds.interleave(
            lambda data: self.extract_crops_and_gt(
                **data, cls_type=self.cls_type, mesh_kpts_tf=self.mesh_kpts_tf
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        ).shuffle(100)
        self._tfds_iter = iter(self._tfds)

        print("Initialized 6IMPOSE Dataset.")
        # print(f"\t# of all images: {total_n_imgs}")
        print(f"\tCls root: {self.data_root}")
        # print(f"\t# of images for this split: {len(self.file_ids)}")
        print(f"\t# of augmented datapoints: {len(self)}")
        # print(f"\nIntrinsic matrix: {self.intrinsic_matrix}")
        print()

    def to_tf_dataset(self):
        def arrange_as_xy_tuple(d):
            return (d["rgb"], d["depth"], d["intrinsics"], d["roi"], d["mesh_kpts"]), (
                d["RT"],
                d["mask"],
            )

        return (
            self._tfds.map(
                arrange_as_xy_tuple, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
            )
            .batch(self.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

    def visualize_example(self, example):
        color_depth = lambda x: cv2.applyColorMap(
            cv2.convertScaleAbs(x, alpha=255 / 2), cv2.COLORMAP_JET
        )

        rgb = example["rgb"]
        depth = example["depth"]

        intrinsics = example["intrinsics"].astype(np.float32)
        bboxes = example["roi"]
        kpts = example["mesh_kpts"]

        RT = example["RT"]
        mask = example["mask"]

        (
            y1,
            x1,
            y2,
            x2,
        ) = bboxes[:4]
        out_rgb = cv2.rectangle(rgb.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
        rvec = cv2.Rodrigues(RT[:3, :3])[0]
        tvec = RT[:3, 3]
        cv2.drawFrameAxes(out_rgb, intrinsics, np.zeros((4,)), rvec=rvec, tvec=tvec, length=0.1)

        c1, c2 = st.columns(2)
        c1.image(out_rgb, caption=f"RGB_L {rgb.shape} ({rgb.dtype})")
        c1.image(
            color_depth(depth),
            caption=f"Depth {depth.shape} ({depth.dtype})",
        )
        c1.image(mask * 255, caption=f"Mask {mask.shape} ({mask.dtype})")

        c2.write(intrinsics)
        c2.write(kpts)
        c2.write(RT)

        from losses.pvn_loss import PvnLoss
        from models.pvn3d_e2e import PVN3D_E2E

        num_samples = st.select_slider("num_samples", [2**i for i in range(5, 13)])
        margin = st.slider("margin", 0, 200, 0, step=50)

        h, w = rgb.shape[:2]
        _bbox, _crop_factor = PVN3D_E2E.get_crop_index(
            bboxes[None], h, w, 160, 160
        )  # bbox: [b, 4], crop_factor: [b]

        # xyz: [b, num_points, 3]
        # inds:  # [b, num_points, 3] with last 3 is index into original image
        xyz, feats, inds = PVN3D_E2E.pcld_processor_tf(
            (rgb[None] / 255.0).astype(np.float32),
            depth[None].astype(np.float32),
            intrinsics[None].astype(np.float32),
            _bbox,
            num_samples,
        )
        mask_selected = tf.gather_nd(mask[None], inds)
        kp_offsets, cp_offsets = PvnLoss.get_offst(
            RT[None].astype(np.float32),
            xyz,
            mask_selected,
            self.mesh_kpts[None].astype(np.float32),
        )
        # [1, n_pts, 8, 3] | [1, n_pts, 1, 3]

        all_offsets = np.concatenate([kp_offsets, cp_offsets], axis=-2)  # [b, n_pts, 9, 3]

        offset_views = {}
        cam_cx, cam_cy = intrinsics[0, 2], intrinsics[1, 2]  # [b]
        cam_fx, cam_fy = intrinsics[0, 0], intrinsics[1, 1]  # [b]

        def to_image(pts):
            coors = (
                pts[..., :2] / pts[..., 2:] * tf.stack([cam_fx, cam_fy], axis=0)[tf.newaxis, :]
                + tf.stack([cam_cx, cam_cy], axis=0)[tf.newaxis, :]
            )
            coors = tf.floor(coors)
            return tf.concat([coors, pts[..., 2:]], axis=-1).numpy()

        projected_keypoints = self.mesh_kpts @ RT[:3, :3].T + RT[:3, 3]
        projected_keypoints = to_image(projected_keypoints)

        # for each pcd point add the offset
        keypoints_from_pcd = xyz[:, :, None, :].numpy() + all_offsets  # [b, n_pts, 9, 3]
        keypoints_from_pcd = to_image(keypoints_from_pcd.astype(np.float32))
        projected_pcd = to_image(xyz)  # [b, n_pts, 3]

        for i in range(9):
            # offset_view = np.zeros_like(rgb, dtype=np.uint8)
            offset_view = rgb.copy() // 3

            # get color hue from offset
            hue = np.arctan2(all_offsets[0, :, i, 1], all_offsets[0, :, i, 0]) / np.pi
            hue = (hue + 1) / 2
            hue = (hue * 180).astype(np.uint8)
            # value = np.ones_like(hue) * 255
            value = (np.linalg.norm(all_offsets[0, :, i, :], axis=-1) / 0.1 * 255).astype(np.uint8)
            hsv = np.stack([hue, np.ones_like(hue) * 255, value], axis=-1)
            colored_offset = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB).astype(np.uint8)

            sorted_inds = np.argsort(np.linalg.norm(all_offsets[0, :, i, :], axis=-1), axis=-1)[
                ::-1
            ]
            keypoints_from_pcd[0, :, i, :] = keypoints_from_pcd[0, sorted_inds, i, :]
            colored_offset[0] = colored_offset[0, sorted_inds, :]
            sorted_xyz = projected_pcd[0, sorted_inds, :]
            for start, target, color in zip(
                sorted_xyz, keypoints_from_pcd[0, :, i, :], colored_offset[0]
            ):
                # over all pcd points
                cv2.line(
                    offset_view,
                    tuple(map(int, start[:2])),
                    tuple(map(int, target[:2])),
                    tuple(map(int, color)),
                    1,
                )

            # # mark correct keypoint
            cv2.drawMarker(
                offset_view,
                (int(projected_keypoints[i, 0]), int(projected_keypoints[i, 1])),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=1,
            )

            h, w = offset_view.shape[:2]
            y1, x1, y2, x2 = _bbox[0]
            x1 = np.clip(x1 - margin, 0, w)
            x2 = np.clip(x2 + margin, 0, w)
            y1 = np.clip(y1 - margin, 0, h)
            y2 = np.clip(y2 + margin, 0, h)
            offset_view = offset_view[y1:y2, x1:x2]
            name = f"Keypoint {i}" if i < 8 else "Center"
            offset_views.update({name: offset_view})

        cols = it.cycle(st.columns(3))

        for col, (name, offset_view) in zip(cols, offset_views.items()):
            col.image(offset_view, caption=name)

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        # WE IGNORE INDICES AND JUST ITERATE THROUGH THE DATASET
        data = next(self._tfds_iter)
        return {
            "rgb": data["rgb"].numpy().astype(np.uint8),
            "depth": data["depth"].numpy().astype(np.float32),
            "intrinsics": data["intrinsics"].numpy().astype(np.float32),
            "roi": data["roi"].numpy().astype(np.int32),
            "RT": data["RT"].numpy().astype(np.float32),
            "mask": data["mask"].numpy().astype(np.uint8),
            "mesh_kpts": self.mesh_kpts,
        }

    @staticmethod
    @tf.function
    def quat_to_matrix(quat):
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        matrix = tf.stack(
            (
                1.0 - (tyy + tzz),
                txy - twz,
                txz + twy,
                txy + twz,
                1.0 - (txx + tzz),
                tyz - twx,
                txz - twy,
                tyz + twx,
                1.0 - (txx + tyy),
            ),
            axis=-1,
        )  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=quat)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)

    @staticmethod
    @tf.function
    def extract_crops_and_gt(
        *,
        rgb,
        depth,
        mask,
        cam_location,
        cam_rotation,
        cam_matrix,
        obj_bbox_visib,
        obj_classes,
        obj_ids,
        obj_pos,
        obj_rot,
        obj_visib_fract,
        cls_type,
        mesh_kpts_tf,
    ):
        depth = depth[..., tf.newaxis]
        mask = mask[..., tf.newaxis]
        cam_rot_matrix = _SPTFRecord.quat_to_matrix(cam_rotation)
        cam_location = cam_location[..., tf.newaxis]
        cam_rot_matrix = tf.transpose(cam_rot_matrix)
        h, w = tf.shape(rgb)[0], tf.shape(rgb)[1]

        @tf.function
        def is_chosen_object(data):
            return data["obj_classes"] == cls_type

        @tf.function
        def is_valid_box(data):
            # here bbox is still in x1,y1,x2,y2 format
            bbox = data["obj_bbox_visib"]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            not_at_border = bbox[0] > 0 and bbox[1] > 0 and bbox[2] < w - 1 and bbox[3] < h - 1
            return bbox_w > 39 and bbox_h > 39 and not_at_border and data["obj_visib_fract"] > 0.3

        @tf.function
        def assemble(data):
            # convert bbox to y1,x1,y2,x2
            bbox = data["obj_bbox_visib"]
            bbox_permuted = tf.stack([bbox[1], bbox[0], bbox[3], bbox[2]])

            # calculate RT
            obj_rot_mat = _SPTFRecord.quat_to_matrix(data["obj_rot"])
            rot_matrix = cam_rot_matrix @ obj_rot_mat  # (3,3)
            translation = cam_rot_matrix @ (
                data["obj_pos"][..., tf.newaxis] - cam_location
            )  # (3,1)
            RT = tf.concat([rot_matrix, translation], axis=-1)  # (3,4)
            RT = tf.concat((RT, tf.constant([[0, 0, 0, 1]], dtype=tf.float32)), axis=0)  # (4,4)

            obj_mask = tf.where(mask == tf.cast(data["obj_ids"], tf.uint8), 1, 0)

            return {
                "rgb": rgb,
                "depth": depth,
                "intrinsics": cam_matrix,
                "roi": bbox_permuted,
                "RT": RT,
                "mask": obj_mask,
                "mesh_kpts": mesh_kpts_tf,
            }

        return (
            tf.data.Dataset.from_tensor_slices(
                {
                    "obj_classes": obj_classes,
                    "obj_bbox_visib": obj_bbox_visib,
                    "obj_visib_fract": obj_visib_fract,
                    "obj_pos": obj_pos,
                    "obj_rot": obj_rot,
                    "obj_ids": obj_ids,
                }
            )
            .filter(is_chosen_object)
            .filter(is_valid_box)
            .map(assemble, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        )


class TrainSPTFRecord(_SPTFRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=True, is_train=True, **kwargs)


class ValSPTFRecord(_SPTFRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=False, is_train=False, **kwargs)
