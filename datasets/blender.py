import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image
import cv2
import json
from scipy.spatial.transform import Rotation as R
import pathlib
import albumentations as A
import tensorflow as tf
from cvde.tf import Dataset as _Dataset
import streamlit as st
from functools import cached_property


class _Blender(_Dataset):
    cls_dict = {
        "cpsduck": 1,
        "stapler": 2,
        "cpsglue": 3,
        "wrench_13": 4,
        "chew_toy": 5,
        "pliers": 6,
        "all": 100,  # hacked to cpsduck for now
    }
    colormap = [
        [0, 0, 0],
        [255, 255, 0],
        [0, 0, 255],
        [240, 240, 240],
        [0, 255, 0],
        [255, 0, 50],
        [0, 255, 255],
    ]
    labelmap = {v: k for k, v in cls_dict.items()}

    def __init__(
        self,
        *,
        data_name,
        if_augment,
        is_train,
        cls_type,
        im_size,
        batch_size,
        use_cache,
        root,
        train_split,
        cutoff = None,
    ):
        self.cls_type = cls_type
        self.cls_id = self.cls_dict[self.cls_type]
        self.current_cls_root = 0
        self.if_augment = if_augment
        self._current = 0
        self.im_size = im_size
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.is_train = is_train

        data_root = pathlib.Path(root) / data_name

        self.kpts_root = data_root / "kpts"
        if self.cls_type != "all":
            self.cls_root = data_root / f"{self.cls_id:02}"
            self.roots_and_ids = self.get_roots_and_ids_for_cls_root(self.cls_root)

        else:
            self.all_cls_roots = [data_root / x for x in data_root.iterdir()]
            self.cls_root = self.all_cls_roots[0]
            all_roots_and_ids = [
                self.get_roots_and_ids_for_cls_root(x) for x in self.all_cls_roots
            ]
            self.roots_and_ids = []
            [self.roots_and_ids.extend(x) for x in zip(*all_roots_and_ids)]

        if cutoff is not None:
            self.roots_and_ids = self.roots_and_ids[:cutoff]

        total_n_imgs = len(self.roots_and_ids)

        split_ind = np.floor(len(self.roots_and_ids) * train_split).astype(
            int
        )
        if is_train:
            self.roots_and_ids = self.roots_and_ids[:split_ind]
        else:
            self.roots_and_ids = self.roots_and_ids[split_ind:]

        with open(os.path.join(self.cls_root, "gt.json")) as f:
            json_dict = json.load(f)
        self.intrinsic_matrix = np.array(json_dict["camera_matrix"])
        self.baseline = np.array(json_dict["stereo_baseline"])

        self.rgbmask_augment = A.Compose(
            [
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5
                ),
                A.RandomGamma(p=0.2),
                A.AdvancedBlur(p=0.2),
                A.GaussNoise(p=0.2),
                A.FancyPCA(p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            ],
        )

        print("Initialized Blender Dataset.")
        print(f"\tData name: {data_name}")
        print(f"\tTotal split # of images: {len(self.roots_and_ids)}")
        print(f"\tCls root: {self.cls_root}")
        print(f"\t# of all images: {total_n_imgs}")
        print()




    def to_tf_dataset(self):
        def generator():
            for i in range(len(self.roots_and_ids)):
                yield self[i]



        tfdata = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                    (
                    tf.TensorSpec(
                        shape=(*self.im_size, 3), dtype=tf.uint8, name="rgb"
                    ),
                    tf.TensorSpec(
                        shape=(*self.im_size, 1), dtype=tf.float32, name="depth"
                    ),
                    tf.TensorSpec(shape=(3,3), dtype=tf.float32, name="intrinsics"),
                    tf.TensorSpec(shape=(4,), dtype=tf.int32, name="roi"),
                    tf.TensorSpec(shape=(9,3), dtype=tf.float32, name="mesh_kpts"),
                ),
                (tf.TensorSpec(shape=(4,4), dtype=tf.float32, name="RT"),
                tf.TensorSpec(shape=(*self.im_size, 1), dtype=tf.uint8, name="mask")
                )
            )
        )
        if self.use_cache:
            h = hash(str(self.cls_root)+str(self.cls_id)+self.cls_type+str(self.if_augment)+str(self.is_train))
            tfdata = tfdata.cache(f'cached_data_{h}')

        tfdata = tfdata.batch(self.batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE
        )
        return tfdata

    def get_roots_and_ids_for_cls_root(self, cls_root: pathlib.Path):
        numeric_file_ids = (cls_root / "rgb").glob("*")
        numeric_file_ids = [x for x in numeric_file_ids if "_R" not in str(x)]
        numeric_file_ids = list([int(x.stem.split("_")[1]) for x in numeric_file_ids])
        numeric_file_ids.sort()
        return [(cls_root, id) for id in numeric_file_ids]

    def color_seg(self, data):
        flattened = data.reshape((-1, 1)).astype(int)
        out = np.array(self.colormap)[flattened].astype(np.uint8)
        return out.reshape((*data.shape[:2], 3))

    def visualize_example(self, example):
        color_depth = lambda x: cv2.applyColorMap(
            cv2.convertScaleAbs(x, alpha=255 / 2), cv2.COLORMAP_JET
        )

        rgb = example[0][0]
        depth = example[0][1]
        intrinsics = example[0][2]
        bboxes = example[0][3]
        kpts = example[0][4]

        RT = example[1][0]
        mask = example[1][1]

        y1, x1, y2, x2,= bboxes[:4]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        rvec = cv2.Rodrigues(RT[:3, :3])[0]
        tvec = RT[:3, 3]
        cv2.drawFrameAxes(rgb, intrinsics, np.zeros((4,)), rvec=rvec, tvec=tvec, length=0.1)

        c1, c2 = st.columns(2)
        c1.image(rgb, caption=f"RGB_L {rgb.shape} ({rgb.dtype})")
        c1.image(
            color_depth(depth),
            caption=f"Depth {depth.shape} ({depth.dtype})",
        )
        c1.image(mask*255, caption=f"Mask {mask.shape} ({mask.dtype})")
        c2.write(intrinsics)
        c2.write(kpts)
        c2.write(RT)


    def __next__(self):
        data = self[self._current]
        self._current += 1
        return data

    def __len__(self):
        return len(self.roots_and_ids) // self.batch_size

    def __getitem__(self, dataset_ind):
        self.cls_root, i = self.roots_and_ids[dataset_ind]

        rgb = self.get_rgb(i)
        mask = self.get_mask(i)
        depth = self.get_depth(i)
        bboxes = self.get_gt_bbox(i, mask=mask)[0] # FOR SINGLE OBJECT
        rt =self.get_RT_list(i)[0] # FOR SINGLE OBJECT

        mask = np.where(mask==self.cls_id, 1, 0).astype(np.uint8)   

        # TODO augment depth

        if self.if_augment:
            res = self.rgbmask_augment(image=rgb, mask=mask)
            rgb = res["image"]
            mask = res["mask"]

        out = (rgb, depth, self.intrinsic_matrix, bboxes[:4], self.mesh_kpts), (rt,mask) # get rid of cls_id in bboxes
        return out

    def get_rgb(self, index):
        rgb_path = os.path.join(self.cls_root, "rgb", f"rgb_{index:04}.png")
        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(np.uint8)
        return rgb

    def get_mask(self, index):
        mask_path = os.path.join(self.cls_root, "mask", f"segmentation_{index:04}.exr")
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        mask = mask[:, :, :1]
        return mask  # .astype(np.uint8)

    def get_depth(self, index):
        depth_path = os.path.join(self.cls_root, "depth", f"depth_{index:04}.exr")
        dpt = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        dpt = dpt[:, :, :1]
        dpt_mask = dpt < 5  # in meters, we filter out the background( > 5m)
        dpt = dpt * dpt_mask
        return dpt

    @cached_property
    def mesh_kpts(self):
        mesh_kpts_path = self.kpts_root / self.cls_type
        kpts = np.loadtxt(mesh_kpts_path / "farthest.txt")
        center = [np.loadtxt(mesh_kpts_path / "center.txt")]
        kpts_cpts = np.concatenate([kpts, center], axis=0)
        return kpts_cpts

    def get_RT_list(self, index):
        """return a list of tuples of RT matrix and cls_id [(RT_0, cls_id_0), (RT_1,, cls_id_1) ..., (RT_N, cls_id_N)]"""
        with open(os.path.join(self.cls_root, "gt", f"gt_{index:05}.json")) as f:
            shot = json.load(f)

        cam_quat = shot["cam_rotation"]
        cam_rot = R.from_quat([*cam_quat[1:], cam_quat[0]])
        cam_pos = np.array(shot["cam_location"])
        cam_Rt = np.eye(4)
        cam_Rt[:3, :3] = cam_rot.as_matrix().T
        cam_Rt[:3, 3] = -cam_rot.as_matrix() @ cam_pos

        objs = shot["objs"]

        RT_list = []

        if self.cls_type == "all":
            for obj in objs:
                cls_type = obj["name"]
                cls_id = self.cls_dict[cls_type]
                pos = np.array(obj["pos"])
                quat = obj["rotation"]
                rot = R.from_quat([*quat[1:], quat[0]])
                Rt = np.eye(4)
                Rt[:3, :3] = cam_rot.as_matrix().T @ rot.as_matrix()
                Rt[:3, 3] = cam_rot.as_matrix().T @ (pos - cam_pos)
                #RT_list.append((Rt, cls_id))

        else:
            for obj in objs:  # here we only consider the single obj
                if obj["name"] == self.cls_type:
                    cls_type = obj["name"]
                    pos = np.array(obj["pos"])
                    quat = obj["rotation"]
                    rot = R.from_quat([*quat[1:], quat[0]])
                    Rt = np.eye(4)
                    Rt[:3, :3] = cam_rot.as_matrix().T @ rot.as_matrix()
                    Rt[:3, 3] = cam_rot.as_matrix().T @ (pos - cam_pos)
                    #RT_list.append((Rt, self.cls_id))
                    RT_list.append(Rt)
        return RT_list

    def get_gt_bbox(self, index, mask=None) -> np.ndarray:
        bboxes = []
        if mask is None:
            mask = self.get_mask(index)
        if self.cls_type == "all":
            for cls, gt_mask_value in self.cls_dict.items():
                bbox = self.get_bbox_from_mask(mask, gt_mask_value)
                if bbox is None:
                    continue
                bbox = list(bbox)
                bbox.append(self.cls_dict[cls])
                bboxes.append(bbox)
        else:
            bbox = self.get_bbox_from_mask(mask, gt_mask_value=self.cls_id)
            bbox = list(bbox)
            bbox.append(self.cls_id)
            bboxes.append(bbox)

        return np.array(bboxes).astype(np.int32)

    def get_bbox_from_mask(self, mask, gt_mask_value):
        """mask with object as 255 -> bbox [x1,y1, x2, y2]"""

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        y, x = np.where(mask == gt_mask_value)
        inds = np.stack([x, y])
        if 0 in inds.shape:
            return None
        x1, y1 = np.min(inds, 1)
        x2, y2 = np.max(inds, 1)

        if (x2 - x1) * (y2 - y1) < 1600:
            return None

        return (y1, x1, y2, x2)


class TrainBlender(_Blender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=True, is_train=True, **kwargs)


class ValBlender(_Blender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, if_augment=False, is_train=False, **kwargs)
