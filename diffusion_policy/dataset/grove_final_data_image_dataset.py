from typing import Dict, List, Optional, Tuple

import copy
import hashlib
import json
import os

import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _sorted_demo_dirs(dataset_path: str, episode_prefix: str) -> List[str]:
    root = os.path.expanduser(dataset_path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset_path is not a directory: {root}")
    dirs = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if name.startswith(episode_prefix) and os.path.isdir(full):
            dirs.append(full)

    def key_fn(path: str):
        base = os.path.basename(path)
        try:
            return int(base.split("_")[-1])
        except Exception:
            return base

    return sorted(dirs, key=key_fn)


def _resize_rgb(frames: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = out_hw
    if tuple(frames.shape[1:3]) == (oh, ow):
        return frames
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("opencv-python is required to resize RGB final_data frames.") from e
    out = np.empty((frames.shape[0], oh, ow, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (ow, oh), interpolation=cv2.INTER_AREA)
    return out


def _load_image(
    ep_dir: str,
    image_mode: str,
    out_hw: Tuple[int, int],
    rgb_file: str,
    cam4_file: str,
) -> np.ndarray:
    if image_mode == "rgb":
        rgb = np.load(os.path.join(ep_dir, rgb_file), mmap_mode="r")
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB (T,H,W,3), got {rgb.shape} in {ep_dir}")
        return _resize_rgb(np.asarray(rgb, dtype=np.uint8), out_hw)
    if image_mode == "cam4":
        cam4 = np.load(os.path.join(ep_dir, cam4_file), mmap_mode="r")
        if cam4.ndim != 4:
            raise ValueError(f"Expected cam4 4D array, got {cam4.shape} in {ep_dir}")
        if cam4.shape[1] == 4:
            cam4 = np.moveaxis(cam4, 1, -1)
        if cam4.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels in cam4 array, got {cam4.shape} in {ep_dir}")
        if tuple(cam4.shape[1:3]) != out_hw:
            raise ValueError(f"Expected cam4 spatial size {out_hw}, got {cam4.shape[1:3]} in {ep_dir}")
        return np.clip(np.asarray(cam4, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image_mode: {image_mode}")


def _convert_final_data_to_replay(
    *,
    store,
    dataset_path: str,
    shape_meta: dict,
    episode_prefix: str,
    image_mode: str,
    rgb_file: str,
    cam4_file: str,
    robot_states_file: str,
    eef_slice: Tuple[int, int],
    alignment_only: bool,
    alignment_end_file: str,
    max_episodes: Optional[int],
) -> ReplayBuffer:
    obs_meta = shape_meta["obs"]
    rgb_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb"]
    lowdim_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "low_dim"]
    if len(rgb_keys) != 1:
        raise ValueError(f"Expected exactly one image obs key, got {rgb_keys}")
    image_key = rgb_keys[0]
    channels, height, width = tuple(obs_meta[image_key]["shape"])
    expected_channels = 3 if image_mode == "rgb" else 4
    if channels != expected_channels:
        raise ValueError(f"{image_mode} expects {expected_channels} channels, got shape {obs_meta[image_key]['shape']}")

    demo_dirs = _sorted_demo_dirs(dataset_path, episode_prefix)
    if max_episodes is not None:
        demo_dirs = demo_dirs[: int(max_episodes)]
    if not demo_dirs:
        raise FileNotFoundError(f"No demos found under {dataset_path} with prefix={episode_prefix}")

    replay = ReplayBuffer.create_empty_zarr(storage=store)
    eef_start, eef_stop = eef_slice

    with threadpool_limits(1):
        for ep_dir in demo_dirs:
            image = _load_image(
                ep_dir=ep_dir,
                image_mode=image_mode,
                out_hw=(height, width),
                rgb_file=rgb_file,
                cam4_file=cam4_file,
            )
            states = np.load(os.path.join(ep_dir, robot_states_file), mmap_mode="r")
            if states.ndim != 2 or states.shape[1] < eef_stop:
                raise ValueError(f"Expected robot_states with at least {eef_stop} columns, got {states.shape} in {ep_dir}")

            if alignment_only:
                align_path = os.path.join(ep_dir, alignment_end_file)
                if not os.path.exists(align_path):
                    raise FileNotFoundError(f"alignment_only=True but missing {alignment_end_file} in {ep_dir}")
                end_idx = int(np.load(align_path))
                if end_idx < 0 or end_idx >= image.shape[0] or end_idx >= states.shape[0]:
                    raise ValueError(
                        f"Invalid alignment end index {end_idx} for image T={image.shape[0]}, "
                        f"state T={states.shape[0]} in {ep_dir}"
                    )
                align_slice = slice(0, end_idx + 1)
                image = image[align_slice]
                states = states[align_slice]

            eef_pose = np.asarray(states[:, eef_start:eef_stop], dtype=np.float32)
            if image.shape[0] != eef_pose.shape[0]:
                raise ValueError(f"Length mismatch in {ep_dir}: image T={image.shape[0]}, eef T={eef_pose.shape[0]}")

            episode: Dict[str, np.ndarray] = {
                image_key: image,
                "action": eef_pose,
            }
            for key in lowdim_keys:
                if key in ("eef_pose", "robot_eef_pose"):
                    episode[key] = eef_pose
                else:
                    raise KeyError(f"Unsupported low_dim key for final_data: {key}")
            replay.add_episode(episode, compressors="disk")

    return replay


class GroveFinalDataImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        n_latency_steps: int = 0,
        use_cache: bool = True,
        seed: int = 42,
        val_ratio: float = 0.05,
        max_train_episodes: Optional[int] = None,
        max_episodes: Optional[int] = None,
        episode_prefix: str = "demo_task_full_",
        image_mode: str = "rgb",
        rgb_file: str = "head_camera_rgb.npy",
        cam4_file: str = "head_camera_cam4.npy",
        robot_states_file: str = "robot_states.npy",
        eef_slice: Tuple[int, int] = (7, 14),
        alignment_only: bool = False,
        alignment_end_file: str = "alignment_end_idx.npy",
    ):
        dataset_path = os.path.expanduser(dataset_path)
        assert os.path.isdir(dataset_path), f"Not a directory: {dataset_path}"

        obs_meta = shape_meta["obs"]
        self.rgb_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb"]
        self.lowdim_keys = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "low_dim"]

        shape_meta_container = OmegaConf.to_container(shape_meta, resolve=True)
        fp_src = json.dumps(
            {
                "shape_meta": shape_meta_container,
                "image_mode": image_mode,
                "episode_prefix": episode_prefix,
                "eef_slice": list(eef_slice),
                "alignment_only": alignment_only,
                "alignment_end_file": alignment_end_file,
                "max_episodes": max_episodes,
                "max_train_episodes": max_train_episodes,
            },
            sort_keys=True,
        )
        fingerprint = hashlib.md5(fp_src.encode("utf-8")).hexdigest()
        cache_zarr_path = os.path.join(dataset_path, f"final_data_{image_mode}_cache_{fingerprint}.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"

        if use_cache:
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    replay_buffer = _convert_final_data_to_replay(
                        store=zarr.MemoryStore(),
                        dataset_path=dataset_path,
                        shape_meta=shape_meta,
                        episode_prefix=episode_prefix,
                        image_mode=image_mode,
                        rgb_file=rgb_file,
                        cam4_file=cam4_file,
                        robot_states_file=robot_states_file,
                        eef_slice=eef_slice,
                        alignment_only=alignment_only,
                        alignment_end_file=alignment_end_file,
                        max_episodes=max_episodes,
                    )
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                else:
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
        else:
            replay_buffer = _convert_final_data_to_replay(
                store=zarr.MemoryStore(),
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                episode_prefix=episode_prefix,
                image_mode=image_mode,
                rgb_file=rgb_file,
                cam4_file=cam4_file,
                robot_states_file=robot_states_file,
                eef_slice=eef_slice,
                alignment_only=alignment_only,
                alignment_end_file=alignment_end_file,
                max_episodes=max_episodes,
            )

        key_first_k: Dict[str, int] = {}
        if n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)

        self.replay_buffer = replay_buffer
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"])
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)
        obs_dict = {}

        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][t_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
