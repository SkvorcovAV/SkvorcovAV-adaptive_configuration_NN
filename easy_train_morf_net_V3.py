import math
import itertools
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import imageio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import cv2
import pyvista as pv
from PIL import Image

# Запускаем виртуальный фреймбуфер для headless-режима
pv.start_xvfb()

class DiffusionNet3D(nn.Module):
    
    def __init__(self,
                 H: int = 28,
                 W: int = 28,
                 D: int = 32,
                 num_classes: int = 10,
                 stripes_dim: str = "height",
                 device: Union[str, torch.device] = "cpu",
                 gif_dir: Union[str, Path] = "gifs") -> None:
        super().__init__()
        self.H, self.W, self.D = H, W, D
        self.num_classes = num_classes
        self.stripes_dim = stripes_dim
        self.device = torch.device(device)

        # --- уникальные 3×3×3 фильтры: (D,H,W,3,3,3) ---------------
        filt = torch.empty(D, H, W, 3, 3, 3, device=self.device) \
                 .uniform_(-0.5, 0.5)
        self.filters = nn.Parameter(filt)

        # --- директория для GIF -----------------------------------------------
        self.gif_dir = Path(gif_dir); self.gif_dir.mkdir(parents=True, exist_ok=True)
        self._gif_counter = 0

    # ===================== внутренние вспомогательные методы ===================
    # --- шаг свободной эволюции сигнала-----------------------------------------
    def _step(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B,1,D,H,W)  → такое же, но после tanh(Σ соседей*w)
        """
        B, _, D, H, W = state.shape
        padded = F.pad(state, (1, 1, 1, 1, 1, 1))        # (B,1,D+2,H+2,W+2)
        conv = torch.zeros_like(state)

        for dz, dy, dx in itertools.product((-1, 0, 1), repeat=3):
            src = padded[:, 0,
                         1+dz : 1+dz+D,
                         1+dy : 1+dy+H,
                         1+dx : 1+dx+W]                  # (B,D,H,W)
            w = self.filters[..., dz+1, dy+1, dx+1]      # (D,H,W)
            conv[:, 0] += src * w.unsqueeze(0)
        return torch.tanh(conv)

    # --- read-out: полосами -> (B,C) -------------------------------
    def _stripe_readout(self, last_plane: torch.Tensor) -> torch.Tensor:
        """
        last_plane: (B,1,H,W)  →  (B, num_classes)
        """
        if self.stripes_dim == "height":
            pooled = F.adaptive_avg_pool2d(last_plane,
                                           output_size=(self.num_classes, 1))
        else:  # 'width'
            pooled = F.adaptive_avg_pool2d(last_plane,
                                           output_size=(1, self.num_classes))
        return pooled.flatten(1)                          # (B,C)

    # --- 3D визуализация с помощью PyVista ------------------------
    def _render_voxels(self, 
                      state_slice: torch.Tensor) -> np.ndarray:
        """
        Рендерит 3D воксели и возвращает изображение
        state_slice: (D, H, W) - срез состояния для одного образца
        """
        data = state_slice.detach().cpu().numpy()
        normalized = (data + 1) * 125.5
        normalized [(data<0.6)&(data> -0.6)] = 0 
        grid = pv.ImageData()
        grid.dimensions = np.array(data.shape)
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.point_data["values"] = normalized.flatten(order="F")
        plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
        plotter.set_background('black')
        opacity = np.linspace(0.01, 0.1, 200).tolist()
        plotter.add_volume(grid,
                           scalars="values",
                           cmap="jet",
#                            opacity="linear",
                           opacity=opacity,
                           shade=True,
                           clim=[0, 255])

        plotter.add_bounding_box(color='green')
        d, h, w = data.shape
        distance_factor = 2.5
        camera_position = (
            w * distance_factor,
            h * distance_factor,
            d * distance_factor
        )
        
        plotter.camera_position = [
            camera_position,
            (w//2, h//2, d//2),
            (0, 0, 1)
        ]
        
        # Рендерим и получаем изображение
#         plotter.show(auto_close=False)
        img = plotter.screenshot(return_img=True)
        plotter.close()
        
        return img

    # --- Визуализация активного класса ----------------------------
    def _add_prediction_label(self, 
                             img: np.ndarray, 
                             pred_class: int) -> np.ndarray:
        """Добавляет текст с предсказанием на изображение"""
        img_rgb = img.copy()
        cv2.putText(img_rgb, f"Prediction: {pred_class}", 
                   (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, (255, 50, 50), 3, cv2.LINE_AA)
        return img_rgb

    def _save_gif(self,
                  frames: List[np.ndarray],
                  fname: Optional[Union[str, Path]] = None,
                  fps: int = 10) -> None:
        """Сохраняет список кадров как GIF"""
        if not frames:
            return
        if fname is None:
            self._gif_counter += 1
            fname = self.gif_dir / f"3d_evo_{self._gif_counter:04d}.gif"
        imageio.mimsave(str(fname), frames, fps=fps)
        print(f"[3D GIF] saved → {fname}")

    # ===================== основной forward ========================
    def forward(self,
                x: torch.Tensor,                # (B,1,H,W)
                T: Optional[int] = None,
                record: bool = False,
                record_idx: int = 0,            # Индекс в батче для записи
                every: int = 2) -> Tuple[torch.Tensor, List[np.ndarray]|None]:
        """
        Возвращает (logits, frames|None)
        logits — (B, num_classes)
        frames — список RGB-кадров, если record=True
        """
        B = x.size(0)
        T = T or 2 * self.D

        # --- начальное состояние -----------------------------------
        state = torch.zeros(B, 1, self.D, self.H, self.W, device=self.device)
        state[:, :, 0] = x                       # вводим изображение

        frames = [] if record else None

        # --- эволюция ---------------------------------------------
        for t in range(T):
            state = self._step(state)
            
            if record and t % every == 0:
                state_slice = state[record_idx, 0]  # (D, H, W)
                img = self._render_voxels(state_slice)
                frames.append(img)

        # --- read-out ---------------------------------------------
        last_plane = state[:, :, -1]             # (B,1,H,W)
        logits = self._stripe_readout(last_plane)  # (B,C)

        if record and frames:
            pred_class = torch.argmax(logits[record_idx]).item()
            frames[-1] = self._add_prediction_label(frames[-1], pred_class)
            self._save_gif(frames)

        return logits, frames

# =================== Обучение модели =============================
if __name__ == "__main__":
    # ------------------------- параметры ----------------------------
    BATCH_SIZE = 4096
    EPOCHS = 40
    T_STEPS = 64
    GIF_EVERY_EPOCH = 1
    
    # ------------------------- подготовка данных ---------------------
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    val_ds   = datasets.MNIST("data", train=False, download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------- модель --------------------------------
    device = "cuda:1"
    print(f"Using device: {device}")
    
    net = DiffusionNet3D(D=32, num_classes=10, device=device,
                         gif_dir="3d_gifs_mnist").to(device)

    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    # ------------------------- цикл обучения -------------------------
    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        net.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        if epoch == 1:
            print("\nGenerating initial 3D GIF before training...")
            sample, label = train_ds[0]
            sample = sample.unsqueeze(0).to(device)
            net(sample, T=10, record=True, every=1)
        
        for x, y in tqdm(train_dl, desc=f"train {epoch}/{EPOCHS}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, _ = net(x, T=T_STEPS)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_loss += loss.item() * x.size(0)
                train_correct += (preds == y).sum().item()
                train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # ---- validation ----
        net.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits, _ = net(x, T=T_STEPS)
                loss = loss_fn(logits, y)
                
                preds = torch.argmax(logits, dim=1)
                val_loss += loss.item() * x.size(0)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:2d}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
              f"LR: {opt.param_groups[0]['lr']:.2e}")

        if epoch % GIF_EVERY_EPOCH == 0:
            print(f"\nGenerating 3D GIF at epoch {epoch}")
            idx = random.randint(0, len(val_ds) - 1)
            sample, true_label = val_ds[idx]
            sample = sample.unsqueeze(0).to(device)
            print(f"  Sample index: {idx}, True label: {true_label}")
            logits, _ = net(sample, T=T_STEPS, record=True, every=2)
            pred_label = torch.argmax(logits[0]).item()
            print(f"  Predicted label: {pred_label}")

    print("\nTraining completed!")
    
    print("\nGenerating final 3D GIFs on test set...")
    test_samples = 3
    for i in range(test_samples):
        idx = random.randint(0, len(val_ds) - 1)
        sample, true_label = val_ds[idx]
        sample = sample.unsqueeze(0).to(device)
        print(f"Sample {i+1}/{test_samples}: True label = {true_label}")
        logits, _ = net(sample, T=T_STEPS, record=True, every=2)
        pred_label = torch.argmax(logits[0]).item()
        print(f"  Predicted: {pred_label}")
