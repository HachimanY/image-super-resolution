import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image

from model import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


class SuperResolutionDemo:
    def __init__(self, root):
        self.root = root
        self.root.title('SRCNN Image Super-Resolution Demo')
        self.root.geometry('600x500')
        self.root.resizable(False, False)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cudnn.benchmark = True
        self.model = None
        self.loaded_weights = None

        self._build_ui()

    def _build_ui(self):
        frame_top = ttk.Frame(self.root, padding=10)
        frame_top.pack(fill='x')

        ttk.Label(frame_top, text='Model Weights:').grid(row=0, column=0, sticky='w')
        self.weights_var = tk.StringVar(value='./model/best.pth')
        ttk.Entry(frame_top, textvariable=self.weights_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(frame_top, text='Browse', command=self._browse_weights).grid(row=0, column=2)

        ttk.Label(frame_top, text='Input Image:').grid(row=1, column=0, sticky='w', pady=5)
        self.image_var = tk.StringVar()
        ttk.Entry(frame_top, textvariable=self.image_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(frame_top, text='Browse', command=self._browse_image).grid(row=1, column=2)

        ttk.Label(frame_top, text='Scale Factor:').grid(row=2, column=0, sticky='w', pady=5)
        self.scale_var = tk.IntVar(value=2)
        scale_combo = ttk.Combobox(frame_top, textvariable=self.scale_var, values=[2, 3, 4], width=5, state='readonly')
        scale_combo.grid(row=2, column=1, sticky='w', padx=5)

        self.run_btn = ttk.Button(frame_top, text='Run Super-Resolution', command=self._run)
        self.run_btn.grid(row=3, column=0, columnspan=3, pady=15)

        frame_info = ttk.LabelFrame(self.root, text='Results', padding=10)
        frame_info.pack(fill='both', expand=True, padx=10, pady=5)

        self.status_var = tk.StringVar(value='Ready. Select weights and image to begin.')
        ttk.Label(frame_info, textvariable=self.status_var, wraplength=560).pack(anchor='w')

        self.psnr_var = tk.StringVar(value='')
        ttk.Label(frame_info, textvariable=self.psnr_var, font=('Consolas', 11)).pack(anchor='w', pady=5)

        self.output_var = tk.StringVar(value='')
        ttk.Label(frame_info, textvariable=self.output_var, wraplength=560, foreground='blue').pack(anchor='w')

        ttk.Label(frame_info, text='\nDevice: ' + str(self.device), foreground='gray').pack(anchor='sw')

    def _browse_weights(self):
        path = filedialog.askopenfilename(filetypes=[('PyTorch Weights', '*.pth'), ('All Files', '*.*')])
        if path:
            self.weights_var.set(path)

    def _browse_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ('Image Files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('All Files', '*.*'),
        ])
        if path:
            self.image_var.set(path)

    def _load_model(self, weights_path):
        if self.loaded_weights == weights_path and self.model is not None:
            return
        self.model = SRCNN().to(self.device)
        state_dict = self.model.state_dict()
        for n, p in torch.load(weights_path, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
        self.model.eval()
        self.loaded_weights = weights_path

    def _run(self):
        weights_path = self.weights_var.get()
        image_path = self.image_var.get()
        scale = self.scale_var.get()

        if not os.path.exists(weights_path):
            messagebox.showerror('Error', f'Weights file not found:\n{weights_path}')
            return
        if not os.path.exists(image_path):
            messagebox.showerror('Error', f'Image file not found:\n{image_path}')
            return

        self.run_btn.config(state='disabled')
        self.status_var.set('Processing...')
        self.psnr_var.set('')
        self.output_var.set('')
        self.root.update()

        thread = threading.Thread(target=self._process, args=(weights_path, image_path, scale), daemon=True)
        thread.start()

    def _process(self, weights_path, image_path, scale):
        try:
            self._load_model(weights_path)

            image = pil_image.open(image_path).convert('RGB')
            image_width = (image.width // scale) * scale
            image_height = (image.height // scale) * scale
            image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
            image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

            image_np = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image_np)

            new_images = []
            psnr_list = []
            for i in range(3):
                y = ycbcr[..., i]
                y /= 255.
                y_tensor = torch.from_numpy(y).to(self.device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    preds = self.model(y_tensor).clamp(0.0, 1.0)
                psnr_list.append(calc_psnr(y_tensor, preds).cpu())
                preds_np = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                new_images.append(preds_np)

            avg_psnr = np.mean(psnr_list)

            output = np.array([new_images[0], new_images[1], new_images[2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output_img = pil_image.fromarray(output)

            base, ext = os.path.splitext(image_path)
            save_path = f'{base}_srcnn_x{scale}{ext}'
            output_img.save(save_path)

            self.root.after(0, lambda: self._on_done(avg_psnr, save_path))

        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _on_done(self, psnr, save_path):
        self.status_var.set('Done!')
        self.psnr_var.set(f'PSNR: {psnr:.2f} dB')
        self.output_var.set(f'Saved to: {save_path}')
        self.run_btn.config(state='normal')

    def _on_error(self, msg):
        self.status_var.set('Error occurred.')
        messagebox.showerror('Error', msg)
        self.run_btn.config(state='normal')


def main():
    root = tk.Tk()
    SuperResolutionDemo(root)
    root.mainloop()


if __name__ == '__main__':
    main()
