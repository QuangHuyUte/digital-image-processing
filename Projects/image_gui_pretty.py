import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox

# =========================
# 1. HÀM XỬ LÝ ẢNH
# =========================

def apply_negative_bgr(img_bgr):
    # âm bản trên ảnh màu -> vẫn còn màu, chỉ đảo
    return 255 - img_bgr

def log_transform_gray(gray, c=1.0, eps=0.0, normalize=False):
    gray_f = gray.astype(np.float32)
    out = c * np.log(1.0 + gray_f + eps)
    if normalize:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    else:
        denom = np.log(1.0 + 255.0 + eps)
        out = out * (255.0 / denom)
    return np.clip(out, 0, 255).astype(np.uint8)

def gamma_transform_gray(gray, c=1.0, gamma=1.0):
    g = gray.astype(np.float32) / 255.0
    out = c * (g ** gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def piecewise_linear_gray(gray, r1=70, s1=10, r2=140, s2=200):
    def comp(x, r1, s1, r2, s2):
        if x <= r1:
            return (s1 / max(r1, 1)) * x
        elif x <= r2:
            return ((s2 - s1) / max((r2 - r1), 1)) * (x - r1) + s1
        else:
            return ((255 - s2) / max((255 - r2), 1)) * (x - r2) + s2
    vec = np.vectorize(lambda x: comp(int(x), r1, s1, r2, s2))
    return np.clip(vec(gray), 0, 255).astype(np.uint8)

def mean_blur(img, ksize=3):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.blur(img, (k, k))

def gaussian_blur(img, ksize=3, sigma=1.0):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma))

def median_blur(img, ksize=3):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)

def hist_equalize_luma(img_bgr, use_clahe=False, clip_limit=2.0, tile=8):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    if use_clahe:
        tile = max(1, int(tile))
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile, tile))
        y_eq = clahe.apply(y)
    else:
        y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

def apply_on_luma(img_bgr, func_gray, **kwargs):
    """
    Áp một hàm xử lý mức xám lên kênh Y rồi ghép lại -> GIỮ MÀU
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_new = func_gray(y, **kwargs)
    return cv2.cvtColor(cv2.merge([y_new, cr, cb]), cv2.COLOR_YCrCb2BGR)

# =========================
# 2. GUI CUSTOMTKINTER
# =========================

class ImageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Image Transform Studio – OpenCV (stacked, color, realtime)")
        self.geometry("1500x900")
        self.minsize(1300, 800)

        self.original_bgr = None
        self.current_bgr = None
        self._after_id = None

        self._build_layout()
        self._build_controls()
        self._bind_resize()

    # ---------- Layout ----------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # LEFT
        self.ctrl_frame = ctk.CTkFrame(self, corner_radius=12)
        self.ctrl_frame.grid(row=0, column=0, sticky="nsw", padx=(14, 8), pady=14)

        btns = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        btns.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkButton(btns, text="Chọn ảnh", command=self.choose_image, width=110).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="Cập nhật", command=self.update_preview, width=110).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="Lưu ra file", command=self.save_image, width=110).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="Close", command=self.destroy, width=80).pack(side="left", padx=4)

        self.scroll = ctk.CTkScrollableFrame(self.ctrl_frame, label_text="CÔNG CỤ BIẾN ĐỔI", width=420, height=760)
        self.scroll.pack(padx=12, pady=(6, 12), fill="both", expand=True)

        # RIGHT
        self.preview_frame = ctk.CTkFrame(self, corner_radius=12)
        self.preview_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 14), pady=14)
        self.preview_frame.grid_rowconfigure(1, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        title.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 6))
        ctk.CTkLabel(title, text="Original", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkLabel(title, text="Result (stacked, color preserved)", font=ctk.CTkFont(size=16, weight="bold")).pack(side="right")

        self.left_area = ctk.CTkLabel(self.preview_frame, text="", corner_radius=8, fg_color="#0e0e12")
        self.left_area.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=(6, 12))
        self.right_area = ctk.CTkLabel(self.preview_frame, text="", corner_radius=8, fg_color="#0e0e12")
        self.right_area.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(6, 12))

    # ---------- Controls ----------
    def _section_header(self, parent, text, color_hex):
        bar = ctk.CTkFrame(parent, height=30, corner_radius=8, fg_color=color_hex)
        bar.pack(fill="x", pady=(10, 6))
        ctk.CTkLabel(bar, text="  " + text, text_color="white",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")

    def _toggle(self, parent, text, varname, default=False):
        var = ctk.BooleanVar(value=default)
        sw = ctk.CTkSwitch(parent, text=text, variable=var,
                           command=lambda: self._schedule_preview(0))
        sw.pack(anchor="w", pady=(0, 6))
        setattr(self, varname, var)
        return var

    def _slider(self, parent, text, varname, frm, to, init, step=0.1, fmt="{:.2f}"):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", pady=2)
        row = ctk.CTkFrame(wrap, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text=text).pack(side="left")
        lbl = ctk.CTkLabel(row, text=fmt.format(init))
        lbl.pack(side="right")
        var = ctk.DoubleVar(value=init)
        sld = ctk.CTkSlider(
            wrap, from_=frm, to=to,
            number_of_steps=int((to - frm) / step) if step > 0 else 0
        )
        sld.set(init)

        def on_change(v):
            v = float(v)
            var.set(v)
            lbl.configure(text=fmt.format(v))
            # realtime
            self._schedule_preview(0)

        sld.configure(command=on_change)
        sld.pack(fill="x", padx=2, pady=(4, 8))
        setattr(self, varname, var)
        return var

    def _build_controls(self):
        col = {
            "neg": "#455a64", "log": "#2db6f4", "pw": "#9c27b0",
            "gm": "#ff9800", "mn": "#ff4081", "gs": "#00bcd4",
            "md": "#f44336", "he": "#4caf50",
        }

        # Negative
        self._section_header(self.scroll, "Negative image (BGR)", col["neg"])
        self.neg_enable = self._toggle(self.scroll, "Áp dụng Negative", "neg_enable", default=False)

        # Log (kênh Y)
        self._section_header(self.scroll, "Biến đổi Log (trên kênh Y)", col["log"])
        self.log_enable = self._toggle(self.scroll, "Bật Log", "log_enable", default=False)
        self.log_c     = self._slider(self.scroll, "Hệ số c (0.1–10)", "log_c", 0.1, 10.0, 1.0, step=0.1)
        self.log_eps   = self._slider(self.scroll, "Epsilon (0–20)", "log_eps", 0.0, 20.0, 0.0, step=0.1)
        self.log_norm  = self._toggle(self.scroll, "Normalize về [0,255]", "log_norm", default=False)

        # Piecewise
        self._section_header(self.scroll, "Biến đổi Piecewise-Linear (Y)", col["pw"])
        self.pw_enable = self._toggle(self.scroll, "Bật Piecewise-Linear", "pw_enable", default=False)
        self.pw_r1 = self._slider(self.scroll, "r1 (0–255)", "pw_r1", 0, 255, 70, step=1, fmt="{:.0f}")
        self.pw_s1 = self._slider(self.scroll, "s1 (0–255)", "pw_s1", 0, 255, 10, step=1, fmt="{:.0f}")
        self.pw_r2 = self._slider(self.scroll, "r2 (0–255)", "pw_r2", 0, 255, 140, step=1, fmt="{:.0f}")
        self.pw_s2 = self._slider(self.scroll, "s2 (0–255)", "pw_s2", 0, 255, 200, step=1, fmt="{:.0f}")

        # Gamma
        self._section_header(self.scroll, "Biến đổi Gamma (Y)", col["gm"])
        self.gm_enable = self._toggle(self.scroll, "Bật Gamma", "gm_enable", default=False)
        self.gm_c      = self._slider(self.scroll, "Hệ số c (0.1–3)", "gm_c", 0.1, 3.0, 1.0, step=0.1)
        self.gm_gamma  = self._slider(self.scroll, "Gamma (0.1–5)", "gm_gamma", 0.1, 5.0, 1.0, step=0.1)

        # Mean
        self._section_header(self.scroll, "Làm trơn (Lọc trung bình)", col["mn"])
        self.mn_enable = self._toggle(self.scroll, "Bật Mean blur", "mn_enable", default=False)
        self.mn_k      = self._slider(self.scroll, "Kích thước lọc (odd)", "mn_k", 1, 31, 3, step=2, fmt="{:.0f}")

        # Gauss
        self._section_header(self.scroll, "Làm trơn (Lọc Gauss)", col["gs"])
        self.gs_enable = self._toggle(self.scroll, "Bật Gaussian blur", "gs_enable", default=False)
        self.gs_k      = self._slider(self.scroll, "Kích thước lọc (odd)", "gs_k", 1, 31, 5, step=2, fmt="{:.0f}")
        self.gs_sigma  = self._slider(self.scroll, "Sigma (0.1–10)", "gs_sigma", 0.1, 10.0, 1.0, step=0.1)

        # Median
        self._section_header(self.scroll, "Làm trơn (Lọc trung vị)", col["md"])
        self.md_enable = self._toggle(self.scroll, "Bật Median blur", "md_enable", default=False)
        self.md_k      = self._slider(self.scroll, "Kích thước lọc (odd)", "md_k", 1, 31, 3, step=2, fmt="{:.0f}")

        # Histogram
        self._section_header(self.scroll, "Cân bằng sáng dùng Histogram (Y)", col["he"])
        self.he_enable = self._toggle(self.scroll, "Bật Histogram Equalization", "he_enable", default=False)
        self.he_clahe  = self._toggle(self.scroll, "Dùng CLAHE", "he_clahe", default=True)
        self.he_clip   = self._slider(self.scroll, "clipLimit (1–10)", "he_clip", 1.0, 10.0, 2.0, step=0.1)
        self.he_tile   = self._slider(self.scroll, "tileGridSize (1–32)", "he_tile", 1.0, 32.0, 8.0, step=1, fmt="{:.0f}")

    # ---------- Resize ----------
    def _bind_resize(self):
        def on_resize(event):
            self._schedule_preview(0)
        self.preview_frame.bind("<Configure>", on_resize)

    # ---------- Helpers ----------
    def _schedule_preview(self, delay_ms=0):
        # 0 = update ngay khi kéo
        if self._after_id:
            self.after_cancel(self._after_id)
        if delay_ms <= 0:
            self.update_preview()
        else:
            self._after_id = self.after(delay_ms, self.update_preview)

    def _fit_to_area(self, bgr_img, target_w, target_h):
        h, w = bgr_img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _show_on(self, widget, bgr_img):
        if bgr_img is None:
            widget.configure(image=None, text="No Image")
            return
        avail_w = max(200, widget.winfo_width())
        avail_h = max(200, widget.winfo_height())
        img = self._fit_to_area(bgr_img, avail_w - 16, avail_h - 16)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(pil.width, pil.height))
        widget.configure(image=ctk_img, text="")
        widget.image = ctk_img

    # ---------- Actions ----------
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh...",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return
        self.original_bgr = img
        self.update_preview()

    def save_image(self):
        if self.current_bgr is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy chọn ảnh và chỉnh tham số trước.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("BMP","*.bmp"),("TIFF","*.tif;*.tiff")]
        )
        if not path:
            return
        if cv2.imwrite(path, self.current_bgr):
            messagebox.showinfo("Đã lưu", f"Lưu ảnh thành công:\n{path}")
        else:
            messagebox.showerror("Lỗi", "Không lưu được ảnh.")

    def update_preview(self):
        if self.original_bgr is None:
            self._show_on(self.left_area, None)
            self._show_on(self.right_area, None)
            return

        img = self.original_bgr.copy()
        left = img.copy()

        try:
            # 1. Negative (BGR)
            if self.neg_enable.get():
                img = apply_negative_bgr(img)

            # 2. Log (Y)
            if self.log_enable.get():
                img = apply_on_luma(
                    img,
                    log_transform_gray,
                    c=float(self.log_c.get()),
                    eps=float(self.log_eps.get()),
                    normalize=bool(self.log_norm.get())
                )

            # 3. Piecewise (Y)
            if self.pw_enable.get():
                img = apply_on_luma(
                    img,
                    piecewise_linear_gray,
                    r1=int(self.pw_r1.get()),
                    s1=int(self.pw_s1.get()),
                    r2=int(self.pw_r2.get()),
                    s2=int(self.pw_s2.get())
                )

            # 4. Gamma (Y)
            if self.gm_enable.get():
                img = apply_on_luma(
                    img,
                    gamma_transform_gray,
                    c=float(self.gm_c.get()),
                    gamma=float(self.gm_gamma.get())
                )

            # 5. Smooth (BGR)
            if self.mn_enable.get():
                img = mean_blur(img, ksize=int(self.mn_k.get()))
            if self.gs_enable.get():
                img = gaussian_blur(img, ksize=int(self.gs_k.get()), sigma=float(self.gs_sigma.get()))
            if self.md_enable.get():
                img = median_blur(img, ksize=int(self.md_k.get()))

            # 6. Histogram (Y)
            if self.he_enable.get():
                img = hist_equalize_luma(
                    img,
                    use_clahe=bool(self.he_clahe.get()),
                    clip_limit=float(self.he_clip.get()),
                    tile=int(self.he_tile.get())
                )

            self.current_bgr = img

            # render
            self._show_on(self.left_area, left)
            self._show_on(self.right_area, img)

        except Exception as e:
            # in ra console để không spam popup khi kéo
            print("Process error:", e)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
