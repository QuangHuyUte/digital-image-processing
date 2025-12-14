import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox

# =========================================================
#                LUT HELPERS (siêu nhanh)
# =========================================================
def build_log_lut(c=1.0, eps=0.0, normalize=False):
    x = np.arange(256, dtype=np.float32)
    y = c * np.log(1.0 + x + eps)
    if normalize:
        y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
    else:
        y *= (255.0 / np.log(1.0 + 255.0 + eps))
    return np.clip(y, 0, 255).astype(np.uint8)


def build_gamma_lut(c=1.0, gamma=1.0):
    x = np.arange(256, dtype=np.float32) / 255.0
    y = c * (x ** gamma) * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)


def build_piecewise_lut(r1, s1, r2, s2):
    lut = np.empty(256, dtype=np.uint8)
    r1 = int(np.clip(r1, 0, 255))
    r2 = int(np.clip(r2, 0, 255))
    s1 = int(np.clip(s1, 0, 255))
    s2 = int(np.clip(s2, 0, 255))
    if r1 == 0:
        r1 = 1
    if r2 == r1:
        r2 = r1 + 1
    for x in range(256):
        if x <= r1:
            y = (s1 / r1) * x
        elif x <= r2:
            y = ((s2 - s1) / (r2 - r1)) * (x - r1) + s1
        else:
            denom = max(1, 255 - r2)
            y = ((255 - s2) / denom) * (x - r2) + s2
        lut[x] = np.clip(y, 0, 255)
    return lut


def apply_luts_on_luma(bgr_img, luts):
    """Áp lần lượt LUT trên kênh Y, giữ màu."""
    if not luts:
        return bgr_img
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    for lut in luts:
        if lut is not None:
            y = cv2.LUT(y, lut)
    merged = cv2.merge([y, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# =========================================================
#          SPATIAL FILTER HELPERS (trên kênh Y)
# =========================================================
def _ensure_odd(k):
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def _apply_on_y(bgr_img, func, **kwargs):
    """Chuyển YCrCb, áp func lên Y, ghép lại (giữ màu)."""
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y2 = func(y, **kwargs)
    merged = cv2.merge([y2, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def lowpass_box_on_y(y, k=3):
    k = _ensure_odd(k)
    return cv2.blur(y, (k, k))  # Averaging filter (low-pass)


def gaussian_on_y(y, k=3, sigma=1.0):
    k = _ensure_odd(k)
    return cv2.GaussianBlur(y, (k, k), sigmaX=float(sigma))


def median_on_y(y, k=3):
    k = _ensure_odd(k)
    return cv2.medianBlur(y, k)


def max_on_y(y, k=3):
    k = _ensure_odd(k)
    ker = np.ones((k, k), np.uint8)
    return cv2.dilate(y, ker)   # local maximum


def min_on_y(y, k=3):
    k = _ensure_odd(k)
    ker = np.ones((k, k), np.uint8)
    return cv2.erode(y, ker)    # local minimum


def midpoint_on_y(y, k=3):
    y_max = max_on_y(y, k).astype(np.uint16)
    y_min = min_on_y(y, k).astype(np.uint16)
    return ((y_max + y_min) // 2).astype(np.uint8)


def range_on_y(y, k=3):
    y_max = max_on_y(y, k).astype(np.int16)
    y_min = min_on_y(y, k).astype(np.int16)
    r = y_max - y_min
    return np.clip(r, 0, 255).astype(np.uint8)


# =========================================================
#          SHARPENING HELPERS (trên kênh Y)
# =========================================================
def laplacian_sharpen_y(y, k=3, alpha=1.0, use_8conn=False):
    """
    Sharpen bằng Laplacian:
        - Làm mịn nhẹ với kernel k (Gaussian) để giảm noise
        - Tính Laplacian
        - g = y - alpha * Laplacian(blur)
    """
    k = _ensure_odd(k)
    y_f = y.astype(np.float32)

    if k > 1:
        base = cv2.GaussianBlur(y_f, (k, k), sigmaX=0)
    else:
        base = y_f

    if use_8conn:
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], np.float32)
    else:
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], np.float32)

    lap = cv2.filter2D(base, cv2.CV_32F, kernel)
    g = y_f - alpha * lap
    return np.clip(g, 0, 255).astype(np.uint8)


def unsharp_mask_y(y, blur_k=5, sigma=1.0, k=1.0):
    """
    Unsharp masking / High-boost:
        blur = Gaussian(y)
        mask = y - blur
        g = y + k * mask
    """
    blur_k = _ensure_odd(blur_k)
    y_f = y.astype(np.float32)
    blur = cv2.GaussianBlur(y, (blur_k, blur_k), sigmaX=float(sigma))
    blur_f = blur.astype(np.float32)

    mask = y_f - blur_f
    g = y_f + k * mask
    return np.clip(g, 0, 255).astype(np.uint8)


def sobel_sharpen_y(y, ksize=3, alpha=1.0):
    """
    Gradient (Sobel) sharpening:
        mag = |grad(y)|
        g = y + alpha * mag_norm
    """
    ksize = _ensure_odd(ksize)
    y_f = y.astype(np.float32)

    gx = cv2.Sobel(y_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(y_f, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    g = y_f + alpha * mag_norm
    return np.clip(g, 0, 255).astype(np.uint8)


def combined_enhance_y(y, sobel_k=3, smooth_k=5, alpha=1.0):
    """
    Kết hợp Laplacian + Sobel:
    - mag = Sobel magnitude (smooth)
    - lap = |Laplacian|
    - mask = alpha * mag_norm * lap_norm
    - g = y + mask
    """
    sobel_k = _ensure_odd(sobel_k)
    smooth_k = _ensure_odd(smooth_k)
    y_f = y.astype(np.float32)

    gx = cv2.Sobel(y_f, cv2.CV_32F, 1, 0, ksize=sobel_k)
    gy = cv2.Sobel(y_f, cv2.CV_32F, 0, 1, ksize=sobel_k)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (smooth_k, smooth_k), 0)
    mag_norm = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], np.float32)
    lap = cv2.filter2D(y_f, cv2.CV_32F, kernel)
    lap_abs = np.abs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0.0, 1.0, cv2.NORM_MINMAX)

    mask = alpha * (mag_norm * lap_norm) * 255.0
    g = y_f + mask
    return np.clip(g, 0, 255).astype(np.uint8)


# =========================================================
#           EDGE DETECTION (Roberts / Prewitt / Sobel)
# =========================================================
def roberts_edge_y(y):
    y_f = y.astype(np.float32)
    kx = np.array([[1, 0],
                   [0, -1]], np.float32)
    ky = np.array([[0, 1],
                   [-1, 0]], np.float32)
    gx = cv2.filter2D(y_f, cv2.CV_32F, kx)
    gy = cv2.filter2D(y_f, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def prewitt_edge_y(y):
    y_f = y.astype(np.float32)
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]], np.float32)
    gx = cv2.filter2D(y_f, cv2.CV_32F, kx)
    gy = cv2.filter2D(y_f, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def sobel_edge_y(y, ksize=3):
    ksize = _ensure_odd(ksize)
    y_f = y.astype(np.float32)
    gx = cv2.Sobel(y_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(y_f, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


# =========================================================
#                        APP
# =========================================================
class DemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Digital Image Processing – Intensity & Spatial Filters")
        self.geometry("1300x800")  # khởi tạo, sau đó full màn

        # state ảnh
        self.original_bgr = None
        self.preview_base = None
        self.max_preview_side = 1400
        self._after_id = None
        self.show_original = False

        # state undo/redo
        self.history = []          # list các dict tham số
        self.history_index = -1
        self._history_lock = False

        # UI
        self._build_layout()
        self._build_controls()
        self._bind_resize()

        # Mở lên là full màn hình
        self.after(100, self._maximize_window)

        # Ghi lại state ban đầu vào history
        self._push_history()
        # Preview placeholder
        self.after(150, self.update_preview)

    # ---------------- Window helpers ----------------
    def _maximize_window(self):
        """Thử các cách full màn hình (Windows / các hệ khác)."""
        try:
            self.state("zoomed")          # Windows
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                self.attributes("-fullscreen", True)

    # ---------------- Layout ----------------
    def _build_layout(self):
        # 2 cột: trái = menu, phải = preview
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # LEFT: controls
        self.ctrl = ctk.CTkFrame(self, corner_radius=12, width=340)
        self.ctrl.grid(row=0, column=0, sticky="nsw", padx=(12, 8), pady=12)

        # ==== Toolbar (2 hàng) ====
        toolbar = ctk.CTkFrame(self.ctrl, fg_color="transparent")
        toolbar.pack(fill="x", padx=8, pady=(8, 4))

        row1 = ctk.CTkFrame(toolbar, fg_color="transparent")
        row1.pack(fill="x")
        ctk.CTkButton(row1, text="Chọn ảnh", width=80,
                      command=self.choose_image).pack(side="left", padx=2)
        ctk.CTkButton(row1, text="Lưu ảnh", width=80,
                      command=self.save_full_res).pack(side="left", padx=2)
        ctk.CTkButton(row1, text="Reset", width=70,
                      command=self.reset_params).pack(side="left", padx=2)

        row2 = ctk.CTkFrame(toolbar, fg_color="transparent")
        row2.pack(fill="x", pady=(4, 0))
        ctk.CTkButton(row2, text="Undo", width=70,
                      command=self.undo).pack(side="left", padx=2)
        ctk.CTkButton(row2, text="Redo", width=70,
                      command=self.redo).pack(side="left", padx=2)

        self.btn_show_orig = ctk.CTkButton(
            row2, text="Giữ xem ảnh gốc", width=140
        )
        self.btn_show_orig.pack(side="left", padx=4)
        self.btn_show_orig.bind("<ButtonPress-1>",
                                lambda e: self._set_show_original(True))
        self.btn_show_orig.bind("<ButtonRelease-1>",
                                lambda e: self._set_show_original(False))

        # Quick view
        quick = ctk.CTkFrame(self.ctrl, fg_color="transparent")
        quick.pack(fill="x", padx=8, pady=(4, 4))
        ctk.CTkLabel(
            quick, text="Hiển thị",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")
        self.fast_preview = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            quick,
            text="Fast preview",
            variable=self.fast_preview,
            command=lambda: self._schedule_preview(0),
        ).pack(side="left", padx=6)

        # Scroll control
        self.scroll = ctk.CTkScrollableFrame(
            self.ctrl,
            label_text="BIẾN ĐỔI CƯỜNG ĐỘ & SPATIAL FILTERS",
            width=320,
            height=650,
        )
        self.scroll.pack(padx=8, pady=(4, 8), fill="both", expand=True)

        # RIGHT: preview
        self.view = ctk.CTkFrame(self, corner_radius=12)
        self.view.grid(row=0, column=1, sticky="nsew", padx=(4, 12), pady=12)
        self.view.grid_rowconfigure(1, weight=1)
        self.view.grid_columnconfigure(0, weight=1)

        title = ctk.CTkFrame(self.view, fg_color="transparent")
        title.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        ctk.CTkLabel(
            title, text="Preview",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left")

        self.preview_area = ctk.CTkLabel(
            self.view,
            text="No Image",
            corner_radius=10,
            fg_color="#0e0e12",
            width=900,
            height=680
        )
        self.preview_area.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=(4, 10)
        )

    # ---------------- Controls builders ----------------
    def _header(self, text, color):
        bar = ctk.CTkFrame(
            self.scroll, height=26, corner_radius=8, fg_color=color
        )
        bar.pack(fill="x", pady=(8, 4))
        ctk.CTkLabel(
            bar,
            text="  " + text,
            text_color="white",
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w")

    def _toggle(self, text, varname, default=False):
        var = ctk.BooleanVar(value=default)
        sw = ctk.CTkSwitch(
            self.scroll,
            text=text,
            variable=var,
            command=lambda: self._schedule_preview(0),
        )
        sw.pack(anchor="w", pady=(0, 4))
        setattr(self, varname, var)

    def _slider(self, text, varname, frm, to, init,
                step=0.1, fmt="{:.2f}"):
        wrap = ctk.CTkFrame(self.scroll, fg_color="transparent")
        wrap.pack(fill="x", pady=2)

        row = ctk.CTkFrame(wrap, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text=text, font=ctk.CTkFont(size=11)).pack(side="left")
        lbl = ctk.CTkLabel(row, text=fmt.format(init), font=ctk.CTkFont(size=11))
        lbl.pack(side="right")

        var = ctk.DoubleVar(value=init)
        slider = ctk.CTkSlider(
            wrap,
            from_=frm,
            to=to,
            number_of_steps=int((to - frm) / step) if step > 0 else 0,
        )
        slider.set(init)

        def on_change(v):
            v = float(v)
            var.set(v)
            lbl.configure(text=fmt.format(v))
            self._schedule_preview(0)

        slider.configure(command=on_change)
        slider.pack(fill="x", padx=2, pady=(2, 6))
        setattr(self, varname, var)

    def _build_controls(self):
        # LOG
        self._header("LOG TRANSFORM (kênh Y – giữ màu)", "#2db6f4")
        self._toggle("Bật LOG", "log_on", default=True)
        self._slider("Hệ số c (0.1–10)", "log_c", 0.1, 10.0, 1.5,
                     step=0.1)
        self._slider("Epsilon (0–20)", "log_eps", 0.0, 20.0, 0.0,
                     step=0.1)
        self._toggle("Normalize về [0,255]", "log_norm", default=False)

        # PIECEWISE
        self._header("PIECEWISE-LINEAR (kênh Y)", "#9c27b0")
        self._toggle("Bật Piecewise-Linear", "pw_on", default=False)
        self._slider("r1 (0–255)", "pw_r1", 0, 255, 70,
                     step=1, fmt="{:.0f}")
        self._slider("s1 (0–255)", "pw_s1", 0, 255, 10,
                     step=1, fmt="{:.0f}")
        self._slider("r2 (0–255)", "pw_r2", 0, 255, 140,
                     step=1, fmt="{:.0f}")
        self._slider("s2 (0–255)", "pw_s2", 0, 255, 200,
                     step=1, fmt="{:.0f}")

        # GAMMA
        self._header("GAMMA (kênh Y)", "#ff9800")
        self._toggle("Bật Gamma", "gm_on", default=False)
        self._slider("Hệ số c (0.1–3)", "gm_c", 0.1, 3.0, 1.0,
                     step=0.1)
        self._slider("Gamma (0.1–5)", "gm_gamma", 0.1, 5.0, 0.8,
                     step=0.1)

        # Smoothing Spatial Filters (linear)
        self._header("SMOOTHING SPATIAL FILTERS (Linear – kênh Y)",
                     "#4caf50")
        self._toggle("Low-pass (Box / Averaging filter)",
                     "lp_on", default=False)
        self._slider("Low-pass: kernel size (odd)", "lp_k",
                     1, 31, 3, step=2, fmt="{:.0f}")

        self._toggle("Gaussian smoothing filter",
                     "gauss_on", default=False)
        self._slider("Gaussian: kernel size (odd)", "gauss_k",
                     1, 31, 5, step=2, fmt="{:.0f}")
        self._slider("Gaussian: sigma (0.1–10)", "gauss_sigma",
                     0.1, 10.0, 1.0, step=0.1)

        # Nonlinear Spatial Filters
        self._header("NONLINEAR SPATIAL FILTERS (Order-Statistic – kênh Y)",
                     "#ff9800")
        self._toggle("Median filter", "med_on", default=False)
        self._slider("Median: kernel size (odd)", "med_k",
                     1, 31, 3, step=2, fmt="{:.0f}")

        self._toggle("Max filter", "max_on", default=False)
        self._slider("Max: kernel size (odd)", "max_k",
                     1, 31, 3, step=2, fmt="{:.0f}")

        self._toggle("Min filter", "min_on", default=False)
        self._slider("Min: kernel size (odd)", "min_k",
                     1, 31, 3, step=2, fmt="{:.0f}")

        self._toggle("Midpoint filter ( (Max+Min)/2 )",
                     "mid_on", default=False)
        self._slider("Midpoint: kernel size (odd)", "mid_k",
                     1, 31, 3, step=2, fmt="{:.0f}")

        self._toggle("Max-min filter (Range = Max−Min)",
                     "rng_on", default=False)
        self._slider("Range: kernel size (odd)", "rng_k",
                     1, 31, 3, step=2, fmt="{:.0f}")

        # SHARPENING SPATIAL FILTERS
        self._header("SHARPENING SPATIAL FILTERS (kênh Y)", "#e91e63")

        # Laplacian
        self._toggle("Laplacian sharpening", "lap_on", default=False)
        self._slider("Laplacian: blur kernel (odd)", "lap_k",
                     1, 15, 3, step=2, fmt="{:.0f}")
        self._slider("Laplacian: strength α", "lap_alpha",
                     0.1, 3.0, 1.0, step=0.1)
        self.lap_use8 = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            self.scroll,
            text="Laplacian 8-neighbors",
            variable=self.lap_use8,
            command=lambda: self._schedule_preview(0),
        ).pack(anchor="w", pady=(0, 6))

        # Unsharp / High-boost
        self._toggle("Unsharp masking / High-boost", "us_on", default=False)
        self._slider("Unsharp: blur kernel (odd)", "us_k",
                     3, 21, 7, step=2, fmt="{:.0f}")
        self._slider("Unsharp: sigma (Gaussian)", "us_sigma",
                     0.1, 5.0, 1.0, step=0.1)
        self._slider("High-boost k (≥1)", "us_kfactor",
                     0.5, 5.0, 1.5, step=0.1)

        # Gradient (Sobel)
        self._toggle("Gradient (Sobel) sharpening", "grad_on", default=False)
        self._slider("Sobel: kernel size (odd)", "grad_ksize",
                     1, 7, 3, step=2, fmt="{:.0f}")
        self._slider("Sobel: strength α", "grad_alpha",
                     0.1, 5.0, 1.0, step=0.1)

        # Combining methods
        self._toggle("Combine Laplacian + Sobel", "comb_on", default=False)
        self._slider("Combine: Sobel kernel", "comb_sobel_k",
                     1, 7, 3, step=2, fmt="{:.0f}")
        self._slider("Combine: smooth kernel", "comb_smooth_k",
                     3, 21, 5, step=2, fmt="{:.0f}")
        self._slider("Combine: strength α", "comb_alpha",
                     0.1, 5.0, 1.0, step=0.1)

        # EDGE-ONLY MODES
        self._header("EDGE DETECTION – CHỈ XEM BIÊN (kênh Y)", "#3f51b5")
        self.edge_mode = ctk.StringVar(value="none")

        def make_edge_radio(text, value):
            rb = ctk.CTkRadioButton(
                self.scroll,
                text=text,
                variable=self.edge_mode,
                value=value,
                command=lambda: self._schedule_preview(0),
            )
            rb.pack(anchor="w", pady=(0, 2))

        make_edge_radio("Không (None)", "none")
        make_edge_radio("Roberts edge", "roberts")
        make_edge_radio("Prewitt edge", "prewitt")
        make_edge_radio("Sobel edge-only", "sobel")

    # ---------------- Undo / Redo helpers ----------------
    def _get_param_state(self):
        """
        Lấy toàn bộ state các biến (BooleanVar, DoubleVar, StringVar).
        Dùng để lưu vào history.
        """
        state = {}
        for name, val in self.__dict__.items():
            if hasattr(val, "get") and hasattr(val, "set"):
                try:
                    state[name] = val.get()
                except Exception:
                    pass
        return state

    def _set_param_state(self, state):
        """
        Gán lại tất cả biến từ state (dùng cho Undo/Redo).
        """
        self._history_lock = True
        for name, value in state.items():
            var = getattr(self, name, None)
            if hasattr(var, "set"):
                try:
                    var.set(value)
                except Exception:
                    pass
        self._history_lock = False
        # Sau khi khôi phục tham số, render lại preview
        self._schedule_preview(0)

    def _push_history(self):
        """
        Lưu state hiện tại vào history (nếu khác state trước).
        """
        if self._history_lock:
            return

        state = self._get_param_state()
        if self.history and state == self.history[self.history_index]:
            return

        # Bỏ hết history phía sau nếu người dùng vừa chỉnh lại sau khi undo
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]

        self.history.append(state)
        self.history_index = len(self.history) - 1

        # Giới hạn history cho nhẹ
        MAX_HISTORY = 50
        if len(self.history) > MAX_HISTORY:
            overflow = len(self.history) - MAX_HISTORY
            self.history = self.history[overflow:]
            self.history_index = len(self.history) - 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self._set_param_state(self.history[self.history_index])

    def redo(self):
        if self.history_index + 1 < len(self.history):
            self.history_index += 1
            self._set_param_state(self.history[self.history_index])

    # ---------------- Resize / Preview ----------------
    def _bind_resize(self):
        def on_resize(_event):
            self._schedule_preview(0)
        self.view.bind("<Configure>", on_resize)

    def _schedule_preview(self, delay_ms=0):
        # Mỗi lần tham số đổi → lưu vào history
        self._push_history()

        if self._after_id:
            self.after_cancel(self._after_id)
        if delay_ms <= 0:
            self.update_preview()
        else:
            self._after_id = self.after(delay_ms, self.update_preview)

    def _show_on(self, widget, bgr):
        """
        Phóng to/thu nhỏ ảnh sao cho:
        - Luôn nằm trọn trong khung preview (không bị cắt).
        - Dùng tối đa diện tích khung (phóng to nếu nhỏ, thu nhỏ nếu lớn).
        """
        if bgr is None:
            widget.configure(image=None, text="No Image")
            widget.image = None
            return

        widget.update_idletasks()
        aw = widget.winfo_width()
        ah = widget.winfo_height()

        if aw <= 1:
            try:
                aw = int(widget.cget("width"))
            except Exception:
                aw = 800
        if ah <= 1:
            try:
                ah = int(widget.cget("height"))
            except Exception:
                ah = 600

        margin = max(12, int(min(aw, ah) * 0.04))
        inner_w = max(1, aw - 2 * margin)
        inner_h = max(1, ah - 2 * margin)

        h, w = bgr.shape[:2]

        scale_fit = min(inner_w / w, inner_h / h)
        scale = scale_fit * 0.98
        if scale <= 0:
            scale = 0.01

        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if (target_w, target_h) != (w, h):
            pil = pil.resize((target_w, target_h), Image.LANCZOS)

        cimg = ctk.CTkImage(
            light_image=pil,
            dark_image=pil,
            size=(target_w, target_h)
        )

        widget.configure(image=cimg, text="")
        widget.image = cimg

    def _make_preview_base(self, bgr):
        h, w = bgr.shape[:2]
        side = max(h, w)
        if side <= self.max_preview_side:
            return bgr.copy()
        scale = self.max_preview_side / side
        return cv2.resize(
            bgr,
            (int(w * scale), int(h * scale)),
            cv2.INTER_AREA,
        )

    # ---------------- File actions ----------------
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh...",
            filetypes=[
                ("Images",
                 "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp")
            ],
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return

        h, w = img.shape[:2]
        max_side = 4000
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_AREA)

        self.original_bgr = img
        self.preview_base = self._make_preview_base(img)
        self._set_show_original(False)

        self._schedule_preview(50)

    def save_full_res(self):
        if self.original_bgr is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy mở ảnh trước.")
            return
        full = self._run_pipeline(self.original_bgr)
        from tkinter.filedialog import asksaveasfilename

        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif;*.tiff"),
            ],
        )
        if not path:
            return
        if cv2.imwrite(path, full):
            messagebox.showinfo("Đã lưu", f"Lưu ảnh thành công:\n{path}")
        else:
            messagebox.showerror("Lỗi", "Không lưu được ảnh.")

    def reset_params(self):
        # tránh push history từng bước khi reset
        self._history_lock = True

        # LOG
        self.log_on.set(True)
        self.log_c.set(1.5)
        self.log_eps.set(0.0)
        self.log_norm.set(False)
        # PIECEWISE
        self.pw_on.set(False)
        self.pw_r1.set(70)
        self.pw_s1.set(10)
        self.pw_r2.set(140)
        self.pw_s2.set(200)
        # GAMMA
        self.gm_on.set(False)
        self.gm_c.set(1.0)
        self.gm_gamma.set(0.8)
        # Linear
        self.lp_on.set(False)
        self.lp_k.set(3)
        self.gauss_on.set(False)
        self.gauss_k.set(5)
        self.gauss_sigma.set(1.0)
        # Nonlinear
        self.med_on.set(False)
        self.med_k.set(3)
        self.max_on.set(False)
        self.max_k.set(3)
        self.min_on.set(False)
        self.min_k.set(3)
        self.mid_on.set(False)
        self.mid_k.set(3)
        self.rng_on.set(False)
        self.rng_k.set(3)
        # Sharpening
        self.lap_on.set(False)
        self.lap_k.set(3)
        self.lap_alpha.set(1.0)
        self.lap_use8.set(False)

        self.us_on.set(False)
        self.us_k.set(7)
        self.us_sigma.set(1.0)
        self.us_kfactor.set(1.5)

        self.grad_on.set(False)
        self.grad_ksize.set(3)
        self.grad_alpha.set(1.0)

        self.comb_on.set(False)
        self.comb_sobel_k.set(3)
        self.comb_smooth_k.set(5)
        self.comb_alpha.set(1.0)

        # Edge-only
        self.edge_mode.set("none")

        self._history_lock = False
        self._push_history()
        self._schedule_preview(0)

    # ---------------- Core pipeline ----------------
    def _run_pipeline(self, src_bgr):
        # 1. Intensity transforms
        luts = []
        if self.log_on.get():
            luts.append(
                build_log_lut(
                    c=float(self.log_c.get()),
                    eps=float(self.log_eps.get()),
                    normalize=bool(self.log_norm.get()),
                )
            )
        if self.pw_on.get():
            luts.append(
                build_piecewise_lut(
                    r1=int(self.pw_r1.get()),
                    s1=int(self.pw_s1.get()),
                    r2=int(self.pw_r2.get()),
                    s2=int(self.pw_s2.get()),
                )
            )
        if self.gm_on.get():
            luts.append(
                build_gamma_lut(
                    c=float(self.gm_c.get()),
                    gamma=float(self.gm_gamma.get()),
                )
            )

        out = apply_luts_on_luma(src_bgr, luts) if luts else src_bgr

        # 2. Linear smoothing
        if self.lp_on.get():
            out = _apply_on_y(out, lowpass_box_on_y,
                              k=int(self.lp_k.get()))
        if self.gauss_on.get():
            out = _apply_on_y(
                out,
                gaussian_on_y,
                k=int(self.gauss_k.get()),
                sigma=float(self.gauss_sigma.get()),
            )

        # 3. Nonlinear filters
        if self.med_on.get():
            out = _apply_on_y(out, median_on_y,
                              k=int(self.med_k.get()))
        if self.max_on.get():
            out = _apply_on_y(out, max_on_y,
                              k=int(self.max_k.get()))
        if self.min_on.get():
            out = _apply_on_y(out, min_on_y,
                              k=int(self.min_k.get()))
        if self.mid_on.get():
            out = _apply_on_y(out, midpoint_on_y,
                              k=int(self.mid_k.get()))
        if self.rng_on.get():
            out = _apply_on_y(out, range_on_y,
                              k=int(self.rng_k.get()))

        # 4. Sharpening filters
        if self.lap_on.get():
            out = _apply_on_y(
                out,
                laplacian_sharpen_y,
                k=int(self.lap_k.get()),
                alpha=float(self.lap_alpha.get()),
                use_8conn=bool(self.lap_use8.get()),
            )

        if self.us_on.get():
            out = _apply_on_y(
                out,
                unsharp_mask_y,
                blur_k=int(self.us_k.get()),
                sigma=float(self.us_sigma.get()),
                k=float(self.us_kfactor.get()),
            )

        if self.grad_on.get():
            out = _apply_on_y(
                out,
                sobel_sharpen_y,
                ksize=int(self.grad_ksize.get()),
                alpha=float(self.grad_alpha.get()),
            )

        if self.comb_on.get():
            out = _apply_on_y(
                out,
                combined_enhance_y,
                sobel_k=int(self.comb_sobel_k.get()),
                smooth_k=int(self.comb_smooth_k.get()),
                alpha=float(self.comb_alpha.get()),
            )

        # 5. EDGE-ONLY (ghi đè kết quả, chỉ để xem biên)
        mode = self.edge_mode.get()
        if mode != "none":
            ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
            y, _, _ = cv2.split(ycrcb)
            if mode == "roberts":
                edge = roberts_edge_y(y)
            elif mode == "prewitt":
                edge = prewitt_edge_y(y)
            elif mode == "sobel":
                edge = sobel_edge_y(y, ksize=int(self.grad_ksize.get()))
            else:
                edge = y
            out = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        return out

    def _set_show_original(self, flag: bool):
        self.show_original = flag
        self.update_preview()

    def update_preview(self):
        if self.original_bgr is None:
            self._show_on(self.preview_area, None)
            return

        src = (
            self.preview_base
            if self.fast_preview.get() and self.preview_base is not None
            else self.original_bgr
        )

        if self.show_original:
            img_to_show = src
        else:
            img_to_show = self._run_pipeline(src)

        self._show_on(self.preview_area, img_to_show)


# ---------------- RUN ----------------
if __name__ == "__main__":
    cv2.setUseOptimized(True)
    app = DemoApp()
    app.mainloop()
