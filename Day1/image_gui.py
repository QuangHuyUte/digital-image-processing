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
    r1 = int(np.clip(r1, 0, 255)); r2 = int(np.clip(r2, 0, 255))
    s1 = int(np.clip(s1, 0, 255)); s2 = int(np.clip(s2, 0, 255))
    if r1 == 0: r1 = 1
    if r2 == r1: r2 = r1 + 1
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
    """Áp lần lượt các LUT (list) trên kênh Y rồi ghép lại -> giữ màu."""
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
#                        APP
# =========================================================
class DemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Intensity Transforms – LOG • PIECEWISE • GAMMA (Color Preserved, Realtime)")
        self.geometry("1380x840")
        self.minsize(1200, 740)

        # state
        self.original_bgr = None
        self.preview_bgr  = None   # ảnh downscale để kéo mượt
        self.current_bgr  = None   # ảnh kết quả (preview)
        self.max_preview_side = 1000  # demo mượt mắt
        self._after_id = None

        # UI
        self._build_layout()
        self._build_controls()
        self._bind_resize()

    # ---------------- Layout ----------------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left panel
        self.ctrl = ctk.CTkFrame(self, corner_radius=12)
        self.ctrl.grid(row=0, column=0, sticky="nsw", padx=(14, 8), pady=14)

        topbar = ctk.CTkFrame(self.ctrl, fg_color="transparent")
        topbar.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkButton(topbar, text="Chọn ảnh", width=110, command=self.choose_image).pack(side="left", padx=4)
        ctk.CTkButton(topbar, text="Lưu ảnh",  width=110, command=self.save_full_res).pack(side="left", padx=4)
        ctk.CTkButton(topbar, text="Close",    width=90,  command=self.destroy).pack(side="left", padx=4)

        # fast preview toggle
        quick = ctk.CTkFrame(self.ctrl, fg_color="transparent")
        quick.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(quick, text="Hiển thị", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        self.fast_preview = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(quick, text="Fast preview (downscale)", variable=self.fast_preview,
                      command=lambda: self._schedule_preview(0)).pack(side="left", padx=10)

        # Scrollable controls
        self.scroll = ctk.CTkScrollableFrame(self.ctrl, label_text="BIẾN ĐỔI CƯỜNG ĐỘ", width=420, height=700)
        self.scroll.pack(padx=12, pady=(6, 12), fill="both", expand=True)

        # Right preview
        self.view = ctk.CTkFrame(self, corner_radius=12)
        self.view.grid(row=0, column=1, sticky="nsew", padx=(8, 14), pady=14)
        self.view.grid_rowconfigure(1, weight=1)
        self.view.grid_columnconfigure(0, weight=1)
        self.view.grid_columnconfigure(1, weight=1)

        title = ctk.CTkFrame(self.view, fg_color="transparent")
        title.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 6))
        ctk.CTkLabel(title, text="Original", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkLabel(title, text="Result (stacked)", font=ctk.CTkFont(size=16, weight="bold")).pack(side="right")

        self.left_area  = ctk.CTkLabel(self.view,  text="", corner_radius=10, fg_color="#0e0e12")
        self.right_area = ctk.CTkLabel(self.view, text="", corner_radius=10, fg_color="#0e0e12")
        self.left_area.grid (row=1, column=0, sticky="nsew", padx=(12, 6), pady=(6, 12))
        self.right_area.grid(row=1, column=1, sticky="nsew", padx=(6, 12),  pady=(6, 12))

    # ---------------- Controls ----------------
    def _header(self, text, color):
        bar = ctk.CTkFrame(self.scroll, height=30, corner_radius=8, fg_color=color)
        bar.pack(fill="x", pady=(10, 6))
        ctk.CTkLabel(bar, text="  " + text, text_color="white",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")

    def _toggle(self, text, varname, default=False):
        var = ctk.BooleanVar(value=default)
        sw = ctk.CTkSwitch(self.scroll, text=text, variable=var,
                           command=lambda: self._schedule_preview(0))
        sw.pack(anchor="w", pady=(0, 6))
        setattr(self, varname, var)

    def _slider(self, text, varname, frm, to, init, step=0.1, fmt="{:.2f}"):
        wrap = ctk.CTkFrame(self.scroll, fg_color="transparent"); wrap.pack(fill="x", pady=3)
        row  = ctk.CTkFrame(wrap,   fg_color="transparent"); row.pack(fill="x")
        ctk.CTkLabel(row, text=text).pack(side="left")
        lbl = ctk.CTkLabel(row, text=fmt.format(init)); lbl.pack(side="right")

        var = ctk.DoubleVar(value=init)
        slider = ctk.CTkSlider(
            wrap, from_=frm, to=to,
            number_of_steps=int((to-frm)/step) if step>0 else 0
        )
        slider.set(init)
        def on_change(v):
            v = float(v); var.set(v); lbl.configure(text=fmt.format(v)); self._schedule_preview(0)
        slider.configure(command=on_change)
        slider.pack(fill="x", padx=2, pady=(4, 10))
        setattr(self, varname, var)

    def _build_controls(self):
        # LOG
        self._header("LOG TRANSFORM (kênh Y – giữ màu)", "#2db6f4")
        self._toggle("Bật LOG", "log_on", default=True)
        self._slider("Hệ số c (0.1–10)", "log_c", 0.1, 10.0, 1.5, step=0.1)
        self._slider("Epsilon (0–20)",  "log_eps", 0.0, 20.0, 0.0, step=0.1)
        self._toggle("Normalize về [0,255]", "log_norm", default=False)

        # PIECEWISE
        self._header("PIECEWISE-LINEAR (kênh Y)", "#9c27b0")
        self._toggle("Bật Piecewise-Linear", "pw_on", default=False)
        self._slider("r1 (0–255)", "pw_r1", 0, 255, 70, step=1, fmt="{:.0f}")
        self._slider("s1 (0–255)", "pw_s1", 0, 255, 10, step=1, fmt="{:.0f}")
        self._slider("r2 (0–255)", "pw_r2", 0, 255, 140, step=1, fmt="{:.0f}")
        self._slider("s2 (0–255)", "pw_s2", 0, 255, 200, step=1, fmt="{:.0f}")

        # GAMMA
        self._header("GAMMA (kênh Y)", "#ff9800")
        self._toggle("Bật Gamma", "gm_on", default=False)
        self._slider("Hệ số c (0.1–3)", "gm_c", 0.1, 3.0, 1.0, step=0.1)
        self._slider("Gamma (0.1–5)",  "gm_gamma", 0.1, 5.0, 0.8, step=0.1)

    # ---------------- Resize / Preview ----------------
    def _bind_resize(self):
        def on_resize(_):
            self._schedule_preview(0)
        self.view.bind("<Configure>", on_resize)

    def _schedule_preview(self, delay_ms=0):
        if self._after_id:
            self.after_cancel(self._after_id)
        if delay_ms <= 0:
            self.update_preview()
        else:
            self._after_id = self.after(delay_ms, self.update_preview)

    def _fit_to_area(self, bgr, w_max, h_max):
        h, w = bgr.shape[:2]
        scale = min(w_max / max(w,1), h_max / max(h,1))
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    def _show_on(self, widget, bgr):
        if bgr is None:
            widget.configure(image=None, text="No Image"); return
        aw = max(200, widget.winfo_width()); ah = max(200, widget.winfo_height())
        vis = self._fit_to_area(bgr, aw-16, ah-16)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cimg = ctk.CTkImage(light_image=pil, dark_image=pil, size=(pil.width, pil.height))
        widget.configure(image=cimg, text=""); widget.image = cimg

    def _make_preview(self, bgr):
        h, w = bgr.shape[:2]
        side = max(h, w)
        if side <= self.max_preview_side:
            return bgr.copy()
        scale = self.max_preview_side / side
        return cv2.resize(bgr, (int(w*scale), int(h*scale)), cv2.INTER_AREA)

    # ---------------- File actions ----------------
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh...",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh."); return
        self.original_bgr = img
        self.preview_bgr  = self._make_preview(img)
        self.update_preview()

    def save_full_res(self):
        if self.original_bgr is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy mở ảnh trước."); return
        # render full-res với tham số hiện tại
        full = self._run_pipeline(self.original_bgr)
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(defaultextension=".png",
                                 filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("BMP","*.bmp"),("TIFF","*.tif;*.tiff")])
        if not path: return
        if cv2.imwrite(path, full):
            messagebox.showinfo("Đã lưu", f"Lưu ảnh thành công:\n{path}")
        else:
            messagebox.showerror("Lỗi", "Không lưu được ảnh.")

    # ---------------- Core pipeline ----------------
    def _run_pipeline(self, src_bgr):
        # build LUTs theo trạng thái UI
        luts = []
        if self.log_on.get():
            luts.append(build_log_lut(c=float(self.log_c.get()),
                                      eps=float(self.log_eps.get()),
                                      normalize=bool(self.log_norm.get())))
        if self.pw_on.get():
            luts.append(build_piecewise_lut(r1=int(self.pw_r1.get()),
                                            s1=int(self.pw_s1.get()),
                                            r2=int(self.pw_r2.get()),
                                            s2=int(self.pw_s2.get())))
        if self.gm_on.get():
            luts.append(build_gamma_lut(c=float(self.gm_c.get()),
                                        gamma=float(self.gm_gamma.get())))
        # áp lần lượt trên kênh Y
        out = apply_luts_on_luma(src_bgr, luts)
        return out

    def update_preview(self):
        if self.original_bgr is None:
            self._show_on(self.left_area, None); self._show_on(self.right_area, None); return

        # chọn ảnh nguồn cho preview
        src = self.preview_bgr if self.fast_preview.get() and self.preview_bgr is not None else self.original_bgr
        left = src.copy()
        out  = self._run_pipeline(src)

        self.current_bgr = out
        self._show_on(self.left_area, left)
        self._show_on(self.right_area, out)

# ---------------- RUN ----------------
if __name__ == "__main__":
    cv2.setUseOptimized(True)
    app = DemoApp()
    app.mainloop()
