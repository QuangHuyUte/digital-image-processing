import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps
import os
import time
import threading

# Cấu hình giao diện chuẩn
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SubWindow(ctk.CTkToplevel):
    """
    Class tùy biến cho cửa sổ con để quản lý hành vi tốt hơn.
    Đảm bảo cửa sổ nổi lên trên và không bị lỗi mất ảnh.
    """
    def __init__(self, parent, title, geometry="800x600"):
        super().__init__(parent)
        self.title(title)
        self.geometry(geometry)
        self.lift()  # Đưa cửa sổ lên trên cùng
        self.focus_force() # Ép trỏ chuột vào cửa sổ này
        self.after(100, self.lift) # Đảm bảo nổi lên lần nữa sau khi render

class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Cấu hình cửa sổ chính ---
        self.title("Phần Mềm Xử Lý Ảnh Chuyên Nghiệp v2.0")
        self.geometry("1100x750")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Biến dữ liệu
        self.current_image_path = None
        self.original_pil_image = None # Ảnh gốc chất lượng cao
        self.display_image_ref = None  # Giữ tham chiếu ảnh để không bị mất

        # --- GIAO DIỆN: SIDEBAR (Trái) ---
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(3, weight=1)

        lbl_logo = ctk.CTkLabel(self.sidebar, text="AI IMAGE LAB", font=ctk.CTkFont(size=24, weight="bold"))
        lbl_logo.grid(row=0, column=0, padx=20, pady=(30, 10))

        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Tải Thư Mục Ảnh", height=40, command=self.load_images)
        self.btn_load.grid(row=1, column=0, padx=20, pady=20)

        lbl_list = ctk.CTkLabel(self.sidebar, text="Danh sách tập tin:", anchor="w")
        lbl_list.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        self.scroll_list = ctk.CTkScrollableFrame(self.sidebar, label_text="")
        self.scroll_list.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        # --- GIAO DIỆN: MAIN AREA (Phải) ---
        self.main_area = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.main_area.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # Khung hiển thị ảnh
        self.preview_frame = ctk.CTkFrame(self.main_area, corner_radius=15, border_width=2, border_color="#3B8ED0")
        self.preview_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        self.lbl_display = ctk.CTkLabel(self.preview_frame, text="<< Vui lòng chọn ảnh từ danh sách bên trái >>")
        self.lbl_display.place(relx=0.5, rely=0.5, anchor="center")

        # Khung công cụ (Các nút bấm)
        self.tools_frame = ctk.CTkFrame(self.main_area, height=120, corner_radius=10)
        self.tools_frame.grid(row=1, column=0, sticky="ew")
        
        # Tạo các nút chức năng
        self.create_tool_buttons()

    def create_tool_buttons(self):
        buttons = [
            ("Tách Lớp RGB (Màu)", self.open_rgb_window),
            ("Ảnh Xám (Greyscale)", self.open_grey_window),
            ("Xoay Ảnh (Animation)", self.open_rotate_window),
            ("Cắt 1/4 Tâm (Crop)", self.open_crop_window)
        ]
        
        self.tool_btns = []
        for i, (text, cmd) in enumerate(buttons):
            btn = ctk.CTkButton(self.tools_frame, text=text, command=cmd, state="disabled", height=50, font=("Arial", 14))
            btn.pack(side="left", padx=10, pady=20, expand=True, fill="x")
            self.tool_btns.append(btn)

    def load_images(self):
        """HW1: Tải danh sách ảnh"""
        path = ctk.filedialog.askdirectory()
        if not path: return

        # Xóa danh sách cũ
        for w in self.scroll_list.winfo_children(): w.destroy()
        
        valid_exts = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(path) if f.lower().endswith(valid_exts)]

        if not files:
            tk.messagebox.showwarning("Cảnh báo", "Không tìm thấy ảnh nào trong thư mục này!")
            return

        for f in files:
            full_path = os.path.join(path, f)
            btn = ctk.CTkButton(
                self.scroll_list, 
                text=f, 
                fg_color="transparent", 
                border_width=1, 
                border_color="gray",
                text_color=("black", "white"),
                anchor="w",
                command=lambda p=full_path: self.display_main_image(p)
            )
            btn.pack(fill="x", pady=2)

    def display_main_image(self, path):
        """Hiển thị ảnh lên màn hình chính"""
        try:
            self.current_image_path = path
            pil_img = Image.open(path)
            self.original_pil_image = pil_img.copy() # Lưu bản gốc
            
            # Resize thông minh để vừa khung nhìn mà không vỡ tỉ lệ
            display_img = self.resize_image_maintain_aspect(pil_img, 800, 500)
            
            ctk_img = ctk.CTkImage(display_img, size=display_img.size)
            self.lbl_display.configure(image=ctk_img, text="")
            self.display_image_ref = ctk_img # QUAN TRỌNG: Giữ tham chiếu

            # Mở khóa các nút
            for btn in self.tool_btns: btn.configure(state="normal")

        except Exception as e:
            print(f"Lỗi: {e}")

    def resize_image_maintain_aspect(self, img, max_w, max_h):
        ratio = min(max_w/img.width, max_h/img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    # ================= CÁC CHỨC NĂNG XỬ LÝ (HW2) =================

    def open_rgb_window(self):
        """Tách 3 lớp RGB và hiển thị CÓ MÀU (Không bị xám)"""
        if not self.original_pil_image: return

        win = SubWindow(self, "Phân Tách Lớp Màu RGB", "1000x450")
        
        # Tách kênh
        r, g, b = self.original_pil_image.split()
        zero = Image.new("L", r.size, 0) # Tạo kênh rỗng (đen)

        # Hợp nhất lại để tạo hiệu ứng màu thị giác
        # Kênh Đỏ: (R, 0, 0)
        img_r = Image.merge("RGB", (r, zero, zero))
        # Kênh Xanh lá: (0, G, 0)
        img_g = Image.merge("RGB", (zero, g, zero))
        # Kênh Xanh dương: (0, 0, B)
        img_b = Image.merge("RGB", (zero, zero, b))

        # Hiển thị 3 ảnh
        images = [("Kênh RED", img_r), ("Kênh GREEN", img_g), ("Kênh BLUE", img_b)]
        
        win.grid_columnconfigure((0,1,2), weight=1)
        
        for idx, (title, img) in enumerate(images):
            # Resize nhỏ để hiển thị
            thumb = self.resize_image_maintain_aspect(img, 300, 300)
            ctk_thumb = ctk.CTkImage(thumb, size=thumb.size)
            
            lbl_t = ctk.CTkLabel(win, text=title, font=("Arial", 16, "bold"))
            lbl_t.grid(row=0, column=idx, pady=10)
            
            lbl_i = ctk.CTkLabel(win, image=ctk_thumb, text="")
            lbl_i.grid(row=1, column=idx, padx=10)
            
            # Lưu tham chiếu cục bộ vào widget để không bị mất ảnh
            lbl_i.image = ctk_thumb 

    def open_grey_window(self):
        """Chuyển ảnh xám"""
        if not self.original_pil_image: return
        
        win = SubWindow(self, "Ảnh Đen Trắng (Grayscale)")
        
        grey_img = self.original_pil_image.convert("L")
        
        # Hiển thị
        display = self.resize_image_maintain_aspect(grey_img, 750, 550)
        ctk_img = ctk.CTkImage(display, size=display.size)
        
        lbl = ctk.CTkLabel(win, image=ctk_img, text="")
        lbl.pack(expand=True)
        lbl.image = ctk_img # Giữ tham chiếu

    def open_crop_window(self):
        """Cắt 1/4 từ tâm"""
        if not self.original_pil_image: return
        
        win = SubWindow(self, "Kết quả Cắt 1/4 Tâm")
        
        w, h = self.original_pil_image.size
        
        # Logic: 1/4 diện tích tức là chiều dài / 2 và chiều rộng / 2
        # Tọa độ crop: (left, top, right, bottom)
        left = w / 4
        top = h / 4
        right = w * 3/4
        bottom = h * 3/4
        
        cropped = self.original_pil_image.crop((left, top, right, bottom))
        
        # Hiển thị
        display = self.resize_image_maintain_aspect(cropped, 600, 600)
        ctk_img = ctk.CTkImage(display, size=display.size)
        
        lbl = ctk.CTkLabel(win, image=ctk_img, text="")
        lbl.pack(expand=True)
        lbl.image = ctk_img # Giữ tham chiếu

    def open_rotate_window(self):
        """Xoay ảnh Animation"""
        if not self.original_pil_image: return
        
        # Tạo cửa sổ mới
        self.rot_win = SubWindow(self, "Xoay Ảnh (Nhấn ESC để dừng sớm)", "600x650")
        
        self.lbl_rot = ctk.CTkLabel(self.rot_win, text="Đang chuẩn bị xoay...", font=("Arial", 14))
        self.lbl_rot.pack(pady=10)
        
        self.lbl_rot_img = ctk.CTkLabel(self.rot_win, text="")
        self.lbl_rot_img.pack(expand=True)
        
        # Cờ điều khiển
        self.stop_rotation = False
        
        # Bắt sự kiện ESC và đóng cửa sổ
        def stop_process(event=None):
            self.stop_rotation = True
            
        self.rot_win.bind("<Escape>", stop_process)
        self.rot_win.protocol("WM_DELETE_WINDOW", lambda: [stop_process(), self.rot_win.destroy()])
        
        # Nút dừng thủ công
        btn_stop = ctk.CTkButton(self.rot_win, text="Dừng Xoay", command=stop_process, fg_color="red")
        btn_stop.pack(pady=10)

        # Chạy luồng xoay
        threading.Thread(target=self.run_rotation_logic, daemon=True).start()

    def run_rotation_logic(self):
        """Logic xoay ảnh chạy ngầm"""
        angle = 0
        img_copy = self.original_pil_image.copy()
        
        # Resize trước khi xoay để mượt mà hơn
        img_copy.thumbnail((500, 500))
        
        for i in range(101): # 0 -> 100 lần
            if self.stop_rotation: break
            if not self.rot_win.winfo_exists(): break
            
            # Xoay
            rotated = img_copy.rotate(angle)
            ctk_rot = ctk.CTkImage(rotated, size=rotated.size)
            
            # Cập nhật GUI từ luồng phụ
            try:
                self.lbl_rot.configure(text=f"Lần: {i}/100 - Góc: {angle}°")
                self.lbl_rot_img.configure(image=ctk_rot)
                self.lbl_rot_img.image = ctk_rot # Giữ tham chiếu
            except:
                break
                
            angle += 5
            time.sleep(0.1)
            
        if not self.stop_rotation and self.rot_win.winfo_exists():
            self.lbl_rot.configure(text="Đã hoàn tất 100 lần xoay!")

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()