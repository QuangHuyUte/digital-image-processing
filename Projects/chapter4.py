import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


# ===========================
#   Helper: CV -> QPixmap
# ===========================
def cvimg_to_qpix(img):
    """
    Convert a grayscale or BGR OpenCV image to QPixmap.
    """
    if img is None:
        return QtGui.QPixmap()

    if len(img.shape) == 2:
        h, w = img.shape
        bytes_per_line = w
        qimg = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        if ch == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported channel number: {}".format(ch))
    return QtGui.QPixmap.fromImage(qimg.copy())


def ensure_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_structuring_element(shape_name, ksize):
    """
    shape_name: 'Rectangle', 'Ellipse', 'Cross'
    ksize: integer size (kernel will be ksize x ksize)
    """
    ksize = max(1, int(ksize))
    if shape_name == "Ellipse":
        shape = cv2.MORPH_ELLIPSE
    elif shape_name == "Cross":
        shape = cv2.MORPH_CROSS
    else:
        shape = cv2.MORPH_RECT
    return cv2.getStructuringElement(shape, (ksize, ksize))


# ===========================
#   Custom Image Label
# ===========================
class ImageLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        # khung to hơn
        self.setMinimumSize(700, 700)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._pixmap = None

        # Shadow effect để nhìn như card
        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(24)
        effect.setOffset(0, 0)
        effect.setColor(QtGui.QColor(0, 0, 0, 180))
        self.setGraphicsEffect(effect)

    def setPixmap(self, pixmap: QtGui.QPixmap):
        self._pixmap = pixmap
        super().setPixmap(self.scaled_pixmap())

    def resizeEvent(self, event):
        if self._pixmap is not None:
            super().setPixmap(self.scaled_pixmap())
        super().resizeEvent(event)

    def scaled_pixmap(self):
        if self._pixmap is None:
            return QtGui.QPixmap()
        return self._pixmap.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )


# ===========================
#   Main App
# ===========================
class MorphologyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Morphological Demo – Erosion, Dilation, Opening, Closing")

        self.original_img = None  # BGR
        self.gray_img = None
        self.current_result = None

        self._init_ui()

    # ---------- UI ----------
    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ==== Left control panel ====
        control_panel = QtWidgets.QFrame()
        control_panel.setObjectName("controlPanel")
        control_panel.setMinimumWidth(340)
        control_panel.setMaximumWidth(380)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(18, 18, 18, 18)
        control_layout.setSpacing(14)

        title = QtWidgets.QLabel("Morphological Operations")
        title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        font = title.font()
        font.setPointSize(18)
        font.setBold(True)
        title.setFont(font)
        control_layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Real-time demo of Erosion, Dilation, Opening, and Closing.\n"
            "Use the sliders to explore effects interactively."
        )
        subtitle.setWordWrap(True)
        control_layout.addWidget(subtitle)

        # --- Load/Save ---
        btn_box = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Image")
        self.btn_save = QtWidgets.QPushButton("Save Result")
        self.btn_load.setObjectName("primaryButton")
        self.btn_save.setObjectName("secondaryButton")
        btn_box.addWidget(self.btn_load)
        btn_box.addWidget(self.btn_save)
        control_layout.addLayout(btn_box)

        # --- Binarization group ---
        bin_group = QtWidgets.QGroupBox("Binarization")
        bin_layout = QtWidgets.QVBoxLayout(bin_group)

        self.chk_auto_thresh = QtWidgets.QCheckBox("Use Otsu automatic threshold")
        self.chk_auto_thresh.setChecked(True)
        bin_layout.addWidget(self.chk_auto_thresh)

        thresh_line = QtWidgets.QHBoxLayout()
        self.lbl_thresh = QtWidgets.QLabel("Threshold: 128")
        self.slider_thresh = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(128)
        self.slider_thresh.setEnabled(False)
        thresh_line.addWidget(self.lbl_thresh)
        thresh_line.addWidget(self.slider_thresh)
        bin_layout.addLayout(thresh_line)

        info_thresh = QtWidgets.QLabel("Left image: original grayscale.\n"
                                       "Right image: binary + morphological result.")
        info_thresh.setWordWrap(True)
        bin_layout.addWidget(info_thresh)

        control_layout.addWidget(bin_group)

        # --- SE group ---
        se_group = QtWidgets.QGroupBox("Structuring Element (SE)")
        se_layout = QtWidgets.QVBoxLayout(se_group)

        # shape selection as "pill" buttons
        shape_label = QtWidgets.QLabel("Shape:")
        se_layout.addWidget(shape_label)

        shape_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_rect = QtWidgets.QPushButton("Rectangle")
        self.btn_ellipse = QtWidgets.QPushButton("Ellipse")
        self.btn_cross = QtWidgets.QPushButton("Cross")
        for b in (self.btn_rect, self.btn_ellipse, self.btn_cross):
            b.setCheckable(True)
            b.setObjectName("pillButton")
            shape_btn_layout.addWidget(b)
        self.btn_rect.setChecked(True)
        self.shape_group = QtWidgets.QButtonGroup()
        self.shape_group.addButton(self.btn_rect, 0)
        self.shape_group.addButton(self.btn_ellipse, 1)
        self.shape_group.addButton(self.btn_cross, 2)
        se_layout.addLayout(shape_btn_layout)

        # size slider
        size_line = QtWidgets.QHBoxLayout()
        self.lbl_se_size = QtWidgets.QLabel("Size: 3 x 3")
        self.slider_se_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_se_size.setRange(1, 31)  # 1..31
        self.slider_se_size.setValue(3)
        size_line.addWidget(self.lbl_se_size)
        size_line.addWidget(self.slider_se_size)
        se_layout.addLayout(size_line)

        # iteration slider
        iter_line = QtWidgets.QHBoxLayout()
        self.lbl_iterations = QtWidgets.QLabel("Iterations: 1")
        self.slider_iterations = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_iterations.setRange(1, 10)
        self.slider_iterations.setValue(1)
        iter_line.addWidget(self.lbl_iterations)
        iter_line.addWidget(self.slider_iterations)
        se_layout.addLayout(iter_line)

        control_layout.addWidget(se_group)

        # --- Operation group ---
        op_group = QtWidgets.QGroupBox("Operation")
        op_layout = QtWidgets.QVBoxLayout(op_group)

        self.combo_operation = QtWidgets.QComboBox()
        self.combo_operation.addItems([
            "Erosion",
            "Dilation",
            "Opening",
            "Closing"
        ])
        op_layout.addWidget(self.combo_operation)

        self.lbl_formula = QtWidgets.QLabel("")
        self.lbl_formula.setWordWrap(True)
        self.lbl_formula.setObjectName("formulaLabel")
        op_layout.addWidget(self.lbl_formula)

        control_layout.addWidget(op_group)

        # status hint
        self.lbl_status_hint = QtWidgets.QLabel(
            "Hint: adjust sliders and see the right image update immediately."
        )
        self.lbl_status_hint.setWordWrap(True)
        control_layout.addWidget(self.lbl_status_hint)

        control_layout.addStretch()

        # ==== Right panel: images (centered) ====
        view_panel = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(view_panel)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(8)

        # thêm stretch trên và dưới để cụm ảnh nằm giữa dọc
        view_layout.addStretch(1)

        # Titles
        titles_layout = QtWidgets.QHBoxLayout()
        self.lbl_original_title = QtWidgets.QLabel("Original (Grayscale)")
        self.lbl_result_title = QtWidgets.QLabel("Morphological Result")

        for lbl in (self.lbl_original_title, self.lbl_result_title):
            lf = lbl.font()
            lf.setPointSize(13)
            lf.setBold(True)
            lbl.setFont(lf)

        titles_layout.addWidget(self.lbl_original_title, alignment=QtCore.Qt.AlignLeft)
        titles_layout.addStretch()
        titles_layout.addWidget(self.lbl_result_title, alignment=QtCore.Qt.AlignRight)

        # Image area
        img_layout = QtWidgets.QHBoxLayout()
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(84)
        img_layout.setAlignment(QtCore.Qt.AlignCenter)

        self.img_original = ImageLabel()
        self.img_result = ImageLabel()
        self.img_original.setObjectName("imageCard")
        self.img_result.setObjectName("imageCard")
        img_layout.addWidget(self.img_original, 1)
        img_layout.addWidget(self.img_result, 1)

        # gói titles + images lại trong 1 widget để căn giữa
        images_container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(images_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)
        container_layout.addLayout(titles_layout)
        container_layout.addLayout(img_layout)

        view_layout.addWidget(images_container, alignment=QtCore.Qt.AlignCenter)

        view_layout.addStretch(1)

        # add 2 panel vào main layout, cho panel phải chiếm nhiều chỗ hơn
        main_layout.addWidget(control_panel)
        main_layout.addWidget(view_panel, 1)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        # --- Status bar ---
        self.statusBar().showMessage("Load an image to start.")

        # Connections
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_result)

        self.slider_thresh.valueChanged.connect(self.on_controls_changed)
        self.chk_auto_thresh.toggled.connect(self.on_auto_thresh_toggled)

        self.slider_se_size.valueChanged.connect(self.on_controls_changed)
        self.slider_iterations.valueChanged.connect(self.on_controls_changed)

        self.shape_group.idClicked.connect(self.on_controls_changed)
        self.combo_operation.currentIndexChanged.connect(self.on_controls_changed)

        # Init formula text
        self.update_formula_label()

    # ---------- Logic helpers ----------
    def get_se_shape_name(self):
        if self.btn_ellipse.isChecked():
            return "Ellipse"
        elif self.btn_cross.isChecked():
            return "Cross"
        return "Rectangle"

    def get_binary_image(self):
        if self.gray_img is None:
            return None
        if self.chk_auto_thresh.isChecked():
            _, bin_img = cv2.threshold(
                self.gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            t = self.slider_thresh.value()
            _, bin_img = cv2.threshold(self.gray_img, t, 255, cv2.THRESH_BINARY)
        return bin_img

    def update_original_view(self):
        if self.gray_img is None:
            return
        # ảnh gốc grayscale
        pix = cvimg_to_qpix(self.gray_img)
        self.img_original.setPixmap(pix)

    def apply_morph_operation(self):
        if self.gray_img is None:
            return None

        bin_img = self.get_binary_image()
        op = self.combo_operation.currentText()
        size = self.slider_se_size.value()
        iterations = self.slider_iterations.value()
        shape_name = self.get_se_shape_name()

        kernel = get_structuring_element(shape_name, size)

        if op == "Erosion":
            result = cv2.erode(bin_img, kernel, iterations=iterations)
        elif op == "Dilation":
            result = cv2.dilate(bin_img, kernel, iterations=iterations)
        elif op == "Opening":
            result = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif op == "Closing":
            result = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            result = bin_img
        return result

    def update_result_view(self):
        if self.gray_img is None:
            return
        result = self.apply_morph_operation()
        if result is None:
            return
        self.current_result = result
        pix = cvimg_to_qpix(result)
        self.img_result.setPixmap(pix)

    def update_formula_label(self):
        op = self.combo_operation.currentText()
        if op == "Erosion":
            text = "Erosion (⊖): A ⊖ B = { z | B_z ⊆ A } – shrinks bright objects."
        elif op == "Dilation":
            text = "Dilation (⊕): A ⊕ B = { z | (B̂)_z ∩ A ≠ ∅ } – grows bright objects."
        elif op == "Opening":
            text = "Opening: A ∘ B = (A ⊖ B) ⊕ B – removes small bright noise, smooths contours."
        elif op == "Closing":
            text = "Closing: A • B = (A ⊕ B) ⊖ B – fills small dark holes, connects gaps."
        else:
            text = ""
        self.lbl_formula.setText(text)

    # ---------- Slots ----------
    def on_auto_thresh_toggled(self, checked):
        # Otsu bật -> khóa slider; tắt -> slider hoạt động
        self.slider_thresh.setEnabled(not checked)
        self.on_controls_changed()

    def on_controls_changed(self, *args):
        # Update labels
        self.lbl_thresh.setText(f"Threshold: {self.slider_thresh.value()}")
        size = self.slider_se_size.value()
        self.lbl_se_size.setText(f"Size: {size} x {size}")
        self.lbl_iterations.setText(f"Iterations: {self.slider_iterations.value()}")

        self.update_formula_label()

        # Recompute views in real-time
        if self.gray_img is not None:
            self.update_original_view()
            self.update_result_view()

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Could not load image.")
            return
        self.original_img = img
        self.gray_img = ensure_gray(img)
        self.statusBar().showMessage(f"Loaded: {path}")
        self.update_original_view()
        self.update_result_view()

    def save_result(self):
        if self.current_result is None:
            QtWidgets.QMessageBox.information(self, "Info", "No result to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Result", "result.png", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        cv2.imwrite(path, self.current_result)
        QtWidgets.QMessageBox.information(self, "Saved", f"Result saved to:\n{path}")


# ===========================
#   Main
# ===========================
def main():
    app = QtWidgets.QApplication(sys.argv)

    # --- Global modern dark style ---
    app.setStyle("Fusion")

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 33, 40))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(24, 26, 32))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(30, 33, 40))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(40, 44, 52))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(100, 180, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    # Stylesheet cho UI đẹp hơn
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1E2128;
        }
        #controlPanel {
            background-color: #252934;
            border-radius: 16px;
        }
        QGroupBox {
            border: 1px solid #3A3F4B;
            border-radius: 10px;
            margin-top: 12px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: #D6D9E0;
            font-weight: bold;
        }
        QLabel {
            color: #D6D9E0;
        }
        #formulaLabel {
            color: #9FA5B5;
            font-size: 11px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3A3F4B;
            height: 6px;
            background: #3A3F4B;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #64B5F6;
            border: 1px solid #64B5F6;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #90CAF9;
            border-color: #90CAF9;
        }
        #primaryButton {
            background-color: #64B5F6;
            color: #0B1018;
            border-radius: 20px;
            padding: 8px 16px;
            font-weight: bold;
        }
        #primaryButton:hover {
            background-color: #90CAF9;
        }
        #secondaryButton {
            background-color: #3A3F4B;
            color: #E0E4EC;
            border-radius: 20px;
            padding: 8px 16px;
        }
        #secondaryButton:hover {
            background-color: #4A5060;
        }
        #pillButton {
            background-color: #2F3441;
            border-radius: 18px;
            padding: 6px 12px;
            color: #D6D9E0;
            border: 1px solid transparent;
        }
        #pillButton:hover {
            background-color: #3A3F4B;
        }
        #pillButton:checked {
            background-color: #64B5F6;
            color: #0B1018;
            border: 1px solid #90CAF9;
        }
        QComboBox {
            background-color: #2F3441;
            border: 1px solid #3A3F4B;
            border-radius: 8px;
            padding: 4px 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #2F3441;
            selection-background-color: #64B5F6;
            selection-color: #0B1018;
        }
        QCheckBox {
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid #868E96;
            background-color: transparent;
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #64B5F6;
            background-color: #64B5F6;
        }
        #imageCard {
            border-radius: 16px;
            background-color: #181A20;
        }
        QStatusBar {
            background-color: #252934;
            color: #D6D9E0;
        }
    """)

    win = MorphologyApp()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
