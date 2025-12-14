<!-- Copilot instructions for contributors and AI coding agents -->
# Guidance for AI coding agents

Summary
- This repository contains small GUI demos for realtime image processing using OpenCV + CustomTkinter.
- Main entry points: `image_gui.py`, `image_gui_pretty.py`, `image_more_functions.py` (each is runnable with `python <file>`).

Quick commands
- Install deps: `pip install -r requirements.txt`
- Run demo: `python image_gui.py` (other files are alternative demos)

Big picture / architecture
- UI layer: implemented with `customtkinter` in each `*gui*.py` file. The GUI creates controls (toggles/sliders) and preview areas.
- Processing layer: image transforms are implemented as small pure helpers at the top of the same files (examples: `apply_on_luma`, `build_log_lut`, `build_gamma_lut`, `build_piecewise_lut`, `hist_equalize_luma`). Prefer editing these helpers when changing algorithms.
- Pipeline pattern: GUI reads control state → builds parameterized transforms (LUTs or functions) → applies them to an input image via a single pipeline function (examples: `_run_pipeline` in `image_gui_pretty.py`, `update_preview` in `image_gui.py`). Keep changes localized to the pipeline or helper functions.

Project-specific conventions & patterns
- Color-preserving transforms operate on the luma (Y) channel: code converts BGR ↔ YCrCb, modifies Y, then merges back (see `apply_on_luma` / `apply_luts_on_luma`). Follow this pattern to preserve colors.
- Naming conventions in GUIs:
  - boolean toggles commonly end with `_on` or `_enable` (e.g., `log_on`, `pw_enable`).
  - kernel sizes use `_k` or `_ksize` and are expected to be odd; helpers include `_ensure_odd` or manual checks.
  - parameters: `_c`, `_gamma`, `_eps`, `_sigma`, `_clip`, `_tile`.
- Preview behaviour: a `fast_preview` toggle uses a downscaled image (`preview_bgr`, `preview_base`, `max_preview_side`) for fast UI updates. Respect this when adding heavy/slow ops — implement a fast approximation for preview and full-res rendering in save functions.
- Saving: Save actions use full-resolution re-run of the pipeline (see `save_full_res` / `save_image`) — do not assume preview image equals saved result.

Files to look at for examples
- `image_more_functions.py` — rich collection of LUT builders, spatial filters, and the `DemoApp` implementation (color-preserving LUT pipeline).
- `image_gui.py` — stacked original/result UI and a pipeline built inside `update_preview()`; shows mixing BGR and Y-channel ops.
- `image_gui_pretty.py` — another demo focused on intensity transforms with `_run_pipeline()` function (good model for extracting pure pipeline code).
- `requirements.txt` — dependency list: `customtkinter`, `opencv-python`, `pillow`, `numpy`.

Guidance for code changes
- Prefer editing the small helper functions (top of each file) for algorithmic changes. Those functions are easy to unit-test outside the GUI.
- When adding features that affect UI state, mirror existing naming and layout patterns (use `_slider`, `_toggle` helpers where present).
- If adding computationally expensive operations, add a fast-preview path (use `fast_preview` toggle and a lower-resolution pipeline) and ensure full-res rendering is performed only in save/export.

Integration points & external assumptions
- Image IO uses OpenCV (`cv2.imread`, `cv2.imwrite`) and expects BGR arrays.
- UI images are shown by converting BGR→RGB→PIL→`ctk.CTkImage`.
- File dialogs use `tkinter.filedialog`; error messages use `tkinter.messagebox`.

What I couldn't discover automatically
- There are no tests, CI config, or additional build scripts. If you need test or CI guidance, say so and I'll propose a minimal setup.

If something is unclear or you want a different focus (tests, CI, refactor into modules), tell me which parts to expand or any style preferences.
