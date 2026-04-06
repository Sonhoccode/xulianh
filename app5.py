import os
import cv2
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ultralytics import YOLO
import threading

# ================== CONFIG ==================
MODEL1_PATH = r"runs/detect/YOLO_Stage2/weights/best.pt"    # Model 1
MODEL2_PATH = r"runs/detect/yolov5_run/weights/best.pt"     # Model 2

FONT_PATH = "Arial.Unicode.ttf"   # Font tiếng Việt cho annotate
CONF_THRES = 0.2                  # Ngưỡng confidence
# ============================================

print("Đang tải model 1...")
try:
    model1 = YOLO(MODEL1_PATH)
    print("Tải model 1 thành công!")
except Exception as e:
    print(f"LỖI: Không thể load model 1 '{MODEL1_PATH}'!")
    messagebox.showerror("Lỗi model 1", str(e))
    exit()

print("Đang tải model 2...")
try:
    model2 = YOLO(MODEL2_PATH)
    print("Tải model 2 thành công!")
except Exception as e:
    print(f"LỖI: Không thể load model 2 '{MODEL2_PATH}'!")
    messagebox.showerror("Lỗi model 2", str(e))
    exit()


class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("So sánh 2 model YOLO - Nhận diện biển báo giao thông")
        self.root.geometry("1300x800")

        # Gắn model vào instance (cho dễ quản lý)
        self.model1 = model1
        self.model2 = model2

        self.cap_cam = None
        self.cap_vid = None
        self.is_cam_running = False
        self.is_vid_running = False

        # Tabs
        self.tab = ttk.Notebook(root, bootstyle="primary")
        self.tab_img = ttk.Frame(self.tab, padding=10)
        self.tab_cam = ttk.Frame(self.tab, padding=10)
        self.tab_vid = ttk.Frame(self.tab, padding=10)

        self.tab.add(self.tab_img, text=" Ảnh (So sánh 2 model) ")
        self.tab.add(self.tab_cam, text=" Webcam (So sánh 2 model) ")
        self.tab.add(self.tab_vid, text=" Video (So sánh 2 model) ")
        self.tab.pack(expand=True, fill="both")

        self.setup_image_tab()
        self.setup_webcam_tab()
        self.setup_video_tab()

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # =========================================================
    # TAB ẢNH – hiển thị 2 model cạnh nhau
    # =========================================================
    def setup_image_tab(self):
        top_frame = ttk.Frame(self.tab_img)
        top_frame.pack(fill="x", pady=10)

        title_label = ttk.Label(
            top_frame,
            text="So sánh 2 model YOLO trên cùng một ảnh",
            font=("Arial", 22, "bold"),
            bootstyle="primary"
        )
        title_label.pack(side="left", padx=10)

        btn = ttk.Button(
            top_frame,
            text="Chọn ảnh",
            bootstyle="success",
            command=self.load_image
        )
        btn.pack(side="right", padx=10)

        # ===== Khung chính 2 cột bằng nhau =====
        main_frame = ttk.Frame(self.tab_img)
        main_frame.pack(expand=True, fill="both", pady=10)

        # cấu hình 2 cột = nhau
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # ===== CỘT MODEL 1 =====
        left_frame = ttk.Labelframe(
            main_frame,
            text=f"YOLOv8: {os.path.basename(MODEL1_PATH)}",
            padding=5,
            bootstyle="info"
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5)

        self.img_label_1 = ttk.Label(
            left_frame,
            text="Ảnh model 1 sẽ hiển thị ở đây",
            relief="solid"
        )
        self.img_label_1.pack(padx=5, pady=5, expand=True, fill="both")

        ttk.Label(left_frame, text="Kết quả model 1:", bootstyle="secondary").pack(anchor="w", padx=5)
        self.text_img_1 = ttk.Text(left_frame, height=6, state="disabled")
        self.text_img_1.pack(fill="x", padx=5, pady=5)
        self.text_img_1.config(font=("Arial", 14, "bold"))

        # ===== CỘT MODEL 2 =====
        right_frame = ttk.Labelframe(
            main_frame,
            text=f"YOLOv5: {os.path.basename(MODEL2_PATH)}",
            padding=5,
            bootstyle="warning"
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        self.img_label_2 = ttk.Label(
            right_frame,
            text="Ảnh model 2 sẽ hiển thị ở đây",
            relief="solid"
        )
        self.img_label_2.pack(padx=5, pady=5, expand=True, fill="both")

        ttk.Label(right_frame, text="Kết quả model 2:", bootstyle="secondary").pack(anchor="w", padx=5)
        self.text_img_2 = ttk.Text(right_frame, height=6, state="disabled")
        self.text_img_2.pack(fill="x", padx=5, pady=5)
        self.text_img_2.config(font=("Arial", 14, "bold"))

    # =========================================================
    # TAB WEBCAM – 2 model cạnh nhau
    # =========================================================
    def setup_webcam_tab(self):
        main_frame = ttk.Frame(self.tab_cam)
        main_frame.pack(expand=True, fill="both", pady=10)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Cột model 1
        left_frame = ttk.Labelframe(
            main_frame,
            text=f"Model 1: {os.path.basename(MODEL1_PATH)}",
            padding=5,
            bootstyle="info"
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5)

        self.cam_label_1 = ttk.Label(
            left_frame,
            text="Webcam model 1 sẽ hiển thị ở đây",
            relief="solid"
        )
        self.cam_label_1.pack(padx=5, pady=5, expand=True, fill="both")

        ttk.Label(left_frame, text="Thông tin model 1:", bootstyle="secondary").pack(anchor="w", padx=5)
        self.text_cam_1 = ttk.Text(left_frame, height=6, state="disabled")
        self.text_cam_1.pack(fill="x", padx=5, pady=5)
        self.text_cam_1.config(font=("Arial", 14, "bold"))

        # Cột model 2
        right_frame = ttk.Labelframe(
            main_frame,
            text=f"Model 2: {os.path.basename(MODEL2_PATH)}",
            padding=5,
            bootstyle="warning"
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        self.cam_label_2 = ttk.Label(
            right_frame,
            text="Webcam model 2 sẽ hiển thị ở đây",
            relief="solid"
        )
        self.cam_label_2.pack(padx=5, pady=5, expand=True, fill="both")

        ttk.Label(right_frame, text="Thông tin model 2:", bootstyle="secondary").pack(anchor="w", padx=5)
        self.text_cam_2 = ttk.Text(right_frame, height=6, state="disabled")
        self.text_cam_2.pack(fill="x", padx=5, pady=5)
        self.text_cam_2.config(font=("Arial", 14, "bold"))

        btn_frame = ttk.Frame(self.tab_cam)
        btn_frame.pack(pady=5)

        start_btn = ttk.Button(btn_frame, text="Bật camera", bootstyle="success", command=self.start_cam)
        stop_btn = ttk.Button(btn_frame, text="Tắt", bootstyle="danger", command=self.stop_cam)

        start_btn.grid(row=0, column=0, padx=20, pady=10)
        stop_btn.grid(row=0, column=1, padx=20, pady=10)

   
    # =========================================================
    # TAB VIDEO – 2 model cạnh nhau
    # =========================================================
    def setup_video_tab(self):
        main_frame = ttk.Frame(self.tab_vid)
        main_frame.pack(expand=True, fill="both", pady=10)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Cột model 1
        left_frame = ttk.Labelframe(
            main_frame,
            text=f"YOLOv8: {os.path.basename(MODEL1_PATH)}",
            padding=5,
            bootstyle="info"
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5)

        self.vid_label_1 = ttk.Label(
            left_frame,
            text="Video YOLOv8 sẽ hiển thị ở đây",
            relief="solid"
        )
        self.vid_label_1.pack(padx=5, pady=5, expand=True, fill="both")

        ttk.Label(left_frame, text="Thông tin model 1:", bootstyle="secondary").pack(anchor="w", padx=5)
        self.text_vid_1 = ttk.Text(left_frame, height=6, state="disabled")
        self.text_vid_1.pack(fill="x", padx=5, pady=5)
        self.text_vid_1.config(font=("Arial", 14, "bold"))

        # Cột model 2
        right_frame = ttk.Labelframe(
            main_frame,
            text=f"YOLOv5: {os.path.basename(MODEL2_PATH)}",
            padding=5,
            bootstyle="warning"
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        self.vid_label_2 = ttk.Label(
            right_frame,
            text="Video YOLOv5 sẽ hiển thị ở đây",
            relief="solid"
        )
        self.vid_label_2.pack(padx=5, pady=5, expand=True, fill="both")

        ttk.Label(right_frame, text="Thông tin model 2:", bootstyle="secondary").pack(anchor="w", padx=5)
        self.text_vid_2 = ttk.Text(right_frame, height=6, state="disabled")
        self.text_vid_2.pack(fill="x", padx=5, pady=5)
        self.text_vid_2.config(font=("Arial", 14, "bold"))

        btn_frame = ttk.Frame(self.tab_vid)
        btn_frame.pack(pady=5)

        start_btn = ttk.Button(btn_frame, text="Chọn video", bootstyle="success", command=self.start_video)
        stop_btn = ttk.Button(btn_frame, text="Dừng", bootstyle="danger", command=self.stop_video)

        start_btn.grid(row=0, column=0, padx=20, pady=10)
        stop_btn.grid(row=0, column=1, padx=20, pady=10)


    # =========================================================
    # HÀM DỰ ĐOÁN CHUNG CHO MỖI MODEL
    # =========================================================
    def predict_frame(self, frame, model):
        results = model(frame, conf=CONF_THRES, verbose=False)
        annotated = results[0].plot(font=FONT_PATH)
        return annotated, results[0]

    def display_result(self, frame, label_widget, text_widget, model, target_size=None):
        annotated, result_obj = self.predict_frame(frame, model)

        # ==== Update text ====
        text_widget.config(state="normal")
        text_widget.delete("1.0", tk.END)
        if len(result_obj.boxes) == 0:
            text_widget.insert("1.0", "Không phát hiện biển báo nào.")
        else:
            text_widget.insert(
                "1.0",
                f"Tìm thấy {len(result_obj.boxes)} biển báo:\n------------------------------\n"
            )
            for b in result_obj.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                class_name = model.names[cls] if hasattr(model, "names") else str(cls)
                text_widget.insert(tk.END, f"{class_name}  ({conf*100:.1f}%)\n")
        text_widget.config(state="disabled")

        # ==== Chuẩn bị ảnh để hiển thị ====
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        if target_size is None:
            # Lấy size widget hiện tại
            label_widget.update_idletasks()
            widget_w = label_widget.winfo_width()
            widget_h = label_widget.winfo_height()
            if widget_w <= 10 or widget_h <= 10:
                widget_w, widget_h = 600, 400
        else:
            widget_w, widget_h = target_size

        r = min(widget_w / w, widget_h / h)
        new_w = max(1, int(w * r))
        new_h = max(1, int(h * r))

        pil_img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)

        label_widget.configure(image=img_tk, text="")
        label_widget.image = img_tk

    # =========================================================
    # ẢNH – chạy 2 model cùng lúc
    # =========================================================
    def load_image(self):
        self.stop_cam()
        self.stop_video()
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return

        # Model 1
        self.display_result(frame.copy(), self.img_label_1, self.text_img_1, self.model1)
        # Model 2
        self.display_result(frame, self.img_label_2, self.text_img_2, self.model2)

    # =========================================================
    # CAMERA – 2 model cùng lúc
    # =========================================================
    def start_cam(self):
        self.stop_video()
        if self.is_cam_running:
            return
        self.cap_cam = cv2.VideoCapture(0)
        if not self.cap_cam.isOpened():
            messagebox.showerror("Lỗi", "Không mở được camera.")
            return
        self.is_cam_running = True
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        while self.is_cam_running:
            ret, frame = self.cap_cam.read()
            if not ret:
                break

            # Dùng after để update UI, mỗi model một label/text
            self.root.after(
                0,
                self.update_cam_frame,
                frame.copy()
            )
            time.sleep(0.03)
        self.is_cam_running = False

    def update_cam_frame(self, frame):
        # mỗi model 1 output, scale nhỏ lại cho vừa 2 cột
        self.display_result(frame.copy(), self.cam_label_1, self.text_cam_1, self.model1, target_size=(600, 400))
        self.display_result(frame, self.cam_label_2, self.text_cam_2, self.model2, target_size=(600, 400))

    def stop_cam(self):
        self.is_cam_running = False
        if self.cap_cam:
            self.cap_cam.release()

    # =========================================================
    # VIDEO – 2 model cùng lúc
    # =========================================================
    def start_video(self):
        self.stop_cam()
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.cap_vid = cv2.VideoCapture(path)
        if not self.cap_vid.isOpened():
            messagebox.showerror("Lỗi", "Không mở được video.")
            return
        self.is_vid_running = True
        threading.Thread(target=self.video_loop, daemon=True).start()

    def video_loop(self):
        while self.is_vid_running:
            ret, frame = self.cap_vid.read()
            if not ret:
                break

            self.root.after(
                0,
                self.update_video_frame,
                frame.copy()
            )
            time.sleep(0.03)
        self.is_vid_running = False

    def update_video_frame(self, frame):
        self.display_result(frame.copy(), self.vid_label_1, self.text_vid_1, self.model1, target_size=(600, 400))
        self.display_result(frame, self.vid_label_2, self.text_vid_2, self.model2, target_size=(600, 400))

    def stop_video(self):
        self.is_vid_running = False
        if self.cap_vid:
            self.cap_vid.release()

    # =========================================================
    def on_close(self):
        self.stop_cam()
        self.stop_video()
        self.root.destroy()


if __name__ == "__main__":
    app = ttk.Window(themename="cosmo")
    TrafficSignApp(app)
    app.mainloop()
