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
MODEL_PATH = r"runs/detect/YOLO_Stage2/weights/best.pt"
# MODEL_PATH = r"runs/detect/YOLO_Stage22/weights/best.pt"   
FONT_PATH = "Arial.Unicode.ttf"                          # Font tiếng Việt
CONF_THRES = 0.2                                     # Ngưỡng confidence
# =============================================
 

print("Đang tải model YOLOv8...")
try:
    model = YOLO(MODEL_PATH)
    print("Tải model thành công!")
except Exception as e:
    print(f"LỖI: Không thể load model '{MODEL_PATH}'!")
    messagebox.showerror("Lỗi model", str(e))
    exit()


class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận Diện Biển Báo Giao Thông - YOLOv8")
        self.root.geometry("1000x800")

        self.cap_cam = None
        self.cap_vid = None
        self.is_cam_running = False
        self.is_vid_running = False

        self.tab = ttk.Notebook(root, bootstyle="primary")
        self.tab_img = ttk.Frame(self.tab, padding=10)
        self.tab_cam = ttk.Frame(self.tab, padding=10)
        self.tab_vid = ttk.Frame(self.tab, padding=10)

        self.tab.add(self.tab_img, text=" Ảnh ")
        self.tab.add(self.tab_cam, text=" Webcam ")
        self.tab.add(self.tab_vid, text=" Video ")
        self.tab.pack(expand=True, fill="both")

        self.setup_image_tab()
        self.setup_webcam_tab()
        self.setup_video_tab()

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------
    # IMAGE TAB
    # -------------------------------------------
    def setup_image_tab(self):
        # ===== Frame ngang để chứa tiêu đề và nút chọn ảnh =====
        top_frame = ttk.Frame(self.tab_img)
        top_frame.pack(fill="x", pady=10)

        # ===== Tiêu đề bên trái =====
        title_label = ttk.Label(
            top_frame,
            text="Nhận diện biển báo giao thông - Nhóm 1",
            font=("Arial", 24, "bold"),      # tăng cỡ chữ tại đây
            bootstyle="primary"
        )
        title_label.pack(side="left", padx=10)

        # ===== Nút Chọn ảnh bên phải =====
        btn = ttk.Button(top_frame, text="Chọn ảnh", bootstyle="success", command=self.load_image)
        btn.pack(side="right", padx=10)

        # ===== Khu vực ảnh hiển thị =====
        self.img_label = ttk.Label(self.tab_img, text="Kết quả sẽ hiển thị ở đây", relief="solid")
        self.img_label.pack(padx=10, pady=10, expand=True, fill="both")

        ttk.Label(self.tab_img, text="Kết quả chi tiết:", bootstyle="info").pack()
        self.text_img = ttk.Text(self.tab_img, height=6, state="disabled")
        self.text_img.pack(fill="x", padx=5, pady=5)
        self.text_img.config(font=("Arial", 17, "bold"))  # tăng cỡ chữ kết quả

    # -------------------------------------------
    # WEBCAM TAB
    # -------------------------------------------
    def setup_webcam_tab(self):
        self.cam_label = ttk.Label(self.tab_cam, text="Webcam sẽ hiển thị ở đây", relief="solid")
        self.cam_label.pack(padx=10, pady=10, expand=True, fill="both")

        btn_frame = ttk.Frame(self.tab_cam)
        btn_frame.pack()

        start_btn = ttk.Button(btn_frame, text="Bật camera", bootstyle="success", command=self.start_cam)
        stop_btn = ttk.Button(btn_frame, text="Tắt", bootstyle="danger", command=self.stop_cam)

        start_btn.grid(row=0, column=0, padx=20, pady=10)
        stop_btn.grid(row=0, column=1, padx=20, pady=10)

        ttk.Label(self.tab_cam, text="Thông tin:", bootstyle="info").pack()

        self.text_cam = ttk.Text(self.tab_cam, height=6, state="disabled")
        self.text_cam.pack(fill="x", padx=5, pady=5)
        self.text_cam.config(font=("Arial", 17, "bold"))   # FONT LỚN CAMERA

    # -------------------------------------------
    # VIDEO TAB
    # -------------------------------------------
    def setup_video_tab(self):
        self.vid_label = ttk.Label(self.tab_vid, text="Video sẽ hiển thị ở đây", relief="solid")
        self.vid_label.pack(padx=10, pady=10, expand=True, fill="both")

        btn_frame = ttk.Frame(self.tab_vid)
        btn_frame.pack()

        start_btn = ttk.Button(btn_frame, text="Chọn video", bootstyle="success", command=self.start_video)
        stop_btn = ttk.Button(btn_frame, text="Dừng", bootstyle="danger", command=self.stop_video)

        start_btn.grid(row=0, column=0, padx=20, pady=10)
        stop_btn.grid(row=0, column=1, padx=20, pady=10)

        ttk.Label(self.tab_vid, text="Thông tin:", bootstyle="info").pack()

        self.text_vid = ttk.Text(self.tab_vid, height=6, state="disabled")
        self.text_vid.pack(fill="x", padx=5, pady=5)
        self.text_vid.config(font=("Arial", 17, "bold"))   # FONT LỚN VIDEO

    # -------------------------------------------
    # DETECTION PROCESS
    # -------------------------------------------
    def predict_frame(self, frame):
        results = model(frame, conf=CONF_THRES, verbose=False)
        annotated = results[0].plot(font=FONT_PATH)
        return annotated, results[0]

    def display_result(self, frame, label_widget, text_widget, target_size=None):
        annotated, result_obj = self.predict_frame(frame)

        # ==== Cập nhật text kết quả ====
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
                text_widget.insert(tk.END, f"{model.names[cls]}  ({conf*100:.1f}%)\n")
        text_widget.config(state="disabled")

        # ==== Chuẩn bị frame để hiển thị ====
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        # Nếu không truyền target_size → lấy size widget (dùng cho tab Ảnh)
        if target_size is None:
            label_widget.update_idletasks()
            widget_w = label_widget.winfo_width()
            widget_h = label_widget.winfo_height()

            if widget_w <= 10 or widget_h <= 10:
                widget_w, widget_h = 800, 600
        else:
            widget_w, widget_h = target_size

        # Tính tỉ lệ scale, giữ đúng tỉ lệ ảnh
        r = min(widget_w / w, widget_h / h)
        new_w = max(1, int(w * r))
        new_h = max(1, int(h * r))

        pil_img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)

        label_widget.configure(image=img_tk, text="")
        label_widget.image = img_tk

    # -------------------------------------------
    def load_image(self):
        self.stop_cam()
        self.stop_video()
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        frame = cv2.imread(path)
        self.display_result(frame, self.img_label, self.text_img)

    # -------------------------------------------
    # CAMERA
    # -------------------------------------------
    def start_cam(self):
        self.stop_video()
        if self.is_cam_running:
            return
        self.cap_cam = cv2.VideoCapture(0)
        self.is_cam_running = True
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        while self.is_cam_running:
            ret, frame = self.cap_cam.read()
            if not ret:
                break
            # scale webcam về 900x600 (tuỳ bạn chỉnh)
            self.root.after(
                0,
                self.display_result,
                frame,
                self.cam_label,
                self.text_cam,
                (900, 600)
            )
            time.sleep(0.03)


    def stop_cam(self):
        self.is_cam_running = False
        if self.cap_cam:
            self.cap_cam.release()

    # -------------------------------------------
    # VIDEO
    # -------------------------------------------
    def start_video(self):
        self.stop_cam()
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.cap_vid = cv2.VideoCapture(path)
        self.is_vid_running = True
        threading.Thread(target=self.video_loop, daemon=True).start()

    def video_loop(self):
        while self.is_vid_running:
            ret, frame = self.cap_vid.read()
            if not ret:
                break
            # scale video về 900x600
            self.root.after(
                0,
                self.display_result,
                frame,
                self.vid_label,
                self.text_vid,
                (900, 600)
            )
            time.sleep(0.03)
        self.is_vid_running = False


    def stop_video(self):
        self.is_vid_running = False
        if self.cap_vid:
            self.cap_vid.release()

    # -------------------------------------------
    def on_close(self):
        self.stop_cam()
        self.stop_video()
        self.root.destroy()


if __name__ == "__main__":
    app = ttk.Window(themename="cosmo")
    TrafficSignApp(app)
    app.mainloop()
