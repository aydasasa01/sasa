"""
SASA Polyester - POY Fabrikası Masura Bölümü
Video Etiketleme Sistemi (Zaman Etüdü)

HEVC/H.265 uyumlu - ffmpeg tabanlı video okuma

Kullanım:
  python labeling_app.py

Kısayollar:
  Space      : Oynat / Duraklat
  K          : Katma Değerli İş - segment başlat/bitir
  D          : Diğer - segment başlat/bitir
  Left/Right : 5 saniye geri/ileri
  Shift+Left/Right : 30 saniye geri/ileri
  S          : Mevcut etiketi kaydet
  Z          : Son etiketi geri al
  Q          : Çıkış
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import json
import os
import subprocess
import struct
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageTk

# ─── Sabitler ───────────────────────────────────────────────────────
LABEL_KATMA_DEGERLI = "katma_degerli_is"
LABEL_DIGER = "diger"

LABEL_COLORS = {
    LABEL_KATMA_DEGERLI: "#2ecc71",  # Yeşil
    LABEL_DIGER: "#e74c3c",          # Kırmızı
}

LABEL_DISPLAY = {
    LABEL_KATMA_DEGERLI: "Katma Değerli İş",
    LABEL_DIGER: "Diğer",
}

CANVAS_W = 960
CANVAS_H = 540


class FFmpegVideoReader:
    """HEVC/H.265 uyumlu ffmpeg tabanlı video okuyucu."""

    def __init__(self, path):
        self.path = path
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self.total_frames = 0
        self.duration = 0.0
        self._pipe_proc = None
        self._current_time = 0.0  # saniye cinsinden mevcut pozisyon

        self._probe()

    def _probe(self):
        """ffprobe ile video bilgilerini al."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", self.path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)

        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                self.width = int(s["width"])
                self.height = int(s["height"])
                # FPS
                r_fps = s.get("r_frame_rate", "25/1")
                num, den = r_fps.split("/")
                self.fps = float(num) / float(den) if float(den) != 0 else 25.0
                # Süre
                self.duration = float(info.get("format", {}).get("duration", 0))
                self.total_frames = int(self.duration * self.fps)
                break

    def read_frame_at(self, time_sec):
        """Belirtilen zamandaki tek frame'i oku (seek için)."""
        time_sec = max(0, min(time_sec, self.duration - 0.1))
        cmd = [
            "ffmpeg", "-ss", f"{time_sec:.3f}",
            "-i", self.path,
            "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet", "-"
        ]
        proc = subprocess.run(cmd, capture_output=True)
        expected = self.width * self.height * 3
        if len(proc.stdout) >= expected:
            frame = np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape(self.height, self.width, 3)
            self._current_time = time_sec
            return True, frame
        return False, None

    def start_streaming(self, start_time=0.0):
        """Belirtilen zamandan itibaren sıralı frame akışı başlat."""
        self.stop_streaming()
        start_time = max(0, min(start_time, self.duration - 0.1))
        self._current_time = start_time
        cmd = [
            "ffmpeg", "-ss", f"{start_time:.3f}",
            "-i", self.path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet", "-"
        ]
        self._pipe_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read_next_frame(self):
        """Akıştan sonraki frame'i oku."""
        if not self._pipe_proc or self._pipe_proc.poll() is not None:
            return False, None
        expected = self.width * self.height * 3
        raw = self._pipe_proc.stdout.read(expected)
        if len(raw) < expected:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._current_time += 1.0 / self.fps
        return True, frame

    def stop_streaming(self):
        """Akışı durdur."""
        if self._pipe_proc:
            try:
                self._pipe_proc.stdout.close()
                self._pipe_proc.terminate()
                self._pipe_proc.wait(timeout=2)
            except Exception:
                try:
                    self._pipe_proc.kill()
                except Exception:
                    pass
            self._pipe_proc = None

    @property
    def current_time(self):
        return self._current_time

    @property
    def current_frame_number(self):
        return int(self._current_time * self.fps)

    def release(self):
        self.stop_streaming()


class VideoLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SASA POY - Masura Bölümü Video Etiketleme Sistemi")
        self.root.configure(bg="#1e1e2e")
        self.root.state("zoomed")

        # Video durumu
        self.reader = None
        self.video_path = None
        self.playing = False
        self.total_frames = 0
        self.fps = 25
        self.current_frame = 0
        self.playback_speed = 1.0

        # Etiketleme durumu
        self.labels = []
        self.current_label = None
        self.label_file = None

        # Video listesi
        self.video_dir = "C:/Users/USER/Desktop/Video_20260212091342"
        self.video_files = []

        # Threading
        self.play_thread = None
        self.lock = threading.Lock()

        self._build_ui()
        self._bind_keys()
        self._load_video_list()

    # ─── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", background="#313244", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))

        # Ana çerçeve
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sol panel - Video listesi
        left_panel = ttk.Frame(main, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        ttk.Label(left_panel, text="Videolar", style="Header.TLabel").pack(pady=(5, 5))

        self.video_listbox = tk.Listbox(
            left_panel, bg="#313244", fg="#cdd6f4", selectbackground="#585b70",
            font=("Segoe UI", 9), relief=tk.FLAT, bd=0
        )
        self.video_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_listbox.bind("<<ListboxSelect>>", self._on_video_select)

        ttk.Button(left_panel, text="Klasör Seç", command=self._select_folder).pack(fill=tk.X, padx=5, pady=5)

        # Orta panel
        center = ttk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.header_label = ttk.Label(center, text="SASA POY - Masura Bölümü Zaman Etüdü", style="Header.TLabel")
        self.header_label.pack(pady=(0, 5))

        self.canvas = tk.Canvas(center, width=CANVAS_W, height=CANVAS_H, bg="#11111b", highlightthickness=0)
        self.canvas.pack()

        # Timeline
        self.timeline_canvas = tk.Canvas(center, height=50, bg="#181825", highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, padx=5, pady=(5, 0))
        self.timeline_canvas.bind("<Button-1>", self._on_timeline_click)

        # Slider
        slider_frame = ttk.Frame(center)
        slider_frame.pack(fill=tk.X, padx=5)

        self.time_label_left = ttk.Label(slider_frame, text="00:00:00")
        self.time_label_left.pack(side=tk.LEFT)

        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.time_label_right = ttk.Label(slider_frame, text="00:00:00")
        self.time_label_right.pack(side=tk.RIGHT)

        # Kontrol butonları
        ctrl_frame = ttk.Frame(center)
        ctrl_frame.pack(pady=5)

        ttk.Button(ctrl_frame, text="<< 30s", command=lambda: self._seek(-30)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="< 5s", command=lambda: self._seek(-5)).pack(side=tk.LEFT, padx=2)

        self.play_btn = ttk.Button(ctrl_frame, text="Oynat", command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(ctrl_frame, text="5s >", command=lambda: self._seek(5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="30s >>", command=lambda: self._seek(30)).pack(side=tk.LEFT, padx=2)

        # Hız kontrolü
        speed_frame = ttk.Frame(ctrl_frame)
        speed_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(speed_frame, text="Hız:").pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var,
                                   values=["0.25x", "0.5x", "1x", "1.5x", "2x", "4x", "8x"],
                                   width=5, state="readonly")
        speed_combo.pack(side=tk.LEFT, padx=5)
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Etiketleme butonları
        label_frame = ttk.Frame(center)
        label_frame.pack(pady=10)

        self.btn_katma = tk.Button(
            label_frame, text="[K] Katma Değerli İş\nBaşlat",
            bg="#2ecc71", fg="white", font=("Segoe UI", 12, "bold"),
            width=22, height=2, relief=tk.FLAT,
            command=lambda: self._toggle_label(LABEL_KATMA_DEGERLI)
        )
        self.btn_katma.pack(side=tk.LEFT, padx=10)

        self.btn_diger = tk.Button(
            label_frame, text="[D] Diğer\nBaşlat",
            bg="#e74c3c", fg="white", font=("Segoe UI", 12, "bold"),
            width=22, height=2, relief=tk.FLAT,
            command=lambda: self._toggle_label(LABEL_DIGER)
        )
        self.btn_diger.pack(side=tk.LEFT, padx=10)

        self.btn_undo = tk.Button(
            label_frame, text="[Z] Geri Al",
            bg="#585b70", fg="white", font=("Segoe UI", 10),
            width=12, height=2, relief=tk.FLAT,
            command=self._undo_label
        )
        self.btn_undo.pack(side=tk.LEFT, padx=10)

        # Aktif etiket göstergesi
        self.active_label_var = tk.StringVar(value="Etiket aktif değil")
        self.active_label_display = ttk.Label(center, textvariable=self.active_label_var,
                                               font=("Segoe UI", 12, "bold"))
        self.active_label_display.pack(pady=5)

        # Sağ panel
        right_panel = ttk.Frame(main, width=320)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)

        ttk.Label(right_panel, text="Etiketler", style="Header.TLabel").pack(pady=(5, 5))

        columns = ("start", "end", "label", "duration")
        self.label_tree = ttk.Treeview(right_panel, columns=columns, show="headings", height=15)
        self.label_tree.heading("start", text="Başlangıç")
        self.label_tree.heading("end", text="Bitiş")
        self.label_tree.heading("label", text="Etiket")
        self.label_tree.heading("duration", text="Süre")
        self.label_tree.column("start", width=70)
        self.label_tree.column("end", width=70)
        self.label_tree.column("label", width=100)
        self.label_tree.column("duration", width=60)

        tree_scroll = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.label_tree.yview)
        self.label_tree.configure(yscrollcommand=tree_scroll.set)

        self.label_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.LEFT)
        tree_scroll.pack(fill=tk.Y, side=tk.LEFT, pady=5)

        self.label_tree.bind("<Double-1>", self._on_label_double_click)

        # İstatistik paneli
        stats_frame = ttk.Frame(right_panel)
        stats_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM)

        ttk.Label(stats_frame, text="İstatistikler", style="Header.TLabel").pack()

        self.stats_text = tk.Text(
            stats_frame, height=14, bg="#313244", fg="#cdd6f4",
            font=("Consolas", 9), relief=tk.FLAT, bd=0, wrap=tk.WORD
        )
        self.stats_text.pack(fill=tk.X, pady=5)

        # Alt durum çubuğu
        self.status_var = tk.StringVar(value="Video seçin veya klasörden yükleyin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=3, padx=2)

        # Dışa aktarma butonları
        export_frame = ttk.Frame(right_panel)
        export_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM, before=stats_frame)

        ttk.Button(export_frame, text="JSON Kaydet", command=self._save_labels).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="CSV Dışa Aktar", command=self._export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Rapor Oluştur", command=self._generate_report).pack(fill=tk.X, pady=2)

    def _bind_keys(self):
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<k>", lambda e: self._toggle_label(LABEL_KATMA_DEGERLI))
        self.root.bind("<K>", lambda e: self._toggle_label(LABEL_KATMA_DEGERLI))
        self.root.bind("<d>", lambda e: self._toggle_label(LABEL_DIGER))
        self.root.bind("<D>", lambda e: self._toggle_label(LABEL_DIGER))
        self.root.bind("<Left>", lambda e: self._seek(-5))
        self.root.bind("<Right>", lambda e: self._seek(5))
        self.root.bind("<Shift-Left>", lambda e: self._seek(-30))
        self.root.bind("<Shift-Right>", lambda e: self._seek(30))
        self.root.bind("<s>", lambda e: self._save_labels())
        self.root.bind("<S>", lambda e: self._save_labels())
        self.root.bind("<z>", lambda e: self._undo_label())
        self.root.bind("<Z>", lambda e: self._undo_label())
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())

    # ─── Video Yükleme ─────────────────────────────────────────────
    def _load_video_list(self):
        self.video_listbox.delete(0, tk.END)
        self.video_files = []
        if os.path.isdir(self.video_dir):
            for f in sorted(os.listdir(self.video_dir)):
                if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    self.video_files.append(f)
                    display = f
                    if len(f) > 35:
                        display = f[:15] + "..." + f[-20:]
                    self.video_listbox.insert(tk.END, display)

    def _select_folder(self):
        folder = filedialog.askdirectory(initialdir=self.video_dir)
        if folder:
            self.video_dir = folder
            self._load_video_list()

    def _on_video_select(self, event):
        sel = self.video_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        self._load_video(video_path)

    def _load_video(self, path):
        if self.video_path and self.labels:
            self._save_labels(auto=True)

        self.playing = False
        if self.reader:
            self.reader.release()

        self.status_var.set("Video yükleniyor...")
        self.root.update_idletasks()

        self.reader = FFmpegVideoReader(path)
        if self.reader.total_frames == 0:
            messagebox.showerror("Hata", f"Video açılamadı:\n{path}")
            self.reader = None
            return

        self.video_path = path
        self.total_frames = self.reader.total_frames
        self.fps = self.reader.fps
        self.current_frame = 0

        self.slider.configure(to=self.total_frames)

        self.label_file = path + ".labels.json"
        self._load_labels()

        self._show_frame()
        self._update_time_display()
        self._update_timeline()

        filename = os.path.basename(path)
        self.header_label.configure(text=f"SASA POY - {filename[:60]}")
        self.status_var.set(
            f"Video yüklendi: {filename} | "
            f"{self._format_time(self.reader.duration)} | "
            f"{self.total_frames} kare | {self.fps:.1f} FPS | "
            f"{self.reader.width}x{self.reader.height}"
        )

    # ─── Oynatma ────────────────────────────────────────────────────
    def _toggle_play(self):
        if not self.reader:
            return
        if self.playing:
            self.playing = False
            self.play_btn.configure(text="Oynat")
            self.reader.stop_streaming()
        else:
            self.playing = True
            self.play_btn.configure(text="Duraklat")
            # Akış başlat
            current_time = self.current_frame / self.fps
            self.reader.start_streaming(current_time)
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

    def _play_loop(self):
        skip = max(1, int(self.playback_speed)) - 1  # Hızlı oynatmada frame atla
        while self.playing and self.reader:
            start_t = time.time()

            ret, frame = self.reader.read_next_frame()
            if not ret:
                self.playing = False
                self.root.after(0, lambda: self.play_btn.configure(text="Oynat"))
                break

            # Hızlı oynatmada frame atla
            for _ in range(skip):
                if not self.playing:
                    break
                self.reader.read_next_frame()

            self.current_frame = self.reader.current_frame_number
            self._display_frame(frame)
            self.root.after(0, self._update_ui_during_play)

            elapsed = time.time() - start_t
            target_delay = 1.0 / (self.fps * self.playback_speed)
            delay = max(0, target_delay - elapsed)
            if delay > 0:
                time.sleep(delay)

    def _update_ui_during_play(self):
        self._update_time_display()
        self.slider.set(self.current_frame)

    def _show_frame(self):
        """Mevcut frame'i ffmpeg ile oku ve göster (seek)."""
        if not self.reader:
            return
        time_sec = self.current_frame / self.fps
        ret, frame = self.reader.read_frame_at(time_sec)
        if ret:
            self._display_frame(frame)

    def _display_frame(self, frame):
        frame = cv2.resize(frame, (CANVAS_W, CANVAS_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aktif etiket overlay
        if self.current_label:
            label = self.current_label["label"]
            color = (46, 204, 113) if label == LABEL_KATMA_DEGERLI else (231, 76, 60)
            overlay = frame_rgb.copy()
            cv2.rectangle(overlay, (0, 0), (CANVAS_W, 40), color, -1)
            cv2.addWeighted(overlay, 0.7, frame_rgb, 0.3, 0, frame_rgb)

            text = f"KAYIT: {LABEL_DISPLAY[label]}"
            cv2.putText(frame_rgb, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elapsed = (self.current_frame - self.current_label["start_frame"]) / self.fps
            cv2.putText(frame_rgb, f"{elapsed:.1f}s", (CANVAS_W - 100, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Zaman damgası
        current_time = self._format_time(self.current_frame / self.fps)
        cv2.putText(frame_rgb, current_time, (CANVAS_W - 130, CANVAS_H - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Frame numarası
        cv2.putText(frame_rgb, f"F:{self.current_frame}", (10, CANVAS_H - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        img = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    # ─── Navigasyon ─────────────────────────────────────────────────
    def _seek(self, seconds):
        if not self.reader:
            return
        was_playing = self.playing
        if was_playing:
            self.playing = False
            self.reader.stop_streaming()

        delta_frames = int(seconds * self.fps)
        self.current_frame = max(0, min(self.total_frames - 1, self.current_frame + delta_frames))
        self.slider.set(self.current_frame)
        self._show_frame()
        self._update_time_display()

        if was_playing:
            self.playing = True
            self.play_btn.configure(text="Duraklat")
            current_time = self.current_frame / self.fps
            self.reader.start_streaming(current_time)
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

    def _on_slider(self, val):
        if not self.reader:
            return
        frame = int(float(val))
        if abs(frame - self.current_frame) > 2:
            was_playing = self.playing
            if was_playing:
                self.playing = False
                self.reader.stop_streaming()

            self.current_frame = frame
            self._show_frame()
            self._update_time_display()

            if was_playing:
                self.playing = True
                current_time = self.current_frame / self.fps
                self.reader.start_streaming(current_time)
                self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
                self.play_thread.start()

    def _on_timeline_click(self, event):
        if not self.reader:
            return
        w = self.timeline_canvas.winfo_width()
        ratio = event.x / w
        was_playing = self.playing
        if was_playing:
            self.playing = False
            self.reader.stop_streaming()

        self.current_frame = int(ratio * self.total_frames)
        self.slider.set(self.current_frame)
        self._show_frame()
        self._update_time_display()
        self._update_timeline()

        if was_playing:
            self.playing = True
            self.play_btn.configure(text="Duraklat")
            current_time = self.current_frame / self.fps
            self.reader.start_streaming(current_time)
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

    def _on_speed_change(self, event):
        speed_str = self.speed_var.get().replace("x", "")
        self.playback_speed = float(speed_str)

    # ─── Etiketleme ─────────────────────────────────────────────────
    def _toggle_label(self, label_type):
        if not self.reader:
            return

        if self.current_label is not None:
            start_frame = self.current_label["start_frame"]
            end_frame = self.current_frame

            if end_frame <= start_frame:
                self.status_var.set("Hata: Bitiş zamanı başlangıçtan önce olamaz!")
                return

            entry = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_frame / self.fps,
                "end_time": end_frame / self.fps,
                "start_str": self._format_time(start_frame / self.fps),
                "end_str": self._format_time(end_frame / self.fps),
                "label": self.current_label["label"],
                "duration": (end_frame - start_frame) / self.fps,
            }
            self.labels.append(entry)
            self.current_label = None

            self._update_label_list()
            self._update_stats()
            self._update_timeline()
            self._update_buttons()
            self._save_labels(auto=True)

            duration_str = self._format_time(entry["duration"])
            self.status_var.set(
                f"Etiket kaydedildi: {LABEL_DISPLAY[entry['label']]} | "
                f"{entry['start_str']} - {entry['end_str']} ({duration_str})"
            )
            self.active_label_var.set("Etiket aktif değil")
        else:
            self.current_label = {
                "label": label_type,
                "start_frame": self.current_frame,
                "start_time": self.current_frame / self.fps,
            }
            self._update_buttons()
            self.active_label_var.set(f"KAYIT: {LABEL_DISPLAY[label_type]} - {self._format_time(self.current_frame / self.fps)}")
            self.status_var.set(f"{LABEL_DISPLAY[label_type]} etiketi başlatıldı: {self._format_time(self.current_frame / self.fps)}")

    def _undo_label(self):
        if self.current_label is not None:
            self.current_label = None
            self._update_buttons()
            self.active_label_var.set("Etiket aktif değil")
            self.status_var.set("Aktif etiket iptal edildi")
        elif self.labels:
            removed = self.labels.pop()
            self._update_label_list()
            self._update_stats()
            self._update_timeline()
            self._save_labels(auto=True)
            self.status_var.set(f"Son etiket silindi: {LABEL_DISPLAY[removed['label']]} ({removed['start_str']} - {removed['end_str']})")

    def _update_buttons(self):
        if self.current_label:
            label = self.current_label["label"]
            if label == LABEL_KATMA_DEGERLI:
                self.btn_katma.configure(text="[K] Katma Değerli İş\nBİTİR", bg="#27ae60")
                self.btn_diger.configure(text="[D] Diğer\nBaşlat", bg="#e74c3c", state=tk.DISABLED)
            else:
                self.btn_diger.configure(text="[D] Diğer\nBİTİR", bg="#c0392b")
                self.btn_katma.configure(text="[K] Katma Değerli İş\nBaşlat", bg="#2ecc71", state=tk.DISABLED)
        else:
            self.btn_katma.configure(text="[K] Katma Değerli İş\nBaşlat", bg="#2ecc71", state=tk.NORMAL)
            self.btn_diger.configure(text="[D] Diğer\nBaşlat", bg="#e74c3c", state=tk.NORMAL)

    # ─── Etiket Listesi ─────────────────────────────────────────────
    def _update_label_list(self):
        for item in self.label_tree.get_children():
            self.label_tree.delete(item)

        for i, entry in enumerate(self.labels):
            tag = "katma" if entry["label"] == LABEL_KATMA_DEGERLI else "diger"
            self.label_tree.insert("", tk.END, values=(
                entry["start_str"],
                entry["end_str"],
                LABEL_DISPLAY[entry["label"]],
                self._format_time(entry["duration"]),
            ), tags=(tag,))

        self.label_tree.tag_configure("katma", foreground="#2ecc71")
        self.label_tree.tag_configure("diger", foreground="#e74c3c")

    def _on_label_double_click(self, event):
        sel = self.label_tree.selection()
        if not sel:
            return
        idx = self.label_tree.index(sel[0])
        if idx < len(self.labels):
            self.current_frame = self.labels[idx]["start_frame"]
            self.slider.set(self.current_frame)
            if not self.playing:
                self._show_frame()
            self._update_time_display()

    # ─── İstatistikler ───────────────────────────────────────────────
    def _calculate_cycle_times(self):
        """Katma değerli işler arası çevrim sürelerini hesapla.
        Çevrim süresi = Bir KDİ başlangıcından sonraki KDİ başlangıcına kadar geçen süre.
        """
        katma_entries = sorted(
            [e for e in self.labels if e["label"] == LABEL_KATMA_DEGERLI],
            key=lambda x: x["start_time"]
        )
        cycle_times = []
        for i in range(1, len(katma_entries)):
            ct = katma_entries[i]["start_time"] - katma_entries[i - 1]["start_time"]
            cycle_times.append(ct)
        return cycle_times, katma_entries

    def _update_stats(self):
        self.stats_text.delete("1.0", tk.END)

        if not self.labels:
            self.stats_text.insert(tk.END, "Henüz etiket yok.\n")
            return

        katma_total = sum(e["duration"] for e in self.labels if e["label"] == LABEL_KATMA_DEGERLI)
        diger_total = sum(e["duration"] for e in self.labels if e["label"] == LABEL_DIGER)
        total = katma_total + diger_total

        katma_count = sum(1 for e in self.labels if e["label"] == LABEL_KATMA_DEGERLI)
        diger_count = sum(1 for e in self.labels if e["label"] == LABEL_DIGER)

        video_duration = self.total_frames / self.fps if self.fps else 0
        labeled_pct = (total / video_duration * 100) if video_duration else 0
        katma_pct = (katma_total / total * 100) if total else 0

        stats = (
            f"Toplam Etiket    : {len(self.labels)}\n"
            f"─────────────────────────\n"
            f"Katma Değerli İş : {katma_count} adet\n"
            f"  Toplam Süre    : {self._format_time(katma_total)}\n"
            f"  Oran           : {katma_pct:.1f}%\n"
            f"─────────────────────────\n"
            f"Diğer            : {diger_count} adet\n"
            f"  Toplam Süre    : {self._format_time(diger_total)}\n"
            f"─────────────────────────\n"
            f"Etiketlenen      : {self._format_time(total)}\n"
            f"Video Süresi     : {self._format_time(video_duration)}\n"
            f"Kapsam           : {labeled_pct:.1f}%\n"
        )

        # Çevrim Süresi Analizi
        cycle_times, katma_entries = self._calculate_cycle_times()
        if cycle_times:
            avg_cycle = sum(cycle_times) / len(cycle_times)
            min_cycle = min(cycle_times)
            max_cycle = max(cycle_times)
            avg_katma_dur = katma_total / katma_count if katma_count else 0

            stats += (
                f"═════════════════════════\n"
                f"ÇEVRİM SÜRESİ ANALİZİ\n"
                f"─────────────────────────\n"
                f"  Çevrim Sayısı  : {len(cycle_times)}\n"
                f"  Ort. Çevrim    : {self._format_time(avg_cycle)}\n"
                f"  Min Çevrim     : {self._format_time(min_cycle)}\n"
                f"  Max Çevrim     : {self._format_time(max_cycle)}\n"
                f"  Ort. KDİ Süresi: {self._format_time(avg_katma_dur)}\n"
                f"  Verimlilik     : {avg_katma_dur / avg_cycle * 100:.1f}%\n"
            )

        self.stats_text.insert(tk.END, stats)

    # ─── Timeline ────────────────────────────────────────────────────
    def _update_timeline(self):
        self.timeline_canvas.delete("all")
        w = self.timeline_canvas.winfo_width()
        h = self.timeline_canvas.winfo_height()

        if not self.reader or self.total_frames == 0 or w < 10:
            return

        self.timeline_canvas.create_rectangle(0, 0, w, h, fill="#181825", outline="")

        for entry in self.labels:
            x1 = int(entry["start_frame"] / self.total_frames * w)
            x2 = int(entry["end_frame"] / self.total_frames * w)
            color = LABEL_COLORS[entry["label"]]
            self.timeline_canvas.create_rectangle(x1, 5, x2, h - 5, fill=color, outline="")

        cx = int(self.current_frame / self.total_frames * w)
        self.timeline_canvas.create_line(cx, 0, cx, h, fill="white", width=2)

        if self.fps > 0:
            interval_sec = 600
            total_sec = self.total_frames / self.fps
            t = 0
            while t < total_sec:
                x = int(t / total_sec * w)
                self.timeline_canvas.create_line(x, h - 8, x, h, fill="#585b70")
                if t % 3600 == 0:
                    hour_str = self._format_time(t)
                    self.timeline_canvas.create_text(x + 2, h - 12, text=hour_str,
                                                      fill="#585b70", font=("Consolas", 7), anchor=tk.SW)
                t += interval_sec

    # ─── Kayıt / Yükleme ────────────────────────────────────────────
    def _save_labels(self, auto=False):
        if not self.label_file:
            if not auto:
                messagebox.showwarning("Uyarı", "Önce video yükleyin")
            return

        cycle_times, katma_entries = self._calculate_cycle_times()
        cycle_data = {}
        if cycle_times:
            avg_katma_dur = sum(e["duration"] for e in self.labels if e["label"] == LABEL_KATMA_DEGERLI) / max(1, sum(1 for e in self.labels if e["label"] == LABEL_KATMA_DEGERLI))
            avg_cycle = sum(cycle_times) / len(cycle_times)
            cycle_data = {
                "cycle_count": len(cycle_times),
                "cycle_times": [round(ct, 2) for ct in cycle_times],
                "avg_cycle_time": round(avg_cycle, 2),
                "min_cycle_time": round(min(cycle_times), 2),
                "max_cycle_time": round(max(cycle_times), 2),
                "avg_kdi_duration": round(avg_katma_dur, 2),
                "efficiency_pct": round(avg_katma_dur / avg_cycle * 100, 2) if avg_cycle else 0,
            }

        data = {
            "video_file": os.path.basename(self.video_path),
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "total_duration": self.total_frames / self.fps,
            "created": datetime.now().isoformat(),
            "cycle_time_analysis": cycle_data,
            "labels": self.labels,
        }

        with open(self.label_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if not auto:
            self.status_var.set(f"Etiketler kaydedildi: {self.label_file}")

    def _load_labels(self):
        self.labels = []
        self.current_label = None

        if self.label_file and os.path.exists(self.label_file):
            try:
                with open(self.label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.labels = data.get("labels", [])
                self.status_var.set(f"{len(self.labels)} etiket yüklendi")
            except Exception as e:
                self.status_var.set(f"Etiket dosyası okunamadı: {e}")

        self._update_label_list()
        self._update_stats()
        self._update_buttons()

    def _export_csv(self):
        if not self.labels:
            messagebox.showinfo("Bilgi", "Dışa aktarılacak etiket yok")
            return

        csv_path = self.video_path + ".labels.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("video_file,start_time,end_time,start_str,end_str,label,label_tr,duration_sec\n")
            for entry in self.labels:
                f.write(
                    f"{os.path.basename(self.video_path)},"
                    f"{entry['start_time']:.2f},"
                    f"{entry['end_time']:.2f},"
                    f"{entry['start_str']},"
                    f"{entry['end_str']},"
                    f"{entry['label']},"
                    f"{LABEL_DISPLAY[entry['label']]},"
                    f"{entry['duration']:.2f}\n"
                )

        self.status_var.set(f"CSV kaydedildi: {csv_path}")
        messagebox.showinfo("CSV Dışa Aktarma", f"Etiketler CSV olarak kaydedildi:\n{csv_path}")

    def _generate_report(self):
        if not self.labels:
            messagebox.showinfo("Bilgi", "Rapor oluşturmak için etiket gerekli")
            return

        report_path = self.video_path + ".report.txt"

        katma_total = sum(e["duration"] for e in self.labels if e["label"] == LABEL_KATMA_DEGERLI)
        diger_total = sum(e["duration"] for e in self.labels if e["label"] == LABEL_DIGER)
        total = katma_total + diger_total
        video_duration = self.total_frames / self.fps

        katma_entries = [e for e in self.labels if e["label"] == LABEL_KATMA_DEGERLI]
        diger_entries = [e for e in self.labels if e["label"] == LABEL_DIGER]

        avg_katma = katma_total / len(katma_entries) if katma_entries else 0
        avg_diger = diger_total / len(diger_entries) if diger_entries else 0

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  SASA POY - MASURA BÖLÜMÜ ZAMAN ETÜDÜ RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Video     : {os.path.basename(self.video_path)}\n")
            f.write(f"Tarih     : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Video Süresi: {self._format_time(video_duration)}\n\n")

            f.write("-" * 60 + "\n")
            f.write("ÖZET\n")
            f.write("-" * 60 + "\n")
            f.write(f"Toplam Etiket Sayısı    : {len(self.labels)}\n")
            f.write(f"Etiketlenen Süre        : {self._format_time(total)}\n")
            f.write(f"Kapsam Oranı            : {total / video_duration * 100:.1f}%\n\n")

            f.write(f"KATMA DEĞERLİ İŞ\n")
            f.write(f"  Adet               : {len(katma_entries)}\n")
            f.write(f"  Toplam Süre        : {self._format_time(katma_total)}\n")
            f.write(f"  Ortalama Süre      : {self._format_time(avg_katma)}\n")
            if total:
                f.write(f"  Oran (etiketli)    : {katma_total / total * 100:.1f}%\n\n")

            f.write(f"DİĞER\n")
            f.write(f"  Adet               : {len(diger_entries)}\n")
            f.write(f"  Toplam Süre        : {self._format_time(diger_total)}\n")
            f.write(f"  Ortalama Süre      : {self._format_time(avg_diger)}\n")
            if total:
                f.write(f"  Oran (etiketli)    : {diger_total / total * 100:.1f}%\n\n")

            # Çevrim Süresi Analizi
            cycle_times, katma_sorted = self._calculate_cycle_times()
            if cycle_times:
                avg_cycle = sum(cycle_times) / len(cycle_times)
                avg_katma_dur = katma_total / len(katma_entries) if katma_entries else 0

                f.write("-" * 60 + "\n")
                f.write("ÇEVRİM SÜRESİ ANALİZİ\n")
                f.write("-" * 60 + "\n")
                f.write(f"  Çevrim Sayısı      : {len(cycle_times)}\n")
                f.write(f"  Ortalama Çevrim    : {self._format_time(avg_cycle)}\n")
                f.write(f"  Minimum Çevrim     : {self._format_time(min(cycle_times))}\n")
                f.write(f"  Maksimum Çevrim    : {self._format_time(max(cycle_times))}\n")
                f.write(f"  Ort. KDİ Süresi    : {self._format_time(avg_katma_dur)}\n")
                f.write(f"  Verimlilik         : {avg_katma_dur / avg_cycle * 100:.1f}%\n\n")

                f.write(f"  {'#':>4}  {'Çevrim Başı':>12}  {'Çevrim Süresi':>14}\n")
                f.write("  " + "-" * 35 + "\n")
                for i, ct in enumerate(cycle_times, 1):
                    start_str = katma_sorted[i - 1]["start_str"]
                    f.write(f"  {i:>4}  {start_str:>12}  {self._format_time(ct):>14}\n")
                f.write("\n")

            f.write("-" * 60 + "\n")
            f.write("DETAYLI ETİKET LİSTESİ\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'#':>4}  {'Başlangıç':>10}  {'Bitiş':>10}  {'Süre':>8}  {'Etiket'}\n")
            f.write("-" * 60 + "\n")
            for i, entry in enumerate(self.labels, 1):
                f.write(
                    f"{i:>4}  {entry['start_str']:>10}  {entry['end_str']:>10}  "
                    f"{self._format_time(entry['duration']):>8}  {LABEL_DISPLAY[entry['label']]}\n"
                )

        self.status_var.set(f"Rapor oluşturuldu: {report_path}")
        messagebox.showinfo("Rapor", f"Zaman etüdü raporu oluşturuldu:\n{report_path}")

    # ─── Yardımcılar ─────────────────────────────────────────────────
    def _format_time(self, seconds):
        seconds = max(0, seconds)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _update_time_display(self):
        if self.reader and self.fps > 0:
            current = self._format_time(self.current_frame / self.fps)
            total = self._format_time(self.total_frames / self.fps)
            self.time_label_left.configure(text=current)
            self.time_label_right.configure(text=total)

    def _quit(self):
        if self.labels and self.video_path:
            self._save_labels(auto=True)
        self.playing = False
        if self.reader:
            self.reader.release()
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VideoLabelingApp(root)
    root.protocol("WM_DELETE_WINDOW", app._quit)
    root.mainloop()


if __name__ == "__main__":
    main()
