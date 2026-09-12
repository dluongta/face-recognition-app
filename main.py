import os
import cv2
import shutil
import numpy as np
import tkinter as tk

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import face_recognition


# ============================================================
# CONFIG
# ============================================================

FACES_DIR = "faces"

# Ngưỡng nhận diện.
# Càng nhỏ thì càng khó nhận nhầm nhưng dễ nhận thành Unknown.
FACE_DISTANCE_THRESHOLD = 0.50

# Kích thước ảnh webcam đem đi nhận diện
PROCESS_SCALE = 0.25


# ============================================================
# GLOBAL DATA
# ============================================================

# known_face_encodings = []
# known_face_names = []

# camera = None
# camera_running = False

known_face_encodings = []
known_face_names = []

camera = None
camera_running = False

# ID của vòng lặp camera root.after()
camera_after_id = None

# Đánh dấu app đang đóng
app_closing = False


# ============================================================
# CREATE FOLDER
# ============================================================

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)


# ============================================================
# LOAD FACE DATABASE
# ============================================================

def load_known_faces():
    """
    Đọc toàn bộ ảnh trong thư mục faces/
    và tạo face encoding.
    """

    global known_face_encodings
    global known_face_names

    known_face_encodings = []
    known_face_names = []

    print("\n========== LOAD DATABASE ==========")

    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)

    # Mỗi folder con là một người
    for person_name in os.listdir(FACES_DIR):

        person_folder = os.path.join(FACES_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):

            file_path = os.path.join(person_folder, filename)

            # Chỉ đọc các file ảnh
            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                continue

            try:
                image = face_recognition.load_image_file(file_path)

                face_locations = face_recognition.face_locations(image)

                if len(face_locations) == 0:
                    print(
                        f"[WARNING] Không tìm thấy khuôn mặt: {file_path}"
                    )
                    continue

                if len(face_locations) > 1:
                    print(
                        f"[WARNING] Có nhiều khuôn mặt: {file_path}"
                    )
                    print(
                        "         Bỏ qua ảnh này để tránh đăng ký sai."
                    )
                    continue

                encodings = face_recognition.face_encodings(
                    image,
                    face_locations
                )

                if len(encodings) == 0:
                    continue

                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)

                print(f"[OK] {person_name}: {filename}")

            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

    print(
        f"\nĐã load {len(known_face_encodings)} ảnh khuôn mặt."
    )
    print("==================================\n")


# ============================================================
# REGISTER PERSON
# ============================================================

def register_person():
    """
    Nhập tên + chọn nhiều ảnh.
    Ảnh được copy vào:
        faces/TEN_NGUOI/

    Nếu người đó đã có ảnh:
        001.jpg
        002.jpg

    thì ảnh mới sẽ tiếp tục:
        003.jpg
        004.jpg
        ...
    """

    name = name_entry.get().strip()

    if not name:
        messagebox.showwarning(
            "Thiếu tên",
            "Vui lòng nhập tên người cần đăng ký."
        )
        return

    # Lọc một số ký tự nguy hiểm cho tên folder
    invalid_chars = '<>:"/\\|?*'

    for char in invalid_chars:
        name = name.replace(char, "_")

    if not name:
        messagebox.showwarning(
            "Tên không hợp lệ",
            "Tên người không hợp lệ."
        )
        return

    # ========================================================
    # CHỌN NHIỀU ẢNH
    # ========================================================

    file_paths = filedialog.askopenfilenames(
        title="Chọn ảnh khuôn mặt",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            ),
            ("JPG", "*.jpg"),
            ("PNG", "*.png"),
            ("All files", "*.*")
        ]
    )

    if not file_paths:
        return

    # ========================================================
    # TẠO FOLDER NGƯỜI
    # ========================================================

    person_folder = os.path.join(
        FACES_DIR,
        name
    )

    os.makedirs(
        person_folder,
        exist_ok=True
    )

    # ========================================================
    # TÌM SỐ THỨ TỰ LỚN NHẤT ĐANG CÓ
    # ========================================================

    max_number = 0

    for filename in os.listdir(person_folder):

        file_path = os.path.join(
            person_folder,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        # Lấy tên file không có extension
        file_name_without_ext = os.path.splitext(
            filename
        )[0]

        # Nếu tên file là 001, 002, 003...
        if file_name_without_ext.isdigit():

            number = int(
                file_name_without_ext
            )

            max_number = max(
                max_number,
                number
            )

    # Ảnh mới sẽ bắt đầu từ số tiếp theo
    next_number = max_number + 1

    success_count = 0

    # ========================================================
    # XỬ LÝ TỪNG ẢNH
    # ========================================================

    for file_path in file_paths:

        try:

            # ------------------------------------------------
            # Đọc ảnh
            # ------------------------------------------------

            image = face_recognition.load_image_file(
                file_path
            )

            # ------------------------------------------------
            # Tìm khuôn mặt
            # ------------------------------------------------

            face_locations = face_recognition.face_locations(
                image
            )

            # Không có mặt
            if len(face_locations) == 0:

                print(
                    f"[SKIP] Không có khuôn mặt: {file_path}"
                )

                continue

            # Có nhiều mặt
            if len(face_locations) > 1:

                print(
                    f"[SKIP] Có nhiều khuôn mặt: {file_path}"
                )

                continue

            # ------------------------------------------------
            # Tạo tên file mới
            # ------------------------------------------------

            extension = os.path.splitext(
                file_path
            )[1].lower()

            new_filename = (
                f"{next_number:03d}{extension}"
            )

            destination = os.path.join(
                person_folder,
                new_filename
            )

            # ------------------------------------------------
            # Copy ảnh
            # ------------------------------------------------

            shutil.copy2(
                file_path,
                destination
            )

            print(
                f"[OK] Đã thêm ảnh: "
                f"{name}/{new_filename}"
            )

            success_count += 1

            # Tăng số cho ảnh tiếp theo
            next_number += 1

        except Exception as e:

            print(
                f"[ERROR] Không thể xử lý "
                f"{file_path}: {e}"
            )

    # ========================================================
    # KHÔNG CÓ ẢNH HỢP LỆ
    # ========================================================

    if success_count == 0:

        messagebox.showerror(
            "Đăng ký thất bại",
            "Không có ảnh hợp lệ chứa đúng 1 khuôn mặt."
        )

        return

    # ========================================================
    # LOAD LẠI DATABASE
    # ========================================================

    load_known_faces()

    # ========================================================
    # THÔNG BÁO
    # ========================================================

    messagebox.showinfo(
        "Đăng ký thành công",
        f"Người: {name}\n"
        f"Đã thêm: {success_count} ảnh\n\n"
        f"Tổng số ảnh hiện có: "
        f"{next_number - 1}"
    )

    # Xóa ô nhập tên
    name_entry.delete(
        0,
        tk.END
    )


def delete_person():
    """Xóa toàn bộ ảnh của một người trong thư mục faces/"""

    people = []

    if os.path.exists(FACES_DIR):
        for name in os.listdir(FACES_DIR):
            folder = os.path.join(FACES_DIR, name)

            if os.path.isdir(folder):
                people.append(name)

    if not people:
        messagebox.showinfo(
            "Xóa người",
            "Chưa có người nào được đăng ký."
        )
        return

    # Tạo cửa sổ chọn người
    delete_window = tk.Toplevel(root)
    delete_window.title("Xóa người")
    delete_window.geometry("400x300")
    delete_window.configure(bg="#303134")

    tk.Label(
        delete_window,
        text="Chọn người cần xóa:",
        font=("Arial", 14, "bold"),
        fg="white",
        bg="#303134"
    ).pack(pady=15)

    listbox = tk.Listbox(
        delete_window,
        font=("Arial", 13),
        height=8
    )
    listbox.pack(
        padx=20,
        pady=5,
        fill="both",
        expand=True
    )

    # Đánh số 1, 2, 3...
    for index, name in enumerate(people, start=1):
        listbox.insert(
            tk.END,
            f"{index}. {name}"
        )

    def confirm_delete():

        selection = listbox.curselection()

        if not selection:
            messagebox.showwarning(
                "Chưa chọn",
                "Vui lòng chọn người cần xóa.",
                parent=delete_window
            )
            return

        index = selection[0]
        name = people[index]

        confirm = messagebox.askyesno(
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa '{name}' không?\n\n"
            f"Tất cả ảnh của {name} sẽ bị xóa.",
            parent=delete_window
        )

        if not confirm:
            return

        person_folder = os.path.join(
            FACES_DIR,
            name
        )

        try:
            # Xóa toàn bộ thư mục người đó
            shutil.rmtree(person_folder)

            # Load lại database
            load_known_faces()

            messagebox.showinfo(
                "Xóa thành công",
                f"Đã xóa người: {name}",
                parent=delete_window
            )

            delete_window.destroy()

        except Exception as e:

            messagebox.showerror(
                "Lỗi",
                f"Không thể xóa:\n{e}",
                parent=delete_window
            )

    tk.Button(
        delete_window,
        text="Xóa người",
        font=("Arial", 12, "bold"),
        bg="#ea4335",
        fg="white",
        activebackground="#c5221f",
        activeforeground="white",
        padx=20,
        pady=8,
        command=confirm_delete
    ).pack(pady=15)

# ============================================================
# VERIFY UPLOADED IMAGE
# ============================================================

def verify_uploaded_image():

    if len(known_face_encodings) == 0:
        messagebox.showwarning(
            "Database trống",
            "Chưa có người nào được đăng ký."
        )
        return

    # Chọn ảnh
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh cần xác minh",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            ),
            ("JPG", "*.jpg"),
            ("PNG", "*.png"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:

        # Đọc ảnh
        image = cv2.imread(file_path)

        if image is None:
            messagebox.showerror(
                "Lỗi",
                "Không thể đọc ảnh."
            )
            return

        # Resize để nhận diện nhanh hơn
        small_image = cv2.resize(
            image,
            (0, 0),
            fx=PROCESS_SCALE,
            fy=PROCESS_SCALE
        )

        # BGR -> RGB
        rgb_small_image = cv2.cvtColor(
            small_image,
            cv2.COLOR_BGR2RGB
        )

        # Tìm khuôn mặt
        face_locations = face_recognition.face_locations(
            rgb_small_image,
            model="hog"
        )

        if len(face_locations) == 0:

            messagebox.showinfo(
                "Kết quả",
                "Không tìm thấy khuôn mặt trong ảnh."
            )

            return

        # Encoding
        face_encodings = face_recognition.face_encodings(
            rgb_small_image,
            face_locations
        )

        results = []

        # ====================================================
        # XÁC MINH TỪNG KHUÔN MẶT
        # ====================================================

        for face_encoding in face_encodings:

            name = "Unknown"
            distance_text = ""

            # Tính khoảng cách với database
            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            best_match_index = np.argmin(
                face_distances
            )

            best_distance = face_distances[
                best_match_index
            ]

            # Kiểm tra threshold
            if best_distance <= FACE_DISTANCE_THRESHOLD:

                name = known_face_names[
                    best_match_index
                ]

                confidence = max(
                    0,
                    min(
                        100,
                        (1 - best_distance) * 100
                    )
                )

                distance_text = (
                    f"{confidence:.1f}%"
                )

            results.append(
                (name, distance_text)
            )

        # ====================================================
        # VẼ KẾT QUẢ LÊN ẢNH
        # ====================================================

        for face_location, result in zip(
            face_locations,
            results
        ):

            name, confidence_text = result

            top, right, bottom, left = face_location

            # Chuyển về kích thước ảnh gốc
            top = int(top / PROCESS_SCALE)
            right = int(right / PROCESS_SCALE)
            bottom = int(bottom / PROCESS_SCALE)
            left = int(left / PROCESS_SCALE)

            # Màu
            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 200, 0)

            # Vẽ box
            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                color,
                3
            )

            # Label
            label = name

            if confidence_text:
                label += f" ({confidence_text})"

            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                2
            )[0]

            label_y = bottom + 35

            if label_y > image.shape[0] - 5:
                label_y = bottom - 10

            # Background
            cv2.rectangle(
                image,
                (
                    left,
                    label_y - text_size[1] - 10
                ),
                (
                    left + text_size[0] + 10,
                    label_y + 5
                ),
                color,
                -1
            )

            # Text
            cv2.putText(
                image,
                label,
                (
                    left + 5,
                    label_y - 3
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # ====================================================
        # HIỂN THỊ ẢNH KẾT QUẢ
        # ====================================================

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            image_rgb
        )

        # Tạo cửa sổ kết quả
        result_window = tk.Toplevel(root)

        result_window.title(
            "Kết quả xác minh"
        )

        result_window.geometry(
            "1000x700"
        )

        result_window.configure(
            bg="#202124"
        )

        # ====================================================
        # FIT ẢNH
        # ====================================================

        max_width = 950
        max_height = 600

        original_width, original_height = pil_image.size

        scale_width = (
            max_width / original_width
        )

        scale_height = (
            max_height / original_height
        )

        scale = min(
            scale_width,
            scale_height,
            1
        )

        new_width = max(
            1,
            int(original_width * scale)
        )

        new_height = max(
            1,
            int(original_height * scale)
        )

        pil_image = pil_image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            pil_image
        )

        image_label = tk.Label(
            result_window,
            image=photo,
            bg="black"
        )

        image_label.pack(
            padx=20,
            pady=20
        )

        # Giữ reference
        image_label.image = photo

        # ====================================================
        # HIỂN THỊ TEXT KẾT QUẢ
        # ====================================================

        result_text = "Kết quả xác minh:\n\n"

        for index, (name, confidence) in enumerate(
            results,
            start=1
        ):

            if confidence:
                result_text += (
                    f"Khuôn mặt {index}: "
                    f"{name} - {confidence}\n"
                )
            else:
                result_text += (
                    f"Khuôn mặt {index}: "
                    f"{name}\n"
                )

        result_label = tk.Label(
            result_window,
            text=result_text,
            font=("Arial", 14, "bold"),
            fg="white",
            bg="#202124"
        )

        result_label.pack(
            pady=10
        )

        tk.Button(
            result_window,
            text="Đóng",
            font=("Arial", 12, "bold"),
            bg="#ea4335",
            fg="white",
            padx=25,
            pady=8,
            command=result_window.destroy
        ).pack(
            pady=10
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi xác minh",
            f"Không thể xử lý ảnh:\n\n{e}"
        )


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize_faces(frame):
    """
    Nhận diện khuôn mặt trong frame webcam.

    Return:
        frame đã được vẽ box + tên
    """

    # Resize để tăng tốc
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=PROCESS_SCALE,
        fy=PROCESS_SCALE
    )

    # OpenCV dùng BGR
    # face_recognition dùng RGB
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    # Tìm khuôn mặt
    face_locations = face_recognition.face_locations(
        rgb_small_frame,
        model="hog"
    )

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )

    # Duyệt từng khuôn mặt
    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        name = "Unknown"
        confidence_text = ""

        # Nếu database có người
        if len(known_face_encodings) > 0:

            # Tính khoảng cách
            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            # Vị trí khoảng cách nhỏ nhất
            best_match_index = np.argmin(
                face_distances
            )

            best_distance = face_distances[
                best_match_index
            ]

            # Chỉ nhận nếu khoảng cách đủ nhỏ
            if best_distance <= FACE_DISTANCE_THRESHOLD:

                name = known_face_names[
                    best_match_index
                ]

                # Chuyển distance thành điểm tương đối
                confidence = max(
                    0,
                    min(
                        100,
                        (1 - best_distance) * 100
                    )
                )

                confidence_text = (
                    f"{confidence:.1f}%"
                )

        # Tọa độ trên frame nhỏ
        top, right, bottom, left = face_location

        # Đưa về kích thước frame gốc
        top = int(top / PROCESS_SCALE)
        right = int(right / PROCESS_SCALE)
        bottom = int(bottom / PROCESS_SCALE)
        left = int(left / PROCESS_SCALE)

        # Màu box
        if name == "Unknown":
            color = (0, 0, 255)       # Đỏ
        else:
            color = (0, 200, 0)       # Xanh

        # Vẽ khuôn mặt
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )

        # Vùng tên
        label = name

        if confidence_text:
            label += f" ({confidence_text})"

        label_y = bottom + 30

        # Không để label ra ngoài ảnh
        if label_y > frame.shape[0] - 5:
            label_y = bottom - 10

        # Background cho text
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )[0]

        cv2.rectangle(
            frame,
            (left, label_y - text_size[1] - 10),
            (left + text_size[0] + 10, label_y + 5),
            color,
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (left + 5, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return frame


# ============================================================
# CAMERA
# ============================================================

# def start_camera():

#     global camera
#     global camera_running

#     if camera_running:
#         return

#     camera = cv2.VideoCapture(0)

#     if not camera.isOpened():

#         messagebox.showerror(
#             "Camera Error",
#             "Không thể mở camera.\n"
#             "Hãy kiểm tra webcam."
#         )

#         camera = None
#         return

#     camera_running = True

#     camera_button.config(
#         text="Đang nhận diện..."
#     )

#     update_camera()

def start_camera():

    global camera
    global camera_running
    global camera_after_id

    if app_closing:
        return

    if camera_running:
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        messagebox.showerror(
            "Camera Error",
            "Không thể mở camera.\n"
            "Hãy kiểm tra webcam."
        )

        camera.release()
        camera = None

        return

    camera_running = True

    camera_button.config(
        text="Đang nhận diện...",
        bg="#34a853"
    )

    update_camera()



# 
def stop_camera():

    global camera
    global camera_running
    global camera_after_id

    camera_running = False

    # Hủy vòng lặp after()
    if camera_after_id is not None:

        try:
            root.after_cancel(camera_after_id)
        except:
            pass

        camera_after_id = None

    # Giải phóng camera
    if camera is not None:

        try:
            camera.release()
        except:
            pass

        camera = None

    # Reset giao diện
    if root.winfo_exists():

        camera_button.config(
            text="Bật camera",
            bg="#34a853"
        )

        camera_label.config(
            image="",
            text="Camera đã tắt"
        )

        camera_label.image = None


# def update_camera():

#     global camera_running

#     if not camera_running:
#         return

#     if camera is None:
#         return

#     ret, frame = camera.read()

#     if not ret:

#         stop_camera()

#         messagebox.showerror(
#             "Camera Error",
#             "Không đọc được hình ảnh từ camera."
#         )

#         return

#     # Lật camera giống camera selfie
#     frame = cv2.flip(
#         frame,
#         1
#     )

#     # Nhận diện
#     frame = recognize_faces(
#         frame
#     )

#     # OpenCV BGR -> RGB
#     frame_rgb = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2RGB
#     )

#     # PIL
#     image = Image.fromarray(
#         frame_rgb
#     )

#     # Resize để hiển thị
#     display_width = 800
#     display_height = int(
#         image.height *
#         display_width /
#         image.width
#     )

#     image = image.resize(
#         (
#             display_width,
#             display_height
#         )
#     )

#     photo = ImageTk.PhotoImage(
#         image=image
#     )

#     camera_label.config(
#         image=photo,
#         text=""
#     )

#     # Giữ reference
#     camera_label.image = photo

#     # Lặp lại
#     root.after(
#         10,
#         update_camera
#     )

# def update_camera():

#     global camera_running

#     if not camera_running:
#         return

#     if camera is None:
#         return

#     ret, frame = camera.read()

#     if not ret:
#         stop_camera()

#         messagebox.showerror(
#             "Camera Error",
#             "Không đọc được hình ảnh từ camera."
#         )

#         return

#     # Lật camera giống camera selfie
#     frame = cv2.flip(frame, 1)

#     # Nhận diện khuôn mặt
#     frame = recognize_faces(frame)

#     # BGR -> RGB
#     frame_rgb = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2RGB
#     )

#     # Chuyển sang PIL
#     image = Image.fromarray(frame_rgb)

#     # ========================================================
#     # FIT CAMERA VÀO KHUNG HIỂN THỊ
#     # ========================================================

#     # Lấy kích thước thực tế của vùng camera
#     camera_frame.update_idletasks()

#     max_width = camera_frame.winfo_width()
#     max_height = camera_frame.winfo_height()

#     # Nếu cửa sổ chưa lấy được kích thước
#     if max_width <= 1 or max_height <= 1:
#         root.after(10, update_camera)
#         return

#     # Kích thước ảnh gốc
#     original_width, original_height = image.size

#     # Tính tỷ lệ để ảnh nằm trọn trong khung
#     scale_width = max_width / original_width
#     scale_height = max_height / original_height

#     scale = min(
#         scale_width,
#         scale_height
#     )

#     new_width = int(
#         original_width * scale
#     )

#     new_height = int(
#         original_height * scale
#     )

#     # Resize giữ nguyên tỉ lệ
#     image = image.resize(
#         (
#             new_width,
#             new_height
#         ),
#         Image.Resampling.LANCZOS
#     )

#     # ========================================================
#     # TẠO NỀN ĐEN ĐỂ ẢNH KHÔNG BỊ CẮT
#     # ========================================================

#     background = Image.new(
#         "RGB",
#         (
#             max_width,
#             max_height
#         ),
#         "black"
#     )

#     # Căn giữa ảnh
#     x = (max_width - new_width) // 2
#     y = (max_height - new_height) // 2

#     background.paste(
#         image,
#         (x, y)
#     )

#     # PIL -> Tkinter
#     photo = ImageTk.PhotoImage(
#         background
#     )

#     camera_label.config(
#         image=photo,
#         text=""
#     )

#     # Giữ reference để ảnh không bị garbage collector
#     camera_label.image = photo

#     # Lặp lại
#     root.after(
#         10,
#         update_camera
#     )

def update_camera():

    global camera_running
    global camera_after_id

    # Nếu app đang đóng thì không chạy nữa
    if app_closing:
        return

    if not camera_running:
        return

    if camera is None:
        return

    try:

        ret, frame = camera.read()

        if not ret:

            stop_camera()

            if not app_closing:
                messagebox.showerror(
                    "Camera Error",
                    "Không đọc được hình ảnh từ camera."
                )

            return

        # ====================================================
        # LẬT CAMERA
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )

        # ====================================================
        # NHẬN DIỆN
        # ====================================================

        frame = recognize_faces(
            frame
        )

        # ====================================================
        # BGR -> RGB
        # ====================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame_rgb
        )

        # ====================================================
        # LẤY KÍCH THƯỚC KHUNG CAMERA
        # ====================================================

        camera_frame.update_idletasks()

        max_width = camera_frame.winfo_width()
        max_height = camera_frame.winfo_height()

        if max_width <= 1 or max_height <= 1:

            camera_after_id = root.after(
                10,
                update_camera
            )

            return

        # ====================================================
        # FIT ẢNH VÀO CAMERA FRAME
        # ====================================================

        original_width, original_height = image.size

        scale_width = (
            max_width / original_width
        )

        scale_height = (
            max_height / original_height
        )

        scale = min(
            scale_width,
            scale_height
        )

        new_width = max(
            1,
            int(original_width * scale)
        )

        new_height = max(
            1,
            int(original_height * scale)
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        # ====================================================
        # TẠO NỀN ĐEN
        # ====================================================

        background = Image.new(
            "RGB",
            (
                max_width,
                max_height
            ),
            "black"
        )

        # Căn giữa camera
        x = (
            max_width - new_width
        ) // 2

        y = (
            max_height - new_height
        ) // 2

        background.paste(
            image,
            (
                x,
                y
            )
        )

        # ====================================================
        # HIỂN THỊ
        # ====================================================

        photo = ImageTk.PhotoImage(
            background
        )

        camera_label.config(
            image=photo,
            text=""
        )

        camera_label.image = photo

        # ====================================================
        # TIẾP TỤC CAMERA
        # ====================================================

        if camera_running and not app_closing:

            camera_after_id = root.after(
                10,
                update_camera
            )

    except tk.TclError:
        # Cửa sổ đã bị destroy
        return

    except Exception as e:

        print(
            f"[CAMERA ERROR] {e}"
        )

        if not app_closing:
            stop_camera()



# ============================================================
# CLEAR DATABASE INFO
# ============================================================

# def show_database():

#     people = []

#     if os.path.exists(FACES_DIR):

#         for name in os.listdir(FACES_DIR):

#             folder = os.path.join(
#                 FACES_DIR,
#                 name
#             )

#             if os.path.isdir(folder):

#                 image_count = 0

#                 for file in os.listdir(folder):

#                     if file.lower().endswith(
#                         (
#                             ".jpg",
#                             ".jpeg",
#                             ".png",
#                             ".bmp",
#                             ".webp"
#                         )
#                     ):
#                         image_count += 1

#                 people.append(
#                     f"- {name}: {image_count} ảnh"
#                 )

#     if not people:

#         text = "Chưa có người nào được đăng ký."

#     else:

#         text = (
#             "Danh sách người đã đăng ký:\n\n"
#             + "\n".join(people)
#         )

#     messagebox.showinfo(
#         "Database",
#         text
#     )

def show_database():

    people = []

    if os.path.exists(FACES_DIR):

        for name in os.listdir(FACES_DIR):

            folder = os.path.join(
                FACES_DIR,
                name
            )

            if os.path.isdir(folder):

                image_count = 0

                for file in os.listdir(folder):

                    if file.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".bmp",
                            ".webp"
                        )
                    ):
                        image_count += 1

                people.append(
                    (name, image_count)
                )

    if not people:

        text = "Chưa có người nào được đăng ký."

    else:

        text = "Danh sách người đã đăng ký:\n\n"

        for index, (name, image_count) in enumerate(people, start=1):

            text += (
                f"{index}. {name}: {image_count} ảnh\n"
            )

    messagebox.showinfo(
        "Database",
        text
    )



# ============================================================
# WINDOW CLOSE
# ============================================================

# 

# def on_close():

#     global app_closing
#     global camera
#     global camera_running
#     global camera_after_id

#     # Tránh gọi hàm đóng nhiều lần
#     if app_closing:
#         return

#     app_closing = True

#     print("Đang đóng ứng dụng...")

#     # ========================================================
#     # DỪNG CAMERA
#     # ========================================================

#     camera_running = False

#     # ========================================================
#     # HỦY ROOT.AFTER
#     # ========================================================

#     if camera_after_id is not None:

#         try:
#             root.after_cancel(
#                 camera_after_id
#             )
#         except:
#             pass

#         camera_after_id = None

#     # ========================================================
#     # GIẢI PHÓNG CAMERA
#     # ========================================================

#     if camera is not None:

#         try:
#             camera.release()
#         except:
#             pass

#         camera = None

#     # ========================================================
#     # ĐÓNG OPENCV
#     # ========================================================

#     try:
#         cv2.destroyAllWindows()
#     except:
#         pass

#     # ========================================================
#     # ĐÓNG TKINTER
#     # ========================================================

#     try:
#         root.quit()
#     except:
#         pass

#     try:
#         root.destroy()
#     except:
#         pass

def on_close():

    global app_closing
    global camera
    global camera_running
    global camera_after_id

    # Tránh gọi đóng nhiều lần
    if app_closing:
        return

    app_closing = True

    print("Đang đóng ứng dụng...")

    # Dừng camera
    camera_running = False

    # Hủy vòng lặp camera
    if camera_after_id is not None:

        try:
            root.after_cancel(
                camera_after_id
            )
        except:
            pass

        camera_after_id = None

    # Giải phóng camera
    if camera is not None:

        try:
            camera.release()
        except:
            pass

        camera = None

    # Đóng OpenCV
    try:
        cv2.destroyAllWindows()
    except:
        pass

    # Thoát Tkinter
    try:
        root.quit()
    except:
        pass

    try:
        root.destroy()
    except:
        pass



# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Face Recognition App"
)

# root.geometry(
#     "1100x800"
# )

root.state("zoomed")

root.configure(
    bg="#202124"
)

root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)

# Nhấn ESC để thoát ứng dụng
root.bind(
    "<Escape>",
    lambda event: on_close()
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="FACE RECOGNITION",
    font=("Arial", 26, "bold"),
    fg="white",
    bg="#202124"
)

title_label.pack(
    pady=15
)


# ============================================================
# REGISTER FRAME
# ============================================================

register_frame = tk.Frame(
    root,
    bg="#303134",
    padx=15,
    pady=15
)

register_frame.pack(
    fill="x",
    padx=20
)


name_label = tk.Label(
    register_frame,
    text="Tên người:",
    font=("Arial", 13),
    fg="white",
    bg="#303134"
)

name_label.pack(
    side="left",
    padx=5
)


name_entry = tk.Entry(
    register_frame,
    font=("Arial", 13),
    width=25
)

name_entry.pack(
    side="left",
    padx=10
)


register_button = tk.Button(
    register_frame,
    text="Đăng ký + Tải ảnh",
    font=("Arial", 12, "bold"),
    bg="#1a73e8",
    fg="white",
    padx=15,
    pady=8,
    command=register_person
)

register_button.pack(
    side="left",
    padx=5
)


# database_button = tk.Button(
#     register_frame,
#     text="Xem danh sách",
#     font=("Arial", 12),
#     bg="#5f6368",
#     fg="white",
#     padx=15,
#     pady=8,
#     command=show_database
# )

database_button = tk.Button(
    register_frame,
    text="Xem danh sách",
    font=("Arial", 12, "bold"),
    bg="#f96800",
    fg="white",
    activebackground="#e65400",
    activeforeground="white",
    padx=15,
    pady=8,
    command=show_database
)


database_button.pack(
    side="left",
    padx=5
)

delete_button = tk.Button(
    register_frame,
    text="Xóa người",
    font=("Arial", 12, "bold"),
    bg="#ea4335",
    fg="white",
    activebackground="#c5221f",
    activeforeground="white",
    padx=15,
    pady=8,
    command=delete_person
)

verify_button = tk.Button(
    register_frame,
    text="Xác minh ảnh",
    font=("Arial", 12, "bold"),
    bg="#8e44ad",
    fg="white",
    activebackground="#71368a",
    activeforeground="white",
    padx=15,
    pady=8,
    command=verify_uploaded_image
)

verify_button.pack(
    side="left",
    padx=5
)


delete_button.pack(
    side="left",
    padx=5
)


# ============================================================
# CAMERA BUTTON
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#202124"
)

button_frame.pack(
    pady=15
)


# camera_button = tk.Button(
#     button_frame,
#     text="Bật camera",
#     font=("Arial", 14, "bold"),
#     bg="#34a853",
#     fg="white",
#     padx=30,
#     pady=10,
#     command=start_camera
# )

camera_button = tk.Button(
    button_frame,
    text="Bật camera",
    font=("Arial", 14, "bold"),
    bg="#34a853",
    fg="white",
    activebackground="#188038",
    activeforeground="white",
    padx=30,
    pady=10,
    command=start_camera
)


camera_button.pack(
    side="left",
    padx=10
)


# stop_button = tk.Button(
#     button_frame,
#     text="Tắt camera",
#     font=("Arial", 14, "bold"),
#     bg="#ea4335",
#     fg="white",
#     padx=30,
#     pady=10,
#     command=stop_camera
# )

stop_button = tk.Button(
    button_frame,
    text="Tắt camera",
    font=("Arial", 14, "bold"),
    bg="#ea4335",
    fg="white",
    activebackground="#c5221f",
    activeforeground="white",
    padx=30,
    pady=10,
    command=stop_camera
)


stop_button.pack(
    side="left",
    padx=10
)


# ============================================================
# CAMERA DISPLAY
# ============================================================

# camera_frame = tk.Frame(
#     root,
#     bg="#000000"
# )

# camera_frame.pack(
#     padx=20,
#     pady=5,
#     fill="both",
#     expand=True
# )

camera_frame = tk.Frame(
    root,
    bg="#000000"
)

camera_frame.pack(
    padx=20,
    pady=5,
    fill="both",
    expand=True
)

camera_frame.pack_propagate(False)


camera_label = tk.Label(
    camera_frame,
    text="Camera đã tắt",
    font=("Arial", 20),
    fg="white",
    bg="#000000"
)

camera_label.pack(
    fill="both",
    expand=True
)


# ============================================================
# INFO
# ============================================================

# info_label = tk.Label(
#     root,
#     text=(
#         "Màu xanh = đã nhận diện    |    "
#         "Màu đỏ = Unknown"
#     ),
#     font=("Arial", 11),
#     fg="#bbbbbb",
#     bg="#202124"
# )

# info_label.pack(
#     pady=8
# )

info_label = tk.Label(
    root,
    text=(
        "Xanh = Đã nhận diện    |    "
        "Đỏ = Unknown    |    "
        "Nhấn ESC để thoát ứng dụng"
    ),
    font=("Arial", 11),
    fg="#bbbbbb",
    bg="#202124"
)

info_label.pack(
    pady=(5, 10)
)



# ============================================================
# LOAD DATABASE
# ============================================================

load_known_faces()


# ============================================================
# START APP
# ============================================================

root.mainloop()
