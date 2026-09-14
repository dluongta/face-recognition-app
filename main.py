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
FACE_DISTANCE_THRESHOLD = 0.5
# Số chữ số hiển thị distance
DISTANCE_DECIMALS = 3
# Kích thước ảnh webcam đem đi nhận diện
# 0.25 = nhanh, nhẹ CPU
# 0.5 = nhận diện chính xác hơn nhưng nặng hơn
PROCESS_SCALE = 0.25

last_face_results = []
recognition_frame_counter = 0

# Nhận diện 1 lần sau mỗi N frame
RECOGNITION_INTERVAL = 3

DISTANCE_DECIMALS = 3

# Camera hiển thị khoảng 25 FPS
CAMERA_FPS = 25

# Kết quả nhận diện được giữ lại giữa các frame
last_face_results = []

recognition_frame_counter = 0

# Số frame liên tiếp không phát hiện mặt
no_face_counter = 0

# Sau bao nhiêu frame không thấy mặt thì mới xóa box
NO_FACE_TOLERANCE = 8


# Kích thước hiển thị tối đa
CAMERA_MAX_WIDTH = 960
CAMERA_MAX_HEIGHT = 720

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

    global person_encodings
    global known_face_encodings
    global known_face_names
    global last_face_results

    person_encodings = {}
    known_face_encodings = []
    known_face_names = []

    # Reset kết quả nhận diện cũ
    last_face_results = []

    print("\n========== LOAD DATABASE ==========")

    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)

    try:
        people = sorted(
            os.listdir(FACES_DIR)
        )

    except Exception as e:

        print(
            f"[ERROR] Không thể đọc database: {e}"
        )

        return

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for person_name in people:

        person_folder = os.path.join(
            FACES_DIR,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        person_encodings[person_name] = []

        try:

            files = sorted(
                os.listdir(person_folder)
            )

        except Exception as e:

            print(
                f"[ERROR] Không thể đọc folder "
                f"{person_name}: {e}"
            )

            continue

        for filename in files:

            if not filename.lower().endswith(
                valid_extensions
            ):
                continue

            file_path = os.path.join(
                person_folder,
                filename
            )

            if not os.path.isfile(file_path):
                continue

            try:

                # ============================================
                # ĐỌC ẢNH
                # ============================================

                image = face_recognition.load_image_file(
                    file_path
                )

                # ============================================
                # TÌM KHUÔN MẶT
                # ============================================

                face_locations = face_recognition.face_locations(
                    image,
                    model="hog"
                )

                if len(face_locations) == 0:

                    print(
                        f"[WARNING] Không có mặt: "
                        f"{person_name}/{filename}"
                    )

                    continue

                if len(face_locations) > 1:

                    print(
                        f"[WARNING] Có nhiều mặt: "
                        f"{person_name}/{filename}"
                    )

                    continue

                # ============================================
                # TẠO ENCODING
                # ============================================

                encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=face_locations,
                    num_jitters=1,
                    model="small"
                )

                if not encodings:

                    print(
                        f"[WARNING] Không tạo được encoding: "
                        f"{person_name}/{filename}"
                    )

                    continue

                encoding = encodings[0]

                # ============================================
                # LƯU DATABASE
                # ============================================

                person_encodings[
                    person_name
                ].append(
                    encoding
                )

                known_face_encodings.append(
                    encoding
                )

                known_face_names.append(
                    person_name
                )

                print(
                    f"[OK] {person_name}: {filename}"
                )

            except Exception as e:

                print(
                    f"[ERROR] {file_path}: {e}"
                )

        # ================================================
        # XÓA NGƯỜI KHÔNG CÓ ENCODING HỢP LỆ
        # ================================================

        if not person_encodings[person_name]:

            del person_encodings[
                person_name
            ]

    print(
        f"\nĐã load "
        f"{len(known_face_encodings)} encoding."
    )

    print(
        f"Số người: "
        f"{len(person_encodings)}"
    )

    for name, encodings in person_encodings.items():

        print(
            f"  - {name}: "
            f"{len(encodings)} ảnh"
        )

    print(
        "==================================\n"
    )




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
            "Lỗi thiếu tên",
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

            # ========================================================
            # Đọc ảnh
            # ========================================================

            image = face_recognition.load_image_file(
                file_path
            )

            # ========================================================
            # Tìm khuôn mặt
            # ========================================================

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

            # ========================================================
            # Tạo tên file mới
            # ========================================================

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

            # ========================================================
            # Copy ảnh
            # ========================================================

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

    if not person_encodings:

        messagebox.showwarning(
            "Database trống",
            "Chưa có người nào được đăng ký."
        )

        return

    # ========================================================
    # CHỌN ẢNH
    # ========================================================

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

        # ====================================================
        # ĐỌC ẢNH
        # ====================================================

        image = cv2.imread(
            file_path
        )

        if image is None:

            messagebox.showerror(
                "Lỗi",
                "Không thể đọc ảnh."
            )

            return

        original_image = image.copy()

        # ====================================================
        # RESIZE
        # ====================================================

        small_image = cv2.resize(
            image,
            None,
            fx=PROCESS_SCALE,
            fy=PROCESS_SCALE,
            interpolation=cv2.INTER_LINEAR
        )

        # ====================================================
        # BGR -> RGB
        # ====================================================

        rgb_small_image = cv2.cvtColor(
            small_image,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # FACE LOCATIONS
        # ====================================================

        face_locations = (
            face_recognition.face_locations(
                rgb_small_image,
                model="hog"
            )
        )

        if not face_locations:

            messagebox.showinfo(
                "Kết quả",
                "Không tìm thấy khuôn mặt trong ảnh."
            )

            return

        # ====================================================
        # FACE ENCODINGS
        # ====================================================

        face_encodings = (
            face_recognition.face_encodings(
                rgb_small_image,
                known_face_locations=face_locations,
                num_jitters=1,
                model="small"
            )
        )

        results = []

        # ====================================================
        # XÁC MINH
        # ====================================================

        for face_encoding in face_encodings:

            best_person = None
            best_distance = float("inf")

            # ================================================
            # SO SÁNH TỪNG NGƯỜI
            # ================================================

            for person_name, encodings in (
                person_encodings.items()
            ):

                if not encodings:
                    continue

                distances = (
                    face_recognition.face_distance(
                        encodings,
                        face_encoding
                    )
                )

                person_best_distance = float(
                    np.min(distances)
                )

                if (
                    person_best_distance
                    < best_distance
                ):

                    best_distance = (
                        person_best_distance
                    )

                    best_person = (
                        person_name
                    )

            # ================================================
            # CONFIDENCE
            # ================================================

            if (
                best_person is not None
                and best_distance != float("inf")
            ):

                confidence = (
                    1.0 - best_distance
                ) * 100

                confidence = max(
                    0.0,
                    min(
                        100.0,
                        confidence
                    )
                )

            else:

                confidence = 0.0

            # ================================================
            # THRESHOLD
            # ================================================

            if (
                best_person is not None
                and best_distance
                <= FACE_DISTANCE_THRESHOLD
            ):

                name = best_person

            else:

                name = "Unknown"

            results.append(
                (
                    name,
                    f"{confidence:.1f}%"
                )
            )

        # ====================================================
        # VẼ KẾT QUẢ
        # ====================================================

        for (
            face_location,
            result
        ) in zip(
            face_locations,
            results
        ):

            name, confidence_text = result

            top, right, bottom, left = (
                face_location
            )

            # Chuyển về ảnh gốc
            top = int(
                top / PROCESS_SCALE
            )

            right = int(
                right / PROCESS_SCALE
            )

            bottom = int(
                bottom / PROCESS_SCALE
            )

            left = int(
                left / PROCESS_SCALE
            )

            # ================================================
            # MÀU
            # ================================================

            if name == "Unknown":

                color = (
                    0,
                    0,
                    255
                )

            else:

                color = (
                    0,
                    200,
                    0
                )

            # ================================================
            # BOX
            # ================================================

            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                color,
                3,
                cv2.LINE_AA
            )

            # ================================================
            # LABEL
            # ================================================

            label = (
                f"{name} ({confidence_text})"
            )

            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                2
            )[0]

            label_width = (
                text_size[0] + 12
            )

            label_height = (
                text_size[1] + 12
            )

            label_top = (
                top - label_height
            )

            if label_top < 0:

                label_top = bottom

            label_bottom = (
                label_top
                + label_height
            )

            cv2.rectangle(
                image,
                (left, label_top),
                (
                    left + label_width,
                    label_bottom
                ),
                color,
                -1
            )

            cv2.putText(
                image,
                label,
                (
                    left + 6,
                    label_top
                    + text_size[1]
                    + 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # HIỂN THỊ ẢNH
        # ====================================================

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            image_rgb
        )

        # ====================================================
        # CỬA SỔ
        # ====================================================

        result_window = tk.Toplevel(
            root
        )

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

        original_width, original_height = (
            pil_image.size
        )

        scale = min(
            max_width / original_width,
            max_height / original_height,
            1.0
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

        image_label.image = photo

        # ====================================================
        # TEXT RESULT
        # ====================================================

        result_text = (
            "Kết quả xác minh:\n\n"
        )

        for index, (
            name,
            confidence_text
        ) in enumerate(
            results,
            start=1
        ):

            result_text += (
                f"Khuôn mặt {index}: "
                f"{name} - "
                f"confidence: "
                f"{confidence_text}\n"
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

        # ====================================================
        # BUTTON
        # ====================================================

        tk.Button(
            result_window,
            text="Đóng",
            font=("Arial", 12, "bold"),
            bg="#ea4335",
            fg="white",
            activebackground="#c5221f",
            activeforeground="white",
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

    global last_face_results
    global recognition_frame_counter
    global no_face_counter

    recognition_frame_counter += 1

    # ========================================================
    # CHỈ NHẬN DIỆN MỖI N FRAME
    # ========================================================

    should_recognize = (
        recognition_frame_counter % RECOGNITION_INTERVAL == 0
        or not last_face_results
    )

    # ========================================================
    # NHẬN DIỆN
    # ========================================================

    if should_recognize:

        try:

            # ------------------------------------------------
            # Resize frame
            # ------------------------------------------------

            small_frame = cv2.resize(
                frame,
                None,
                fx=PROCESS_SCALE,
                fy=PROCESS_SCALE,
                interpolation=cv2.INTER_LINEAR
            )

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            rgb_small_frame = cv2.cvtColor(
                small_frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # Detect face
            # ------------------------------------------------

            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                model="hog"
            )

            # =================================================
            # KHÔNG THẤY MẶT
            # =================================================

            if not face_locations:

                no_face_counter += 1

                # Không xóa ngay.
                # Giữ box cũ trong một khoảng thời gian.
                if no_face_counter >= NO_FACE_TOLERANCE:

                    last_face_results = []

            # =================================================
            # CÓ MẶT
            # =================================================

            else:

                no_face_counter = 0

                # =================================================
                # DATABASE TRỐNG
                # =================================================

                if not person_encodings:

                    last_face_results = []

                    for face_location in face_locations:

                        last_face_results.append(
                            (
                                face_location,
                                "Unknown",
                                ""
                            )
                        )

                # =================================================
                # DATABASE CÓ DỮ LIỆU
                # =================================================

                else:

                    face_encodings = (
                        face_recognition.face_encodings(
                            rgb_small_frame,
                            known_face_locations=face_locations,
                            num_jitters=1,
                            model="small"
                        )
                    )

                    new_results = []

                    # =================================================
                    # NHẬN DIỆN TỪNG KHUÔN MẶT
                    # =================================================

                    for face_encoding, face_location in zip(
                        face_encodings,
                        face_locations
                    ):

                        best_person = None
                        best_distance = float("inf")

                        # -----------------------------------------
                        # So sánh với database
                        # -----------------------------------------

                        for person_name, encodings in (
                            person_encodings.items()
                        ):

                            if not encodings:
                                continue

                            distances = (
                                face_recognition.face_distance(
                                    encodings,
                                    face_encoding
                                )
                            )

                            person_distance = float(
                                np.min(distances)
                            )

                            if person_distance < best_distance:

                                best_distance = (
                                    person_distance
                                )

                                best_person = (
                                    person_name
                                )

                        # -----------------------------------------
                        # Confidence
                        # -----------------------------------------

                        if (
                            best_person is not None
                            and best_distance != float("inf")
                        ):

                            confidence = (
                                1.0 - best_distance
                            ) * 100

                            confidence = max(
                                0.0,
                                min(
                                    100.0,
                                    confidence
                                )
                            )

                            confidence_text = (
                                f"{confidence:.1f}%"
                            )

                        else:

                            confidence_text = ""

                        # -----------------------------------------
                        # Threshold
                        # -----------------------------------------

                        if (
                            best_person is not None
                            and best_distance
                            <= FACE_DISTANCE_THRESHOLD
                        ):

                            name = best_person

                        else:

                            name = "Unknown"

                        # -----------------------------------------
                        # Lưu kết quả mới
                        # -----------------------------------------

                        new_results.append(
                            (
                                face_location,
                                name,
                                confidence_text
                            )
                        )

                    # =================================================
                    # CHỈ CẬP NHẬT KHI CÓ KẾT QUẢ
                    # =================================================

                    if new_results:

                        last_face_results = new_results

        except Exception as e:

            print(
                f"[RECOGNITION ERROR] {e}"
            )

            # Quan trọng:
            # KHÔNG xóa last_face_results khi có lỗi.
            # Giữ kết quả cũ để tránh nhấp nháy.

    # ========================================================
    # VẼ KẾT QUẢ CŨ / MỚI
    # ========================================================

    for (
        face_location,
        name,
        confidence_text
    ) in last_face_results:

        top, right, bottom, left = face_location

        # ====================================================
        # SCALE NGƯỢC VỀ FRAME GỐC
        # ====================================================

        top = int(
            top / PROCESS_SCALE
        )

        right = int(
            right / PROCESS_SCALE
        )

        bottom = int(
            bottom / PROCESS_SCALE
        )

        left = int(
            left / PROCESS_SCALE
        )

        # ====================================================
        # GIỚI HẠN FRAME
        # ====================================================

        height, width = frame.shape[:2]

        left = max(
            0,
            min(left, width - 1)
        )

        right = max(
            0,
            min(right, width - 1)
        )

        top = max(
            0,
            min(top, height - 1)
        )

        bottom = max(
            0,
            min(bottom, height - 1)
        )

        # ====================================================
        # MÀU
        # ====================================================

        if name == "Unknown":

            color = (
                0,
                0,
                255
            )

        else:

            color = (
                0,
                200,
                0
            )

        # ====================================================
        # VẼ BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # LABEL
        # ====================================================

        label = name

        if confidence_text:

            label += (
                f" ({confidence_text})"
            )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )[0]

        label_width = (
            text_size[0] + 10
        )

        label_height = (
            text_size[1] + 10
        )

        # Đặt label phía trên mặt
        label_top = top - label_height

        # Cạnh phải label = cạnh phải mặt
        label_right = right

        label_left = (
            label_right
            - label_width
        )

        # Không vượt trái
        if label_left < 0:

            label_left = 0

            label_right = label_width

        # Không vượt trên
        if label_top < 0:

            label_top = 0

        label_bottom = (
            label_top
            + label_height
        )

        # Không vượt dưới
        if label_bottom > height:

            label_bottom = height

        # ====================================================
        # LABEL BACKGROUND
        # ====================================================

        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_right, label_bottom),
            color,
            -1
        )

        # ====================================================
        # TEXT
        # ====================================================

        text_y = (
            label_top
            + text_size[1]
            + 4
        )

        cv2.putText(
            frame,
            label,
            (
                label_left + 5,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame



# ============================================================
# CAMERA
# ============================================================

def start_camera():

    global camera
    global camera_running
    global camera_after_id
    global recognition_frame_counter
    global last_face_results

    if app_closing:
        return

    if camera_running:
        return

    # Reset recognition
    recognition_frame_counter = 0
    last_face_results = []

    # ================================================
    # MỞ CAMERA
    # ================================================

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():


        camera.release()

        camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        camera = None

        messagebox.showerror(
            "Camera Error",
            "Không thể mở camera.\n"
            "Hãy kiểm tra webcam."
        )

        return

    # ================================================
    # CAMERA SETTINGS
    # ================================================

    # Độ phân giải camera
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Giảm buffer để tránh hình bị trễ
    try:

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except:
        pass

    # ================================================
    # START
    # ================================================

    camera_running = True

    camera_button.config(
        text="Đang nhận diện...",
        bg="#188038"
    )

    update_camera()



 
def stop_camera():

    global camera
    global camera_running
    global camera_after_id
    global last_face_results
    global recognition_frame_counter

    camera_running = False

    recognition_frame_counter = 0
    last_face_results = []

    # ====================================================
    # HỦY AFTER
    # ====================================================

    if camera_after_id is not None:

        try:
            root.after_cancel(
                camera_after_id
            )
        except:
            pass

        camera_after_id = None

    # ====================================================
    # RELEASE CAMERA
    # ====================================================

    if camera is not None:

        try:
            camera.release()
        except:
            pass

        camera = None

    # ====================================================
    # RESET UI
    # ====================================================

    if not app_closing:

        try:

            camera_button.config(
                text="Bật camera",
                bg="#34a853"
            )

            camera_label.config(
                image="",
                text="Camera đã tắt"
            )

            camera_label.image = None

        except tk.TclError:

            pass


def update_camera():

    global camera_running
    global camera_after_id

    # ========================================================
    # KIỂM TRA TRẠNG THÁI
    # ========================================================

    if app_closing:
        camera_after_id = None
        return

    if not camera_running:
        camera_after_id = None
        return

    if camera is None:
        camera_after_id = None
        return

    try:

        # ====================================================
        # ĐỌC CAMERA
        # ====================================================

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
        # LẤY KÍCH THƯỚC CAMERA FRAME
        # ====================================================

        max_width = (
            camera_frame.winfo_width()
        )

        max_height = (
            camera_frame.winfo_height()
        )

        if (
            max_width <= 1
            or max_height <= 1
        ):

            camera_after_id = root.after(
                30,
                update_camera
            )

            return

        # ====================================================
        # GIỚI HẠN KÍCH THƯỚC HIỂN THỊ
        # ====================================================

        max_width = min(
            max_width,
            CAMERA_MAX_WIDTH
        )

        max_height = min(
            max_height,
            CAMERA_MAX_HEIGHT
        )

        # ====================================================
        # BGR -> RGB
        # ====================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # PIL
        # ====================================================

        image = Image.fromarray(
            frame_rgb
        )

        original_width, original_height = (
            image.size
        )

        # ====================================================
        # TÍNH SCALE
        # ====================================================

        scale_width = (
            max_width
            / original_width
        )

        scale_height = (
            max_height
            / original_height
        )

        scale = min(
            scale_width,
            scale_height,
            1.0
        )

        new_width = max(
            1,
            int(
                original_width
                * scale
            )
        )

        new_height = max(
            1,
            int(
                original_height
                * scale
            )
        )

        # ====================================================
        # RESIZE
        # ====================================================

        if (
            new_width != original_width
            or new_height != original_height
        ):

            image = image.resize(
                (
                    new_width,
                    new_height
                ),
                Image.Resampling.BILINEAR
            )

        # ====================================================
        # TẠO BACKGROUND
        # ====================================================

        background = Image.new(
            "RGB",
            (
                max_width,
                max_height
            ),
            "black"
        )

        # ====================================================
        # CĂN GIỮA
        # ====================================================

        x = (
            max_width
            - new_width
        ) // 2

        y = (
            max_height
            - new_height
        ) // 2

        background.paste(
            image,
            (
                x,
                y
            )
        )

        # ====================================================
        # PIL -> TK
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
        # LOOP
        # ====================================================

        if (
            camera_running
            and not app_closing
        ):

            # 25 FPS ≈ 40 ms/frame
            camera_after_id = root.after(
                int(1000 / CAMERA_FPS),
                update_camera
            )

        else:

            camera_after_id = None

    except tk.TclError:

        camera_after_id = None

    except Exception as e:

        print(
            f"[CAMERA ERROR] {e}"
        )

        camera_after_id = None

        if not app_closing:

            try:
                stop_camera()
            except:
                pass




# ============================================================
# CLEAR DATABASE INFO
# ============================================================


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