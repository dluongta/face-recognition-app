## Overview
- Register photos with person's name.
- Recognize person's name in the camera in real time.
- Delete registered person's photos.
- Recognize person's name in photos.

## Install

```
python -m pip install -r requirements.txt
```

## Build

```
python -m PyInstaller --onefile --windowed --name FaceRecognitionApp --add-data "C:\Users\DELL\AppData\Local\Programs\Python\Python310\lib\site-packages\face_recognition_models;face_recognition_models" main.py

```

## Run

```
python main.py
```