# buildozer.spec (parte crítica)
[app]
title = SeñasCol
package.name = senascol
package.domain = org.senascol

[buildozer]
log_level = 2

[app]
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,tflite

requirements = 
    python3,
    kivy==2.1.0,
    opencv-python-headless,
    mediapipe,
    numpy,
    requests,
    pyttsx3,
    pillow,
    android

android.permissions = CAMERA,INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE