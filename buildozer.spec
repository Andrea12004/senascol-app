[app]
title = SeñasCol
package.name = senascol
package.domain = org.senascol

source.dir = .
source.include_exts = py,png,jpg,kv,json,tflite

version = 1.0

# Requirements SIN MediaPipe Python (usaremos la versión Android nativa)
requirements = python3,kivy==2.1.0,opencv,numpy,pillow,requests,pyjnius,android

# Permisos
android.permissions = CAMERA,INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# Agregar MediaPipe Android nativo + TensorFlow Lite
android.gradle_dependencies = com.google.mediapipe:tasks-vision:0.10.14,org.tensorflow:tensorflow-lite:2.9.0,org.tensorflow:tensorflow-lite-select-tf-ops:2.9.0

# AAR personalizado (si es necesario)
android.add_aars = 

# Configuración Android
orientation = portrait
fullscreen = 0

android.api = 33
android.minapi = 21
android.ndk = 23b
android.accept_sdk_license = True

# Arquitecturas
android.archs = arm64-v8a,armeabi-v7a

# Bootstrap
p4a.bootstrap = sdl2
p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1