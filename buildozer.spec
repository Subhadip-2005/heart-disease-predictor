[app]

# Application title
title = Heart Disease Predictor

# Package name and domain
package.name = heartdisease
package.domain = org.example

# Source directory
source.dir = .

# Only include necessary files - EXCLUDE .pkl files
source.include_exts = py,png,jpg,kv,atlas

# Exclude large model files from APK
source.exclude_exts = pkl,csv,db

# Exclude directories
source.exclude_dirs = instance,.buildozer,bin,__pycache__,.git

# Application version
version = 0.1.0

# Application requirements
requirements = python3,kivy,requests

# Python version
python_version = 3.11

# Orientation
orientation = portrait

# Fullscreen
fullscreen = 0

# Android configuration
android.permissions = INTERNET
android.api = 33
android.minapi = 21
android.ndk = 25b

# Architecture (ARM64 recommended for modern phones)
android.archs = arm64-v8a,armeabi-v7a

# Java version
android.java_version = 11

# Gradle version
gradle_version = 7.0

# Logcat filters
log_level = 2

# Internet permission
android.features = android.hardware.internet