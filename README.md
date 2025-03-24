# Occlusion-Proof-Tracker
An occlusion-resistant tracker was developed to avoid losing the tracked item during prolonged occlusions.

It is a C++ application designed for real-time video processing and object tracking. It leverages OpenCV's CUDA capabilities for GPU acceleration, enabling efficient handling of video streams. The program supports multiple tracking algorithms, can process video from various sources (e.g., webcam, RTSP streams), and includes fallback mechanisms to CPU if GPU resources are unavailable.
Features

    GPU Acceleration: Uses CUDA for faster video processing when a CUDA-capable GPU is available.

    Multiple Tracking Algorithms: Supports trackers like CSRT, KCF, BOOSTING, MIL, TLD, MEDIANFLOW, and MOSSE.

    Video Source Flexibility:

        Webcam

        RTSP streams

        Local video files

    Frame Preprocessing:

        Gaussian blur for denoising

        Resize functionality

    Robust Object Tracking:

        Feature-based tracking using AKAZE descriptors.

        Motion model prediction for improved tracking accuracy.

        Recovery mechanism for lost objects.

    Dynamic Fallback: Automatically falls back to CPU processing if GPU is unavailable or encounters an error.

Dependencies

To build and run this application, ensure the following dependencies are installed:

    C++ Compiler: GCC or MSVC with C++17 support.

    OpenCV: With CUDA support enabled (version 4.x or later recommended).

    CUDA Toolkit: Required for GPU acceleration.

    Linux/Windows Headers: Depending on the platform.

