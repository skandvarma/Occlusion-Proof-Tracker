#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

// OpenCV CUDA headers
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudacodec.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// Standard C++ headers
#include <iostream>
#include <string>
#include <queue>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <thread>
#include<fstream>
#include <unistd.h> // for getpagesize()
#include <sys/resource.h> // for getrusage
#include <numeric>   // For std::accumulate
#include <condition_variable>

// System-specific headers
#ifdef __linux__
    #include <unistd.h>
    #include <sys/resource.h>
#endif

// CUDA check function
bool USE_GPU = false;
void check_cuda() {
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            USE_GPU = true;
            std::cout << "Using GPU: " << USE_GPU << std::endl;
        } else {
            USE_GPU = false;
            std::cout << "Using CPU: " << USE_GPU << std::endl;
        }
    } catch (const cv::Exception& e) {
        USE_GPU = false;
        std::cout << "Using CPU: " << USE_GPU << std::endl;
    }
}

// Parse command-line arguments
struct Arguments {
    std::string video;
    std::string tracker_type;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    
    // Default values
    args.video = "0";  // Default to webcam
    args.tracker_type = "CSRT";  // Default to CSRT tracker

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-v" || arg == "--video") {
            if (i + 1 < argc) {
                args.video = argv[++i];
            } else {
                std::cerr << "Video path is required after " << arg << std::endl;
                exit(1);
            }
        } else if (arg == "-t" || arg == "--tracker") {
            if (i + 1 < argc) {
                args.tracker_type = argv[++i];
                // Check for valid tracker choices
                if (args.tracker_type != "CSRT" && args.tracker_type != "KCF" && 
                    args.tracker_type != "BOOSTING" && args.tracker_type != "MIL" && 
                    args.tracker_type != "TLD" && args.tracker_type != "MEDIANFLOW" && 
                    args.tracker_type != "MOSSE") {
                    std::cerr << "Invalid tracker type. Available choices: CSRT, KCF, BOOSTING, MIL, TLD, MEDIANFLOW, MOSSE" << std::endl;
                    exit(1);
                }
            } else {
                std::cerr << "Tracker type is required after " << arg << std::endl;
                exit(1);
            }
        }
    }

    return args;
}


class VideoCapture {
private:
    cv::VideoCapture cap;
    cv::Size resize;
    int skip_frames;
    const int RTSP_BUFFER = 0;
    bool use_gpu;
    bool is_rtsp;
    
    // GPU resources
    std::unique_ptr<cv::cuda::GpuMat> gpu_frame;
    std::unique_ptr<cv::cuda::GpuMat> gpu_resized;
    std::unique_ptr<cv::cuda::GpuMat> gpu_denoised;
    std::unique_ptr<cv::cuda::GpuMat> gpu_gray;
    std::unique_ptr<cv::cuda::GpuMat> gpu_hsv;
    cv::Ptr<cv::cuda::Filter> gpu_blur;
    std::unique_ptr<cv::cuda::Stream> stream;

    // Preallocated CPU matrices
    cv::Mat cpu_frame;
    cv::Mat cpu_resized;
    cv::Mat cpu_denoised;

public:
    VideoCapture(const std::string& source, cv::Size resize = cv::Size(600, 300), int skip_frames = 15) 
        : resize(resize), skip_frames(skip_frames), use_gpu(false) {
        try {
            safeInitializeGPU();
            openVideoSource(source);
            configureVideoSource();
        } catch (const std::exception& e) {
            std::cerr << "Error initializing video capture: " << e.what() << std::endl;
            use_gpu = false;
            openVideoSource(source);
            configureVideoSource();
        }
    }

    VideoCapture(int camera_index, cv::Size resize = cv::Size(600, 300), int skip_frames = 15) 
        : resize(resize), skip_frames(skip_frames), use_gpu(false) {
        try {
            safeInitializeGPU();
            openVideoSource(camera_index);
            configureVideoSource();
        } catch (const std::exception& e) {
            std::cerr << "Error initializing video capture: " << e.what() << std::endl;
            use_gpu = false;
            openVideoSource(camera_index);
            configureVideoSource();
        }
    }

private:
    void safeInitializeGPU() {
        try {
            if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                // Create stream first
                stream = std::make_unique<cv::cuda::Stream>();
                
                // Initialize GPU matrices
                gpu_frame = std::make_unique<cv::cuda::GpuMat>();
                gpu_resized = std::make_unique<cv::cuda::GpuMat>();
                gpu_denoised = std::make_unique<cv::cuda::GpuMat>();
                gpu_gray = std::make_unique<cv::cuda::GpuMat>();
                gpu_hsv = std::make_unique<cv::cuda::GpuMat>();
                
                // Initialize GPU operations
                gpu_blur = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(3,3), 1.5);
                
                use_gpu = true;
                std::cout << "GPU acceleration enabled successfully" << std::endl;
                
                // Print GPU info
                cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
            } else {
                std::cout << "No CUDA-capable devices found, using CPU mode" << std::endl;
                use_gpu = false;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "GPU initialization failed, falling back to CPU: " << e.what() << std::endl;
            cleanupGPUResources();
            use_gpu = false;
        }
    }

    void cleanupGPUResources() {
        gpu_frame.reset();
        gpu_resized.reset();
        gpu_denoised.reset();
        gpu_gray.reset();
        gpu_hsv.reset();
        gpu_blur.release();
        stream.reset();
    }

    void openVideoSource(const std::string& source) {
        bool is_number = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
        
        if (is_number) {
            openVideoSource(std::stoi(source));
            return;
        }

        is_rtsp = source.substr(0, 4) == "rtsp";
        std::string pipeline;
        
        if (is_rtsp) {
            pipeline = createGstreamerPipeline(source);
            if (!cap.open(pipeline, cv::CAP_GSTREAMER)) {
                if (!cap.open(source)) {
                    throw std::runtime_error("Error opening video source: " + source);
                }
            }
        } else {
            if (!cap.open(source)) {
                throw std::runtime_error("Error opening video source: " + source);
            }
        }
    }

    void openVideoSource(int camera_index) {
        is_rtsp = false;
        bool opened = false;
        
        #ifdef _WIN32
            opened = cap.open(camera_index, cv::CAP_DSHOW);
        #else
            opened = cap.open(camera_index, cv::CAP_V4L2);
        #endif
        
        if (!opened) {
            if (!cap.open(camera_index)) {
                throw std::runtime_error("Error opening camera index: " + std::to_string(camera_index));
            }
        }
    }

    std::string createGstreamerPipeline(const std::string& rtsp_url) {
        return "rtspsrc location=" + rtsp_url + 
               " latency=0 ! rtph264depay ! h264parse ! " +
               #ifdef _WIN32
               "d3d11h264dec ! videoconvert ! " +
               #else
               "avdec_h264 max-threads=8 ! videoconvert ! " +
               #endif
               "appsink max-buffers=1 drop=true sync=false";
    }

    void configureVideoSource() {
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video source");
        }

        cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                           cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (is_rtsp) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, RTSP_BUFFER);
            cap.set(cv::CAP_PROP_FPS, 60);
        } else {
            cap.set(cv::CAP_PROP_FPS, 30);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }

        // Preallocate CPU matrices
        cpu_frame = cv::Mat(frame_size, CV_8UC3);
        if (resize.width > 0 && resize.height > 0) {
            cpu_resized = cv::Mat(resize, CV_8UC3);
        }
        cpu_denoised = cv::Mat(frame_size, CV_8UC3);

        // Initialize GPU buffers if using GPU
        if (use_gpu) {
            try {
                gpu_frame->create(frame_size, CV_8UC3);
                if (resize.width > 0 && resize.height > 0) {
                    gpu_resized->create(resize, CV_8UC3);
                }
                gpu_denoised->create(frame_size, CV_8UC3);
                gpu_gray->create(frame_size, CV_8UC1);
                gpu_hsv->create(frame_size, CV_8UC3);
            } catch (const cv::Exception& e) {
                std::cerr << "GPU buffer allocation failed, falling back to CPU: " << e.what() << std::endl;
                use_gpu = false;
            }
        }

        std::cout << "Video source configuration:" << std::endl
                  << "- Type: " << (is_rtsp ? "RTSP Stream" : "Camera") << std::endl
                  << "- Resolution: " << frame_size.width << "x" << frame_size.height << std::endl
                  << "- FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl
                  << "- GPU Acceleration: " << (use_gpu ? "Enabled" : "Disabled") << std::endl;
    }

    void processFrameGPU(const cv::Mat& input, cv::Mat& output) {
        try {
            if (!gpu_frame || !gpu_resized || !stream) {
                throw cv::Exception(0, "GPU resources not initialized", __FUNCTION__, __FILE__, __LINE__);
            }

            // Upload with zero copy if possible
            gpu_frame->upload(input, *stream);
            
            // Apply denoising
            if (gpu_blur) {
                gpu_blur->apply(*gpu_frame, *gpu_denoised, *stream);
            } else {
                gpu_frame->copyTo(*gpu_denoised, *stream);
            }

            // Resize if needed
            if (resize.width > 0 && resize.height > 0) {
                cv::cuda::resize(*gpu_denoised, *gpu_resized, resize, 0, 0, cv::INTER_NEAREST, *stream);
                gpu_resized->download(output, *stream);
            } else {
                gpu_denoised->download(output, *stream);
            }

            stream->waitForCompletion();
        } catch (const cv::Exception& e) {
            std::cerr << "GPU processing error, falling back to CPU for this frame: " << e.what() << std::endl;
            processFrameCPU(input, output);
        }
    }

    void processFrameCPU(const cv::Mat& input, cv::Mat& output) {
        cv::GaussianBlur(input, cpu_denoised, cv::Size(3,3), 1.5);
        
        if (resize.width > 0 && resize.height > 0) {
            cv::resize(cpu_denoised, output, resize, 0, 0, cv::INTER_NEAREST);
        } else {
            cpu_denoised.copyTo(output);
        }
    }

public:
    bool isOpened() const {
        return cap.isOpened();
    }
    
    bool read(cv::Mat& frame) {
        if (!cap.isOpened()) {
            return false;
        }

        bool ret = true;

        // Handle frame skipping for RTSP
        if (is_rtsp) {
            for (int i = 0; i < skip_frames; ++i) {
                ret = cap.grab();
                if (!ret) break;
            }
        }

        // Read frame
        if (ret) {
            ret = cap.retrieve(cpu_frame);
            if (ret && !cpu_frame.empty()) {
                if (use_gpu) {
                    try {
                        processFrameGPU(cpu_frame, frame);
                    } catch (const cv::Exception& e) {
                        std::cerr << "GPU processing failed, falling back to CPU: " << e.what() << std::endl;
                        processFrameCPU(cpu_frame, frame);
                    }
                } else {
                    processFrameCPU(cpu_frame, frame);
                }
            }
        }

        return ret;
    }

    // GPU-optimized conversion methods
    void convertToGray(const cv::Mat& input, cv::Mat& output) {
        if (use_gpu) {
            try {
                gpu_frame->upload(input, *stream);
                cv::cuda::cvtColor(*gpu_frame, *gpu_gray, cv::COLOR_BGR2GRAY, 0, *stream);
                gpu_gray->download(output, *stream);
                stream->waitForCompletion();
            } catch (const cv::Exception& e) {
                std::cerr << "GPU grayscale conversion failed: " << e.what() << std::endl;
                cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
            }
        } else {
            cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
        }
    }

    void convertToHSV(const cv::Mat& input, cv::Mat& output) {
        if (use_gpu) {
            try {
                gpu_frame->upload(input, *stream);
                cv::cuda::cvtColor(*gpu_frame, *gpu_hsv, cv::COLOR_BGR2HSV, 0, *stream);
                gpu_hsv->download(output, *stream);
                stream->waitForCompletion();
            } catch (const cv::Exception& e) {
                std::cerr << "GPU HSV conversion failed: " << e.what() << std::endl;
                cv::cvtColor(input, output, cv::COLOR_BGR2HSV);
            }
        } else {
            cv::cvtColor(input, output, cv::COLOR_BGR2HSV);
        }
    }

    // Utility methods
    bool isUsingGPU() const { return use_gpu; }
    cv::Size getFrameSize() const { return cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)); }
    double getFPS() const { return cap.get(cv::CAP_PROP_FPS); }
    bool isRTSP() const { return is_rtsp; }
    void setSkipFrames(int skip) { skip_frames = skip; }
    cv::Size getResizeResolution() const { return resize; }

    void setResizeResolution(const cv::Size& new_size) {
        resize = new_size;
        if (use_gpu && gpu_resized) {
            try {
                gpu_resized->create(resize, CV_8UC3);
            } catch (const cv::Exception& e) {
                std::cerr << "Failed to update GPU resize buffer: " << e.what() << std::endl;
            }
        }
    }

    void release() {
        cap.release();
        cleanupGPUResources();
        
        // Release CPU resources
        cpu_frame.release();
        cpu_resized.release();
        cpu_denoised.release();
    }

    ~VideoCapture() {
        release();
    }
};


class FeatureTracker {
public:
    struct TrackingResult {
        bool success;
        cv::Rect2d updated_bbox;
        double confidence;
        std::vector<cv::Point2f> matched_points;
        bool is_occluded;
        bool recovered;
        cv::Rect2d predicted_bbox;
        cv::Point2f velocity;
        cv::Point2f acceleration;
        
        TrackingResult(const cv::Rect2d& bbox) : 
            success(false), 
            updated_bbox(bbox), 
            confidence(0.0), 
            is_occluded(false), 
            recovered(false), 
            predicted_bbox(bbox),
            velocity(0, 0),
            acceleration(0, 0) {}
    };

private:
    // Feature detection and matching
    cv::Ptr<cv::AKAZE> feature_detector;
    cv::Ptr<cv::FastFeatureDetector> fast_detector;
    cv::FlannBasedMatcher flann_matcher;
    
    // Kalman filter for motion prediction
    cv::KalmanFilter motion_kalman;
    bool kalman_initialized = false;
    
    // Motion model
    struct MotionModel {
        cv::Point2f velocity;
        cv::Point2f acceleration;
        double confidence;
        std::chrono::steady_clock::time_point last_update;
        
        void reset() {
            velocity = cv::Point2f(0, 0);
            acceleration = cv::Point2f(0, 0);
            confidence = 0.0;
        }
    } motion_model;
    
    // Feature history for robust tracking
    struct FeatureHistory {
        std::deque<cv::Point2f> positions;
        std::deque<cv::Mat> descriptors;
        double reliability;
        const size_t MAX_HISTORY = 60;
        
        void update(const cv::Point2f& pos, const cv::Mat& desc) {
            positions.push_back(pos);
            descriptors.push_back(desc.clone());
            if (positions.size() > MAX_HISTORY) {
                positions.pop_front();
                descriptors.front().release();
                descriptors.pop_front();
            }
            updateReliability();
        }
        
        void updateReliability() {
            if (positions.size() < 2) {
                reliability = 1.0;
                return;
            }
            
            double movement_consistency = 0.0;
            for (size_t i = 1; i < positions.size(); ++i) {
                cv::Point2f motion = positions[i] - positions[i-1];
                movement_consistency += std::exp(-cv::norm(motion) * 0.1);
            }
            reliability = movement_consistency / (positions.size() - 1);
        }
        
        void clear() {
            positions.clear();
            for (auto& desc : descriptors) desc.release();
            descriptors.clear();
            reliability = 0.0;
        }
    };
    std::vector<FeatureHistory> feature_histories;
    
    // Tracking parameters
    const float match_ratio = 0.25f;  // Stricter matching ratio (was 0.72f)
    const int min_matches = 16;  // Increased minimum matches (was 3)
    const double min_confidence = 0.50;  // Increased minimum confidence (was 0.10)
    const int min_area = 24;
    const double max_area_ratio = 0.95;
    const double ransac_threshold = 0.5;  // Reduced for more precise matching (was 2.5)
    const int MAX_FEATURES = 2000;  // Increased max features (was 500)
    const float FAST_THRESHOLD = 10.0f;  // Reduced to detect more features (was 20.0f)


    // Recovery parameters
    const int search_expansion = 20;
    const int max_recovery_attempts = 60;
    int recovery_count = 0;
    cv::Mat template_features;
    cv::Mat original_roi;
    std::vector<cv::KeyPoint> template_keypoint;
    cv::Mat template_descriptors;
    double original_area;
    cv::Mat template_hist;  // Added histogram storage
    int recovery_frames = 0;
    const int recovery_threshold = 1;
    // Motion prediction
    cv::Point2f last_velocity;
    cv::Point2f last_valid_center;
    bool has_last_motion = false;
    const double velocity_weight = 0.85;
    
    // Performance metrics
    struct Metrics {
        double avg_processing_time = 0.0;
        int successful_tracks = 0;
        int failed_tracks = 0;
        int recoveries = 0;
        
        void reset() {
            avg_processing_time = 0.0;
            successful_tracks = 0;
            failed_tracks = 0;
            recoveries = 0;
        }
    } metrics;

    double original_roi_area = 0.0;
    double original_aspect_ratio = 1.0;
    cv::Size original_size;


    void storeTemplate(const cv::Mat& frame, const cv::Rect2d& bbox) {
        try {
            // Convert Rect2d to Rect for integer coordinates
            cv::Rect roi(
                static_cast<int>(bbox.x),
                static_cast<int>(bbox.y),
                static_cast<int>(bbox.width),
                static_cast<int>(bbox.height)
            );
            
            // Validate ROI
            roi = validateSearchArea(roi, frame.size());
            
            // Store multiple scaled versions of the ROI for better matching
            std::vector<cv::Mat> scaled_templates;
            std::vector<std::vector<cv::KeyPoint>> scaled_keypoints;
            std::vector<cv::Mat> scaled_descriptors;
            
            // Convert to grayscale
            cv::Mat gray_frame;
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
            original_roi = gray_frame(roi).clone();
            
            // Store original and scaled versions
            const std::vector<float> scales = {0.8f, 1.0f, 1.2f};
            for (float scale : scales) {
                cv::Mat scaled_roi;
                cv::resize(original_roi, scaled_roi, cv::Size(), scale, scale);
                scaled_templates.push_back(scaled_roi);
                
                std::vector<cv::KeyPoint> kps;
                cv::Mat descs;
                feature_detector->detectAndCompute(scaled_roi, cv::Mat(), kps, descs);
                scaled_keypoints.push_back(kps);
                scaled_descriptors.push_back(descs);
            }
            
            // Combine all features
            template_keypoint.clear();
            template_descriptors.release();
            for (size_t i = 0; i < scaled_keypoints.size(); i++) {
                // Adjust keypoint coordinates for scaled versions
                for (auto& kp : scaled_keypoints[i]) {
                    kp.pt *= scales[i];
                    template_keypoint.push_back(kp);
                }
                if (!scaled_descriptors[i].empty()) {
                    if (template_descriptors.empty()) {
                        template_descriptors = scaled_descriptors[i].clone();
                    } else {
                        cv::vconcat(template_descriptors, scaled_descriptors[i], template_descriptors);
                    }
                }
            }
            
            // Store additional appearance information
            cv::Mat roi_img = frame(roi);
            cv::Mat hsv_roi;
            cv::cvtColor(roi_img, hsv_roi, cv::COLOR_BGR2HSV);
            
            // Calculate color histogram
            int h_bins = 50, s_bins = 60;
            int histSize[] = {h_bins, s_bins};
            float h_ranges[] = {0, 180}, s_ranges[] = {0, 256};
            const float* ranges[] = {h_ranges, s_ranges};
            int channels[] = {0, 1};
            
            // Create and calculate histogram
            template_hist.release();  // Clear any existing histogram
            cv::calcHist(&hsv_roi, 1, channels, cv::Mat(), template_hist, 2, histSize, ranges);
            cv::normalize(template_hist, template_hist, 0, 1, cv::NORM_MINMAX);
            
            // Store original properties
            original_roi_area = roi.area();
            original_aspect_ratio = static_cast<double>(roi.width) / roi.height;
            original_size = roi.size();
            
            // Reset recovery counters
            recovery_count = 0;
            recovery_frames = 0;
            
            // Initialize feature histories with enhanced features
            feature_histories.clear();
            for (const auto& kp : template_keypoint) {
                FeatureHistory history;
                history.positions.push_back(kp.pt);
                feature_histories.push_back(history);
            }
            
            std::cout << "Template stored with " << template_keypoint.size() 
                     << " keypoints and multiple scales" << std::endl;
                     
        } catch (const cv::Exception& e) {
            std::cerr << "Error storing template: " << e.what() << std::endl;
            template_keypoint.clear();
            template_descriptors.release();
            template_hist.release();
        }
    }


    cv::Rect validateBox(const cv::Rect& box, const cv::Size& frame_size) {
        cv::Rect valid = box;
        
        // Ensure coordinates are not negative
        valid.x = std::max(0, valid.x);
        valid.y = std::max(0, valid.y);
        
        // Ensure width and height are positive and not too small
        valid.width = std::max(min_area, valid.width);
        valid.height = std::max(min_area, valid.height);
        
        // Ensure box doesn't exceed frame boundaries
        valid.width = std::min(valid.width, frame_size.width - valid.x);
        valid.height = std::min(valid.height, frame_size.height - valid.y);
        
        // One final check for maximum area
        if (valid.area() > frame_size.area() * max_area_ratio) {
            float scale = std::sqrt((frame_size.area() * max_area_ratio) / valid.area());
            valid.width = static_cast<int>(valid.width * scale);
            valid.height = static_cast<int>(valid.height * scale);
        }
        
        return valid;
    }
public:
    FeatureTracker() {
        initializeDetectors();
        initializeKalmanFilter();
    }

    bool attemptRecovery(const cv::Mat& frame, const cv::Rect& search_area,
                    cv::Rect2d& recovered_bbox) {
    try {
        // Convert to grayscale with error checking
        cv::Mat gray_frame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        } else {
            gray_frame = frame.clone();
        }
        
        // Calculate padded search area
        cv::Rect padded_search = search_area;
        padded_search.x -= search_area.width * 0.25;  // Increased search margin
        padded_search.y -= search_area.height * 0.25;
        padded_search.width *= 1.5;
        padded_search.height *= 1.5;
        cv::Rect valid_search = validateSearchArea(padded_search, frame.size());
        
        // Extract and validate ROI
        if (valid_search.width < 10 || valid_search.height < 10 || 
            valid_search.x < 0 || valid_search.y < 0 || 
            valid_search.x + valid_search.width > frame.cols ||
            valid_search.y + valid_search.height > frame.rows) {
            return false;
        }
        
        cv::Mat search_roi = gray_frame(valid_search);
        
        // Multi-scale feature detection
        std::vector<cv::KeyPoint> curr_keypoints;
        cv::Mat curr_descriptors;
        std::vector<float> scale_factors = {0.8f, 1.0f, 1.2f};
        
        for (float scale : scale_factors) {
            cv::Mat scaled_roi;
            if (std::abs(scale - 1.0f) > 0.01f) {
                cv::resize(search_roi, scaled_roi, cv::Size(), scale, scale);
            } else {
                scaled_roi = search_roi;
            }
            
            // Try AKAZE detection
            std::vector<cv::KeyPoint> scale_keypoints;
            cv::Mat scale_descriptors;
            try {
                feature_detector->detectAndCompute(scaled_roi, cv::Mat(), 
                                                scale_keypoints, scale_descriptors);
                
                // Adjust keypoint coordinates for scale
                for (auto& kp : scale_keypoints) {
                    kp.pt *= 1.0f/scale;
                }
                
                // Add to main keypoint collection
                if (!scale_keypoints.empty() && !scale_descriptors.empty()) {
                    curr_keypoints.insert(curr_keypoints.end(), 
                                        scale_keypoints.begin(), 
                                        scale_keypoints.end());
                    if (curr_descriptors.empty()) {
                        curr_descriptors = scale_descriptors.clone();
                    } else {
                        cv::vconcat(curr_descriptors, scale_descriptors, curr_descriptors);
                    }
                }
            } catch (const cv::Exception& e) {
                std::cerr << "AKAZE detection failed at scale " << scale << ": " 
                         << e.what() << std::endl;
            }
            
            // Try FAST detection as backup
            try {
                std::vector<cv::KeyPoint> fast_keypoints;
                fast_detector->detect(scaled_roi, fast_keypoints);
                
                // Adjust keypoint coordinates for scale
                for (auto& kp : fast_keypoints) {
                    kp.pt *= 1.0f/scale;
                }
                
                // Create simple descriptors for FAST keypoints
                if (!fast_keypoints.empty()) {
                    cv::Mat fast_descriptors = computeSimpleDescriptors(scaled_roi, 
                                                                      fast_keypoints,
                                                                      scale);
                    curr_keypoints.insert(curr_keypoints.end(), 
                                        fast_keypoints.begin(), 
                                        fast_keypoints.end());
                    if (curr_descriptors.empty()) {
                        curr_descriptors = fast_descriptors.clone();
                    } else {
                        cv::vconcat(curr_descriptors, fast_descriptors, curr_descriptors);
                    }
                }
            } catch (const cv::Exception& e) {
                std::cerr << "FAST detection failed at scale " << scale << ": " 
                         << e.what() << std::endl;
            }
        }
        
        // Check if we have enough features
        if (curr_keypoints.empty() || template_keypoint.empty() || 
            curr_descriptors.empty() || template_descriptors.empty()) {
            return false;
        }
        
        // Limit maximum features for efficiency
        if (curr_keypoints.size() > MAX_FEATURES) {
            // Sort keypoints by response strength
            std::vector<size_t> indices(curr_keypoints.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                     [&](size_t a, size_t b) {
                         return curr_keypoints[a].response > curr_keypoints[b].response;
                     });
            
            // Keep only the strongest features
            std::vector<cv::KeyPoint> filtered_keypoints;
            cv::Mat filtered_descriptors;
            filtered_keypoints.reserve(MAX_FEATURES);
            filtered_descriptors.create(MAX_FEATURES, curr_descriptors.cols, 
                                     curr_descriptors.type());
            
            for (size_t i = 0; i < MAX_FEATURES; ++i) {
                filtered_keypoints.push_back(curr_keypoints[indices[i]]);
                curr_descriptors.row(indices[i]).copyTo(filtered_descriptors.row(i));
            }
            
            curr_keypoints = filtered_keypoints;
            curr_descriptors = filtered_descriptors;
        }
        
        // Match features with cross-checking
        std::vector<std::vector<cv::DMatch>> knn_matches;
        try {
            flann_matcher.knnMatch(template_descriptors, curr_descriptors, knn_matches, 2);
        } catch (const cv::Exception& e) {
            std::cerr << "Feature matching failed: " << e.what() << std::endl;
            return false;
        }
        
        // Filter matches using ratio test and collect good matches
        std::vector<cv::Point2f> template_points, curr_points;
        std::vector<cv::DMatch> good_matches;
        
        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() < 2) continue;
            
            if (match_pair[0].distance < match_ratio * match_pair[1].distance) {
                cv::Point2f template_pt = template_keypoint[match_pair[0].queryIdx].pt;
                cv::Point2f curr_pt = curr_keypoints[match_pair[0].trainIdx].pt;
                curr_pt += cv::Point2f(valid_search.x, valid_search.y);
                
                template_points.push_back(template_pt);
                curr_points.push_back(curr_pt);
                good_matches.push_back(match_pair[0]);
            }
        }
        
        // Check if we have enough matches
        if (template_points.size() < min_matches) {
            return false;
        }
        
        // Calculate homography with RANSAC
        std::vector<uchar> inlier_mask;
        cv::Mat H = cv::findHomography(template_points, curr_points, 
                                     cv::RANSAC, ransac_threshold, inlier_mask);
        
        if (H.empty()) {
            return false;
        }
        
        // Transform original template corners
        std::vector<cv::Point2f> template_corners = {
            cv::Point2f(0, 0),
            cv::Point2f(original_size.width, 0),
            cv::Point2f(original_size.width, original_size.height),
            cv::Point2f(0, original_size.height)
        };
        
        std::vector<cv::Point2f> transformed_corners;
        cv::perspectiveTransform(template_corners, transformed_corners, H);
        
        // Calculate recovered box
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        
        for (const auto& pt : transformed_corners) {
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        
        // Validate recovered box
        recovered_bbox = cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y);
        
        // Calculate confidence metrics
        double area_ratio = recovered_bbox.area() / original_roi_area;
        double aspect_ratio = recovered_bbox.width / recovered_bbox.height;
        double aspect_change = std::abs(aspect_ratio - original_aspect_ratio);
        
        // Validate recovery using multiple criteria
        if (area_ratio < 0.3 || area_ratio > 2.5 || aspect_change > 0.5) {
            return false;
        }
        
        // Calculate confidence using multiple metrics
        int inlier_count = cv::countNonZero(inlier_mask);
        double inlier_ratio = static_cast<double>(inlier_count) / inlier_mask.size();
        double area_confidence = std::exp(-std::abs(1.0 - area_ratio));
        double aspect_confidence = std::exp(-aspect_change);
        
        double overall_confidence = (inlier_ratio * 0.4 + 
                                  area_confidence * 0.3 + 
                                  aspect_confidence * 0.3);
        
        return overall_confidence > min_confidence;
        
    } catch (const cv::Exception& e) {
        std::cerr << "Recovery attempt failed: " << e.what() << std::endl;
        return false;
    }
}

private:
    cv::Mat computeSimpleDescriptors(const cv::Mat& img, 
                                   const std::vector<cv::KeyPoint>& keypoints,
                                   float scale) {
        cv::Mat descriptors(keypoints.size(), 64, CV_32F);
        
        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::KeyPoint kp = keypoints[i];
            kp.pt *= scale; // Scale the keypoint location
            
            // Check boundaries
            if (kp.pt.x < 2 || kp.pt.y < 2 || 
                kp.pt.x >= img.cols-2 || kp.pt.y >= img.rows-2) {
                descriptors.row(i) = cv::Scalar(0);
                continue;
            }
            
            // Compute simple gradient-based descriptor
            float* desc = descriptors.ptr<float>(i);
            int idx = 0;
            
            // Sample 8x8 grid around keypoint
            for (int dy = -2; dy <= 2; dy += 2) {
                for (int dx = -2; dx <= 2; dx += 2) {
                    int x = static_cast<int>(kp.pt.x + dx);
                    int y = static_cast<int>(kp.pt.y + dy);
                    
                    // Compute gradients
                    float dx_val = static_cast<float>(
                        img.at<uchar>(y, x+1) - img.at<uchar>(y, x-1)) / 2.0f;
                    float dy_val = static_cast<float>(
                        img.at<uchar>(y+1, x) - img.at<uchar>(y-1, x)) / 2.0f;
                    
                    // Store gradient information
                    desc[idx++] = dx_val;
                    desc[idx++] = dy_val;
                    desc[idx++] = std::sqrt(dx_val*dx_val + dy_val*dy_val);
                    desc[idx++] = std::atan2(dy_val, dx_val);
                }
            }
        }
        
        // Normalize descriptors
        for (int i = 0; i < descriptors.rows; i++) {
            cv::normalize(descriptors.row(i), descriptors.row(i));
        }
        
        return descriptors;
    }
    
    TrackingResult track(const cv::Mat& prev_frame, const cv::Mat& curr_frame,
                        const cv::Rect2d& prev_bbox, bool is_first_frame = false) {
        auto start_time = std::chrono::steady_clock::now();
        TrackingResult result(prev_bbox);

        try {
            if (prev_frame.empty() || curr_frame.empty()) {
                return result;
            }

            // Store template on first frame
            if (is_first_frame) {
                storeTemplate(prev_frame, prev_bbox);
                resetMotionModel();
                return result;
            }

            // Predict using motion model
            predictMotion(result);

            // Perform feature tracking
            result = trackFeatures(prev_frame, curr_frame, prev_bbox, result);

            // Update motion model with new tracking result
            updateMotionModel(result);

            // Handle occlusion if needed
            if (result.is_occluded) {
                handleOcclusion(curr_frame, result);
            }

            // Update metrics
            updateMetrics(start_time, result);

            return result;

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in tracking: " << e.what() << std::endl;
            result.is_occluded = true;
            return result;
        }
    }

    void reset() {
        resetMotionModel();
        feature_histories.clear();
        recovery_count = 0;
        recovery_frames = 0;
        has_last_motion = false;
        last_velocity = cv::Point2f(0, 0);
        template_keypoint.clear();
        template_descriptors.release();
        original_roi.release();
        metrics.reset();
    }

    // Metric accessors
    double getAverageProcessingTime() const { return metrics.avg_processing_time; }
    int getSuccessfulTracks() const { return metrics.successful_tracks; }
    int getFailedTracks() const { return metrics.failed_tracks; }
    int getRecoveries() const { return metrics.recoveries; }

private:
    void initializeDetectors() {
        // Initialize AKAZE detector for accurate feature detection
        feature_detector = cv::AKAZE::create(
            cv::AKAZE::DESCRIPTOR_MLDB,
            0,  // descriptor size
            8,  // descriptor channels (reduced from 8)
            0.0004f,  // threshold (increased from 0.0002f)
            8,  // octaves (reduced from 8)
            16,  // octave layers
            cv::KAZE::DIFF_PM_G2  // Changed from G2 for speed
        );

        // Initialize FAST detector for quick feature detection
        fast_detector = cv::FastFeatureDetector::create(
            FAST_THRESHOLD,
            true,  // nonmaxSuppression
            cv::FastFeatureDetector::TYPE_9_16
        );

        // Initialize FLANN matcher with optimized parameters
        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(16);
        cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(128);
        flann_matcher = cv::FlannBasedMatcher(indexParams, searchParams);
    }

    void initializeKalmanFilter() {
        // Increase state vector size to include acceleration
        motion_kalman.init(6, 4, 0);  // State: [x, y, dx, dy, ax, ay], Measurement: [x, y, dx, dy]
        
        // Enhanced state transition matrix incorporating acceleration
        motion_kalman.transitionMatrix = (cv::Mat_<float>(6, 6) <<
            1, 0, 1, 0, 0.5, 0,    // x = x + dx + 0.5ax
            0, 1, 0, 1, 0, 0.5,    // y = y + dy + 0.5ay
            0, 0, 1, 0, 1, 0,      // dx = dx + ax
            0, 0, 0, 1, 0, 1,      // dy = dy + ay
            0, 0, 0, 0, 1, 0,      // ax = ax
            0, 0, 0, 0, 0, 1);     // ay = ay

        // Enhanced measurement matrix to include velocity measurements
        motion_kalman.measurementMatrix = (cv::Mat_<float>(4, 6) <<
            1, 0, 0, 0, 0, 0,      // measure x
            0, 1, 0, 0, 0, 0,      // measure y
            0, 0, 1, 0, 0, 0,      // measure dx
            0, 0, 0, 1, 0, 0);     // measure dy

        // Fine-tuned noise parameters
        cv::setIdentity(motion_kalman.processNoiseCov, cv::Scalar::all(1e-5));
        motion_kalman.processNoiseCov.at<float>(2,2) = 1e-3;  // Velocity noise
        motion_kalman.processNoiseCov.at<float>(3,3) = 1e-3;
        motion_kalman.processNoiseCov.at<float>(4,4) = 1e-2;  // Acceleration noise
        motion_kalman.processNoiseCov.at<float>(5,5) = 1e-2;

        cv::setIdentity(motion_kalman.measurementNoiseCov, cv::Scalar::all(1e-2));
        motion_kalman.measurementNoiseCov.at<float>(2,2) = 1e-1;  // Velocity measurement noise
        motion_kalman.measurementNoiseCov.at<float>(3,3) = 1e-1;

        cv::setIdentity(motion_kalman.errorCovPost, cv::Scalar::all(0.1));
    }

    // Add adaptive noise adjustment
    void updateKalmanNoise(const std::vector<cv::Point2f>& trajectory, 
                        const cv::Point2f& current_velocity) {
        if (trajectory.size() < 3) return;

        // Calculate trajectory smoothness
        float trajectory_variance = 0;
        for (size_t i = 2; i < trajectory.size(); i++) {
            cv::Point2f expected = trajectory[i-1] * 2 - trajectory[i-2];
            cv::Point2f error = trajectory[i] - expected;
            trajectory_variance += cv::norm(error);
        }
        trajectory_variance /= (trajectory.size() - 2);

        // Adjust process noise based on trajectory smoothness
        float noise_scale = std::min(1.0f, trajectory_variance / 10.0f);
        motion_kalman.processNoiseCov.at<float>(0,0) = 1e-5 * (1 + noise_scale);
        motion_kalman.processNoiseCov.at<float>(1,1) = 1e-5 * (1 + noise_scale);
        
        // Adjust measurement noise based on velocity magnitude
        float velocity_magnitude = cv::norm(current_velocity);
        float velocity_noise_scale = std::min(1.0f, velocity_magnitude / 50.0f);
        motion_kalman.measurementNoiseCov.at<float>(2,2) = 1e-1 * (1 + velocity_noise_scale);
        motion_kalman.measurementNoiseCov.at<float>(3,3) = 1e-1 * (1 + velocity_noise_scale);
    }

    // Enhanced prediction method
    cv::Rect2d predictNextPosition(const cv::Rect2d& current_bbox, 
                                const std::vector<cv::Point2f>& trajectory,
                                double dt) {
        // Update noise parameters based on recent motion
        if (!trajectory.empty()) {
            cv::Point2f current_velocity = trajectory.back() - trajectory.front();
            current_velocity *= 1.0 / trajectory.size();
            updateKalmanNoise(trajectory, current_velocity);
        }

        // Predict next state
        cv::Mat prediction = motion_kalman.predict();
        
        // Extract predicted position and velocity
        float pred_x = prediction.at<float>(0);
        float pred_y = prediction.at<float>(1);
        float pred_dx = prediction.at<float>(2);
        float pred_dy = prediction.at<float>(3);
        float pred_ax = prediction.at<float>(4);
        float pred_ay = prediction.at<float>(5);

        // Calculate confidence in prediction
        float prediction_uncertainty = 
            std::sqrt(motion_kalman.errorCovPre.at<float>(0,0) + 
                    motion_kalman.errorCovPre.at<float>(1,1));

        // Adjust prediction based on uncertainty
        if (prediction_uncertainty > 10.0) {
            // Fall back to simple linear prediction if uncertainty is too high
            pred_x = current_bbox.x + pred_dx * dt;
            pred_y = current_bbox.y + pred_dy * dt;
        } else {
            // Use full acceleration-based prediction
            pred_x += 0.5f * pred_ax * dt * dt;
            pred_y += 0.5f * pred_ay * dt * dt;
        }

        return cv::Rect2d(pred_x, pred_y, current_bbox.width, current_bbox.height);
    }

    // Enhanced update method
    void updateKalmanFilter(const cv::Point2f& measured_pos, 
                        const cv::Point2f& measured_velocity,
                        double confidence) {
        cv::Mat measurement = (cv::Mat_<float>(4,1) <<
            measured_pos.x,
            measured_pos.y,
            measured_velocity.x,
            measured_velocity.y);

        // Weight measurements based on confidence
        motion_kalman.measurementNoiseCov *= (1.0 / std::max(0.1, confidence));
        
        // Update filter
        cv::Mat estimated = motion_kalman.correct(measurement);
        
        // Update state uncertainties based on innovation
        cv::Mat innovation = measurement - motion_kalman.measurementMatrix * motion_kalman.statePost;
        float innovation_magnitude = cv::norm(innovation);
        
        // Adapt process noise if innovations are large
        if (innovation_magnitude > 5.0) {
            motion_kalman.processNoiseCov *= 1.5;
        } else {
            motion_kalman.processNoiseCov *= 0.95;
        }
    }

    void predictMotion(TrackingResult& result) {
        if (!kalman_initialized) return;

        cv::Mat prediction = motion_kalman.predict();
        
        // Update predicted position
        result.predicted_bbox.x += prediction.at<float>(2);  // dx
        result.predicted_bbox.y += prediction.at<float>(3);  // dy
        
        // Store predicted velocity
        result.velocity = cv::Point2f(prediction.at<float>(2), prediction.at<float>(3));
    }

    TrackingResult trackFeatures(const cv::Mat& prev_frame, const cv::Mat& curr_frame,
                               const cv::Rect2d& prev_bbox, TrackingResult& result) {
        cv::Mat prev_gray, curr_gray;
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

        // Detect features using both FAST and AKAZE
        std::vector<cv::KeyPoint> prev_keypoints_fast, prev_keypoints_akaze;
        cv::Mat prev_descriptors_akaze;

        // FAST detection for quick initial points
        fast_detector->detect(prev_gray(prev_bbox), prev_keypoints_fast);

        // AKAZE detection for more robust features
        feature_detector->detectAndCompute(prev_gray(prev_bbox), cv::Mat(), 
                                         prev_keypoints_akaze, prev_descriptors_akaze);

        // Combine keypoints (prioritize AKAZE features)
        std::vector<cv::KeyPoint> prev_keypoints = prev_keypoints_akaze;
        if (prev_keypoints.size() < MAX_FEATURES) {
            prev_keypoints.insert(prev_keypoints.end(), 
                                prev_keypoints_fast.begin(), 
                                prev_keypoints_fast.end());
        }

        // Limit total number of features
        if (prev_keypoints.size() > MAX_FEATURES) {
            prev_keypoints.resize(MAX_FEATURES);
        }

        // Define search region
        cv::Rect search_region = calculateSearchRegion(prev_bbox, curr_frame.size());

        // Match features
        std::vector<cv::KeyPoint> curr_keypoints;
        cv::Mat curr_descriptors;
        feature_detector->detectAndCompute(curr_gray(search_region), cv::Mat(), 
                                         curr_keypoints, curr_descriptors);

        std::vector<std::vector<cv::DMatch>> knn_matches;
        flann_matcher.knnMatch(prev_descriptors_akaze, curr_descriptors, knn_matches, 2);

        // Filter and process matches
        std::vector<cv::Point2f> prev_points, curr_points;
        std::vector<cv::DMatch> good_matches;

        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() < 2) continue;
            
            if (match_pair[0].distance < match_ratio * match_pair[1].distance) {
                good_matches.push_back(match_pair[0]);
                
                cv::Point2f prev_pt = prev_keypoints[match_pair[0].queryIdx].pt;
                prev_pt += cv::Point2f(prev_bbox.x, prev_bbox.y);
                prev_points.push_back(prev_pt);
                
                cv::Point2f curr_pt = curr_keypoints[match_pair[0].trainIdx].pt;
                curr_pt += cv::Point2f(search_region.x, search_region.y);
                curr_points.push_back(curr_pt);
            }
        }

        // Update result based on matches
        if (good_matches.size() >= min_matches) {
            updateTrackingResult(prev_points, curr_points, result);
        } else {
            result.is_occluded = true;
        }

        return result;
    }

    void updateTrackingResult(const std::vector<cv::Point2f>& prev_points,
                            const std::vector<cv::Point2f>& curr_points,
                            TrackingResult& result) {
        // Calculate homography
        std::vector<uchar> inlier_mask;
        cv::Mat H = cv::findHomography(prev_points, curr_points, cv::RANSAC, 
                                     ransac_threshold, inlier_mask);

        if (H.empty()) {
            result.is_occluded = true;
            return;
        }

        // Transform bounding box
        std::vector<cv::Point2f> bbox_corners = {
            cv::Point2f(result.predicted_bbox.x, result.predicted_bbox.y),
            cv::Point2f(result.predicted_bbox.x + result.predicted_bbox.width, result.predicted_bbox.y),
            cv::Point2f(result.predicted_bbox.x + result.predicted_bbox.width, 
                       result.predicted_bbox.y + result.predicted_bbox.height),
            cv::Point2f(result.predicted_bbox.x, result.predicted_bbox.y + result.predicted_bbox.height)
        };

        std::vector<cv::Point2f> transformed_corners;
        cv::perspectiveTransform(bbox_corners, transformed_corners, H);

        // Calculate new bounding box
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();

        for (const auto& pt : transformed_corners) {
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }

        // Update result
        result.updated_bbox = cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y);
        result.success = true;
        result.matched_points = curr_points;
        
        // Calculate confidence
        double inlier_ratio = static_cast<double>(cv::countNonZero(inlier_mask)) / 
                             inlier_mask.size();
        result.confidence = inlier_ratio;
    }

    void handleOcclusion(const cv::Mat& curr_frame, TrackingResult& result) {
        recovery_frames++;
        
        if (recovery_frames >= recovery_threshold) {
            cv::Rect search_area = calculateSearchRegion(result.predicted_bbox, curr_frame.size());
            cv::Rect2d recovered_bbox;
            
            if (attemptRecovery(curr_frame, search_area, recovered_bbox)) {
                result.success = true;
                result.updated_bbox = recovered_bbox;
                result.recovered = true;
                result.is_occluded = false;
                recovery_frames = 0;
                metrics.recoveries++;
                
                // Update template after successful recovery
                storeTemplate(curr_frame, recovered_bbox);
            }
        }
    }


    void resetMotionModel() {
        motion_model.reset();
        if (kalman_initialized) {
            motion_kalman.init(4, 2, 0);
        }
        has_last_motion = false;
        last_velocity = cv::Point2f(0, 0);
    }

    void updateMotionModel(const TrackingResult& result) {
        if (!result.success) return;

        // Calculate current center
        cv::Point2f curr_center(
            result.updated_bbox.x + result.updated_bbox.width/2,
            result.updated_bbox.y + result.updated_bbox.height/2
        );

        auto current_time = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(
            current_time - motion_model.last_update).count();

        if (has_last_motion && dt > 0) {
            // Update velocity
            cv::Point2f new_velocity = (curr_center - last_valid_center) / dt;
            
            // Update acceleration
            motion_model.acceleration = (new_velocity - motion_model.velocity) / dt;
            
            // Smooth velocity update
            motion_model.velocity = velocity_weight * motion_model.velocity + 
                                  (1 - velocity_weight) * new_velocity;

            // Update Kalman filter
            if (kalman_initialized) {
                cv::Mat measurement = (cv::Mat_<float>(2,1) << curr_center.x, curr_center.y);
                motion_kalman.correct(measurement);
            }

            // Update confidence based on motion consistency
            double velocity_change = cv::norm(new_velocity - motion_model.velocity);
            double acc_magnitude = cv::norm(motion_model.acceleration);
            motion_model.confidence = 1.0 / (1.0 + velocity_change + 0.5 * acc_magnitude);
        } else {
            motion_model.velocity = cv::Point2f(0, 0);
            motion_model.acceleration = cv::Point2f(0, 0);
            motion_model.confidence = 1.0;
        }

        last_valid_center = curr_center;
        has_last_motion = true;
        motion_model.last_update = current_time;
    }

    cv::Rect calculateSearchRegion(const cv::Rect2d& bbox, const cv::Size& frame_size) {
        cv::Rect search_region = bbox;
        int margin_x = search_region.width * search_expansion;
        int margin_y = search_region.height * search_expansion;
        
        search_region.x = std::max(0, search_region.x - margin_x);
        search_region.y = std::max(0, search_region.y - margin_y);
        search_region.width = std::min(frame_size.width - search_region.x, 
                                     search_region.width + 2 * margin_x);
        search_region.height = std::min(frame_size.height - search_region.y, 
                                      search_region.height + 2 * margin_y);
        
        return validateSearchArea(search_region, frame_size);
    }

    cv::Rect validateSearchArea(const cv::Rect& area, const cv::Size& frame_size) {
        cv::Rect valid = area;
        valid.x = std::max(0, std::min(valid.x, frame_size.width - 1));
        valid.y = std::max(0, std::min(valid.y, frame_size.height - 1));
        valid.width = std::min(valid.width, frame_size.width - valid.x);
        valid.height = std::min(valid.height, frame_size.height - valid.y);
        return valid;
    }

    void updateMetrics(const std::chrono::steady_clock::time_point& start_time,
                      const TrackingResult& result) {
        auto end_time = std::chrono::steady_clock::now();
        double processing_time = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        // Update running average of processing time
        metrics.avg_processing_time = 0.95 * metrics.avg_processing_time + 
                                    0.05 * processing_time;
        
        // Update track success/failure counts
        if (result.success) {
            metrics.successful_tracks++;
        } else {
            metrics.failed_tracks++;
        }
    }

public:
    void drawDebugInfo(cv::Mat& frame, const TrackingResult& result) {
        if (result.success) {
            // Draw bounding box
            cv::Scalar color = result.is_occluded ? cv::Scalar(0, 0, 255) : 
                             (result.recovered ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0));
            cv::rectangle(frame, result.updated_bbox, color, 2);

            // Draw predicted box if in recovery
            if (result.is_occluded) {
                cv::rectangle(frame, result.predicted_bbox, cv::Scalar(0, 255, 255), 1);
            }

            // Draw matched points and motion vectors
            for (const auto& pt : result.matched_points) {
                cv::circle(frame, pt, 2, color, -1);
            }
            // Draw velocity vector
            cv::Point2f center(result.updated_bbox.x + result.updated_bbox.width/2,
                             result.updated_bbox.y + result.updated_bbox.height/2);
            cv::arrowedLine(frame, center, 
                           center + result.velocity * 10,
                           cv::Scalar(0, 255, 0), 2);

            // Draw confidence and state information
            std::string info = cv::format("Conf: %.2f", result.confidence);
            cv::putText(frame, info, 
                       cv::Point(result.updated_bbox.x, result.updated_bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

            if (result.recovered) {
                cv::putText(frame, "Recovered", 
                           cv::Point(result.updated_bbox.x, result.updated_bbox.y - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }
    }
};


class ObjectTracker {
private:
    // Tracking state and management
    struct TrackerInfo {
        int id;
        cv::Ptr<cv::Tracker> tracker;
        cv::KalmanFilter kalman;
        cv::Mat hist;
        cv::Rect last_box;
        std::deque<cv::Point2f> trajectory;
        std::deque<cv::Point2f> velocity_history;
        cv::Point2f smoothed_velocity;
        int lost_count;
        bool is_active;
        cv::Point2f velocity;
        cv::Point2f acceleration;
        double confidence;
        std::chrono::steady_clock::time_point last_update;

        // New appearance model members
        cv::Mat template_features;
        cv::Mat appearance_model;
        std::deque<cv::Mat> appearance_history;
        
        // Change const members to static constexpr
        static constexpr size_t appearance_history_size = 10;
        static constexpr double appearance_threshold = 0.35;
        static constexpr int max_recovery_attempts = 30;
        
        int recovery_attempts = 0;

        // Constructor
        TrackerInfo(int id, cv::Ptr<cv::Tracker> tracker, const cv::KalmanFilter& kf, 
                   const cv::Mat& hist, const cv::Rect& box) : 
            id(id), tracker(tracker), kalman(kf), hist(hist.clone()), last_box(box), 
            lost_count(0), is_active(true), velocity(0,0), acceleration(0,0), 
            confidence(1.0), smoothed_velocity(0,0) {
            
            trajectory.push_back(cv::Point2f(
                box.x + box.width/2.0f,
                box.y + box.height/2.0f
            ));
            last_update = std::chrono::steady_clock::now();
        }

        // Explicitly delete copy constructor and assignment operator
        TrackerInfo(const TrackerInfo&) = delete;
        TrackerInfo& operator=(const TrackerInfo&) = delete;

        // Implement move constructor
        TrackerInfo(TrackerInfo&& other) noexcept :
            id(other.id),
            tracker(std::move(other.tracker)),
            kalman(std::move(other.kalman)),
            hist(std::move(other.hist)),
            last_box(other.last_box),
            trajectory(std::move(other.trajectory)),
            velocity_history(std::move(other.velocity_history)),
            smoothed_velocity(other.smoothed_velocity),
            lost_count(other.lost_count),
            is_active(other.is_active),
            velocity(other.velocity),
            acceleration(other.acceleration),
            confidence(other.confidence),
            last_update(other.last_update),
            template_features(std::move(other.template_features)),
            appearance_model(std::move(other.appearance_model)),
            appearance_history(std::move(other.appearance_history)),
            recovery_attempts(other.recovery_attempts) {
        }

        // Implement move assignment operator
        TrackerInfo& operator=(TrackerInfo&& other) noexcept {
            if (this != &other) {
                id = other.id;
                tracker = std::move(other.tracker);
                kalman = std::move(other.kalman);
                hist = std::move(other.hist);
                last_box = other.last_box;
                trajectory = std::move(other.trajectory);
                velocity_history = std::move(other.velocity_history);
                smoothed_velocity = other.smoothed_velocity;
                lost_count = other.lost_count;
                is_active = other.is_active;
                velocity = other.velocity;
                acceleration = other.acceleration;
                confidence = other.confidence;
                last_update = other.last_update;
                template_features = std::move(other.template_features);
                appearance_model = std::move(other.appearance_model);
                appearance_history = std::move(other.appearance_history);
                recovery_attempts = other.recovery_attempts;
            }
            return *this;
        }

        // Destructor
        ~TrackerInfo() {
            template_features.release();
            appearance_model.release();
            for(auto& hist : appearance_history) {
                hist.release();
            }
            appearance_history.clear();
        }
    };

    // Update the cleanupTrackers method
    void cleanupTrackers() {
        auto new_end = std::remove_if(
            tracked_objects.begin(),
            tracked_objects.end(),
            [](const TrackerInfo& obj) { 
                return !obj.is_active; 
            }
        );
        tracked_objects.erase(new_end, tracked_objects.end());
    }
    
    std::vector<TrackerInfo> tracked_objects;
    std::map<int, cv::Mat> object_gallery;
    std::vector<cv::Rect> exit_history;
    int next_id;
    
    // Feature tracking and matching
    std::unique_ptr<FeatureTracker> feature_tracker;
    cv::Ptr<cv::AKAZE> feature_detector;
    cv::FlannBasedMatcher flann_matcher;
    
    // Multi-threading support
    const int NUM_THREADS = 4;
    std::vector<std::thread> worker_threads;
    std::queue<TrackerInfo*> processing_queue;
    std::mutex tracker_mutex;
    std::condition_variable processing_condition;
    bool is_processing = true;
    
    // Tracking parameters
    const size_t MAX_TRAJECTORY_LENGTH = 120;
    const double LOST_THRESHOLD = 0.02;
    const int MAX_LOST_FRAMES = 200;
    const float OVERLAP_THRESHOLD = 0.10f;
    const double SIMILARITY_THRESHOLD = 0.01;
    const int MAX_ACTIVE_TRACKERS = 10;
    const double velocity_weight = 0.65;
    const double VELOCITY_SMOOTHING = 0.99;  // High smoothing factor
    const size_t VELOCITY_HISTORY_SIZE = 30; // Keep 10 frames of velocity history
    const double MAX_VELOCITY_CHANGE = 8.0; // Maximum allowed velocity change per frame
    const double APPEARANCE_MEMORY_FACTOR = 0.2;
    const int TEMPLATE_UPDATE_INTERVAL = 2;
    const double RECOVERY_CONFIDENCE_THRESHOLD = 0.75;
    
    
    // Performance metrics
    struct Metrics {
        double avg_processing_time = 0.0;
        int active_trackers = 0;
        int total_tracks = 0;
        int successful_tracks = 0;
        int lost_tracks = 0;
        std::chrono::steady_clock::time_point last_update;  

        
        void reset() {
            avg_processing_time = 0.0;
            active_trackers = 0;
            total_tracks = 0;
            successful_tracks = 0;
            lost_tracks = 0;
        }
    } metrics;

    

public:
    ObjectTracker() : next_id(0) {
        initializeDetectors();
        initializeThreadPool();
        feature_tracker = std::make_unique<FeatureTracker>();
    }
    
    ~ObjectTracker() {
        stopProcessing();
    }   
    
    bool add_new_object(const cv::Mat& frame, const cv::Rect& roi) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        
        try {
            if (frame.empty() || roi.area() <= 0) {
                std::cerr << "Invalid frame or ROI" << std::endl;
                return false;
            }
            
            // Validate ROI
            cv::Rect adjusted_roi = validateBox(roi, frame.size());
            if (adjusted_roi.area() <= 0) {
                std::cerr << "Invalid ROI after adjustment" << std::endl;
                return false;
            }
            
            // Extract ROI and calculate initial features
            cv::Mat roi_img = frame(adjusted_roi);
            if (roi_img.empty()) {
                std::cerr << "Failed to extract ROI image" << std::endl;
                return false;
            }
            
            // Calculate initial features
            cv::Mat initial_features = extractFeatures(roi_img);
            if (initial_features.empty()) {
                std::cerr << "Failed to extract initial features" << std::endl;
                return false;
            }
            
            // Create and initialize tracker
            cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
            cv::Rect2d tracking_roi(adjusted_roi);
            try {
                // Initialize tracker without checking return value
                tracker->init(frame, tracking_roi);
            } catch (const cv::Exception& e) {
                std::cerr << "Failed to initialize tracker: " << e.what() << std::endl;
                return false;
            }
            
            // Initialize Kalman filter
            cv::KalmanFilter kf = initKalmanFilter();
            
            // Create new tracker info
            TrackerInfo new_tracker(next_id++, tracker, kf, initial_features, adjusted_roi);
            new_tracker.template_features = initial_features.clone();
            new_tracker.appearance_model = initial_features.clone();
            new_tracker.appearance_history.push_back(initial_features.clone());
            
            // Add to tracked objects
            tracked_objects.push_back(std::move(new_tracker));
            metrics.total_tracks++;
            
            std::cout << "Successfully added new object with ID: " << (next_id-1) << std::endl;
            return true;
            
        } catch (const cv::Exception& e) {
            std::cerr << "Error adding new object: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool update_tracker(cv::Mat& frame) {
        auto start_time = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(tracker_mutex);
        
        if (frame.empty() || tracked_objects.empty()) {
            return false;
        }

        bool overall_success = false;
        cv::Size frame_size = frame.size();
        
        // Handle occlusions first
        handleOcclusions(frame);
        
        // Update all active trackers
        for (auto& obj : tracked_objects) {
            if (!obj.is_active) continue;
            
            // Predict using Kalman filter
            cv::Mat prediction = obj.kalman.predict();
            
            // Update tracker
            cv::Rect new_box;
            bool success = obj.tracker->update(frame, new_box);
            
            if (success) {
                processSuccessfulTrack(obj, new_box, frame, prediction);
                overall_success = true;
            } else {
                processFailedTrack(obj, frame, prediction);
            }
        }
        
        // Clean up inactive trackers
        cleanupTrackers();
        
        // Update metrics
        updateMetrics(start_time);
        
        return overall_success;
    }
    
    void draw_tracked_objects(cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        
        for (const auto& obj : tracked_objects) {
            if (!obj.is_active) continue;

            // Colors for different states
            const cv::Scalar ACTIVE_COLOR(0, 255, 0);
            const cv::Scalar OCCLUDED_COLOR(0, 0, 255);
            const cv::Scalar PREDICTED_COLOR(255, 255, 0);
            const cv::Scalar TEXT_COLOR(255, 255, 255);

            // Draw bounding box
            cv::Scalar box_color = (obj.lost_count > 0) ? OCCLUDED_COLOR : ACTIVE_COLOR;
            cv::rectangle(frame, obj.last_box, box_color, 2);

            // Draw trajectory
            drawTrajectory(frame, obj);

            // Draw motion vector
            if (obj.trajectory.size() > 1) {
                cv::Point2f current = obj.trajectory.back();
                cv::arrowedLine(frame, current, 
                              current + obj.velocity * 10, 
                              PREDICTED_COLOR, 2);
            }

            // Draw text information
            drawTrackerInfo(frame, obj);
        }

        // Draw overall statistics
        drawStatistics(frame);
    }



    // Add these methods to the ObjectTracker class
    void updateAppearanceModel(TrackerInfo& obj, const cv::Mat& frame) {
        cv::Mat curr_features = extractFeatures(frame(obj.last_box));
        
        // Check similarity with template
        double similarity = compareFeatures(curr_features, obj.template_features);
        
        if (similarity < obj.appearance_threshold) {
            obj.lost_count++;
            return;
        }
        
        // Update appearance history
        obj.appearance_history.push_back(curr_features);
        if (obj.appearance_history.size() > obj.appearance_history_size) {
            obj.appearance_history.pop_front();
        }
        
        // Update appearance model with temporal consistency
        cv::Mat weighted_features = curr_features.clone();
        for (const auto& hist_features : obj.appearance_history) {
            weighted_features += APPEARANCE_MEMORY_FACTOR * hist_features;
        }
        weighted_features /= (1 + APPEARANCE_MEMORY_FACTOR * obj.appearance_history.size());
        
        obj.appearance_model = weighted_features.clone();
    }
    
    double compareFeatures(const cv::Mat& features1, const cv::Mat& features2) {
        return cv::compareHist(features1, features2, cv::HISTCMP_CORREL);
    }
    
    cv::Mat calculateNormalizedHistogram(const cv::Mat& roi) {
        try {
            if (roi.empty()) {
                return cv::Mat();
            }

            // Convert to HSV color space
            cv::Mat hsv;
            cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
            
            // Define histogram parameters - using 2D histogram (H and S channels only)
            int h_bins = 30;
            int s_bins = 32;
            int histSize[] = { h_bins, s_bins };
            
            // H varies from 0 to 179, S from 0 to 255
            float h_ranges[] = { 0, 180 };
            float s_ranges[] = { 0, 256 };
            const float* ranges[] = { h_ranges, s_ranges };
            
            // Use only the first two channels (Hue and Saturation)
            int channels[] = { 0, 1 };
            
            // Calculate 2D histogram
            cv::Mat hist;
            cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
            
            // Normalize histogram
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
            
            // Convert to proper type
            hist.convertTo(hist, CV_32F);
            
            // Reshape to 1D array for easier comparison
            hist = hist.reshape(1, 1);
            
            return hist;

        } catch (const cv::Exception& e) {
            std::cerr << "Error calculating histogram: " << e.what() << std::endl;
            return cv::Mat();
        }
    }

    double compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2) {
        try {
            if (hist1.empty() || hist2.empty()) {
                return 0.0;
            }

            // Make sure both histograms are CV_32F
            cv::Mat h1, h2;
            hist1.convertTo(h1, CV_32F);
            hist2.convertTo(h2, CV_32F);

            // Compare histograms using correlation method
            double correlation = cv::compareHist(h1, h2, cv::HISTCMP_CORREL);
            
            // Ensure result is in [0,1] range
            correlation = (correlation + 1.0) / 2.0;
            
            return correlation;

        } catch (const cv::Exception& e) {
            std::cerr << "Error comparing histograms: " << e.what() << std::endl;
            return 0.0;
        }
    }

    // Add these helper functions for improved appearance matching
    cv::Mat calculateHistogramWithMask(const cv::Mat& roi, const cv::Mat& mask) {
        try {
            cv::Mat hsv;
            cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

            // Calculate weighted histogram
            int h_bins = 30;
            int s_bins = 32;
            int histSize[] = { h_bins, s_bins };
            float h_ranges[] = { 0, 180 };
            float s_ranges[] = { 0, 256 };
            const float* ranges[] = { h_ranges, s_ranges };
            int channels[] = { 0, 1 };

            cv::Mat hist;
            cv::calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges);
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
            hist.convertTo(hist, CV_32F);
            return hist.reshape(1, 1);

        } catch (const cv::Exception& e) {
            std::cerr << "Error calculating masked histogram: " << e.what() << std::endl;
            return cv::Mat();
        }
    }

    double calculateAppearanceSimilarity(const TrackerInfo& obj, const cv::Mat& frame, 
                                    const cv::Rect& box) {
        try {
            // Create center-weighted mask
            cv::Mat mask(box.height, box.width, CV_8UC1);
            cv::Point2f center(box.width / 2.0f, box.height / 2.0f);
            
            for (int y = 0; y < box.height; y++) {
                for (int x = 0; x < box.width; x++) {
                    float dist = cv::norm(cv::Point2f(x, y) - center);
                    float weight = std::exp(-dist * dist / (2 * (box.width * box.width / 16.0f)));
                    mask.at<uchar>(y, x) = static_cast<uchar>(weight * 255);
                }
            }

            // Calculate histograms with center-weighting
            cv::Mat curr_hist = calculateHistogramWithMask(frame(box), mask);
            cv::Mat template_hist = calculateHistogramWithMask(frame(obj.last_box), mask);

            // Compare histograms
            double hist_sim = compareHistograms(curr_hist, template_hist);

            // Calculate edge similarity if possible
            double edge_sim = 1.0;
            try {
                cv::Mat curr_edges, template_edges;
                cv::Canny(frame(box), curr_edges, 100, 200);
                cv::Canny(frame(obj.last_box), template_edges, 100, 200);
                
                cv::resize(curr_edges, curr_edges, template_edges.size());
                edge_sim = 1.0 - cv::norm(curr_edges, template_edges, cv::NORM_L1) / 
                                (curr_edges.total() * 255);
            } catch (const cv::Exception& e) {
                // If edge detection fails, rely only on histogram
                std::cerr << "Edge detection failed: " << e.what() << std::endl;
            }

            // Combine similarities
            return 0.7 * hist_sim + 0.3 * edge_sim;

        } catch (const cv::Exception& e) {
            std::cerr << "Error calculating appearance similarity: " << e.what() << std::endl;
            return 0.0;
        }
    }

    // Update the extractFeatures method in ObjectTracker
    cv::Mat extractFeatures(const cv::Mat& roi) {
        if (roi.empty()) {
            std::cerr << "Empty ROI provided to extractFeatures" << std::endl;
            return cv::Mat();
        }
        
        try {
            return calculateNormalizedHistogram(roi);
        } catch (const cv::Exception& e) {
            std::cerr << "Error extracting features: " << e.what() << std::endl;
            return cv::Mat();
        }
    }

    // Update the processSuccessfulTrack method
    void processSuccessfulTrack(TrackerInfo& obj, const cv::Rect& new_box, 
                            const cv::Mat& frame, const cv::Mat& prediction) {
        try {
            // Validate new box
            cv::Rect validated_box = validateBox(new_box, frame.size());
            
            // Ensure box is within frame bounds
            if (validated_box.x < 0 || validated_box.y < 0 || 
                validated_box.x + validated_box.width > frame.cols ||
                validated_box.y + validated_box.height > frame.rows) {
                obj.lost_count++;
                return;
            }
            
            // Extract ROI and calculate features
            cv::Mat roi = frame(validated_box);
            if (roi.empty()) {
                std::cerr << "Empty ROI in processSuccessfulTrack" << std::endl;
                obj.lost_count++;
                return;
            }
            
            cv::Mat curr_features = extractFeatures(roi);
            if (curr_features.empty()) {
                std::cerr << "Failed to extract features" << std::endl;
                obj.lost_count++;
                return;
            }
            
            // Compare with template features
            double similarity = compareHistograms(curr_features, obj.template_features);
            
            if (similarity < TrackerInfo::appearance_threshold) {
                std::cout << "Low appearance similarity: " << similarity << std::endl;
                obj.lost_count++;
                return;
            }
            
            // Calculate current center
            cv::Point2f curr_center(
                validated_box.x + validated_box.width/2.0f,
                validated_box.y + validated_box.height/2.0f
            );
            
            // Update trajectory and motion model
            if (obj.trajectory.size() >= MAX_TRAJECTORY_LENGTH) {
                obj.trajectory.pop_front();
            }
            obj.trajectory.push_back(curr_center);
            
            updateMotionModel(obj, curr_center);
            
            // Update appearance history
            obj.appearance_history.push_back(curr_features);
            if (obj.appearance_history.size() > TrackerInfo::appearance_history_size) {
                obj.appearance_history.pop_front();
            }
            
            // Update Kalman filter
            cv::Mat measurement = (cv::Mat_<float>(4,1) << 
                validated_box.x, validated_box.y,
                validated_box.width, validated_box.height);
            obj.kalman.correct(measurement);
            
            // Update object state
            obj.last_box = validated_box;
            obj.lost_count = 0;
            metrics.successful_tracks++;
            
        } catch (const cv::Exception& e) {
            std::cerr << "Error in processSuccessfulTrack: " << e.what() << std::endl;
            obj.lost_count++;
        }
    }
    
    void clearAll() {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        tracked_objects.clear();
        object_gallery.clear();
        exit_history.clear();
        next_id = 0;
        metrics.reset();
    }
    
    // Accessors
    int getActiveTrackers() const {
        return metrics.active_trackers;
    }
    
    std::vector<cv::Rect> getExitLocations() const {
        return exit_history;
    }
    
    double getAverageProcessingTime() const {
        return metrics.avg_processing_time;
    }

private:
    void updateSingleTracker(TrackerInfo& obj) {
        // Basic implementation - you can enhance this
        cv::Mat frame;  // You'll need to pass the frame or maintain a reference
        cv::Rect new_box;
        if (obj.tracker->update(frame, new_box)) {
            obj.last_box = validateBox(new_box, frame.size());
            obj.lost_count = 0;
        } else {
            obj.lost_count++;
        }
    }

    void initializeDetectors() {
        feature_detector = cv::AKAZE::create(
            cv::AKAZE::DESCRIPTOR_MLDB,
            0, 8, 0.0002f, 8, 0,
            cv::KAZE::DIFF_PM_G2
        );
        
        cv::Ptr<cv::flann::IndexParams> indexParams = 
            cv::makePtr<cv::flann::KDTreeIndexParams>(5);
        cv::Ptr<cv::flann::SearchParams> searchParams = 
            cv::makePtr<cv::flann::SearchParams>(50);
        flann_matcher = cv::FlannBasedMatcher(indexParams, searchParams);
    }
    
    void initializeThreadPool() {
        for (int i = 0; i < NUM_THREADS; ++i) {
            worker_threads.emplace_back(&ObjectTracker::processTracker, this);
        }
    }
    
    void processTracker() {
        while (is_processing) {
            TrackerInfo* obj = nullptr;
            
            {
                std::unique_lock<std::mutex> lock(tracker_mutex);
                processing_condition.wait(lock, [this] {
                    return !processing_queue.empty() || !is_processing;
                });
                
                if (!is_processing) break;
                
                obj = processing_queue.front();
                processing_queue.pop();
            }
            
            if (obj) {
                updateSingleTracker(*obj);
            }
        }
    }
    
    void stopProcessing() {
        {
            std::lock_guard<std::mutex> lock(tracker_mutex);
            is_processing = false;
        }
        processing_condition.notify_all();
        
        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
    cv::KalmanFilter initKalmanFilter() {
        cv::KalmanFilter kf(6, 4, 0);
        
        kf.measurementMatrix = cv::Mat::zeros(4, 6, CV_32F);
        kf.measurementMatrix.at<float>(0,0) = 1.0f;
        kf.measurementMatrix.at<float>(1,1) = 1.0f;
        kf.measurementMatrix.at<float>(2,2) = 1.0f;
        kf.measurementMatrix.at<float>(3,3) = 1.0f;
        
        kf.transitionMatrix = cv::Mat::zeros(6, 6, CV_32F);
        kf.transitionMatrix.at<float>(0,0) = 1.0f;
        kf.transitionMatrix.at<float>(1,1) = 1.0f;
        kf.transitionMatrix.at<float>(2,2) = 1.0f;
        kf.transitionMatrix.at<float>(3,3) = 1.0f;
        kf.transitionMatrix.at<float>(4,4) = 1.0f;
        kf.transitionMatrix.at<float>(5,5) = 1.0f;
        kf.transitionMatrix.at<float>(0,4) = 0.02f;
        kf.transitionMatrix.at<float>(1,5) = 0.02f;
        
        kf.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.005f;
        kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.005f;
        kf.errorCovPost = cv::Mat::eye(6, 6, CV_32F) * 0.05f;
        
        return kf;
    }
    
    double original_roi_area = 0.0;
    double original_aspect_ratio = 1.0;
    cv::Size original_size;

    void handleOcclusions(const cv::Mat& frame) {
        std::vector<std::pair<size_t, size_t>> overlaps;
        
        // Detect overlapping trackers
        for (size_t i = 0; i < tracked_objects.size(); i++) {
            if (!tracked_objects[i].is_active) continue;
            
            for (size_t j = i + 1; j < tracked_objects.size(); j++) {
                if (!tracked_objects[j].is_active) continue;
                
                float iou = calculateIOU(tracked_objects[i].last_box, 
                                      tracked_objects[j].last_box);
                
                if (iou > OVERLAP_THRESHOLD) {
                    overlaps.push_back({i, j});
                }
            }
        }
        
        // Resolve occlusions
        for (const auto& [i, j] : overlaps) {
            resolveOcclusion(tracked_objects[i], tracked_objects[j], frame);
        }
    }
    
    void resolveOcclusion(TrackerInfo& obj1, TrackerInfo& obj2, const cv::Mat& frame) {
        // Calculate confidence based on multiple factors
        double conf1 = calculateTrackerConfidence(obj1);
        double conf2 = calculateTrackerConfidence(obj2);
        
        // Update tracker states based on confidence
        if (conf1 > conf2) {
            obj2.lost_count++;
            updateOccludedTracker(obj2, frame);
        } else {
            obj1.lost_count++;
            updateOccludedTracker(obj1, frame);
        }
    }
    
    double calculateTrackerConfidence(const TrackerInfo& obj) {
        // Combine multiple confidence factors
        double trajectory_conf = calculateTrajectoryConfidence(obj);
        double velocity_conf = calculateVelocityConfidence(obj);
        double size_conf = calculateSizeConfidence(obj);
        
        return 0.4 * trajectory_conf + 0.3 * velocity_conf + 0.3 * size_conf;
    }
    
    double calculateTrajectoryConfidence(const TrackerInfo& obj) {
        if (obj.trajectory.size() < 2) return 1.0;
        
        double consistency = 0.0;
        for (size_t i = 1; i < obj.trajectory.size(); ++i) {
            cv::Point2f diff = obj.trajectory[i] - obj.trajectory[i-1];
            consistency += std::exp(-cv::norm(diff) * 0.1);
        }
        
        return consistency / (obj.trajectory.size() - 1);
    }
    
    double calculateVelocityConfidence(const TrackerInfo& obj) {
        double velocity_magnitude = cv::norm(obj.velocity);
        double acc_magnitude = cv::norm(obj.acceleration);
        
        return std::exp(-(velocity_magnitude * 0.01 + acc_magnitude * 0.02));
    }
    
    double calculateSizeConfidence(const TrackerInfo& obj) {
        // Check if the object size is reasonable
        double aspect_ratio = static_cast<double>(obj.last_box.width) / 
                            obj.last_box.height;
        double size_ratio = static_cast<double>(obj.last_box.area()) / 
                           original_roi_area;
        
        // Penalize extreme aspect ratios and size changes
        double aspect_confidence = std::exp(-std::abs(aspect_ratio - original_aspect_ratio));
        double size_confidence = std::exp(-std::abs(size_ratio - 1.0));
        
        return 0.5 * aspect_confidence + 0.5 * size_confidence;
    }
    
    void updateOccludedTracker(TrackerInfo& obj, const cv::Mat& frame) {
        // Predict next position using Kalman filter
        cv::Mat prediction = obj.kalman.predict();
        
        // Update position based on prediction
        obj.last_box.x += prediction.at<float>(4);  // velocity x
        obj.last_box.y += prediction.at<float>(5);  // velocity y
        
        // Try to recover if object is lost for too long
        if (obj.lost_count > MAX_LOST_FRAMES) {
            attemptRecovery(obj, frame);
        }
    }
    
    void attemptRecovery(TrackerInfo& obj, const cv::Mat& frame) {
        cv::Rect search_area = calculateSearchArea(obj.last_box, frame.size());
        cv::Rect2d recovered_box;
        
        if (feature_tracker->attemptRecovery(frame, search_area, recovered_box)) {
            reinitializeTracker(obj, frame, recovered_box);
            obj.lost_count = 0;
            metrics.successful_tracks++;
        } else {
            obj.is_active = false;
            exit_history.push_back(obj.last_box);
            metrics.lost_tracks++;
        }
    }
    
    void reinitializeTracker(TrackerInfo& obj, const cv::Mat& frame, const cv::Rect2d& box) {
        obj.tracker = cv::TrackerCSRT::create();
        obj.tracker->init(frame, box);
        obj.last_box = box;
        obj.hist = calculateColorHistogram(frame(box));
        
        // Reset Kalman filter
        cv::Mat state(6, 1, CV_32F);
        state.at<float>(0) = box.x + box.width/2.0f;
        state.at<float>(1) = box.y + box.height/2.0f;
        state.at<float>(2) = box.width;
        state.at<float>(3) = box.height;
        state.at<float>(4) = obj.velocity.x;
        state.at<float>(5) = obj.velocity.y;
        obj.kalman.statePost = state;
    }
    
    
    void processFailedTrack(TrackerInfo& obj, const cv::Mat& frame, 
                           const cv::Mat& prediction) {
        obj.lost_count++;
        
        if (obj.lost_count > MAX_LOST_FRAMES) {
            attemptRecovery(obj, frame);
        } else {
            // Use Kalman prediction for temporary tracking
            cv::Rect predicted_box(
                prediction.at<float>(0) - obj.last_box.width/2,
                prediction.at<float>(1) - obj.last_box.height/2,
                obj.last_box.width,
                obj.last_box.height
            );
            obj.last_box = validateBox(predicted_box, frame.size());
        }
    }
    
    void updateMotionModel(TrackerInfo& obj, const cv::Point2f& curr_center) {
        auto current_time = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(
            current_time - obj.last_update).count();
        
        if (dt > 0) {
            // Calculate center of last box
            cv::Point2f last_center(
                obj.last_box.x + obj.last_box.width/2.0f,
                obj.last_box.y + obj.last_box.height/2.0f
            );

            // Calculate new velocity
            cv::Point2f new_velocity = (curr_center - last_center) / dt;

            // Limit maximum velocity change
            cv::Point2f velocity_change = new_velocity - obj.velocity;
            double change_magnitude = cv::norm(velocity_change);
            if (change_magnitude > MAX_VELOCITY_CHANGE) {
                velocity_change *= MAX_VELOCITY_CHANGE / change_magnitude;
                new_velocity = obj.velocity + velocity_change;
            }

            // Update velocity history
            obj.velocity_history.push_back(new_velocity);
            if (obj.velocity_history.size() > VELOCITY_HISTORY_SIZE) {
                obj.velocity_history.pop_front();
            }

            // Calculate median velocity from history
            if (!obj.velocity_history.empty()) {
                std::vector<float> x_velocities, y_velocities;
                for (const auto& v : obj.velocity_history) {
                    x_velocities.push_back(v.x);
                    y_velocities.push_back(v.y);
                }
                std::sort(x_velocities.begin(), x_velocities.end());
                std::sort(y_velocities.begin(), y_velocities.end());
                
                size_t mid = x_velocities.size() / 2;
                cv::Point2f median_velocity(x_velocities[mid], y_velocities[mid]);

                // Update smoothed velocity using median
                obj.smoothed_velocity = obj.smoothed_velocity * VELOCITY_SMOOTHING + 
                                      median_velocity * (1 - VELOCITY_SMOOTHING);
            }

            // Update final velocity and acceleration
            obj.acceleration = (new_velocity - obj.velocity) / dt;
            obj.velocity = obj.smoothed_velocity; // Use smoothed velocity
        }
        
        obj.last_update = current_time;
    }
    
    cv::Rect calculateSearchArea(const cv::Rect& box, const cv::Size& frame_size) {
        int margin_x = box.width;
        int margin_y = box.height;
        
        cv::Rect search_area(
            box.x - margin_x,
            box.y - margin_y,
            box.width + 2 * margin_x,
            box.height + 2 * margin_y
        );
        
        return validateBox(search_area, frame_size);
    }
    
    cv::Mat calculateColorHistogram(const cv::Mat& roi) {
        cv::Mat hsv_roi;
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
        
        int h_bins = 90;
        int s_bins = 80;
        int histSize[] = { h_bins, s_bins };
        float h_ranges[] = { 0, 180 };
        float s_ranges[] = { 0, 256 };
        const float* ranges[] = { h_ranges, s_ranges };
        int channels[] = { 0, 1 };
        
        cv::Mat hist;
        cv::calcHist(&hsv_roi, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
        
        return hist;
    }
    
    float calculateIOU(const cv::Rect& box1, const cv::Rect& box2) {
        cv::Rect intersection = box1 & box2;
        float inter_area = intersection.area();
        float union_area = box1.area() + box2.area() - inter_area;
        return union_area > 0 ? inter_area / union_area : 0;
    }
    
    // void cleanupTrackers() {
    //     tracked_objects.erase(
    //         std::remove_if(
    //             tracked_objects.begin(),
    //             tracked_objects.end(),
    //             [](const TrackerInfo& obj) { return !obj.is_active; }
    //         ),
    //         tracked_objects.end()
    //     );
    // }
    
    void drawTrajectory(cv::Mat& frame, const TrackerInfo& obj) {
        if (obj.trajectory.size() < 2) return;
        
        for (size_t i = 1; i < obj.trajectory.size(); i++) {
            // Calculate alpha for fading effect
            float alpha = static_cast<float>(i) / obj.trajectory.size();
            cv::Scalar color(0, 255 * alpha, 0);
            
            cv::line(frame, obj.trajectory[i-1], obj.trajectory[i], color, 2);
            
            // Draw points every 5 positions
            if (i % 5 == 0) {
                cv::circle(frame, obj.trajectory[i], 2, color, -1);
            }
        }
    }
    
    void drawTrackerInfo(cv::Mat& frame, const TrackerInfo& obj) {
        // Modified version of drawTrackerInfo that uses smoothed velocity for arrow
        if (!obj.is_active) return;

        // Draw bounding box and other info as before...
        cv::Point2f center(
            obj.last_box.x + obj.last_box.width/2.0f,
            obj.last_box.y + obj.last_box.height/2.0f
        );

        // Draw velocity arrow using smoothed velocity
        cv::Point2f arrow_end = center + obj.smoothed_velocity * 5.0; // Reduced scale factor
        cv::arrowedLine(frame, center, arrow_end, 
                       cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.3); // Added tip size control

        // Draw text information
        std::string info = cv::format("ID: %d V: %.1f", obj.id, cv::norm(obj.smoothed_velocity));
        cv::putText(frame, info,
                   cv::Point(obj.last_box.x, obj.last_box.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
    
    void drawStatistics(cv::Mat& frame) {
        std::string stats = cv::format("Active: %d Total: %d Lost: %d FPS: %.1f",
                                     metrics.active_trackers,
                                     metrics.total_tracks,
                                     metrics.lost_tracks,
                                     1000.0 / metrics.avg_processing_time);
        
        cv::putText(frame, stats, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
    
    void updateMetrics(const std::chrono::steady_clock::time_point& start_time) {
        auto end_time = std::chrono::steady_clock::now();
        double processing_time = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        metrics.avg_processing_time = 0.95 * metrics.avg_processing_time + 
                                    0.05 * processing_time;
        metrics.active_trackers = std::count_if(
            tracked_objects.begin(),
            tracked_objects.end(),
            [](const TrackerInfo& obj) { return obj.is_active; }
        );
    }

    cv::Rect validateBox(const cv::Rect& box, const cv::Size& frame_size) {
        cv::Rect valid = box;
        
        // Ensure coordinates are not negative
        valid.x = std::max(0, valid.x);
        valid.y = std::max(0, valid.y);
        
        // Ensure width and height are positive and not too small
        valid.width = std::max(2, valid.width);
        valid.height = std::max(2, valid.height);
        
        // Ensure box doesn't exceed frame boundaries
        valid.width = std::min(valid.width, frame_size.width - valid.x);
        valid.height = std::min(valid.height, frame_size.height - valid.y);
        
        return valid;
    }
};


class ROISelector {
private:
    static bool drawing;
    static cv::Point start_point;
    static cv::Rect selection;
    static bool roi_selected;

public:
    static void mouse_callback(int event, int x, int y, int flags, void* param) {
        switch (event) {
            case cv::EVENT_LBUTTONDOWN:
                drawing = true;
                roi_selected = false;
                start_point = cv::Point(x, y);
                selection = cv::Rect(x, y, 0, 0);
                break;

            case cv::EVENT_MOUSEMOVE:
                if (drawing) {
                    selection.x = std::min(x, start_point.x);
                    selection.y = std::min(y, start_point.y);
                    selection.width = std::abs(x - start_point.x);
                    selection.height = std::abs(y - start_point.y);
                }
                break;

            case cv::EVENT_LBUTTONUP:
                drawing = false;
                if (selection.width > 0 && selection.height > 0) {
                    roi_selected = true;
                }
                break;
        }
    }

    static void draw_selection(cv::Mat& image) {
        if (drawing || roi_selected) {
            cv::rectangle(image, selection, cv::Scalar(0, 255, 0), 2);
            if (roi_selected) {
                cv::putText(image, "Press SPACE to confirm selection", 
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                           0.75, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    static bool is_roi_selected() { return roi_selected; }
    static void reset_selection() { roi_selected = false; }
    static cv::Rect get_selection() { return selection; }
};

// Initialize static members
bool ROISelector::drawing = false;
cv::Point ROISelector::start_point;
cv::Rect ROISelector::selection;
bool ROISelector::roi_selected = false;

class Visualizer {
public:
    bool handle_key(char key, ObjectTracker& tracker, cv::Mat& frame) {
        switch (key) {
            case 'c':
                tracker.clearAll();
                return true;
            case 'r':
                tracker.clearAll();
                return true;
            case ' ':  // Space key to confirm ROI selection
                if (ROISelector::is_roi_selected()) {
                    cv::Rect2d roi(ROISelector::get_selection());
                    cv::Mat frame_copy = frame.clone();
                    tracker.add_new_object(frame_copy, roi);
                    ROISelector::reset_selection();
                }
                return true;
            default:
                return false;
        }
    }

    static void draw_help_text(cv::Mat& frame) {
        const int font_face = cv::FONT_HERSHEY_SIMPLEX;
        const double font_scale = 0.5;
        const int thickness = 1;
        const cv::Scalar color(255, 255, 255);
        const int line_spacing = 20;
        int y = 20;

        std::vector<std::string> help_text = {
            "Controls:",
            "- Click and drag to select ROI",
            "- SPACE: Confirm selection",
            "- C: Clear all trackers",
            "- R: Reset",
            "- Q/ESC: Quit"
        };

        for (const auto& text : help_text) {
            cv::putText(frame, text, cv::Point(10, y), font_face, font_scale, color, thickness);
            y += line_spacing;
        }
    }
};

int main(int argc, char** argv) {
    // Set real-time scheduling priority if possible
    #ifdef __linux__
        struct sched_param schparam;
        schparam.sched_priority = sched_get_priority_max(SCHED_FIFO);
        if (sched_setscheduler(0, SCHED_FIFO, &schparam) == 0) {
            std::cout << "Set real-time scheduling priority" << std::endl;
        }
    #endif

    // RTSP stream settings
    const std::string rtsp_url = "rtsp://192.168.0.114:8554/webCamStream";
    const int SKIP_FRAMES = 1;
    const cv::Size FRAME_SIZE(640, 480);  

    try {
        // Initialize video capture with RTSP stream
        VideoCapture video_capture(rtsp_url, FRAME_SIZE, SKIP_FRAMES);

        if (!video_capture.isOpened()) {
            std::cerr << "Failed to open RTSP stream" << std::endl;
            return 1;
        }

        // Try to create window with OpenGL support first, fall back to normal window if it fails
        try {
            cv::namedWindow("Object Tracker", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
        } catch (const cv::Exception& e) {
            std::cout << "OpenGL support not available, using standard window" << std::endl;
            cv::namedWindow("Object Tracker", cv::WINDOW_AUTOSIZE);
        }
        
        cv::setMouseCallback("Object Tracker", ROISelector::mouse_callback, nullptr);

        ObjectTracker tracker;
        Visualizer visualizer;

        cv::Mat frame, display_frame;
        bool paused = false;
        bool show_help = true;

        // Display initial information
        std::cout << "Object Tracker initialized with RTSP stream" << std::endl;
        std::cout << "Press 'h' for help" << std::endl;

        // Performance monitoring variables
        auto last_time = std::chrono::high_resolution_clock::now();
        auto last_fps_update = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        double fps = 0;
        double processing_time = 0;
        std::deque<double> processing_times(30, 0.0); // Store last 30 processing times

        // Optional: Set buffer optimization flags for display
        cv::setWindowProperty("Object Tracker", cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
        cv::setWindowProperty("Object Tracker", cv::WND_PROP_AUTOSIZE, cv::WINDOW_AUTOSIZE);

        while (true) {
            auto frame_start = std::chrono::high_resolution_clock::now();

            if (!paused) {
                // Read frame with error handling
                bool read_success = video_capture.read(frame);
                
                if (!read_success || frame.empty()) {
                    std::cerr << "Failed to read frame from RTSP stream. Retrying..." << std::endl;
                    cv::waitKey(1000); // Wait a second before retry
                    continue;
                }

                // Create a copy for display
                display_frame = frame.clone();

                // Update tracking
                bool tracking_success = tracker.update_tracker(display_frame);

                // Calculate processing time for this frame
                auto frame_end = std::chrono::high_resolution_clock::now();
                processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    frame_end - frame_start).count();
                
                // Update processing times history
                processing_times.pop_front();
                processing_times.push_back(processing_time);

                // Calculate average processing time
                double avg_processing_time = std::accumulate(processing_times.begin(), 
                                                          processing_times.end(), 0.0) 
                                          / processing_times.size();

                // Update FPS counter
                frame_count++;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - last_fps_update).count();
                
                if (elapsed >= 1) {
                    fps = frame_count / elapsed;
                    frame_count = 0;
                    last_fps_update = current_time;
                }

                // Draw performance information
                const int text_offset_y = 25;
                const cv::Scalar text_color(0, 255, 0);
                const double font_scale = 0.6;
                const int font_thickness = 2;

                // Draw FPS
                cv::putText(display_frame, 
                           "FPS: " + std::to_string(static_cast<int>(fps)),
                           cv::Point(10, text_offset_y), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           text_color, 
                           font_thickness);

                // Draw processing time
                cv::putText(display_frame, 
                           "Processing Time: " + std::to_string(static_cast<int>(avg_processing_time)) + "ms",
                           cv::Point(10, text_offset_y * 2), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           text_color, 
                           font_thickness);

                // Draw latency estimation
                cv::putText(display_frame, 
                           "Estimated Latency: " + std::to_string(static_cast<int>(avg_processing_time + 
                           1000.0/fps)) + "ms",
                           cv::Point(10, text_offset_y * 3), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           text_color, 
                           font_thickness);
            }

            // Draw tracking visualizations
            tracker.draw_tracked_objects(display_frame);
            ROISelector::draw_selection(display_frame);
            
            if (show_help) {
                Visualizer::draw_help_text(display_frame);
            }

            // Show frame
            if (!display_frame.empty()) {
                cv::imshow("Object Tracker", display_frame);
            }

            // Handle keyboard input with longer wait time to reduce CPU usage
            char key = static_cast<char>(cv::waitKey(1));
            
            switch (key) {
                case 'q':
                case 27: // ESC
                    goto cleanup;
                case 'p':
                    paused = !paused;
                    std::cout << (paused ? "Paused" : "Resumed") << std::endl;
                    break;
                case 'h':
                    show_help = !show_help;
                    break;
                case 'r':
                    tracker.clearAll();
                    std::cout << "Tracking reset" << std::endl;
                    break;
                default:
                    visualizer.handle_key(key, tracker, frame);
                    break;
            }

            // Monitor memory usage
            #ifdef __linux__
                if (frame_count % 30 == 0) {
                    double vm_usage = 0.0;
                    double resident_set = 0.0;
                    std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
                    std::string pid, comm, state, ppid, pgrp, session, tty_nr;
                    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr;
                    vm_usage = std::stod(pid) * getpagesize() / 1024.0 / 1024.0; // Convert to MB
                    std::cout << "Memory Usage (MB): " << vm_usage << std::endl;
                }
            #endif
        }

cleanup:
        // Cleanup
        video_capture.release();
        cv::destroyAllWindows();
        std::cout << "Application terminated normally" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}