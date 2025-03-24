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
        
        TrackingResult(const cv::Rect2d& bbox) : 
            success(false), 
            updated_bbox(bbox), 
            confidence(0.0), 
            is_occluded(false), 
            recovered(false), 
            predicted_bbox(bbox) {}
    };

private:
    // Feature detection and matching
    cv::Ptr<cv::AKAZE> feature_detector;
    cv::FlannBasedMatcher flann_matcher;
    
    // Tracking parameters
    const float match_ratio = 0.72f;
    const int min_matches = 3;
    const double min_confidence = 0.10;
    const int min_area = 12;
    const double max_area_ratio = 0.98;
    const double ransac_threshold = 2.5;

    // Recovery parameters
    const int search_expansion = 5;
    const int max_recovery_attempts = 10;
    int recovery_count = 0;
    cv::Mat template_features;
    cv::Mat original_roi;
    std::vector<cv::KeyPoint> template_keypoints;
    cv::Mat template_descriptors;
    double original_area;
    cv::Size original_size;
    int recovery_frames = 0;
    const int recovery_threshold = 3;

    // Motion model
    cv::Point2f last_velocity;
    cv::Point2f last_valid_center;
    bool has_last_motion = false;
    const double velocity_weight = 0.85;

public:
    FeatureTracker() {
        feature_detector = cv::AKAZE::create(
            cv::AKAZE::DESCRIPTOR_MLDB,
            0,  // descriptor size
            8,  // descriptor channels
            0.0002f,  // threshold
            8,  // octaves
            0,  // octave layers
            cv::KAZE::DIFF_PM_G2
        );

        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
        cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
        flann_matcher = cv::FlannBasedMatcher(indexParams, searchParams);
    }

private:
    cv::Rect validateRoi(const cv::Rect& roi, const cv::Size& frame_size) {
        cv::Rect valid = roi;
        
        // First ensure the ROI is within frame bounds
        valid.x = std::max(0, std::min(valid.x, frame_size.width - 1));
        valid.y = std::max(0, std::min(valid.y, frame_size.height - 1));
        
        // Ensure width and height don't exceed frame bounds
        valid.width = std::min(valid.width, frame_size.width - valid.x);
        valid.height = std::min(valid.height, frame_size.height - valid.y);
        
        // Ensure minimum size
        valid.width = std::max(min_area, valid.width);
        valid.height = std::max(min_area, valid.height);
        
        // Apply maximum size constraints
        if (valid.width > frame_size.width * max_area_ratio) {
            valid.width = static_cast<int>(frame_size.width * max_area_ratio);
        }
        if (valid.height > frame_size.height * max_area_ratio) {
            valid.height = static_cast<int>(frame_size.height * max_area_ratio);
        }
        
        // Final boundary check
        valid.width = std::min(valid.width, frame_size.width - valid.x);
        valid.height = std::min(valid.height, frame_size.height - valid.y);
        
        // If resulting ROI is invalid, create a minimal valid ROI
        if (valid.width <= 0 || valid.height <= 0 || 
            valid.x < 0 || valid.y < 0 || 
            valid.x + valid.width > frame_size.width || 
            valid.y + valid.height > frame_size.height) {
            
            // Create a minimal valid ROI in the center of the frame
            valid.width = std::min(min_area, frame_size.width);
            valid.height = std::min(min_area, frame_size.height);
            valid.x = (frame_size.width - valid.width) / 2;
            valid.y = (frame_size.height - valid.height) / 2;
        }
        
        return valid;
    }


    void storeTemplate(const cv::Mat& frame, const cv::Rect& roi) {
        try {
            cv::Rect valid_roi = validateRoi(roi, frame.size());
            cv::Mat gray_frame;
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
            original_roi = gray_frame(valid_roi).clone();
            original_area = valid_roi.area();
            original_size = valid_roi.size();
            
            feature_detector->detectAndCompute(original_roi, cv::Mat(), 
                                            template_keypoints, template_descriptors);
            
            std::cout << "Template stored with " << template_keypoints.size() 
                      << " keypoints" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error storing template: " << e.what() << std::endl;
        }
    }

    TrackingResult trackFrame(const cv::Mat& prev_gray, const cv::Mat& curr_gray,
                          const cv::Rect2d& prev_bbox) {
        TrackingResult result(prev_bbox);

        try {
            // Predict new location based on motion model
            cv::Rect2d predicted_bbox = prev_bbox;
            if (has_last_motion) {
                predicted_bbox.x += last_velocity.x;
                predicted_bbox.y += last_velocity.y;
            }
            result.predicted_bbox = predicted_bbox;

            // Validate bounding boxes
            cv::Rect validated_bbox = validateRoi(prev_bbox, prev_gray.size());
            cv::Rect validated_pred_bbox = validateRoi(predicted_bbox, curr_gray.size());

            // Create search region
            cv::Rect search_region = validated_pred_bbox;
            int search_margin = std::max(search_region.width, search_region.height) / 2;
            search_region.x = std::max(0, search_region.x - search_margin);
            search_region.y = std::max(0, search_region.y - search_margin);
            search_region.width = std::min(curr_gray.cols - search_region.x, 
                                         search_region.width + 2 * search_margin);
            search_region.height = std::min(curr_gray.rows - search_region.y, 
                                          search_region.height + 2 * search_margin);

            search_region = validateRoi(search_region, curr_gray.size());

            // Extract ROI and detect features
            cv::Mat prev_roi = prev_gray(validated_bbox);
            cv::Mat curr_roi = curr_gray(search_region);

            if (prev_roi.empty() || curr_roi.empty()) {
                result.is_occluded = true;
                result.updated_bbox = validated_pred_bbox;
                return result;
            }

            std::vector<cv::KeyPoint> prev_keypoints, curr_keypoints;
            cv::Mat prev_descriptors, curr_descriptors;

            feature_detector->detectAndCompute(prev_roi, cv::Mat(), prev_keypoints, prev_descriptors);
            feature_detector->detectAndCompute(curr_roi, cv::Mat(), curr_keypoints, curr_descriptors);

            if (prev_keypoints.empty() || curr_keypoints.empty() || 
                prev_descriptors.empty() || curr_descriptors.empty()) {
                result.is_occluded = true;
                result.updated_bbox = validated_pred_bbox;
                return result;
            }

            // Adjust keypoint coordinates to global image space
            for (auto& kp : prev_keypoints) {
                kp.pt.x += validated_bbox.x;
                kp.pt.y += validated_bbox.y;
            }
            for (auto& kp : curr_keypoints) {
                kp.pt.x += search_region.x;
                kp.pt.y += search_region.y;
            }

            // Match features
            std::vector<std::vector<cv::DMatch>> knn_matches;
            try {
                flann_matcher.knnMatch(prev_descriptors, curr_descriptors, knn_matches, 2);
            } catch (const cv::Exception&) {
                result.is_occluded = true;
                result.updated_bbox = validated_pred_bbox;
                return result;
            }

            std::vector<cv::DMatch> good_matches;
            std::vector<cv::Point2f> prev_points, curr_points;

            // Filter matches
            for (const auto& match_pair : knn_matches) {
                if (match_pair.size() < 2) continue;
                
                if (match_pair[0].distance < match_ratio * match_pair[1].distance) {
                    good_matches.push_back(match_pair[0]);
                    prev_points.push_back(prev_keypoints[match_pair[0].queryIdx].pt);
                    curr_points.push_back(curr_keypoints[match_pair[0].trainIdx].pt);
                }
            }

            if (good_matches.size() < min_matches) {
                result.is_occluded = true;
                result.updated_bbox = validated_pred_bbox;
                return result;
            }

            // Find homography
            std::vector<uchar> inlier_mask;
            cv::Mat H = cv::findHomography(prev_points, curr_points, cv::RANSAC, 
                                         ransac_threshold, inlier_mask);

            if (H.empty()) {
                result.is_occluded = true;
                result.updated_bbox = validated_pred_bbox;
                return result;
            }

            // Transform bounding box
            std::vector<cv::Point2f> bbox_corners(4);
            bbox_corners[0] = cv::Point2f(validated_bbox.x, validated_bbox.y);
            bbox_corners[1] = cv::Point2f(validated_bbox.x + validated_bbox.width, validated_bbox.y);
            bbox_corners[2] = cv::Point2f(validated_bbox.x + validated_bbox.width, 
                                        validated_bbox.y + validated_bbox.height);
            bbox_corners[3] = cv::Point2f(validated_bbox.x, validated_bbox.y + validated_bbox.height);

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

            cv::Rect2d new_bbox(min_x, min_y, max_x - min_x, max_y - min_y);
            cv::Rect validated_new_bbox = validateRoi(new_bbox, curr_gray.size());

            // Calculate confidence
            double inlier_ratio = static_cast<double>(cv::countNonZero(inlier_mask)) / 
                                inlier_mask.size();
            double bbox_size_ratio = std::min(
                static_cast<double>(validated_new_bbox.area()) / validated_bbox.area(),
                static_cast<double>(validated_bbox.area()) / validated_new_bbox.area()
            );

            // Update motion model
            cv::Point2f current_center(validated_new_bbox.x + validated_new_bbox.width/2,
                                     validated_new_bbox.y + validated_new_bbox.height/2);
            if (has_last_motion) {
                cv::Point2f current_velocity = current_center - last_valid_center;
                last_velocity = velocity_weight * last_velocity + 
                              (1 - velocity_weight) * current_velocity;
            } else {
                last_velocity = cv::Point2f(0, 0);
                has_last_motion = true;
            }
            last_valid_center = current_center;

            // Combine tracking and prediction
            double tracking_confidence = inlier_ratio * bbox_size_ratio;
            if (tracking_confidence < min_confidence) {
                double alpha = tracking_confidence / min_confidence;
                validated_new_bbox.x = static_cast<int>(alpha * validated_new_bbox.x + 
                                                      (1-alpha) * validated_pred_bbox.x);
                validated_new_bbox.y = static_cast<int>(alpha * validated_new_bbox.y + 
                                                      (1-alpha) * validated_pred_bbox.y);
                result.is_occluded = true;
            }

            result.success = true;
            result.updated_bbox = validated_new_bbox;
            result.confidence = tracking_confidence;
            result.matched_points = curr_points;

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in trackFrame: " << e.what() << std::endl;
            result.is_occluded = true;
            result.updated_bbox = result.predicted_bbox;
        }

        return result;
    }

    bool attemptRecovery(const cv::Mat& frame, const cv::Rect& search_area,
                        cv::Rect2d& recovered_bbox) {
        try {
            cv::Mat gray_frame;
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
            cv::Mat search_roi = gray_frame(validateRoi(search_area, frame.size()));

            std::vector<cv::KeyPoint> curr_keypoints;
            cv::Mat curr_descriptors;
            feature_detector->detectAndCompute(search_roi, cv::Mat(), 
                                            curr_keypoints, curr_descriptors);

            if (curr_keypoints.empty() || template_keypoints.empty() || 
                curr_descriptors.empty() || template_descriptors.empty()) {
                return false;
            }

            std::vector<std::vector<cv::DMatch>> knn_matches;
            flann_matcher.knnMatch(template_descriptors, curr_descriptors, knn_matches, 2);

            std::vector<cv::Point2f> template_points, curr_points;
            for (const auto& match_pair : knn_matches) {
                if (match_pair.size() < 2) continue;

                if (match_pair[0].distance < match_ratio * match_pair[1].distance) {
                    template_points.push_back(template_keypoints[match_pair[0].queryIdx].pt);
                    cv::Point2f curr_pt = curr_keypoints[match_pair[0].trainIdx].pt;
                    curr_pt.x += search_area.x;
                    curr_pt.y += search_area.y;
                    curr_points.push_back(curr_pt);
                }
            }

            if (template_points.size() < min_matches) {
                return false;
            }

            std::vector<uchar> inlier_mask;
            cv::Mat H = cv::findHomography(template_points, curr_points, cv::RANSAC, 
                                         ransac_threshold * 2, inlier_mask);

            if (H.empty()) {
                return false;
            }

            std::vector<cv::Point2f> bbox_corners = {
                cv::Point2f(0, 0),
                cv::Point2f(original_size.width, 0),
                cv::Point2f(original_size.width, original_size.height),
                cv::Point2f(0, original_size.height)
            };

            std::vector<cv::Point2f> transformed_corners;
            cv::perspectiveTransform(bbox_corners, transformed_corners, H);

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

            recovered_bbox = cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y);
            
            double area_ratio = recovered_bbox.area() / original_area;
            if (area_ratio < 0.5 || area_ratio > 2.0) {
                return false;
            }

            double inlier_ratio = static_cast<double>(cv::countNonZero(inlier_mask)) / 
                                inlier_mask.size();
            
            return inlier_ratio > min_confidence;

        } catch (const cv::Exception& e) {
            std::cerr << "Recovery attempt failed: " << e.what() << std::endl;
            return false;
        }
    }

public:
    TrackingResult track(const cv::Mat& prev_frame, const cv::Mat& curr_frame,
                        const cv::Rect2d& prev_bbox, bool is_first_frame = false) {
        TrackingResult result(prev_bbox);

        try {
            if (prev_frame.empty() || curr_frame.empty()) {
                return result;
            }

            // Store template on first frame
            if (is_first_frame) {
                storeTemplate(prev_frame, prev_bbox);
                recovery_count = 0;
                recovery_frames = 0;
            }

            cv::Mat prev_gray, curr_gray;
            cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

            // Normal tracking attempt
            auto tracking_result = trackFrame(prev_gray, curr_gray, prev_bbox);
            if (!tracking_result.is_occluded) {
                recovery_frames = 0;
                return tracking_result;
            }

            // Handle occlusion
            recovery_frames++;

            // Attempt recovery if occlusion persists
            if (recovery_frames >= recovery_threshold) {
                cv::Rect search_area = prev_bbox;
                int margin_x = search_area.width * search_expansion;
                int margin_y = search_area.height * search_expansion;
                
                search_area.x -= margin_x;
                search_area.y -= margin_y;
                search_area.width += 2 * margin_x;
                search_area.height += 2 * margin_y;
                
                search_area = validateRoi(search_area, curr_frame.size());

                cv::Rect2d recovered_bbox;
                if (attemptRecovery(curr_frame, search_area, recovered_bbox)) {
                    result.success = true;
                    result.updated_bbox = validateRoi(recovered_bbox, curr_frame.size());
                    result.recovered = true;
                    result.is_occluded = false;
                    recovery_frames = 0;
                    
                    // Update template after successful recovery
                    storeTemplate(curr_frame, result.updated_bbox);
                    
                    std::cout << "Object recovered after occlusion" << std::endl;
                    return result;
                }
            }

            // If recovery failed, use motion prediction
            result.is_occluded = true;
            result.updated_bbox = tracking_result.predicted_bbox;
            return result;

        } catch (const cv::Exception& e) {
            std::cerr << "Error in tracking: " << e.what() << std::endl;
            result.is_occluded = true;
            return result;
        }
    }

    void reset() {
        recovery_count = 0;
        recovery_frames = 0;
        has_last_motion = false;
        last_velocity = cv::Point2f(0, 0);
        template_keypoints.clear();
        template_descriptors.release();
        original_roi.release();
    }

    bool isRecovering() const {
        return recovery_frames > 0;
    }

    int getRecoveryFrames() const {
        return recovery_frames;
    }

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

            // Draw matched points
            for (const auto& pt : result.matched_points) {
                cv::circle(frame, pt, 2, color, -1);
            }

            // Draw confidence value
            std::string conf_str = "Conf: " + std::to_string(result.confidence).substr(0, 4);
            cv::putText(frame, conf_str, 
                       cv::Point(result.updated_bbox.x, result.updated_bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

            // Draw recovery status if applicable
            if (recovery_frames > 0) {
                std::string recovery_str = "Recovery: " + std::to_string(recovery_frames);
                cv::putText(frame, recovery_str,
                           cv::Point(result.updated_bbox.x, result.updated_bbox.y - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }
        }
    }
};

class ObjectTracker {
private:
    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Mat> prev_frames;
    std::vector<std::deque<cv::Point2f>> trajectories;
    std::vector<cv::KalmanFilter> kalman_filters;
    std::vector<bool> kalman_initialized;
    std::unique_ptr<FeatureTracker> feature_tracker;
    cv::Ptr<cv::AKAZE> feature_detector;
    cv::FlannBasedMatcher flann_matcher;
    std::map<int, cv::Mat> object_gallery;
    std::vector<cv::Rect> exit_history;
    int next_id;

    // Constants
    const size_t MAX_TRAJECTORY_LENGTH = 60;
    const double LOST_THRESHOLD = 0.01;
    const int MAX_LOST_FRAMES = 60;
    const float OVERLAP_THRESHOLD = 0.35f;
    const double SIMILARITY_THRESHOLD = 0.12;

    struct TrackerInfo {
        int id;
        cv::Ptr<cv::Tracker> tracker;
        cv::KalmanFilter kalman;
        cv::Mat hist;
        cv::Rect last_box;
        std::deque<cv::Point2f> trajectory;
        int lost_count;
        bool is_active;
        
        TrackerInfo(int id, cv::Ptr<cv::Tracker> tracker, const cv::KalmanFilter& kf, 
                   const cv::Mat& hist, const cv::Rect& box) : 
            id(id), tracker(tracker), kalman(kf), hist(hist), last_box(box), 
            lost_count(0), is_active(true) {
            trajectory.push_back(cv::Point2f(
                box.x + box.width/2.0f,
                box.y + box.height/2.0f
            ));
        }
    };
    
    std::vector<TrackerInfo> tracked_objects;

    cv::KalmanFilter init_kalman() {
        cv::KalmanFilter kalman(6, 4, 0);
        
        // Measurement Matrix (4x6)
        kalman.measurementMatrix = cv::Mat::zeros(4, 6, CV_32F);
        kalman.measurementMatrix.at<float>(0,0) = 1.0f;
        kalman.measurementMatrix.at<float>(1,1) = 1.0f;
        kalman.measurementMatrix.at<float>(2,2) = 1.0f;
        kalman.measurementMatrix.at<float>(3,3) = 1.0f;
        
        // Transition Matrix (6x6)
        kalman.transitionMatrix = cv::Mat::zeros(6, 6, CV_32F);
        kalman.transitionMatrix.at<float>(0,0) = 1.0f;  // x
        kalman.transitionMatrix.at<float>(1,1) = 1.0f;  // y
        kalman.transitionMatrix.at<float>(2,2) = 1.0f;  // width
        kalman.transitionMatrix.at<float>(3,3) = 1.0f;  // height
        kalman.transitionMatrix.at<float>(4,4) = 1.0f;  // dx
        kalman.transitionMatrix.at<float>(5,5) = 1.0f;  // dy
        kalman.transitionMatrix.at<float>(0,4) = 0.2f;  // x += dx * 0.5
        kalman.transitionMatrix.at<float>(1,5) = 0.2f;  // y += dy * 0.5

        // Process noise
        kalman.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.03f;
        kalman.processNoiseCov.at<float>(4,4) *= 0.3f;  // Reduce noise for velocity x
        kalman.processNoiseCov.at<float>(5,5) *= 0.3f;  // Reduce noise for velocity y
        
        // Measurement noise
        kalman.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.1f;
        
        // Error covariance matrices
        kalman.errorCovPost = cv::Mat::eye(6, 6, CV_32F) * 0.25f;
        kalman.errorCovPre = cv::Mat::eye(6, 6, CV_32F) * 0.15f;
        
        return kalman;
    }

    cv::Mat calculateColorHistogram(const cv::Mat& roi) {
        cv::Mat hsv_roi;
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
        cv::Mat hist;
        int h_bins = 90;
        int s_bins = 128;
        int histSize[] = { h_bins, s_bins };
        float h_ranges[] = { 0, 180 };
        float s_ranges[] = { 0, 256 };
        const float* ranges[] = { h_ranges, s_ranges };
        int channels[] = { 0, 1 };
        cv::calcHist(&hsv_roi, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
        cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
        return hist;
    }

    double calculateHistSimilarity(const cv::Mat& hist1, const cv::Mat& hist2) {
        return cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
    }

    cv::Rect recoverLostObject(const cv::Mat& frame, const cv::Rect& last_box, const cv::Mat& obj_hist) {
        try {
            // Define search area with boundary checks
            cv::Size frame_size = frame.size();
            cv::Rect search_area(
                std::max(0, last_box.x - last_box.width),
                std::max(0, last_box.y - last_box.height),
                std::min(frame_size.width - std::max(0, last_box.x - last_box.width), last_box.width * 3),
                std::min(frame_size.height - std::max(0, last_box.y - last_box.height), last_box.height * 3)
            );
            
            search_area = validateBox(search_area, frame_size);
            
            if (search_area.area() <= 0) {
                return cv::Rect();
            }

            // Try feature matching first
            if (feature_tracker) {
                cv::Rect last_box_in_search = last_box;
                last_box_in_search.x -= search_area.x;
                last_box_in_search.y -= search_area.y;
                
                auto feature_result = feature_tracker->track(
                    frame(validateBox(last_box, frame_size)),
                    frame(search_area),
                    cv::Rect2d(last_box_in_search)
                );
                
                if (feature_result.success && feature_result.confidence > LOST_THRESHOLD) {
                    cv::Rect recovered_box(
                        static_cast<int>(feature_result.updated_bbox.x) + search_area.x,
                        static_cast<int>(feature_result.updated_bbox.y) + search_area.y,
                        static_cast<int>(feature_result.updated_bbox.width),
                        static_cast<int>(feature_result.updated_bbox.height)
                    );
                    return validateBox(recovered_box, frame_size);
                }
            }
            
            // If feature matching fails, try mean shift
            cv::Mat roi = frame(search_area);
            if (roi.empty()) {
                return cv::Rect();
            }
            
            cv::Mat roi_hsv;
            cv::cvtColor(roi, roi_hsv, cv::COLOR_BGR2HSV);
            
            // Calculate back projection
            cv::Mat back_proj;
            int h_bins = 90;
            int s_bins = 128;
            int histSize[] = { h_bins, s_bins };
            float h_ranges[] = { 0, 180 };
            float s_ranges[] = { 0, 256 };
            const float* ranges[] = { h_ranges, s_ranges };
            int channels[] = { 0, 1 };
            
            cv::calcBackProject(&roi_hsv, 1, channels, obj_hist, back_proj, ranges);
            
            // Initialize search window for mean shift
            cv::Rect track_window(
                last_box.x - search_area.x,
                last_box.y - search_area.y,
                last_box.width,
                last_box.height
            );
            
            // Ensure track window is within search area
            track_window = validateBox(track_window, roi.size());
            
            if (track_window.area() <= 0) {
                return cv::Rect();
            }
            
            // Apply mean shift
            cv::TermCriteria criteria(
                cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                10,  // maximum number of iterations
                1    // minimum change in position
            );
            
            int attempts = 0;
            const int MAX_ATTEMPTS = 5;
            double shift_delta = 2.0;
            
            while (attempts < MAX_ATTEMPTS) {
                cv::Rect prev_window = track_window;
                cv::meanShift(back_proj, track_window, criteria);
                
                // Calculate movement
                double dx = std::abs(track_window.x - prev_window.x);
                double dy = std::abs(track_window.y - prev_window.y);
                double movement = std::sqrt(dx*dx + dy*dy);
                
                // If movement is small enough, break
                if (movement < shift_delta) {
                    break;
                }
                
                attempts++;
            }
            
            // Convert back to frame coordinates
            cv::Rect recovered_box(
                track_window.x + search_area.x,
                track_window.y + search_area.y,
                track_window.width,
                track_window.height
            );
            
            // Validate final box
            recovered_box = validateBox(recovered_box, frame_size);
            
            // Verify recovery quality
            if (recovered_box.area() > 0) {
                cv::Mat recovered_roi = frame(recovered_box);
                cv::Mat recovered_hist = calculateColorHistogram(recovered_roi);
                double similarity = calculateHistSimilarity(obj_hist, recovered_hist);
                
                if (similarity > SIMILARITY_THRESHOLD) {
                    return recovered_box;
                }
            }
            
            return cv::Rect();
            
        } catch (const cv::Exception& e) {
            std::cerr << "Error in recoverLostObject: " << e.what() << std::endl;
            return cv::Rect();
        } catch (const std::exception& e) {
            std::cerr << "Error in recoverLostObject: " << e.what() << std::endl;
            return cv::Rect();
        }
    }

    bool boxes_overlap(const cv::Rect& box1, const cv::Rect& box2) {
        return (box1 & box2).area() > 0;
    }

    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
        cv::Rect intersection = box1 & box2;
        float inter_area = intersection.area();
        float union_area = box1.area() + box2.area() - inter_area;
        return union_area > 0 ? inter_area / union_area : 0;
    }

    void handle_occlusions(cv::Mat& frame) {
        std::vector<std::pair<size_t, size_t>> overlaps;
        const float DEEP_OCCLUSION_THRESHOLD = 0.6f;  // Threshold for severe occlusion
        const float RECOVERY_CONFIDENCE_THRESHOLD = 0.25f;  // Confidence threshold for recovery
        
        // Detect overlapping boxes with enhanced precision
        for (size_t i = 0; i < tracked_objects.size(); i++) {
            for (size_t j = i + 1; j < tracked_objects.size(); j++) {
                if (boxes_overlap(tracked_objects[i].last_box, tracked_objects[j].last_box)) {
                    float iou = calculate_iou(tracked_objects[i].last_box, tracked_objects[j].last_box);
                    
                    if (iou > OVERLAP_THRESHOLD) {
                        overlaps.push_back({i, j});
                        
                        // Enhanced handling for deep occlusions
                        if (iou > DEEP_OCCLUSION_THRESHOLD) {
                            auto& obj1 = tracked_objects[i];
                            auto& obj2 = tracked_objects[j];
                            
                            // Predict using velocity and acceleration
                            cv::Mat pred1 = obj1.kalman.predict();
                            cv::Mat pred2 = obj2.kalman.predict();
                            
                            // Calculate confidence based on historical trajectory
                            double conf1 = calculate_trajectory_confidence(obj1);
                            double conf2 = calculate_trajectory_confidence(obj2);
                            
                            // Update based on confidence and predictions
                            if (conf1 > RECOVERY_CONFIDENCE_THRESHOLD || 
                                conf2 > RECOVERY_CONFIDENCE_THRESHOLD) {
                                update_deep_occlusion(obj1, obj2, pred1, pred2, conf1, conf2);
                            }
                        }
                    }
                }
            }
        }
    }

    double calculate_trajectory_confidence(const TrackerInfo& obj) {
        if (obj.trajectory.size() < 2) return 0.0;
        
        // Calculate trajectory smoothness and consistency
        double velocity_consistency = 0.0;
        double direction_consistency = 0.0;
        
        for (size_t i = 1; i < obj.trajectory.size(); i++) {
            cv::Point2f vel = obj.trajectory[i] - obj.trajectory[i-1];
            if (i > 1) {
                cv::Point2f prev_vel = obj.trajectory[i-1] - obj.trajectory[i-2];
                velocity_consistency += std::abs(cv::norm(vel) - cv::norm(prev_vel));
                direction_consistency += std::abs(std::atan2(vel.y, vel.x) - 
                                               std::atan2(prev_vel.y, prev_vel.x));
            }
        }
        
        // Normalize confidence metrics
        velocity_consistency = 1.0 / (1.0 + velocity_consistency / obj.trajectory.size());
        direction_consistency = 1.0 / (1.0 + direction_consistency / obj.trajectory.size());
        
        return 0.6 * velocity_consistency + 0.4 * direction_consistency;
    }

    void update_deep_occlusion(TrackerInfo& obj1, TrackerInfo& obj2, 
                             const cv::Mat& pred1, const cv::Mat& pred2,
                             double conf1, double conf2) {
        // Weight predictions based on confidence
        float w1 = conf1 / (conf1 + conf2);
        float w2 = 1.0f - w1;
        
        // Update object states with weighted predictions
        if (conf1 > conf2) {
            obj1.last_box = cv::Rect(
                pred1.at<float>(0), pred1.at<float>(1),
                pred1.at<float>(2), pred1.at<float>(3)
            );
            obj2.lost_count++;
        } else {
            obj2.last_box = cv::Rect(
                pred2.at<float>(0), pred2.at<float>(1),
                pred2.at<float>(2), pred2.at<float>(3)
            );
            obj1.lost_count++;
        }
    }


    void handleTrajectories() {
        for (auto& obj : tracked_objects) {
            if (obj.trajectory.size() > MAX_TRAJECTORY_LENGTH) {
                obj.trajectory.pop_front();
            }
        }
    }

public:
    // ObjectTracker() : next_id(0) {
    //     feature_detector = cv::AKAZE::create(
    //         cv::AKAZE::DESCRIPTOR_MLDB,
    //         0, 8, 0.0002f, 8, 0,
    //         cv::KAZE::DIFF_PM_G2
    //     );
        
    //     cv::Ptr<cv::flann::IndexParams> indexParams = 
    //         cv::makePtr<cv::flann::KDTreeIndexParams>(5);
    //     cv::Ptr<cv::flann::SearchParams> searchParams = 
    //         cv::makePtr<cv::flann::SearchParams>(50);
    //     flann_matcher = cv::FlannBasedMatcher(indexParams, searchParams);
        
    //     feature_tracker = std::make_unique<FeatureTracker>();
    // }

    void add_new_object(const cv::Mat& frame, const cv::Rect& roi) {
        try {
            if (frame.empty() || roi.area() <= 0) {
                std::cerr << "Invalid frame or ROI" << std::endl;
                return;
            }

            cv::Mat frame_copy = frame.clone();
            cv::Rect adjusted_roi = roi;
            adjusted_roi.x = std::max(0, std::min(adjusted_roi.x, frame.cols - 1));
            adjusted_roi.y = std::max(0, std::min(adjusted_roi.y, frame.rows - 1));
            adjusted_roi.width = std::min(adjusted_roi.width, frame.cols - adjusted_roi.x);
            adjusted_roi.height = std::min(adjusted_roi.height, frame.rows - adjusted_roi.y);

            // Initialize tracker
            cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
            tracker->init(frame_copy, adjusted_roi);

            // Calculate initial histogram
            cv::Mat roi_img = frame_copy(adjusted_roi);
            cv::Mat hist = calculateColorHistogram(roi_img);

            // Initialize Kalman filter
            cv::KalmanFilter kf = init_kalman();
            cv::Mat state(6, 1, CV_32F);
            state.at<float>(0) = adjusted_roi.x + adjusted_roi.width/2.0f;
            state.at<float>(1) = adjusted_roi.y + adjusted_roi.height/2.0f;
            state.at<float>(2) = adjusted_roi.width;
            state.at<float>(3) = adjusted_roi.height;
            state.at<float>(4) = 0;
            state.at<float>(5) = 0;
            kf.statePost = state;

            // Create and add new tracker info
            tracked_objects.emplace_back(next_id, tracker, kf, hist, adjusted_roi);

            // Store in gallery
            object_gallery[next_id] = hist.clone();

            next_id++;
            std::cout << "Added new object with ID: " << (next_id - 1) << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in add_new_object: " << e.what() << std::endl;
        }
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

    bool update_tracker(cv::Mat& frame) {
        if (frame.empty() || tracked_objects.empty()) {
            return false;
        }

        bool overall_success = false;
        cv::Size frame_size = frame.size();
        handle_occlusions(frame);

        for (auto& obj : tracked_objects) {
            if (!obj.is_active) continue;

            // Predict using Kalman filter
            cv::Mat prediction = obj.kalman.predict();
            
            // Update tracker
            cv::Rect new_box;
            bool success = obj.tracker->update(frame, new_box);
            
            if (success) {
                // Validate box position before accessing ROI
                new_box = validateBox(new_box, frame_size);
                
                if (new_box.area() > 0) {
                    // Now safe to access ROI
                    cv::Mat roi = frame(new_box);
                    cv::Mat curr_hist = calculateColorHistogram(roi);
                    double similarity = calculateHistSimilarity(obj.hist, curr_hist);
                    
                    if (similarity > SIMILARITY_THRESHOLD) {
                        // Update Kalman filter with measurement
                        cv::Mat measurement = (cv::Mat_<float>(4,1) << 
                            new_box.x, new_box.y, new_box.width, new_box.height);
                        obj.kalman.correct(measurement);
                        
                        // Combine Kalman prediction with measurement
                        float alpha = 0.7f;
                        cv::Rect smoothed_box;
                        smoothed_box.x = static_cast<int>(alpha * new_box.x + (1-alpha) * prediction.at<float>(0));
                        smoothed_box.y = static_cast<int>(alpha * new_box.y + (1-alpha) * prediction.at<float>(1));
                        smoothed_box.width = static_cast<int>(alpha * new_box.width + (1-alpha) * prediction.at<float>(2));
                        smoothed_box.height = static_cast<int>(alpha * new_box.height + (1-alpha) * prediction.at<float>(3));
                        
                        // Validate smoothed box
                        smoothed_box = validateBox(smoothed_box, frame_size);

                        // Update trajectory
                        obj.trajectory.push_back(cv::Point2f(
                            smoothed_box.x + smoothed_box.width/2.0f,
                            smoothed_box.y + smoothed_box.height/2.0f
                        ));
                        
                        obj.last_box = smoothed_box;
                        obj.hist = curr_hist;
                        obj.lost_count = 0;
                        overall_success = true;
                    } else {
                        obj.lost_count++;
                    }
                } else {
                    obj.lost_count++;
                }
            } else {
                obj.lost_count++;
            }
            
            // Handle lost objects
            if (obj.lost_count > MAX_LOST_FRAMES) {
                cv::Rect recovered_box = recoverLostObject(frame, obj.last_box, obj.hist);
                recovered_box = validateBox(recovered_box, frame_size);
                
                if (recovered_box.area() > 0) {
                    // Validate recovered position
                    cv::Mat recovered_roi = frame(recovered_box);
                    cv::Mat recovered_hist = calculateColorHistogram(recovered_roi);
                    double recovery_similarity = calculateHistSimilarity(obj.hist, recovered_hist);
                    
                    if (recovery_similarity > SIMILARITY_THRESHOLD) {
                        // Reinitialize tracker with recovered position
                        obj.tracker = cv::TrackerCSRT::create();
                        obj.tracker->init(frame, recovered_box);
                        obj.last_box = recovered_box;
                        obj.hist = recovered_hist;
                        obj.lost_count = 0;
                        overall_success = true;
                    } else {
                        obj.is_active = false;
                        exit_history.push_back(obj.last_box);
                    }
                } else {
                    obj.is_active = false;
                    exit_history.push_back(obj.last_box);
                }
            }
        }
        
        // Clean up inactive trackers
        tracked_objects.erase(
            std::remove_if(tracked_objects.begin(), tracked_objects.end(),
                [](const TrackerInfo& info) { return !info.is_active; }),
            tracked_objects.end()
        );
        
        handleTrajectories();
        return overall_success;
    }

    void draw_tracked_objects(cv::Mat& frame) {
        for (const auto& obj : tracked_objects) {
            if (!obj.is_active) continue;

            // Colors
            const cv::Scalar BOX_COLOR(0, 255, 255);    
            const cv::Scalar PRED_COLOR(0, 165, 255);   
            const cv::Scalar TRAJ_COLOR(0, 255, 0);     
            const cv::Scalar TEXT_COLOR(255, 255, 255); 

            // Draw current bounding box
            cv::rectangle(frame, obj.last_box, BOX_COLOR, 2);

            // Instead of predicting, use the current state for visualization
            cv::Point2f curr_center(obj.last_box.x + obj.last_box.width/2, 
                                obj.last_box.y + obj.last_box.height/2);

            // Draw ID and tracking info
            std::string info = "ID: " + std::to_string(obj.id);
            cv::Point text_pos(obj.last_box.x, obj.last_box.y - 10);
            
            // Add background for text
            cv::Size text_size = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
            cv::rectangle(frame, 
                        cv::Point(text_pos.x - 2, text_pos.y - text_size.height - 2),
                        cv::Point(text_pos.x + text_size.width + 2, text_pos.y + 2),
                        cv::Scalar(0, 0, 0), -1);
            
            cv::putText(frame, info, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2);

            // Draw trajectory with fading effect
            if (obj.trajectory.size() > 1) {
                for (size_t i = 1; i < obj.trajectory.size(); i++) {
                    float alpha = static_cast<float>(i) / obj.trajectory.size();
                    cv::Scalar traj_color_fade = TRAJ_COLOR * alpha;
                    
                    cv::line(frame, obj.trajectory[i-1], obj.trajectory[i],
                            traj_color_fade, 1, cv::LINE_AA);

                    if (i % 5 == 0) {
                        cv::circle(frame, obj.trajectory[i], 2, traj_color_fade, -1, cv::LINE_AA);
                    }
                }
            }

            // Draw status info
            std::string status_info = "Lost: " + std::to_string(obj.lost_count);
            cv::Point status_pos(obj.last_box.x, obj.last_box.y + obj.last_box.height + 15);
            
            text_size = cv::getTextSize(status_info, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, nullptr);
            cv::rectangle(frame, 
                        cv::Point(status_pos.x - 2, status_pos.y - text_size.height - 2),
                        cv::Point(status_pos.x + text_size.width + 2, status_pos.y + 2),
                        cv::Scalar(0, 0, 0), -1);
            
            cv::putText(frame, status_info, status_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1);
        }

        // Draw statistics
        std::string stats = "Tracking " + std::to_string(tracked_objects.size()) + 
                        " objects | Exits: " + std::to_string(exit_history.size());
        cv::putText(frame, stats, cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }

        void clearAll() {
            tracked_objects.clear();
            object_gallery.clear();
            exit_history.clear();
            next_id = 0;
        }

        int getActiveTrackers() const {
            return std::count_if(tracked_objects.begin(), tracked_objects.end(),
                [](const TrackerInfo& info) { return info.is_active; });
        }

        std::vector<cv::Rect> getExitLocations() const {
            return exit_history;
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
    const std::string rtsp_url = "rtsp://192.168.1.74:8554/webCamStream";
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