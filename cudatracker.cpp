#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <functional>
#include <exception>
#include <fstream>
#include <unistd.h> // for getpagesize()
#include <sys/resource.h> // for getrusage
#include <numeric>   // For std::accumulate

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
    const int RTSP_BUFFER = 0;  // Minimal buffering
    std::queue<cv::Mat> frame_buffer;
    const size_t MAX_BUFFER_SIZE = 1;  // Keep buffer minimal

public:
    VideoCapture(const std::string& rtsp_url, cv::Size resize = cv::Size(600, 300), int skip_frames = 15) 
        : resize(resize), skip_frames(skip_frames) {
        try {
            // Configure RTSP stream for low latency
            cap.set(cv::CAP_PROP_BUFFERSIZE, RTSP_BUFFER);
            
            // Set additional stream properties before opening
            std::string pipeline = create_gstreamer_pipeline(rtsp_url);
            
            // Open stream with optimized GStreamer pipeline
            if (!cap.open(pipeline, cv::CAP_GSTREAMER)) {
                // Fallback to regular RTSP if GStreamer fails
                if (!cap.open(rtsp_url)) {
                    throw std::runtime_error("Error opening RTSP stream: " + rtsp_url);
                }
            }

            // Configure stream settings after opening
            cap.set(cv::CAP_PROP_BUFFERSIZE, RTSP_BUFFER);
            cap.set(cv::CAP_PROP_FPS, 60);  // Request maximum FPS
            
            // Print stream information
            std::cout << "RTSP Stream opened successfully" << std::endl;
            std::cout << "Frame Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
            std::cout << "Frame Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
            std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error initializing video capture: " << e.what() << std::endl;
            throw;
        }
    }

private:
    std::string create_gstreamer_pipeline(const std::string& rtsp_url) {
        // Create an optimized GStreamer pipeline for low latency
        return "rtspsrc location=" + rtsp_url + " latency=0 buffer-mode=auto ! "
               "rtph264depay ! h264parse ! "
               "avdec_h264 max-threads=8 ! "  // Use hardware acceleration if available
               "videoconvert ! "
               "appsink max-buffers=1 drop=true sync=false"; // Minimize buffering
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
        cv::Mat temp_frame;

        // Clear old frames
        while (!frame_buffer.empty()) {
            frame_buffer.pop();
        }

        // Skip frames using grab() for efficiency
        for (int i = 0; i < skip_frames; ++i) {
            ret = cap.grab();
            if (!ret) break;
        }

        // Read the latest frame
        if (ret) {
            ret = cap.retrieve(temp_frame);
            if (ret && !temp_frame.empty()) {
                if (resize.width > 0 && resize.height > 0) {
                    // Use CUDA for resize if available
                    if (USE_GPU) {
                        cv::cuda::GpuMat gpu_frame, gpu_resized;
                        gpu_frame.upload(temp_frame);
                        cv::resize(gpu_frame, gpu_resized, resize);
                        gpu_resized.download(frame);
                    } else {
                        cv::resize(temp_frame, frame, resize, 0, 0, cv::INTER_NEAREST);
                    }
                } else {
                    frame = temp_frame;
                }
            }
        }

        return ret;
    }

    void release() {
        cap.release();
        // Clear buffer
        while (!frame_buffer.empty()) {
            frame_buffer.pop();
        }
    }
};


class FeatureTracker {
private:
    cv::Ptr<cv::AKAZE> feature_detector;
    cv::FlannBasedMatcher flann_matcher;
    const float match_ratio = 0.7f;
    const int min_matches = 10;

public:
    FeatureTracker() {
        feature_detector = cv::AKAZE::create(
            cv::AKAZE::DESCRIPTOR_MLDB,
            0,
            4,
            0.0001f,
            4,
            0,
            cv::KAZE::DIFF_PM_G2
        );

        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
        cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
        flann_matcher = cv::FlannBasedMatcher(indexParams, searchParams);
    }

    struct TrackingResult {
        bool success;
        cv::Rect2d updated_bbox;
        double confidence;
        std::vector<cv::Point2f> matched_points;
    };

    TrackingResult track(const cv::Mat& prev_frame, const cv::Mat& curr_frame, const cv::Rect2d& prev_bbox) {
        TrackingResult result{false, prev_bbox, 0.0, {}};

        try {
            cv::Mat prev_gray, curr_gray;
            cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

            cv::Mat prev_roi = prev_gray(prev_bbox);

            std::vector<cv::KeyPoint> prev_keypoints, curr_keypoints;
            cv::Mat prev_descriptors, curr_descriptors;

            feature_detector->detectAndCompute(prev_roi, cv::Mat(), prev_keypoints, prev_descriptors);
            feature_detector->detectAndCompute(curr_gray, cv::Mat(), curr_keypoints, curr_descriptors);

            if (prev_keypoints.empty() || curr_keypoints.empty()) {
                return result;
            }

            std::vector<std::vector<cv::DMatch>> knn_matches;
            flann_matcher.knnMatch(prev_descriptors, curr_descriptors, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            std::vector<cv::Point2f> prev_points, curr_points;

            for (const auto& match_pair : knn_matches) {
                if (match_pair.size() < 2) continue;
                
                if (match_pair[0].distance < match_ratio * match_pair[1].distance) {
                    good_matches.push_back(match_pair[0]);

                    cv::Point2f prev_pt = prev_keypoints[match_pair[0].queryIdx].pt;
                    prev_pt.x += prev_bbox.x;
                    prev_pt.y += prev_bbox.y;
                    
                    prev_points.push_back(prev_pt);
                    curr_points.push_back(curr_keypoints[match_pair[0].trainIdx].pt);
                }
            }

            if (good_matches.size() < min_matches) {
                return result;
            }

            std::vector<uchar> inlier_mask;
            cv::Mat H = cv::findHomography(prev_points, curr_points, cv::RANSAC, 3.0, inlier_mask);

            if (H.empty()) {
                return result;
            }

            std::vector<cv::Point2f> bbox_corners(4);
            bbox_corners[0] = cv::Point2f(prev_bbox.x, prev_bbox.y);
            bbox_corners[1] = cv::Point2f(prev_bbox.x + prev_bbox.width, prev_bbox.y);
            bbox_corners[2] = cv::Point2f(prev_bbox.x + prev_bbox.width, prev_bbox.y + prev_bbox.height);
            bbox_corners[3] = cv::Point2f(prev_bbox.x, prev_bbox.y + prev_bbox.height);

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

            result.success = true;
            result.updated_bbox = cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y);
            result.confidence = static_cast<double>(cv::countNonZero(inlier_mask)) / inlier_mask.size();
            result.matched_points = curr_points;

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in feature tracking: " << e.what() << std::endl;
        }

        return result;
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
    const double LOST_THRESHOLD = 0.15;
    const int MAX_LOST_FRAMES = 10;
    const float OVERLAP_THRESHOLD = 0.3f;
    const double SIMILARITY_THRESHOLD = 0.15;

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
        kalman.transitionMatrix.at<float>(0,4) = 0.5f;  // x += dx * 0.5
        kalman.transitionMatrix.at<float>(1,5) = 0.5f;  // y += dy * 0.5

        // Process noise
        kalman.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.01f;
        kalman.processNoiseCov.at<float>(4,4) *= 0.5f;  // Reduce noise for velocity x
        kalman.processNoiseCov.at<float>(5,5) *= 0.5f;  // Reduce noise for velocity y
        
        // Measurement noise
        kalman.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.3f;
        
        // Error covariance matrices
        kalman.errorCovPost = cv::Mat::eye(6, 6, CV_32F) * 0.2f;
        kalman.errorCovPre = cv::Mat::eye(6, 6, CV_32F) * 0.1f;
        
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
        
        // Detect overlapping boxes
        for (size_t i = 0; i < tracked_objects.size(); i++) {
            for (size_t j = i + 1; j < tracked_objects.size(); j++) {
                if (boxes_overlap(tracked_objects[i].last_box, tracked_objects[j].last_box)) {
                    float iou = calculate_iou(tracked_objects[i].last_box, tracked_objects[j].last_box);
                    if (iou > OVERLAP_THRESHOLD) {
                        overlaps.push_back({i, j});
                    }
                }
            }
        }

        // Handle each overlapping pair
        for (const auto& pair : overlaps) {
            auto& obj1 = tracked_objects[pair.first];
            auto& obj2 = tracked_objects[pair.second];

            // Predict future positions
            cv::Mat pred1 = obj1.kalman.predict();
            cv::Mat pred2 = obj2.kalman.predict();

            cv::Rect pred_box1(
                pred1.at<float>(0) - pred1.at<float>(2)/2,
                pred1.at<float>(1) - pred1.at<float>(3)/2,
                pred1.at<float>(2),
                pred1.at<float>(3)
            );

            cv::Rect pred_box2(
                pred2.at<float>(0) - pred2.at<float>(2)/2,
                pred2.at<float>(1) - pred2.at<float>(3)/2,
                pred2.at<float>(2),
                pred2.at<float>(3)
            );

            // Calculate appearance similarity
            cv::Mat roi1 = frame(pred_box1);
            cv::Mat roi2 = frame(pred_box2);
            cv::Mat hist1 = calculateColorHistogram(roi1);
            cv::Mat hist2 = calculateColorHistogram(roi2);

            double sim1 = calculateHistSimilarity(obj1.hist, hist1);
            double sim2 = calculateHistSimilarity(obj2.hist, hist2);

            // Update based on better prediction
            if (sim1 > sim2) {
                obj1.last_box = pred_box1;
                obj2.lost_count++;
            } else {
                obj2.last_box = pred_box2;
                obj1.lost_count++;
            }
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
    ObjectTracker() : next_id(0) {
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
        
        feature_tracker = std::make_unique<FeatureTracker>();
    }

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
    // Check for CUDA availability
    bool use_cuda = false;
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            use_cuda = true;
            std::cout << "CUDA is available. Using GPU acceleration." << std::endl;
            cv::cuda::setDevice(0); // Use the first GPU
        }
    } catch (const cv::Exception& e) {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }

    // Set real-time scheduling priority if possible
    #ifdef __linux__
        struct sched_param schparam;
        schparam.sched_priority = sched_get_priority_max(SCHED_FIFO);
        if (sched_setscheduler(0, SCHED_FIFO, &schparam) == 0) {
            std::cout << "Set real-time scheduling priority" << std::endl;
        }
    #endif

    // RTSP stream settings
    const std::string rtsp_url = "rtsp://192.168.144.25:8554/main.264";
    const int SKIP_FRAMES = 1;
    const cv::Size FRAME_SIZE(640, 480);  

    try {
        // Initialize video capture with RTSP stream
        VideoCapture video_capture(rtsp_url, FRAME_SIZE, SKIP_FRAMES);

        if (!video_capture.isOpened()) {
            std::cerr << "Failed to open RTSP stream" << std::endl;
            return 1;
        }

        // Create display window with OpenGL support for faster rendering
        cv::namedWindow("Object Tracker", cv::WINDOW_AUTOSIZE);
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

        while (true) {
            auto frame_start = std::chrono::high_resolution_clock::now();

            if (!paused) {
                // Read frame with error handling
                bool read_success = video_capture.read(frame);
                
                if (!read_success || frame.empty()) {
                    std::cerr << "Failed to read frame from RTSP stream. Retrying..." << std::endl;
                    cv::waitKey(1); // Minimal wait before retry
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
            
            // Draw help text if enabled
            if (show_help) {
                Visualizer::draw_help_text(display_frame);
            }

            // Show frame
            cv::imshow("Object Tracker", display_frame);

            // Handle keyboard input
            char key = static_cast<char>(cv::waitKey(1));
            
            switch (key) {
                case 'q':
                case 27: // ESC
                    goto cleanup; // Exit the loop
                
                case 'p': // Pause/Unpause
                    paused = !paused;
                    std::cout << (paused ? "Paused" : "Resumed") << std::endl;
                    break;
                
                case 'h': // Toggle help text
                    show_help = !show_help;
                    break;
                
                case 'r': // Reset tracking
                    tracker.clearAll();
                    std::cout << "Tracking reset" << std::endl;
                    break;
                
                default:
                    visualizer.handle_key(key, tracker, frame);
                    break;
            }

            // Monitor memory usage
            #ifdef __linux__
                if (frame_count % 30 == 0) { // Check every 30 frames
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