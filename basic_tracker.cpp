#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Enhanced VideoCapture class for RTSP
class RTSPVideoCapture {
private:
    VideoCapture cap;
    Size resize;
    int skip_frames;
    const int RTSP_BUFFER = 0;  // Minimal buffering for low latency

public:
    RTSPVideoCapture(const string& rtsp_url, Size resize = Size(600, 300), int skip_frames = 1) 
        : resize(resize), skip_frames(skip_frames) {
        try {
            // Configure RTSP stream for low latency
            cap.set(CAP_PROP_BUFFERSIZE, RTSP_BUFFER);
            
            // Create optimized GStreamer pipeline
            string pipeline = "rtspsrc location=" + rtsp_url + " latency=0 ! "
                            "rtph264depay ! h264parse ! avdec_h264 ! "
                            "videoconvert ! appsink max-buffers=1 drop=true sync=false";
            
            // Try GStreamer pipeline first
            if (!cap.open(pipeline, CAP_GSTREAMER)) {
                // Fallback to direct RTSP if GStreamer fails
                if (!cap.open(rtsp_url)) {
                    throw runtime_error("Error opening RTSP stream: " + rtsp_url);
                }
            }

            // Configure stream settings
            cap.set(CAP_PROP_BUFFERSIZE, RTSP_BUFFER);
            cap.set(CAP_PROP_FPS, 30);  // Request maximum FPS
            
            cout << "RTSP Stream opened successfully" << endl;
            cout << "Frame Width: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
            cout << "Frame Height: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
            cout << "FPS: " << cap.get(CAP_PROP_FPS) << endl;

        } catch (const exception& e) {
            cerr << "Error initializing video capture: " << e.what() << endl;
            throw;
        }
    }

    bool read(Mat& frame) {
        if (!cap.isOpened()) {
            return false;
        }

        bool ret = true;
        Mat temp_frame;

        // Skip frames using grab() for efficiency
        for (int i = 0; i < skip_frames; ++i) {
            ret = cap.grab();
            if (!ret) break;
        }

        // Read the latest frame
        if (ret) {
            ret = cap.retrieve(frame);
            if (ret && !frame.empty() && resize.width > 0 && resize.height > 0) {
                cv::resize(frame, frame, resize, 0, 0, INTER_LINEAR);
            }
        }

        return ret;
    }

    void release() {
        cap.release();
    }

    bool isOpened() const {
        return cap.isOpened();
    }
};

// Global variables for ROI selection
Rect roi;
bool roiSelected = false;
Point startPoint;
Point endPoint;
bool selecting = false;

// Mouse callback function for ROI selection
void onMouse(int event, int x, int y, int flags, void* userdata)
{
    switch(event) {
        case EVENT_LBUTTONDOWN:
            selecting = true;
            roiSelected = false;
            startPoint = Point(x, y);
            endPoint = startPoint;
            break;
            
        case EVENT_MOUSEMOVE:
            if (selecting) {
                endPoint = Point(x, y);
            }
            break;
            
        case EVENT_LBUTTONUP:
            selecting = false;
            endPoint = Point(x, y);
            roi = Rect(
                min(startPoint.x, endPoint.x),
                min(startPoint.y, endPoint.y),
                abs(startPoint.x - endPoint.x),
                abs(startPoint.y - endPoint.y)
            );
            roiSelected = true;
            break;
    }
}

int main()
{
    // RTSP stream URL
    string rtsp_url = "rtsp://192.168.144.25:8554/main.264";
    
    try {
        // Initialize RTSP capture
        RTSPVideoCapture cap(rtsp_url, Size(600, 300), 1);

        // Create window and set mouse callback
        string windowName = "RTSP Tracker";
        namedWindow(windowName);
        setMouseCallback(windowName, onMouse);

        Mat frame;
        Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
        bool trackerInitialized = false;

        // Display instructions
        cout << "Instructions:" << endl;
        cout << "1. Draw ROI using mouse" << endl;
        cout << "2. Press SPACE to start tracking" << endl;
        cout << "3. Press 'r' to reset tracker" << endl;
        cout << "4. Press ESC to exit" << endl;

        while (true) {
            // Read frame with error handling
            bool ret = cap.read(frame);
            if (!ret || frame.empty()) {
                cerr << "Failed to read frame. Retrying..." << endl;
                waitKey(1000);  // Wait before retry
                continue;
            }

            // Draw ROI while selecting
            if (selecting) {
                rectangle(frame, startPoint, endPoint, Scalar(0, 255, 0), 2);
            }

            // Show selected ROI before tracking starts
            if (roiSelected && !trackerInitialized) {
                rectangle(frame, roi, Scalar(0, 255, 0), 2);
            }

            // Update tracking
            if (trackerInitialized) {
                Rect bbox = roi;
                bool ok = tracker->update(frame, bbox);
                
                if (ok) {
                    roi = bbox;
                    rectangle(frame, bbox, Scalar(255, 0, 0), 2);
                    
                    // Display tracking info
                    string txt = "Tracking: " + to_string(bbox.x) + "," + to_string(bbox.y);
                    putText(frame, txt, Point(10, 30), FONT_HERSHEY_SIMPLEX, 
                            0.7, Scalar(0, 255, 0), 2);
                } else {
                    putText(frame, "Tracking lost!", Point(10, 30), FONT_HERSHEY_SIMPLEX, 
                            0.7, Scalar(0, 0, 255), 2);
                }
            }

            // Show frame
            imshow(windowName, frame);

            // Handle keyboard input
            char key = (char)waitKey(1);
            switch(key) {
                case 27:  // ESC
                    goto cleanup;
                    
                case ' ':  // Space
                    if (roiSelected && !trackerInitialized) {
                        tracker->init(frame, roi);
                        trackerInitialized = true;
                    }
                    break;
                    
                case 'r':  // Reset
                    tracker = TrackerCSRT::create();
                    trackerInitialized = false;
                    roiSelected = false;
                    break;
            }
        }

cleanup:
        cap.release();
        destroyAllWindows();
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}