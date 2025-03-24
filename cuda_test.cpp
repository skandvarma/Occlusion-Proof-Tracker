#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

using namespace std;

void checkCudaSupport() {
    cout << "\n=== CUDA Support Check ===\n";
    
    try {
        // Check CUDA availability
        int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
        cout << "CUDA Device Count: " << deviceCount << endl;
        
        if (deviceCount < 1) {
            cout << "No CUDA devices found\n";
            return;
        }

        // Print information for each CUDA device
        for (int dev = 0; dev < deviceCount; dev++) {
            cv::cuda::DeviceInfo devInfo(dev);
            cout << "\nCUDA Device #" << dev << " Information:\n";
            cout << "Name: " << devInfo.name() << endl;
            cout << "Compute Capability: " << devInfo.majorVersion() 
                 << "." << devInfo.minorVersion() << endl;
            cout << "Total Memory: " << devInfo.totalMemory() / (1024*1024) << " MB" << endl;
            cout << "Multi-Processor Count: " << devInfo.multiProcessorCount() << endl;
            cout << "Async Engine Count: " << devInfo.asyncEngineCount() << endl;
            cout << "Concurrent Kernels: " << (devInfo.concurrentKernels() ? "Yes" : "No") << endl;
        }

        // Try to use CUDA
        // Create a test image
        cv::Mat cpuImg(1000, 1000, CV_8UC3, cv::Scalar(255, 0, 0));
        
        // Upload to GPU
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);
        
        // Perform some GPU operations
        cv::cuda::GpuMat gpuGray, gpuBlurred;
        cv::cuda::cvtColor(gpuImg, gpuGray, cv::COLOR_BGR2GRAY);
        
        // Create and apply Gaussian filter
        cv::Ptr<cv::cuda::createGaussianFilter> gaussian = 
            cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1.5);
        gaussian->apply(gpuGray, gpuBlurred);
        
        // Try some additional CUDA operations
        cv::cuda::GpuMat gpuEdges;
        cv::Ptr<cv::cuda::CannyEdgeDetector> canny = 
            cv::cuda::createCannyEdgeDetector(50.0, 100.0);
        canny->detect(gpuBlurred, gpuEdges);
        
        // Download results
        cv::Mat result;
        gpuEdges.download(result);
        
        cout << "\nSuccessfully performed CUDA operations!\n";
        cout << "Image processing pipeline completed:\n";
        cout << "1. Color to Grayscale conversion\n";
        cout << "2. Gaussian blur\n";
        cout << "3. Canny edge detection\n";

    } catch (const cv::Exception& e) {
        cerr << "\nOpenCV CUDA error: " << e.what() << endl;
    } catch (const std::exception& e) {
        cerr << "\nGeneral error: " << e.what() << endl;
    }
}

void printCudaBuildInformation() {
    cout << "\n=== OpenCV Build Information ===\n";
    cout << "OpenCV version: " << CV_VERSION << endl;
    
    // Get full build information
    string buildInfo = cv::getBuildInformation();
    
    // Extract and print CUDA-related information
    string searchStr = "CUDA";
    size_t pos = buildInfo.find(searchStr);
    if (pos != string::npos) {
        // Find the start of CUDA information
        size_t sectionStart = buildInfo.rfind("\n", pos) + 1;
        // Find the end of CUDA information (next double newline)
        size_t sectionEnd = buildInfo.find("\n\n", sectionStart);
        if (sectionEnd == string::npos) {
            sectionEnd = buildInfo.length();
        }
        
        // Extract and print CUDA section
        string cudaInfo = buildInfo.substr(sectionStart, sectionEnd - sectionStart);
        cout << "\nCUDA Build Information:\n" << cudaInfo << endl;
    } else {
        cout << "\nNo CUDA information found in build information.\n";
        cout << "OpenCV might have been built without CUDA support.\n";
    }
}

int main() {
    try {
        // Print build information first
        printCudaBuildInformation();
        
        // Then check CUDA support
        checkCudaSupport();
        
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}