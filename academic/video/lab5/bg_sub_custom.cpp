#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
using std::string;

struct Config {
    int min_brightness;
    int max_brightness;
    int min_area;
    double learning_rate;
    int bg_thresh;
    Size morph_kernel;
};

int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <video_path> <mode: fast|slow>" << endl;
        return 1;
    }

    string video_path = argv[1];
    string mode = argv[2];

    Config cfg;
    if (mode == "fast") {
        cfg.min_brightness = 30;
        cfg.max_brightness = 220;
        cfg.min_area = 800;
        cfg.learning_rate = 0.7; // fast adaptation
        cfg.bg_thresh = 900;
        cfg.morph_kernel = Size(3,3);
    } else if (mode == "slow") {
        cfg.min_brightness = 40;
        cfg.max_brightness = 180;
        cfg.min_area = 1200;
        cfg.learning_rate = 0.05; // slow adaptation
        cfg.bg_thresh = 300;
        cfg.morph_kernel = Size(7,7);
    } else {
        cerr << "Mode must be 'fast' or 'slow'" << endl;
        return 1;
    }

    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "❌ Cannot open video: " << video_path << endl;
        return 1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0) fps = 20.0;
    Size frame_size(frame_width, frame_height);

    // ensure outputs directory
    std::filesystem::create_directories("./outputs");
    string output_path = "./outputs/" + mode + "_output_video.avi";
    int fourcc = VideoWriter::fourcc('M','J','P','G');
    VideoWriter writer(output_path, fourcc, fps, frame_size);
    if (!writer.isOpened()) {
        cerr << "Could not open output writer: " << output_path << endl;
        // continue, we can still show windows
    }

    Ptr<BackgroundSubtractorKNN> backSub = createBackgroundSubtractorKNN(false);
    backSub->setDist2Threshold(cfg.bg_thresh);

    cout << "▶ Processing " << mode << " mode... Press 'q' to quit" << endl;

    Mat frame, gray, mask, gray_masked, fgmask;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cfg.morph_kernel);

    while (true) {
        if (!cap.read(frame) || frame.empty())
            break;

        // convert to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // brightness mask
        inRange(gray, Scalar(cfg.min_brightness), Scalar(cfg.max_brightness), mask);
        bitwise_and(gray, gray, gray_masked, mask);

        // apply background subtraction on masked grayscale
        backSub->apply(gray_masked, fgmask, cfg.learning_rate);

        // morphological cleanup (erode then dilate)
        erode(fgmask, fgmask, kernel, Point(-1,-1), 1);
        dilate(fgmask, fgmask, kernel, Point(-1,-1), 2);

        // find contours
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(fgmask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat result = frame.clone();
        for (const auto &cont : contours) {
            double area = contourArea(cont);
            if (area < cfg.min_area) continue;
            Rect r = boundingRect(cont);
            rectangle(result, r, Scalar(0,255,0), 2);
            putText(result, std::to_string(static_cast<int>(area)), Point(r.x, r.y-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
        }

        // write FG mask as 3-channel for video output (if writer opened)
        if (writer.isOpened()) {
            Mat fg_bgr;
            cvtColor(fgmask, fg_bgr, COLOR_GRAY2BGR);
            writer.write(fg_bgr);
        }

        // display
        imshow("Motion Detection (" + mode + ")", fgmask);
        imshow("Result (" + mode + ")", result);

        char c = static_cast<char>(waitKey(30));
        if (c == 'q' || c == 27)
            break;
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    destroyAllWindows();
    cout << "✅ Output saved to " << output_path << endl;

    return 0;
}
