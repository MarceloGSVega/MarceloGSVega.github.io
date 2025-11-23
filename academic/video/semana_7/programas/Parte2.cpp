#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;


int main (int argc, char* argv[])
{
    //entrada: argv[1] se fornecido, caso contrario usa a stream IP local (minha camera)
    std::string source = (argc > 1) ? argv[1] : "http://192.168.15.136:8080/video";

    VideoCapture cap;
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); //reduzir o buffer interno para melhor latencia
    cap.open(source, cv::CAP_ANY);
 


    if (!cap.isOpened()) 
    {
        cout << "Nao foi possivel abrir ou encontrar a webcam: " << argv[1] << endl;
        return -1;
    }
        //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 200;
    Ptr<SURF> detector = SURF::create( minHessian );
    
    Mat frame;
    std::vector<KeyPoint> keypoints;
    Mat img_keypoints;

    for (;;) 
    {
        cap.read(frame);
        cv::resize(frame, frame, Size(), 0.5, 0.5); //redimensionar para melhorar a performance
	    // Initialize a boolean to check if frames are there or not
        detector->detect(frame, keypoints);
        drawKeypoints(frame, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		imshow("Webcam", frame);
        imshow("Features", img_keypoints);
        
		int key = waitKey(1); // very small delay
        if (key == 27 || key == 'q' || key == 'Q') // ESC or q
            break;
    }
     cap.release();
    destroyAllWindows();
    return 0;
}