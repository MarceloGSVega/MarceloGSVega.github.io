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
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <image_path>" << endl;
        return -1;
    } //carrega a imagem
    Mat src;
    src = imread (argv[1], IMREAD_COLOR);
    if (!src.data) {
        cout << "Nao foi possivel abrir ou encontrar a imagem: " << argv[1] << endl;
        return -1;
    }
        //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints;
    detector->detect( src, keypoints );
 
    //-- Draw keypoints
    Mat img_keypoints;
    drawKeypoints( src, keypoints, img_keypoints );

    
    std::string filename = "SURF_" + std::string(argv[1]) + ".png";
    cv::imwrite(filename, img_keypoints);
    std::cout << "Imagem salva como " << filename << std::endl;
}
