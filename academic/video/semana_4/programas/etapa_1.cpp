#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    // Path to your image inside the "sources" folder
    string imagePath = "sources/grupo.png"; // adjust filename as needed

    // Output folder
    string outputDir = "output/etapa_1";
    fs::create_directories(outputDir); // create folder if it doesn't exist

    // Read the image
    Mat src = imread(imagePath, IMREAD_GRAYSCALE);
    if( src.empty() )
    {
        return EXIT_FAILURE;
    }

    vector<Mat> bgr_planes;
    split( src, bgr_planes );

    Mat dst;
    equalizeHist( src, dst );

    int histSize = 256;

    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };

    bool uniform = true, accumulate = false;

    Mat hist, hist_eq;
    calcHist( &bgr_planes[0], 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &dst, 1, 0, Mat(), hist_eq, 1, &histSize, histRange, uniform, accumulate );


    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histDst( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(hist_eq, hist_eq, 0, histDst.rows, NORM_MINMAX, -1, Mat() );

    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histDst, Point( bin_w*(i-1), hist_h - cvRound(hist_eq.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(hist_eq.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
    }

    imshow("Source image", src );
    imshow("Source Equalized", dst);
    imshow("calcHist Demo", histImage );
    imshow("calcHist Equalized", histDst);
    waitKey();

    // Save all images
    imwrite(outputDir + "/src.png", src);
    imwrite(outputDir + "/dst.png", dst);
    imwrite(outputDir + "/hist_original.png", histImage);
    imwrite(outputDir + "/hist_equalized.png", histDst);

    return EXIT_SUCCESS;
}
