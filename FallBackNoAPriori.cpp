#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cmath>




using namespace cv;
using namespace std;



// Custom Grayscale conversion function
Mat customConvertToGrayscale(const Mat& img) {
    Mat grayscaleImg(img.rows, img.cols, CV_8UC1);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // Take the average of the B, G, R values for each pixel
            grayscaleImg.at<uchar>(i, j) = 0.299 * img.at<Vec3b>(i, j)[2] + 0.587 * img.at<Vec3b>(i, j)[1] + 0.114 * img.at<Vec3b>(i, j)[0];
        }
    }
    return grayscaleImg;
}

// END

// Custom Gaussian blur function
Mat customGaussianBlur(const Mat& img) {
    Mat blurredImg(img.rows, img.cols, CV_8UC1);
    // Assume a 3x3 Gaussian kernel with sigma = 1.0
    float kernel[3][3] = {
        {1 / 16.0, 2 / 16.0, 1 / 16.0},
        {2 / 16.0, 4 / 16.0, 2 / 16.0},
        {1 / 16.0, 2 / 16.0, 1 / 16.0}
    };
    // Apply the kernel to each pixel
    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {
            float sum = 0.0;
            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    sum += img.at<uchar>(i + ki, j + kj) * kernel[ki + 1][kj + 1];
                }
            }
            blurredImg.at<uchar>(i, j) = sum;
        }
    }

    // Set the boundary pixels to black
    for (int i = 0; i < blurredImg.rows; ++i) {
        blurredImg.at<uchar>(i, 0) = 0;
        blurredImg.at<uchar>(i, blurredImg.cols - 1) = 0;
    }
    for (int j = 0; j < blurredImg.cols; ++j) {
        blurredImg.at<uchar>(0, j) = 0;
        blurredImg.at<uchar>(blurredImg.rows - 1, j) = 0;
    }
    return blurredImg;
}

// END


// Custom Threshold function
Mat customThreshold(const Mat& img, int threshold) {
    Mat thresholdImg(img.rows, img.cols, CV_8UC1);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            thresholdImg.at<uchar>(i, j) = img.at<uchar>(i, j) > threshold ? 255 : 0;
        }
    }
    return thresholdImg;
}

// END

// CONTOUR CUSTOM FUNCTION

struct MyPoint {
    int x, y;
    MyPoint(int _x, int _y) : x(_x), y(_y) {}
};

// Function to check if a point is within the image bounds


// Function to trace the outermost contour
std::vector<cv::Point> traceContours(const cv::Mat& binaryImg) {
    std::vector<cv::Point> contour;
    int direction = 0;
    int startX = -1;
    int startY = -1;

    // Find the starting point
    for (int y = 0; y < binaryImg.rows; ++y) {
        for (int x = 0; x < binaryImg.cols; ++x) {
            if (binaryImg.at<uchar>(y, x) == 255) {
                startX = x;
                startY = y;
                break;
            }
        }
        if (startX != -1) break;
    }

    if (startX == -1) return contour; // No object found

    // Moore-Neighbor tracing algorithm
    int currentX = startX, currentY = startY;
    int nextX, nextY;
    const int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[] = { 0, 1, 1, 1, 0, -1, -1, -1 };

    do {
        contour.push_back(cv::Point(currentX, currentY));
        direction = (direction + 6) % 8;

        for (int i = 0; i < 8; ++i) {
            nextX = currentX + dx[direction];
            nextY = currentY + dy[direction];

            if (binaryImg.at<uchar>(nextY, nextX) == 255) {
                currentX = nextX;
                currentY = nextY;
                break;
            }

            direction = (direction + 1) % 8;
        }
    } while (currentX != startX || currentY != startY);

    return contour;
}

// END

// CUSTOM HARRIS CORNER DETECTION

// Define Sobel operator kernels
float sobelX[3][3] = {
        {-3, 0, 3},
        {-10, 0, 10},
        {-3, 0, 3}
};
float sobelY[3][3] = {
    {-3, -10, -3},
        {0, 0, 0},
        {3, 10, 3}
};

// Custom Sobel function
Mat customSobel(const Mat& img, const float kernel[3][3]) {
    Mat result = Mat::zeros(img.rows, img.cols, CV_32F);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            float pixelSum = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    pixelSum += img.at<uchar>(y + ky, x + kx) * kernel[ky + 1][kx + 1];
                }
            }
            result.at<float>(y, x) = pixelSum;
        }
    }
    return result;
}

// Custom box filter function
Mat customBoxFilter(const Mat& img, int blockSize) {
    Mat result = Mat::zeros(img.rows, img.cols, CV_32F);
    int radius = blockSize / 2;
    for (int y = radius; y < img.rows - radius; ++y) {
        for (int x = radius; x < img.cols - radius; ++x) {
            float pixelSum = 0.0;
            for (int by = -radius; by <= radius; ++by) {
                for (int bx = -radius; bx <= radius; ++bx) {
                    pixelSum += img.at<float>(y + by, x + bx);
                }
            }
            result.at<float>(y, x) = pixelSum / (blockSize * blockSize);
        }
    }
    return result;
}


// CUSTOM SHI-THOMASI

Mat nonMaxSupp(Mat& img, int blocksize) {
    Mat NMS = Mat::zeros(img.size(), img.type());

    for (int j = blocksize; j < img.rows - blocksize; j++)
    {
        for (int i = blocksize; i < img.cols - blocksize; i++)
        {
            Rect rect(i - blocksize, j - blocksize, 2 * blocksize, 2 * blocksize);
            Mat roi = img(rect);
            double maxVal;
            minMaxLoc(roi, NULL, &maxVal, NULL, NULL);
            if (img.at<float>(j, i) == maxVal)
                NMS.at<float>(j, i) = img.at<float>(j, i);
        }


    }

    return NMS;
}

vector<Point2f>detectCorners(Mat& img, int blockSize, int apertureSize, double k, int threshold) {
    Mat dx = customSobel(img, sobelX);
    Mat dy = customSobel(img, sobelY);

    // Compute the elements of the structure tensor at each pixel
    Mat A = dx.mul(dx);
    Mat B = dx.mul(dy);
    Mat C = dy.mul(dy);

    // Call custom box filter function
    Mat sumA = customBoxFilter(A, blockSize);
    Mat sumB = customBoxFilter(B, blockSize);
    Mat sumC = customBoxFilter(C, blockSize);

    // Compute the determinant and trace of the structure tensor at each pixel
    Mat det = sumA.mul(sumC) - sumB.mul(sumB);
    Mat trace = sumA + sumC;

    // Compute the corner response function at each pixel
    Mat cornerResponse = det - k * trace.mul(trace);

    Mat suppressedResponse = nonMaxSupp(cornerResponse, blockSize);
    // Find the local maxima of the corner response function
    Mat localMaxima = suppressedResponse > threshold;

    // Find the indices of the local maxima
    vector<Point2f> corners;
    for (int i = 0; i < localMaxima.rows; i++) {
        for (int j = 0; j < localMaxima.cols; j++) {
            if (localMaxima.at<bool>(i, j)) {
                corners.push_back(Point2f(j, i));
            }
        }
    }

    return corners;


}

// END

// CUSTOM CARTTOPOLAR FUNCTION

void customCartToPolar(const Mat& x, const Mat& y, Mat& magnitude, Mat& angle) {
    magnitude = Mat::zeros(x.size(), CV_32F);
    angle = Mat::zeros(x.size(), CV_32F);
    for (int i = 0; i < x.rows; ++i) {
        for (int j = 0; j < x.cols; ++j) {
            float xVal = x.at<float>(i, j);
            float yVal = y.at<float>(i, j);
            magnitude.at<float>(i, j) = sqrt(xVal * xVal + yVal * yVal);
            angle.at<float>(i, j) = atan2(yVal, xVal) * 180.0 / CV_PI;
        }
    }
}

// END

// CUSTOM CANNY FUNCTION

Mat customCanny(const Mat& src, int lowThreshold, int highThreshold, int kernelSize = 3) {
    Mat blurred, dx, dy, magnitude, angle, nonMaxSuppressed, result;

    // 1. Apply Gaussian blur
    blurred = customGaussianBlur(src);

    // Define Sobel operator kernels
    float sobelX[3][3] = {
            {-3, 0, 3},
            {-10, 0, 10},
            {-3, 0, 3}
    };
    float sobelY[3][3] = {
        {-3, -10, -3},
            {0, 0, 0},
            {3, 10, 3}
    };

    // 2. Compute gradient magnitudes and directions using Sobel operators
    dx = customSobel(blurred, sobelX);
    dy = customSobel(blurred, sobelY);
    customCartToPolar(dx, dy, magnitude, angle);

    // 3. Apply non-maximum suppression
    nonMaxSuppressed = magnitude.clone();
    for (int y = 1; y < magnitude.rows - 1; ++y) {
        for (int x = 1; x < magnitude.cols - 1; ++x) {
            float current_angle = angle.at<float>(y, x);
            float current_magnitude = magnitude.at<float>(y, x);

            int x1 = x, y1 = y, x2 = x, y2 = y;

            // Normalize the current_angle to between -90 and 90 degrees
            if (current_angle < -90)
                current_angle += 180;

            // Assign directions (here, -90 and 90 degree for vertical edges, -45 degree for the diagonal, 0 degree for horizontal edges)
            if ((current_angle >= -90 && current_angle < -45) || (current_angle >= 45 && current_angle < 90)) {
                // Edge is vertical
                y1 = y - 1;
                y2 = y + 1;
            }
            else if ((current_angle >= -45 && current_angle < 0) || (current_angle >= 90 && current_angle < 135)) {
                // Edge is diagonal from bottom left to top right
                x1 = x - 1;
                y1 = y + 1;
                x2 = x + 1;
                y2 = y - 1;
            }
            else if ((current_angle >= 0 && current_angle < 45) || (current_angle >= -135 && current_angle < -90)) {
                // Edge is horizontal
                x1 = x - 1;
                x2 = x + 1;
            }
            else {
                // Edge is diagonal from top left to bottom right
                x1 = x - 1;
                y1 = y - 1;
                x2 = x + 1;
                y2 = y + 1;
            }

            if (current_magnitude <= magnitude.at<float>(y1, x1) || current_magnitude <= magnitude.at<float>(y2, x2)) {
                nonMaxSuppressed.at<float>(y, x) = 0;
            }
        }
    }

    // Apply double thresholding on the non-maximum suppressed image
    result = Mat::zeros(nonMaxSuppressed.size(), CV_8U);

    float maxMagnitude = 0;
    for (int y = 0; y < nonMaxSuppressed.rows; ++y) {
        for (int x = 0; x < nonMaxSuppressed.cols; ++x) {
            float magnitudeValue = nonMaxSuppressed.at<float>(y, x);
            if (magnitudeValue > maxMagnitude) {
                maxMagnitude = magnitudeValue;
            }
        }
    }

    float highThresholdValue = highThreshold;
    float lowThresholdValue = lowThreshold;

    for (int y = 0; y < nonMaxSuppressed.rows; ++y) {
        for (int x = 0; x < nonMaxSuppressed.cols; ++x) {
            float magnitudeValue = nonMaxSuppressed.at<float>(y, x);
            float angleValue = angle.at<float>(y, x);

            if (magnitudeValue >= highThresholdValue) {
                result.at<uchar>(y, x) = 255;  // Strong edge
            }
            else if (magnitudeValue >= lowThresholdValue && angleValue >= -22.5 && angleValue < 22.5) {
                result.at<uchar>(y, x) = 128;  // Weak edge
            }
        }
    }




    return result;
}


// READ, GRAYSCALE, CROP, BLUR, CONTOUR, CROP AROUND CONTOUR, HARRIS, CANNY, CORNERS ON CANNY

Mat processImage(const std::string& imagePath) {
    Mat img = imread(imagePath);

    // Define the size of the rectangular region around the center


    Mat img_gray = customConvertToGrayscale(img);

    int regionSize = 320;  // Adjust this value according to your needs

    // Define the rectangular region of interest

    int rows = img_gray.rows;
    int cols = img_gray.cols;

    // Calculate the total intensity and weighted sums
    double totalIntensity = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            // Get pixel intensity
            double intensity = static_cast<double>(img_gray.at<uchar>(y, x));

            // Accumulate total intensity and weighted sums
            totalIntensity += intensity;
            sumX += intensity * x;
            sumY += intensity * y;
        }
    }

    // Calculate the center of mass
    double centerX = sumX / totalIntensity;
    double centerY = sumY / totalIntensity;



    int roiX = (centerX - 120) - regionSize / 2;
    int roiY = (centerY + 90) - regionSize / 2;

    // Ensure the ROI boundaries are within the image dimensions
    roiX = std::max(0, roiX);
    roiY = std::max(0, roiY);

    // Calculate the actual size of the ROI based on the boundaries and image dimensions
    int roiWidth = std::min(regionSize, img.cols - roiX);
    int roiHeight = std::min(regionSize, img.rows - roiY);

    // Create a ROI by specifying a range of rows and columns in the image matrix
    Mat roiImage = img_gray(Range(roiY, roiY + roiHeight), Range(roiX, roiX + roiWidth));
    Mat roi_original = img(Range(roiY, roiY + roiHeight), Range(roiX, roiX + roiWidth));

    Mat img_blurred = customGaussianBlur(roiImage);

    Mat img_thresh = customThreshold(img_blurred, 20);

    




    std::vector<cv::Point> contour = traceContours(img_thresh);

    Mat img_contour = Mat::zeros(roiImage.size(), CV_8UC1);
    for (int i = 0; i < contour.size(); i++) {
        img_contour.at<uchar>(contour[i].y, contour[i].x) = 255;
    }





    // CROP CONTOUR IMAGE TO SEARCH FOR CORNERS ONLY THERE

    int minX = roiImage.cols, minY = roiImage.rows, maxX = 0, maxY = 0;
    for (const Point& Point : contour) {
        cv::Point point(Point.x, Point.y);
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
    }


    // Add some margin if needed
    int margin = 5; // Adjust this value based on the desired area around the contour
    minX = std::max(0, minX - margin);
    minY = std::max(0, minY - margin);
    maxX = std::min(roiImage.cols, maxX + margin);
    maxY = std::min(roiImage.rows, maxY + margin);

    cv::Rect roi(minX, minY, maxX - minX, maxY - minY);
    cv::Mat subImage = img_contour(roi); 
    cv::Mat subImage_original = roi_original(roi);
    cv::Mat subImage_gray = roiImage(roi); 




    // APPLY CUSTOM CORNER DETECTION FUNCTION

    int blockSize = 3;
    int apertureSize = 3;
    double k = 0.1;
    int threshold = 100;

    // Call custom Sobel function
    vector<Point2f> corners = detectCorners(subImage, blockSize, apertureSize, k, threshold);


    // Draw circles at the detected corners
    Mat Corner_img = subImage_original.clone();
    for (size_t i = 0; i < corners.size(); i++) {
        circle(Corner_img, corners[i], 0.5, Scalar(0, 255, 0), -1);
    }

    Mat img_corners = Mat::zeros(subImage.size(), CV_8UC1);
    for (int i = 0; i < corners.size(); i++) {
        img_corners.at<uchar>(corners[i].y, corners[i].x) = 255;
    }


    double lowThreshold = 50;
    double highThreshold = 150;
    

    std::vector<cv::Rect> rois;
    std::vector<cv::Rect> expanded_rois;
    int radius = 15; // Define the radius of the area around the corner
    int border = 2;  // Define the border width

    for (const auto& corner : corners) {
        // ROI without border around each corner
        int x = std::max(0, static_cast<int>(corner.x) - radius);
        int y = std::max(0, static_cast<int>(corner.y) - radius);
        int width = x + 2 * radius <= subImage_gray.cols ? 2 * radius : subImage_gray.cols - x;
        int height = y + 2 * radius <= subImage_gray.rows ? 2 * radius : subImage_gray.rows - y;
        rois.push_back(cv::Rect(x, y, width, height));

        // Expanded ROI with border για να φυγει το artifact που σου ελεγα χθες
        x = std::max(0, x - border);
        y = std::max(0, y - border);
        width = x + width + 2 * border <= subImage_gray.cols ? width + 2 * border : subImage_gray.cols - x;
        height = y + height + 2 * border <= subImage_gray.rows ? height + 2 * border : subImage_gray.rows - y;
        expanded_rois.push_back(cv::Rect(x, y, width, height));
    }

    Mat custom_canny_result = Mat::zeros(subImage_gray.size(), CV_8UC1);
    Mat custom_canny_result_2 = Mat::zeros(subImage_gray.size(), CV_8UC1);
    custom_canny_result_2 = customCanny(subImage_gray, lowThreshold, highThreshold);
    for (size_t i = 0; i < rois.size(); ++i) {
        Mat expanded_roi_img = subImage_gray(expanded_rois[i]);
        Mat canny_expanded_roi = customCanny(expanded_roi_img, lowThreshold, highThreshold);

        // Only use the pixels within the original ROI (without the border)
        Mat roi_result = custom_canny_result(rois[i]);
        Mat canny_roi = canny_expanded_roi(cv::Rect(border, border, rois[i].width, rois[i].height));
        canny_roi.copyTo(roi_result);
    }


    // REAPPLY THE CUSTOM CORNER FUNCTION ON THE CANNY IMAGE

    int blockSize_canny = 3;
    int apertureSize_canny = 3;
    float k_canny = 0.1;
    int threshold_canny = 100;

    vector<Point2f> corners_canny = detectCorners(custom_canny_result, blockSize_canny, apertureSize_canny, k_canny, threshold_canny);
    // Call custom Sobel function

    // Draw circles at the detected corners
    Mat Corner_img_canny = subImage_original.clone();
    for (size_t i = 0; i < corners_canny.size(); i++) {
        circle(Corner_img_canny, corners_canny[i], 2, Scalar(0, 255, 0), -1);
    }

    Mat corners_image_canny = Mat::zeros(subImage.size(), CV_8UC1);
    for (int i = 0; i < corners_canny.size(); i++) {
        corners_image_canny.at<uchar>(corners_canny[i].y, corners_canny[i].x) = 255;
    }



    vector<Point2f> Corrected_coordinates = corners_canny;
    for (Point2f& corner : Corrected_coordinates) {
        corner.x += roiX + minX;
        corner.y += roiY + minY;
    }
    Mat Original_img_canny = img.clone();
    for (size_t i = 0; i < Corrected_coordinates.size(); i++) {
        circle(Original_img_canny, Corrected_coordinates[i], 2, Scalar(0, 255, 0), -1);
    }

    Mat Corr_Coor_Canny = Mat::zeros(580, 752, CV_8UC1);
    for (int i = 0; i < corners_canny.size(); i++) {
        Corr_Coor_Canny.at<uchar>(Corrected_coordinates[i].y, Corrected_coordinates[i].x) = 255;
    }


    vector<Point2f> Corrected_coordinates_noCanny = corners;
    for (Point2f& corner1 : Corrected_coordinates_noCanny) {
        corner1.x += roiX + minX;
        corner1.y += roiY + minY;
    }
    Mat Original_img_canny_noCanny = img.clone();
    for (size_t i = 0; i < Corrected_coordinates_noCanny.size(); i++) {
        circle(Original_img_canny_noCanny, Corrected_coordinates_noCanny[i], 0.5, Scalar(0, 255, 0), -1);
    }

    Mat Corr_Coor_noCanny = Mat::zeros(580, 752, CV_8UC1);
    for (int i = 0; i < corners.size(); i++) {
        Corr_Coor_noCanny.at<uchar>(Corrected_coordinates_noCanny[i].y, Corrected_coordinates_noCanny[i].x) = 255;
    }

    std::cout << "Corrected Corner Coordinates (with Canny):\n";
    for (const auto& pt : Corrected_coordinates) {
        std::cout << "(" << pt.x << ", " << pt.y << ")\n";
    }


    imshow("Translated Coordinates", Original_img_canny);
    imshow("Corrected Coordinates", Corr_Coor_Canny);
    waitKey(0);


    return corners_image_canny;
}

// END







// MAIN FUNCTION

int main(int argc, char** argv) {
    string directoryPath = ".\\P3_nonCoop_DynamicTest_png\\";

    vector<string> imagePaths;
    glob(directoryPath + "*.png", imagePaths, false);

    for (const auto& imagePath : imagePaths) {
        Mat processedImage = processImage(imagePath);
    }




    return 0;
}
