#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cmath>
#include <map>
#include <fstream>





using namespace cv;
using namespace std;



// Custom Grayscale conversion function
Mat customConvertToGrayscale(const Mat& img) {
    Mat grayscaleImg(img.rows, img.cols, CV_8UC1);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // Take the average of the B, G, R values for each pixel
            grayscaleImg.at<uchar>(i, j) = (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2]) / 3;
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


// CUSTOM HARRIS CORNER DETECTION


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

    // Find the local maxima of the corner response function
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

Mat customCanny(const Mat& src, double lowThreshold, double highThreshold, int kernelSize = 3) {
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



    // Print the maximum and minimum values


    // 3. Apply non-maximum suppression
    nonMaxSuppressed = magnitude.clone();
    for (int y = 1; y < magnitude.rows - 1; ++y) {
        for (int x = 1; x < magnitude.cols - 1; ++x) {
            float current_angle = angle.at<float>(y, x);
            float current_magnitude = magnitude.at<float>(y, x);

            int x1 = x, y1 = y, x2 = x, y2 = y;
            if (current_angle >= 337.5 || current_angle < 22.5 || (current_angle >= 157.5 && current_angle < 202.5)) {
                x1 = x - 1;
                x2 = x + 1;
            }
            else if ((current_angle >= 22.5 && current_angle < 67.5) || (current_angle >= 202.5 && current_angle < 247.5)) {
                x1 = x + 1;
                y1 = y - 1;
                x2 = x - 1;
                y2 = y + 1;
            }
            else if ((current_angle >= 67.5 && current_angle < 112.5) || (current_angle >= 247.5 && current_angle < 292.5)) {
                y1 = y - 1;
                y2 = y + 1;
            }
            else {
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

    // 4. Apply double thresholding
    result = Mat::zeros(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            float value = nonMaxSuppressed.at<float>(y, x);
            if (value >= highThreshold) {
                result.at<uchar>(y, x) = 255;
            }
            else if (value >= lowThreshold) {
                result.at<uchar>(y, x) = 128;
            }
        }
    }

    // 5. Perform edge tracking by hysteresis
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            if (result.at<uchar>(y, x) == 128) {
                bool hasStrongNeighbor = false;
                for (int j = -1; j <= 1; ++j) {
                    for (int i = -1; i <= 1; ++i) {
                        if (result.at<uchar>(y + j, x + i) == 255) {
                            hasStrongNeighbor = true;
                            break;
                        }
                    }
                    if (hasStrongNeighbor) break;
                }
                result.at<uchar>(y, x) = hasStrongNeighbor ? 255 : 0;
            }
        }
    }



    return result;
}

// END

// CALLING ALL FUNCTIONS AND PROCESSES

vector<Point2f> Contour_Harris_Canny_Harris(const std::string& imagePath) {
    Mat img = imread(imagePath);

    // Define the size of the rectangular region around the center


    Mat img_gray = customConvertToGrayscale(img);

    int regionSize = 270;  

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



    int roiX = (centerX - 90) - regionSize / 2;
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

    Mat img_thresh = customThreshold(img_blurred, 30);

    //Mat grad_x, grad_y, mag, mag_8U;

    //grad_x = horizontal_gradient(img_thresholded);
    //grad_y = vertical_gradient(img_thresholded);
    //mag = magnitude(grad_x, grad_y);
    //mag_8U = magnitude_8U(mag);
    //Mat img_gray;





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
    cv::Mat subImage = img_contour(roi); // ??????? CONTOUR
    cv::Mat subImage_original = roi_original(roi); // ??????? ORIGINAL IMAGE
    cv::Mat subImage_gray = roiImage(roi); // ?????? GRAY IMAGE ???? ??????????? ????!!!




    // APPLY CUSTOM CORNER DETECTION FUNCTION

    int blockSize = 3;
    int apertureSize = 3;
    double k = 0.1;
    int threshold = 100;

    // Call custom Sobel function
    vector<Point2f> corners = detectCorners(subImage, blockSize, apertureSize, k, threshold);


    // APPLY CUSTOM CANNY EDGE DETECTION


    double lowThreshold = 50;
    double highThreshold = 150;
    //Mat custom_canny_result = customCanny(corners_image, lowThreshold, highThreshold);
    //custom_canny_result.setTo(Scalar(0), dilated_corners_mask == 0);
    //cv::imshow("Canny image", custom_canny_result);

    std::vector<cv::Rect> rois;
    std::vector<cv::Rect> expanded_rois;
    int radius = 10; // Define the radius of the area around the corner
    int border = 5;  // Define the border width

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



    // SAVE FINAL PRODUCT


    vector<Point2f> Corrected_coordinates = corners_canny;
    for (Point2f& corner : Corrected_coordinates) {
        corner.x += roiX + minX;
        corner.y += roiY + minY;
    }




    return Corrected_coordinates;
}




vector<Point2f> Contour_Harris(const std::string& imagePath) {
    Mat img = imread(imagePath);

    // Define the size of the rectangular region around the center


    Mat img_gray = customConvertToGrayscale(img);

    int regionSize = 270;  // Adjust this value according to your needs

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



    int roiX = (centerX - 90) - regionSize / 2;
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

    Mat img_thresh = customThreshold(img_blurred, 30);

    //Mat grad_x, grad_y, mag, mag_8U;

    //grad_x = horizontal_gradient(img_thresholded);
    //grad_y = vertical_gradient(img_thresholded);
    //mag = magnitude(grad_x, grad_y);
    //mag_8U = magnitude_8U(mag);
    //Mat img_gray;





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
    cv::Mat subImage = img_contour(roi); // ??????? CONTOUR
    cv::Mat subImage_original = roi_original(roi); // ??????? ORIGINAL IMAGE
    cv::Mat subImage_gray = roiImage(roi); // ?????? GRAY IMAGE ???? ??????????? ????!!!




    // APPLY CUSTOM CORNER DETECTION FUNCTION

    int blockSize = 3;
    int apertureSize = 3;
    double k = 0.1;
    int threshold = 100;

    // Call custom Sobel function
    vector<Point2f> corners = detectCorners(subImage, blockSize, apertureSize, k, threshold);


    // Draw circles at the detected corners




    vector<Point2f> Corrected_coordinates = corners;
    for (Point2f& corner : Corrected_coordinates) {
        corner.x += roiX + minX;
        corner.y += roiY + minY;
    }




    return corners;
}




// MAIN FUNCTION

int main(int argc, char** argv) {
    string directoryPath = "WRITE YOUR FOLDER PATH FOR IMAGES HERE";  // IMAGE FOLDER PATH

    vector<string> imagePaths;
    glob(directoryPath + "*.png", imagePaths, false);

    std::map<std::string, std::vector<Point2f>> imageCornerCoordinates;


    for (const auto& imagePath : imagePaths) {
        vector<Point2f> Corner_Coordinates = Contour_Harris(imagePath);

        imageCornerCoordinates[imagePath] = Corner_Coordinates;

    }

    std::ofstream outfile("corner_coordinates.txt");
    for (const auto& item : imageCornerCoordinates) {
        outfile << "Image: " << item.first << "\n";
        for (const auto& point : item.second) {
            outfile << point.x << " " << point.y << "\n";  // Assuming Point2f has x and y attributes
        }
        outfile << "\n";
    }
    outfile.close();


    return 0;
}

// END