
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace cv;
using namespace std;

int main()
{
    // 棋盘格内角点数量
    Size boardSize(11, 8);

    // 单个棋盘格边长，单位mm
    float squareSize = 50.0f;

   //1.提取图片
    vector<String> imageFiles;
    glob("D:/opencv/rmtest/calibration_images/calibration_images/image*.png",
        imageFiles, false);

    if (imageFiles.empty())
    {
        cout << "Error: No  camera images found." << endl;
        return -1;
    }
    cout << "Found images: " << imageFiles.size() << endl;

    //存储角点
    vector<Point3f> objectTemplate;

    for (int y = 0; y < boardSize.height; y++)
    {
        for (int x = 0; x < boardSize.width; x++)
        {
            objectTemplate.push_back(
                Point3f(x * squareSize, y * squareSize, 0.0f)
            );
        }
    }
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;

    Size imageSize;
    int detectedCount = 0;

    for (size_t i = 0; i < imageFiles.size(); i++)
    {
        Mat image = imread(imageFiles[i]);

        if (image.empty())
        {
            cout << "Cannot load: " << imageFiles[i] << endl;
            continue;
        }


        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        //排除无关或残缺棋盘格
        if (imageSize.width == 0)
        {
            imageSize = gray.size();
        }

        if (gray.size() != imageSize)
        {
            cout << "Skipped image with different size: "
                << imageFiles[i] << endl;
            continue;
        }

        vector<Point2f> corners;

        //2.找角点

        bool found = findChessboardCorners(
            gray,
            boardSize,
            corners,
            CALIB_CB_ADAPTIVE_THRESH |
            CALIB_CB_NORMALIZE_IMAGE
        );

        if (!found)
        {
            cout << "Corners not found: "
                << imageFiles[i] << endl;
            continue;
        }


        //3.亚像素优化
        cornerSubPix(
            gray,
            corners,
            Size(11, 11),
            Size(-1, -1),
            TermCriteria(
                TermCriteria::EPS + TermCriteria::COUNT,
                30,
                0.001
            )
        );
        imagePoints.push_back(corners);
        objectPoints.push_back(objectTemplate);
        detectedCount++;

        cout << "Corners detected: "
            << imageFiles[i] << endl;



        //直观体现角点
        drawChessboardCorners(image, boardSize, corners, found);
        imshow("Calibration -  Camera", image);
        waitKey(300);
   
    destroyAllWindows();

    //防止符合要求的棋盘格过少
    cout << "\nValid calibration images: "
        << detectedCount << endl;

    if (detectedCount < 2)
    {
        cout << "Error: At least 2 valid images are required."
            << endl;
        return -1;
    }
    if (detectedCount < 10)
    {
        cout << "Warning: Fewer than 10 images. "
            << "Results are preliminary." << endl;
    }




    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<Mat> rvecs, tvecs;

    double rms = calibrateCamera(
        objectPoints,
        imagePoints,
        imageSize,
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs
    );

    cout << fixed << setprecision(6);
    //输出关键相机标定参数
    cout << "\n===== Calibration Results =====" << endl;
    cout << "Image size: " << imageSize.width
        << " x " << imageSize.height << endl;

    cout << "RMS reprojection error: "
        << rms << endl;

    cout << "Camera matrix:" << endl;
    cout << cameraMatrix << endl;

    cout << "Distortion coefficients:" << endl;
    cout << distCoeffs << endl;

    FileStorage fs(
        "D:/opencv/rmtest/camera.yaml",
        FileStorage::WRITE
    );

    if (!fs.isOpened())
    {
        cout << "Error: Cannot save calibration file." << endl;
        return -1;
    }

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "rms_reprojection_error" << rms;

    fs.release();

    cout << "\nCalibration results saved to:" << endl;
    cout << "D:/opencv/rm/camera.yaml" << endl;






    // 验证去畸变效果
    Mat testImage = imread(
        "D:/opencv/rmtest/calibration_images/image_000002"
    );

    if (!testImage.empty())
    {
        Mat undistorted;

        undistort(
            testImage,
            undistorted,
            cameraMatrix,
            distCoeffs
        );

        imshow("Original Image", testImage);
        imshow("Undistorted Image", undistorted);

        waitKey(0);
        destroyAllWindows();
    }



    return 0;
}
