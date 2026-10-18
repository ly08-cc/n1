#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
using namespace std;
using namespace cv;
const int INPUT_MODE = 1; 
const string VIDEO_PATH = "D:/opencv/rmtest/video.avi"; 
const string CAMERA_FILE = "D:/opencv/rmtest/camera.yaml";
const string MASK_OUTPUT_PATH = "blue_mask.png";
const string VIDEO_OUTPUT_PATH = "armor_detection_output.avi";

const int INPUT_GEOMETRY_MODE = 1;

const float SMALL_ARMOR_WIDTH_MM = 135.0f;
const float SMALL_ARMOR_HEIGHT_MM = 55.0f;

const int CALIBRATION_WIDTH_FALLBACK = 1280;
const int CALIBRATION_HEIGHT_FALLBACK = 1024;

// 灯条信息
struct Light
{
    Rect box;                 // 轴对齐外接框，用于显示
    Point2f center;           // 灯条中心
    double length;            // 灯条长边长度
    RotatedRect rotatedBox;   // 灯条旋转矩形，用于估计四个角点
};

struct PoseResult
{
    bool ok = false;
    Mat rvec;
    Mat tvec;
    double distanceMm = 0.0;
    double zMm = 0.0;
    double reprojectionRms = 0.0;
};

// 将米格式化为保留两位小数的字符串，避免 cv::format 与 std::string 拼接时的重载歧义。
string formatMeters(double meters)
{
    ostringstream stream;
    stream << fixed << setprecision(2) << meters << "m";
    return stream.str();
}



//2.使用相机参数
bool loadCameraCalibration(Mat& cameraMatrix, Mat& distCoeffs,
    int& calibrationWidth, int& calibrationHeight)
{
    calibrationWidth = 0;
    calibrationHeight = 0;

    FileStorage fs(CAMERA_FILE, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "PnP disabled: cannot open calibration file: "
            << CAMERA_FILE << endl;
        return false;
    }




    // 兼容常见字段名。
    fs["image_width"] >> calibrationWidth;
    fs["image_height"] >> calibrationHeight;

    fs["camera_matrix"] >> cameraMatrix;
    if (cameraMatrix.empty()) fs["cameraMatrix"] >> cameraMatrix;
    if (cameraMatrix.empty()) fs["K"] >> cameraMatrix;

    fs["distortion_coefficients"] >> distCoeffs;
    if (distCoeffs.empty()) fs["distCoeffs"] >> distCoeffs;
    if (distCoeffs.empty()) fs["D"] >> distCoeffs;
    fs.release();

    if (cameraMatrix.empty() || cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
    {
        cerr << "PnP disabled: camera matrix missing or not 3x3." << endl;
        return false;
    }

    if (calibrationWidth <= 0 || calibrationHeight <= 0)
    {
        calibrationWidth = CALIBRATION_WIDTH_FALLBACK;
        calibrationHeight = CALIBRATION_HEIGHT_FALLBACK;
        cerr << "Warning: image size not found in YAML; using known fallback "
            << calibrationWidth << "x" << calibrationHeight << "." << endl;
    }

    if (distCoeffs.empty())
    {
        cerr << "PnP disabled: distortion coefficients are missing." << endl;
        return false;
    }

    cameraMatrix.convertTo(cameraMatrix, CV_64F);
    distCoeffs = distCoeffs.reshape(1, 1);
    distCoeffs.convertTo(distCoeffs, CV_64F);

    if (!std::isfinite(cameraMatrix.at<double>(0, 0)) ||
        !std::isfinite(cameraMatrix.at<double>(1, 1)) ||
        !std::isfinite(cameraMatrix.at<double>(0, 2)) ||
        !std::isfinite(cameraMatrix.at<double>(1, 2)) ||
        cameraMatrix.at<double>(0, 0) <= 0.0 ||
        cameraMatrix.at<double>(1, 1) <= 0.0)
    {
        cerr << "PnP disabled: invalid camera matrix values." << endl;
        return false;
    }








    //图片数据和相机内参
    cout << "Calibration loaded: " << CAMERA_FILE << endl;
    cout << "Calibration image size: " << calibrationWidth << " x "
        << calibrationHeight << endl;
    cout << "K: fx=" << cameraMatrix.at<double>(0, 0)
        << ", fy=" << cameraMatrix.at<double>(1, 1)
        << ", cx=" << cameraMatrix.at<double>(0, 2)
        << ", cy=" << cameraMatrix.at<double>(1, 2) << endl;
    return true;
}

//内参使用
bool buildInputCameraMatrix(const Mat& baseK,
    int calibrationWidth, int calibrationHeight,
    const Size& inputSize,
    Mat& inputK, bool& metricPoseAllowed)
{
    inputK = baseK.clone();
    metricPoseAllowed = false;

    if (inputSize.width <= 0 || inputSize.height <= 0 ||
        calibrationWidth <= 0 || calibrationHeight <= 0)
    {
        cerr << "false" << endl;
        return false;
    }

    double sx = 1.0;
    double sy = 1.0;

   
   
        sx = static_cast<double>(inputSize.width) / calibrationWidth;
        sy = static_cast<double>(inputSize.height) / calibrationHeight;
        metricPoseAllowed = true;
    
        inputK.at<double>(0, 2) *= sx; // cx
        inputK.at<double>(1, 2) *= sy; // cy
    

    return true;
}


//筛选蓝色
Mat createBlueMask(const Mat& image)
{
    Mat hsv, mask;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(100, 70, 110), Scalar(130, 255, 240), mask);
    return mask;

}


//筛选灯条
vector<Light> detectBlueLights(const Mat& mask, Mat& display)
{
    vector<vector<Point> > contours;
    vector<Light> lights;
    Mat contourInput = mask.clone();
    findContours(contourInput, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // 调试：用红框显示所有提取到的外部轮廓，检查是否被后续条件过滤
    for (const auto& c : contours)
    {
        if (c.size() < 3)
            continue;

        // 获取当前轮廓的外接矩形
        cv::Rect r = cv::boundingRect(c);
    }


    for (const auto& contour : contours)
    {
        if (contour.size() < 3)
            continue;

        double area = contourArea(contour);
        RotatedRect rr = minAreaRect(contour);

        double w = rr.size.width;
        double h = rr.size.height;
        double length = max(w, h);
        double width = min(w, h);

        if (width < 1.0)
            continue;
        double ratio = length / width;

        double angle = rr.angle;
        if (w < h)
            angle += 90.0;
        angle = fmod(angle + 180.0, 180.0);

        if (abs(angle - 90.0) > 25.0)
            continue;

        Rect box = rr.boundingRect();
        Light light;
        light.box = box;
        light.center = rr.center;
        light.length = length;
        light.rotatedBox = rr;
        lights.push_back(light);

        rectangle(display, box, Scalar(0, 255, 0), 2);
    }

    return lights;
}

//分上下端
void getLightEndCorners(const RotatedRect& rotatedBox,
    Point2f& top1, Point2f& top2,
    Point2f& bottom1, Point2f& bottom2)
{
    Point2f corners[4];
    rotatedBox.points(corners);
    vector<Point2f> pts(corners, corners + 4);
    sort(pts.begin(), pts.end(),
        [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

    top1 = pts[0];
    top2 = pts[1];
    bottom1 = pts[2];
    bottom2 = pts[3];
}

PoseResult estimateArmorPose(const Light& first, const Light& second,
    float armorWidthMm, float armorHeightMm,
    const Mat& cameraMatrix, const Mat& distCoeffs)
{
    PoseResult result;
    if (armorWidthMm <= 0.0f || armorHeightMm <= 0.0f)
        return result;

    const Light* leftLight = &first;
    const Light* rightLight = &second;
    if (first.center.x > second.center.x)
    {
        leftLight = &second;
        rightLight = &first;
    }

    Point2f lt1, lt2, lb1, lb2;
    Point2f rt1, rt2, rb1, rb2;
    getLightEndCorners(leftLight->rotatedBox, lt1, lt2, lb1, lb2);
    getLightEndCorners(rightLight->rotatedBox, rt1, rt2, rb1, rb2);

    //
    Point2f leftTop = (lt1.x < lt2.x) ? lt1 : lt2;
    Point2f leftBottom = (lb1.x < lb2.x) ? lb1 : lb2;
    Point2f rightTop = (rt1.x > rt2.x) ? rt1 : rt2;
    Point2f rightBottom = (rb1.x > rb2.x) ? rb1 : rb2;

    vector<Point2f> imagePoints;
    imagePoints.push_back(leftTop);       // 左上
    imagePoints.push_back(rightTop);      // 右上
    imagePoints.push_back(rightBottom);   // 右下
    imagePoints.push_back(leftBottom);    // 左下



    // 检查四点是否形成有效的图像区域。
    double topWidth = norm(rightTop - leftTop);
    double bottomWidth = norm(rightBottom - leftBottom);
    double leftHeight = norm(leftBottom - leftTop);
    double rightHeight = norm(rightBottom - rightTop);
    if (topWidth < 2.0 || bottomWidth < 2.0 || leftHeight < 2.0 || rightHeight < 2.0)
        return result;

    // 三维点与图像点顺序一致，坐标单位为毫米。
    vector<Point3f> objectPoints;
    objectPoints.push_back(Point3f(-armorWidthMm / 2.0f, -armorHeightMm / 2.0f, 0.0f));
    objectPoints.push_back(Point3f(armorWidthMm / 2.0f, -armorHeightMm / 2.0f, 0.0f));
    objectPoints.push_back(Point3f(armorWidthMm / 2.0f, armorHeightMm / 2.0f, 0.0f));
    objectPoints.push_back(Point3f(-armorWidthMm / 2.0f, armorHeightMm / 2.0f, 0.0f));

    bool solved = solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
        result.rvec, result.tvec, false, SOLVEPNP_ITERATIVE);
    if (!solved || result.rvec.empty() || result.tvec.empty())
        return result;

    result.rvec.convertTo(result.rvec, CV_64F);
    result.tvec.convertTo(result.tvec, CV_64F);

    Mat rotation;
    Rodrigues(result.rvec, rotation);





    // 检查四个模型点在相机坐标系下是否全部位于相机前方。
    for (const Point3f& p : objectPoints)
    {
        double z = rotation.at<double>(2, 0) * p.x +
            rotation.at<double>(2, 1) * p.y +
            rotation.at<double>(2, 2) * p.z +
            result.tvec.at<double>(2, 0);
        if (!std::isfinite(z) || z <= 0.0)
            return result;
    }

    // 计算重投影 RMS，作为角点/模型匹配质量的参考。
    
    vector<Point2f> projectedPoints;
    projectPoints(objectPoints, result.rvec, result.tvec,
        cameraMatrix, distCoeffs, projectedPoints);
    double sumSquaredError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i)
    {
        Point2f diff = projectedPoints[i] - imagePoints[i];
        sumSquaredError += diff.dot(diff);
    }
    result.reprojectionRms = sqrt(sumSquaredError / imagePoints.size());
    result.distanceMm = norm(result.tvec);
    result.zMm = result.tvec.at<double>(2, 0);

    result.ok = std::isfinite(result.distanceMm) &&
        std::isfinite(result.zMm) &&
        std::isfinite(result.reprojectionRms) &&
        result.distanceMm > 0.0;
    return result;
}

// 输出pnp结果
void printPoseResult(const string& label, const PoseResult& pose,
    bool metricPoseAllowed, const Point& center,
    Mat& display)
{
    if (!pose.ok)
    {
        cout << label << " PnP failed (corner order/geometry may be unsuitable)." << endl;
        return;
    }

    const string validity = metricPoseAllowed ? "METRIC-CANDIDATE" : "no";
    cout << label << " [" << validity << "] translation mm: x="
        << pose.tvec.at<double>(0, 0)
        << ", y=" << pose.tvec.at<double>(1, 0)
        << ", z=" << pose.zMm << endl;
    cout << label << " [" << validity << "] distance="
        << pose.distanceMm / 1000.0
        << "m, reprojection RMS px=" << pose.reprojectionRms << endl;

    if (pose.reprojectionRms > 8.0)
        cout << "Warning: reprojection RMS is high; inspect detected light-bar corners." << endl;

    // 
    string text = metricPoseAllowed ? label + " Z=" : label + " DEMO Z=";
    text += formatMeters(pose.zMm / 1000.0);
    putText(display, text, Point(center.x + 8, center.y + 18),
        FONT_HERSHEY_SIMPLEX, 0.48,
        metricPoseAllowed ? Scalar(0, 255, 255) : Scalar(0, 165, 255),
        1, LINE_AA);
}






//视频
Mat processFrame(const Mat& frame, const Mat& baseK, const Mat& distCoeffs,
    bool calibrationReady, int calibrationWidth, int calibrationHeight,
    bool logDetails, int frameIndex)
{
    Mat display = frame.clone();

    Mat mask = createBlueMask(frame);

    //二值图
   
    
        imshow("Blue Mask - Debug", mask);
    

    vector<Light> lights = detectBlueLights(mask, display);


    if (logDetails)
    {
        cout << "Frame " << frameIndex << ": image size "
            << frame.cols << " x " << frame.rows << endl;
        cout << "Valid lights: " << lights.size() << endl;
    }

    Mat inputK;
    bool metricPoseAllowed = false;
    bool pnpReady = false;
    if (calibrationReady)
    {
        pnpReady = buildInputCameraMatrix(baseK, calibrationWidth, calibrationHeight,
            frame.size(), inputK, metricPoseAllowed);
    }




    //灯条匹配
    int matchCount = 0;
    for (int i = 0; i < static_cast<int>(lights.size()); ++i)
    {
        for (int j = i + 1; j < static_cast<int>(lights.size()); ++j)
        {
            const Light& a = lights[i];
            const Light& b = lights[j];
            double avgH = (a.length + b.length) / 2.0;
           // if (avgH <= 0.0)
            //    continue;

            double heightRatio = min(a.length, b.length) / max(a.length, b.length);
            double yDiff = abs(a.center.y - b.center.y);
            double xDiff = abs(a.center.x - b.center.x);

            if (heightRatio > 0.50&&
                yDiff / avgH < 0.7 &&
                xDiff / avgH > 5.0 &&
                xDiff / avgH <6.5)
            {
                Rect armorBox = a.box | b.box;
                Point center(armorBox.x + armorBox.width / 2,
                    armorBox.y + armorBox.height / 2);

                rectangle(display, armorBox, Scalar(0, 0, 255), 2);
                circle(display, center, 5, Scalar(0, 255, 255), FILLED);
                ++matchCount;

                if (logDetails)
                    cout << "Armor center: " << center << endl;



                if (pnpReady)
                {
                    PoseResult smallPose = estimateArmorPose(
                        a, b, SMALL_ARMOR_WIDTH_MM, SMALL_ARMOR_HEIGHT_MM,
                        inputK, distCoeffs);
                    

                    if (logDetails)
                    {
                        printPoseResult("SMALL 135x55mm", smallPose,
                            metricPoseAllowed, center, display);
                        
                    }
                    else
                    {
                        
                        if (smallPose.ok)
                        {
                            string label = metricPoseAllowed ? "S Z=" : "S DEMO Z=";
                            label += formatMeters(smallPose.zMm / 1000.0);
                            putText(display, label, Point(center.x + 8, center.y + 18),
                                FONT_HERSHEY_SIMPLEX, 0.45,
                                metricPoseAllowed ? Scalar(0, 255, 255) : Scalar(0, 165, 255),
                                1, LINE_AA);
                        }
                        
                    }
                }
                else if (calibrationReady)
                {
                    putText(display, "PnP unavailable: geometry/calibration",
                        Point(center.x + 8, center.y + 18),
                        FONT_HERSHEY_SIMPLEX, 0.42, Scalar(0, 165, 255), 1, LINE_AA);
                }
            }
        }
    }

    if (logDetails)
        cout << "Matched pairs: " << matchCount << endl;


    return display;
}


int main()
{
    Mat baseK, distCoeffs;
    int calibrationWidth = CALIBRATION_WIDTH_FALLBACK;
    int calibrationHeight = CALIBRATION_HEIGHT_FALLBACK;
    bool calibrationReady = loadCameraCalibration(
        baseK, distCoeffs, calibrationWidth, calibrationHeight);


    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened())
    {
        cerr << "Video open failed. Check VIDEO_PATH: " << VIDEO_PATH << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps < 1.0 || fps > 240.0)
        fps = 30.0;


    VideoWriter writer;
    bool writerAttempted = false;
    int frameIndex = 0;
    Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty())
            break;

        
        bool logDetails = (frameIndex % 30 == 0);
        Mat result = processFrame(frame, baseK, distCoeffs, calibrationReady,
            calibrationWidth, calibrationHeight,
            logDetails, frameIndex);

        if (!writerAttempted)
        {
            writerAttempted = true;
            writer.open(VIDEO_OUTPUT_PATH,
                VideoWriter::fourcc('M', 'J', 'P', 'G'),
                fps, result.size(), true);
            if (!writer.isOpened())
                cerr << "Warning: cannot create output video; live display will continue." << endl;
        }
        if (writer.isOpened())
            writer.write(result);

        imshow("RoboMaster Armor Detection", result);
        int key = waitKey(10);
        if (key == 27 )
            break;

        ++frameIndex;
    }

    cap.release();
    if (writer.isOpened())
        writer.release();
   
    cout << "Video processing finished."
        << VIDEO_OUTPUT_PATH << endl;
    return 0;
}
