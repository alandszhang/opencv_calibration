#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <cv.hpp>

using namespace std;
using namespace cv;
const int imageWidth = 752;
const int imageHeight = 480;
const int boardWidth = 11;
//横向的角点数目
const int boardHeight = 8;
//纵向的角点数据
const int boardCorner = boardWidth * boardHeight;
//总的角点数据

//相机标定时需要采用的图像帧数
const int squareSize = 30.00;
//标定板黑白格子的大小单位mm

const int frameNumber = 14;
//图像命名 从1 ～ 58（59-1=58）
string folder_ = "/home/huhu/CLionProjects/biaoding/data/";
string format_R = "R";
string format_L = "L";
//例如： R1.jpg   L58.jpg 置于工程目录的 data文件夹下，
const Size boardSize = Size(boardWidth, boardHeight);
//标定板的总内角点
Size imageSize = Size(imageWidth, imageHeight);Mat R, T, E, F;
//R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
vector<Mat> rvecs;
//旋转向量
vector<Mat> tvecs;
//平移向量
vector<vector<Point2f>>
        imagePointL;
//左边摄像机所有照片角点的坐标集合
vector<vector<Point2f>> imagePointR;
//右边摄像机所有照片角点的坐标集合
vector<vector<Point3f>> objRealPoint;
//各副图像的角点的实际物理坐标集合
vector<Point2f> cornerL;
//左边摄像机某一照片角点坐标集合
vector<Point2f> cornerR;
//右边摄像机某一照片角点坐标集合
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat Rl, Rr, Pl, Pr, Q;
//校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）
Mat mapLx, mapLy, mapRx, mapRy;
//映射表
Rect validROIL, validROIR;
//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域
/*事先标定好的左相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 296.65731645541695, 0, 343.1975436071541,
        0, 300.71016643747646, 246.01183552967473,
        0, 0, 1);
//这时候就需要你把左右相机单目标定的参数给写上
//获得的畸变参数
Mat distCoeffL = (Mat_<double>(4, 1) << -0.23906272129552558, 0.03436102573634348, 0.001517498429211239, -0.005280695866378259);
/*事先标定好的右相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1*/
Mat cameraMatrixR = (Mat_<double>(3, 3) << 296.92709649579353, 0, 313.1873142211607,
        0, 300.0649937238372, 217.0722185756087,
        0, 0, 1);
Mat distCoeffR = (Mat_<double>(4, 1) << -0.23753878535018613, 0.03338842944635466, 0.0026030620085220105, -0.0008840126895030034);

void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
    {
        for (int colIndex = 0; colIndex < boardwidth; colIndex++)
        {
            imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
    {
        obj.push_back(imgpoint);
    }
}
void outputCameraParam(void)
{	/*保存数据*/	/*输出数据*/
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    //文件存储器的初始化
    if (fs.isOpened())
    {
        fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
        fs.release();
        cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
    }
    else
    {
        cout << "Error: can not save the intrinsics!!!!!" << endl;
    }
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
        cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";
}

int main(int argc, char* argv[])
{
    Mat img;
    int goodFrameCount = 1;
    cout << "Total Images：" << frameNumber << endl;
    while (goodFrameCount < frameNumber)
    {
        cout <<"Current image ：" << goodFrameCount << endl;
        string 	filenamel,filenamer;
        //char filename[100];
        /*读取左边的图像*/
        filenamel = folder_ + format_L+	to_string(goodFrameCount)+".jpg";
        rgbImageL = imread(filenamel, CV_LOAD_IMAGE_COLOR);
        cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
        /*读取右边的图像*/
        //sprintf_s(filename, "D:/dual_camera_clibration/dual/R%d.jpg", goodFrameCount );
        filenamer = folder_ + format_R+	to_string(goodFrameCount)+".jpg";
        rgbImageR = imread(filenamer, CV_LOAD_IMAGE_COLOR);
        cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
        bool isFindL, isFindR;
        isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
        isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);
        if (isFindL == true && isFindR == true)
            //如果两幅图像都找到了所有的角点 则说明这两幅图像是可行的
        {
            cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
            imshow("chessboardL", rgbImageL);
            imagePointL.push_back(cornerL);
            cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
            imshow("chessboardR", rgbImageR);
            imagePointR.push_back(cornerR);
            goodFrameCount++;
            cout << "The image" << goodFrameCount << " is good" << endl;
        }
        else
        {
            cout << "The image "<< goodFrameCount <<"is bad please try again" << endl;
            goodFrameCount++;
        }

        if (waitKey(10) == 'q')
        {
            break;
        }
    }
    /*	计算实际的校正点的三维坐标	根据实际标定格子的大小来设置	*/
    calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber-1, squareSize);
    cout << "cal real successful" << endl;
    /*	标定摄像头	由于左右摄像机分别都经过了单目标定	所以在此处选择flag = CALIB_USE_INTRINSIC_GUESS	*/
    double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
                                 cameraMatrixL, distCoeffL,
                                 cameraMatrixR, distCoeffR,
                                 Size(imageWidth, imageHeight), R, T, E, F, CV_CALIB_USE_INTRINSIC_GUESS,
                                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 80, 1e-5));
    cout << "Stereo Calibration done with RMS error = " << rms << endl;
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,		CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);

    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
    Mat rectifyImageL, rectifyImageR;
    cout << "debug"<<endl;
    for (int num = 1; num < frameNumber;num++)
    {
        string 	filenamel,filenamer;
        filenamel = folder_ + format_L+	to_string(num)+".jpg";
        filenamer = folder_ + format_R+	to_string(num)+".jpg";
        rectifyImageL = imread(filenamel);
        rectifyImageR = imread(filenamer);
        imshow("Rectify Before", rectifyImageL);
        /*	经过remap之后，左右相机的图像已经共面并且行对准了	*/
        Mat rectifyImageL2, rectifyImageR2;
        remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
        remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);
        imshow("rectifyImageL", rectifyImageL2);
        imshow("rectifyImageR", rectifyImageR2);
        /*保存并输出数据*/
        outputCameraParam();
        /*	把校正结果显示出来 把左右两幅图像显示到同一个画面上 这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来	*/
        Mat canvas;	double sf;
        int w, h;
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
        /*左图像画到画布上*/
        Mat canvasPart = canvas(Rect(w * 0, 0, w, h));
        //得到画布的一部分
        resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
        //把图像缩放到跟canvasPart一样大小
        Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),   //获得被截取的区域
                   cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
        rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);
        //画上一个矩形
        cout << "Painted ImageL" << endl;
        /*右图像画到画布上*/
        canvasPart = canvas(Rect(w, 0, w, h));
        //获得画布的另一部分
        resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
        Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
                   cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
        rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);	cout << "Painted ImageR" << endl;
        /*画上对应的线条*/
        for (int i = 0; i < canvas.rows; i += 16)
            line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        //cout << "wait key" << endl;
        waitKey();	//system("pause");
    }
    return 0;
}

