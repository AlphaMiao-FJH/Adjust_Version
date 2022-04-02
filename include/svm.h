#include "../AngelSolver/AngleSolver.hpp"
#include "../General/General.h"

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace ml;

const string model_path = "../File/HOG_SVM.xml";

/* hog-svm定义 */
class HOG_SVM
{
private:
    cv::Ptr<cv::ml::SVM> m_svm;
    std::map<int, int> m_label2id;
    cv::HOGDescriptor m_hog;

public:
    HOG_SVM();
    int test(const cv::Mat &src);
};

class NumClassfier
{
public:
    bool runSvm(const Mat &src,ArmorPlate &target_armor);
    NumClassfier();
    HOG_SVM m_svm;                  //数字识别类
    int num;                        //数字
private:
    Ptr<SVM> svm;


    Point2f src_apex_cord[4];
    Point2f dst_apex_cord[4];

    int sample_size;                                     //设置样本大小
    int binary_threshold;                                //设置二值化阈值
    
    bool loadSvmModel(const string &path);               //SVM模型加载
    bool initImg(const Mat &src,Mat &dst,const ArmorPlate &target_armor);
};

