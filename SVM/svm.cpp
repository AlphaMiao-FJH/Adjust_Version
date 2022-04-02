#include "Svm.h"
/**
 * @brief NumClassfier构造函数
*/
NumClassfier::NumClassfier()
{   
    sample_size = 40;
    binary_threshold = 20;
    //loadSvmModel(model_path);
    HOG_SVM();
    dst_apex_cord[0] = Point2f(0,sample_size);
    dst_apex_cord[1] = Point2f(0,0);
    dst_apex_cord[2] = Point2f(sample_size,0);
    dst_apex_cord[3] = Point2f(sample_size,sample_size);
}

//svm类函数
    HOG_SVM::HOG_SVM()
    {
        m_label2id = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 11}, {7, 7}, {8, 8}};
        m_hog.winSize = Size(48, 32);
        m_hog.blockSize = Size(16, 16);
        m_hog.blockStride = Size(8, 8);
        m_hog.cellSize = Size(8, 8);
        m_hog.nbins = 9;
        m_hog.derivAperture = 1;
        m_hog.winSigma = -1;
        m_hog.histogramNormType = HOGDescriptor::L2Hys;
        m_hog.L2HysThreshold = 0.2;
        m_hog.gammaCorrection = false;
        m_hog.free_coef = -1.f;
        m_hog.nlevels = HOGDescriptor::DEFAULT_NLEVELS;
        m_hog.signedGradient = false;
        if (m_svm)
        {
            m_svm->clear();
        }
        m_svm = SVM::load(model_path);
    }
    int HOG_SVM::test(const Mat &src)
    {
        if (m_svm)
        {
            vector<float> descriptors;
            m_hog.compute(src, descriptors, Size(8, 8));
            int label = m_svm->predict(descriptors);
            return m_label2id[label];
        }
        else
        {
            return 0;
        }
    }
/**
 *@brief 载入SVM模型
 *@param path SVM模型路径
 *@return 是否成功载入 
 */
bool NumClassfier::loadSvmModel(const string &path)
{
    svm = StatModel::load<SVM>(path);
    if(svm.empty())
    {
        cout<<"svm model path failure"<<endl;
        return false;
    }

    cout<<"svm load succeed"<<endl;
    return true;
}
/**
 * @brief SVM主函数
 * @param src 原图
 * @param ArmorPlate 目标装甲板
 * @return 是否成功运行
*/
bool NumClassfier::runSvm(const Mat &src,ArmorPlate &target_armor)
{
    Mat sample_img = Mat::zeros(Size(sample_size,sample_size),CV_8UC1);
    Mat sample_img_reshaped;
    //图像初始化
    initImg(src,sample_img,target_armor);
    //样本reshape与改变格式
    //sample_img_reshaped = sample_img.reshape(1, 1);
    //sample_img_reshaped.convertTo(sample_img_reshaped, CV_32FC1);
    num=m_svm.test(sample_img);
    cout<<"num:"<<num<<endl;
    cout<<"---------------------------------------"<<endl;
    //cout<<"num:"<<(int)svm->predict(sample_img_reshaped)<<endl;
    //计算结果
    target_armor.serial = num;
    return true;
}
/**
 * @brief 图像初始化
 * @param src 原图
 * @param dst 处理后图像
 * @param target_armor 目标装甲板
 * @return 是否成功初始化
*/
bool NumClassfier::initImg(const Mat &src,Mat &dst,const ArmorPlate &target_armor)
{
    Mat warped_img = Mat::zeros(Size(sample_size,sample_size),CV_8UC3);
    Mat src_gray;
    //设置扩张高度
    int extented_height = 0.5 * std::min(target_armor.rrect.size.width,target_armor.rrect.size.height);
    //计算归一化的装甲板左右边缘方向向量
    Point2f left_edge_vector = (target_armor.apex[1] - target_armor.apex[0]) / pointsDistance(target_armor.apex[1] , target_armor.apex[0]);//由0指向1
    Point2f right_edge_vector = (target_armor.apex[2] - target_armor.apex[3]) / pointsDistance(target_armor.apex[2] , target_armor.apex[3]);//由3指向2
    src_apex_cord[0] = target_armor.apex[0] - left_edge_vector * extented_height;
    src_apex_cord[1] = target_armor.apex[1] + left_edge_vector * extented_height;
    src_apex_cord[2] = target_armor.apex[2] + right_edge_vector * extented_height;
    src_apex_cord[3] = target_armor.apex[3] - right_edge_vector * extented_height;
    //计算透视变换矩阵
    Mat warp_matrix = getPerspectiveTransform(src_apex_cord,dst_apex_cord);
    //进行图像透视变换
    warpPerspective(src, warped_img, warp_matrix, Size(sample_size,sample_size), INTER_NEAREST, BORDER_CONSTANT, Scalar(0));
    //进行图像的灰度化与二值化
    cvtColor(warped_img,src_gray,COLOR_BGR2GRAY);
    threshold(src_gray,dst,binary_threshold,255,THRESH_BINARY);
    resize(dst, dst, Size(48, 32));
    //转换尺寸
    Rect front_roi(Point(20, 0), Size(10, 32));
    Mat front_roi_img = dst(front_roi);
    //imshow("dstaaaaaaa",dst);
    return true;
}
