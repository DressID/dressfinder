
#ifndef DRESSID_ANNTRAINER_HPP
#define DRESSID_ANNTRAINER_HPP

#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <dirent.h>



typedef std::vector<std::string>::const_iterator vec_iter;

struct ImageData
{
    std::string classname;
    cv::Mat bowFeatures;
};

std::vector<std::string> getFilesInDirectory(const char* directory);


inline std::string getClassName(const std::string& filename);


cv::Mat getDescriptors(const cv::Mat& img);


void readImages(vec_iter begin, vec_iter end, std::function<void (const std::string&, const cv::Mat&)> callback);


int getClassId(const std::set<std::string>& classes, const std::string& classname);


cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname);


cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors, int vocabularySize);


cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples,
                                                 const cv::Mat& trainResponses);


int getPredictedClass(const cv::Mat& predictions);


void saveModels(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary,
                const std::set<std::string>& classes);


double Accuracy(cv::Ptr<cv::ml::ANN_MLP> mlp,
                const cv::Mat& testSamples, const std::vector<int>& testOutputExpected);


void readImage(std::string& filename,std::function<void (const std::string&, const cv::Mat&)> callback);


std::set<std::string> readclasses(std::set<std::string>& list, std::ifstream& input);




#endif //DRESSID_ANNTRAINER_HPP
