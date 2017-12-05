#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>


#include <dirent.h>

#include "ANNtrainer.hpp"



int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: <IMAGES_DIRECTORY>  <NETWORK_INPUT_LAYER_SIZE> <TRAIN_SPLIT_RATIO>" << std::endl;
        exit(-1);
    }
    int networkInputSize = atoi(argv[2]);
    float trainSplitRatio = atof(argv[3]);

    std::cout << "Reading training set" << std::endl;
    double start = (double)cv::getTickCount();
    std::vector<std::string> files = getFilesInDirectory(argv[1]);
    std::random_shuffle(files.begin(), files.end());

    cv::Mat descriptorsSet;
    std::vector<ImageData*> descriptorsMetadata;
    std::set<std::string> classes;
    readImages(files.begin(), files.begin() + (std::size_t)(files.size() * trainSplitRatio),
               [&](const std::string& classname, const cv::Mat& descriptors) {
                   classes.insert(classname);
                   descriptorsSet.push_back(descriptors);
                   ImageData* data = new ImageData;
                   data->classname = classname;
                   data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
                   for (int j = 0; j < descriptors.rows; j++)
                   {
                       descriptorsMetadata.push_back(data);
                   }
               });
    std::cout << "Working time in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;


    std::cout << "Creating vocabulary..." << std::endl;
    start = (double)cv::getTickCount();
    cv::Mat labels;
    cv::Mat vocabulary;

    cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
                                                                          cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
    descriptorsSet.release();
    std::cout << "Working time in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;


    std::cout << "Histograms of visual words" << std::endl;
    int* ptrLabels = (int*)(labels.data);
    int size = labels.rows * labels.cols;
    for (int i = 0; i < size; i++)
    {
        int label = *ptrLabels++;
        ImageData* data = descriptorsMetadata[i];
        data->bowFeatures.at<float>(label)++;
    }


    std::cout << "Preparing neural network for training" << std::endl;
    cv::Mat trainSamples;
    cv::Mat trainResponses;
    std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
    for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); )
    {
        ImageData* data = *it;
        cv::Mat normalizedHist;
        cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
        trainSamples.push_back(normalizedHist);
        trainResponses.push_back(getClassCode(classes, data->classname));
        delete *it;
        it++;
    }
    descriptorsMetadata.clear();


    std::cout << "Training neural network" << std::endl;
    start = cv::getTickCount();
    cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
    std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
    trainSamples.release();
    trainResponses.release();


    std::cout << "Training FLANN" << std::endl;
    start = cv::getTickCount();
    cv::FlannBasedMatcher flann;
    flann.add(vocabulary);
    flann.train();
    std::cout << "Working time in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

    std::cout << "Reading test set" << std::endl;
    start = cv::getTickCount();
    cv::Mat testSamples;
    std::vector<int> testOutputExpected;
    readImages(files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
               [&](const std::string& classname, const cv::Mat& descriptors) {

                   cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
                   cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
                   testSamples.push_back(bowFeatures);
                   testOutputExpected.push_back(getClassId(classes, classname));
               });
    std::cout << "Working time in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;


    double acc = Accuracy(mlp, testSamples, testOutputExpected);
    std::cout << "Accuracy: " << acc << std::endl;
    std::cout << "Saving models" << std::endl;
    saveModels(mlp, vocabulary, classes);



    return 0;
}