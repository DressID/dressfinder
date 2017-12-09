#include <iostream>
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

std::string convert_id(int id)
{
    std::string classname;
    switch (id) {
        case 0:
            classname = "Outerwear";
            break;
        case 1:
            classname = "dress";
            break;
        case 2:
            classname = "pants";
            break;
        case 3:
            classname = "pullover";
            break;
        case 4:
            classname = "shirt";
            break;
        default:
            break;
    }
    return classname;
}

int main(){
    std::cout << "Enter MLP model path: " << std::endl;
    std::string mlp_path = "mlp.yaml";
    //std::cin >> mlp_path;
    std::cout << "Enter vocabulary path: " << std::endl;
    std::string vocabulary_path="vocabulary.yaml";
    //std::cin >> vocabulary_path;
    std::cout << "Enter classes path: " << std::endl;
    std::string classes_path="classes.txt";
    //std::cin >> classes_path;
    cv::Ptr<cv::ml::ANN_MLP> mlp=cv::ml::ANN_MLP::load(mlp_path);

    std::cout << "Give image for check"  <<std::endl;
    std::string filename = "/home/danil/techno/dressid/DressNeural/testme.jpg";
    //std::cin >> filename;
    cv::Mat output;

    int networkInputSize = 512;
    cv::Mat testSamples;
    cv::FileStorage fs(vocabulary_path, cv::FileStorage::READ);
    cv::Mat vocabulary;
    fs["vocabulary"] >> vocabulary;
    fs.release();
    cv::FlannBasedMatcher flann;
    flann.add(vocabulary);
    flann.train();
    std::ifstream input(classes_path);
    std::set<std::string> classes;
    readclasses(classes, input);
    input.close();

    std::cout << "reading img" << std::endl;
    readImage(filename,
              [&](const std::string& classname, const cv::Mat& descriptors) {

                  cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
                  cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
                  testSamples.push_back(bowFeatures);
              });
    mlp->predict(testSamples, output);
    int predictedClass;
    predictedClass = getPredictedClass(output);


    std::cout <<"Predicter class: "<< convert_id(predictedClass) << std::endl;

    std::cout << "is that right? y/n" << std::endl;
    char c;
    std::cin >> c;

    if (c == 'n'){
        std::cout << "enter what is it" << std::endl;
        std::string name;
        std::cin >> name;
        cv::Mat descriptorsSet;
        std::vector<ImageData*> descriptorsMetadata;

        readImage(filename,
                  [&](const std::string& classname, const cv::Mat& descriptors) {
                      descriptorsSet.push_back(descriptors);
                      ImageData* data = new ImageData;
                      data->classname = name;
                      //std::cout << name <<std::endl;
                      data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
                      for (int j = 0; j < descriptors.rows; j++)
                      {
                          descriptorsMetadata.push_back(data);
                      }
                  });
        //std::cout << getClassCode(classes, "shirt") << std::endl;
        std::cout << "Creating new vocabulary" << std::endl;
        cv::Mat labels;
        cv::Mat vocabulary2;
        //int const* rrr = descriptorsSet.size;
        std::cout << descriptorsSet <<std::endl;
        cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
                                                                              cv::TermCriteria::MAX_ITER, 10, 0.01), 1,
                                                                              cv::KMEANS_PP_CENTERS, vocabulary2);

        descriptorsSet.release();


        int* ptrLabels = (int*)(labels.data);
        int size = labels.rows * labels.cols;
        for (int i = 0; i < size; i++)
        {
            int label = *ptrLabels++;
            ImageData* data = descriptorsMetadata[i];
            data->bowFeatures.at<float>(label)++;
        }


        std::cout << "Preparing neural network" << std::endl;
        cv::Mat trainSamples;
        cv::Mat trainResponses;
        std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
        auto it = uniqueMetadata.begin();

        ImageData* data = *it;
        cv::Mat normalizedHist;
        cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
        trainSamples = normalizedHist;
        trainResponses = getClassCode(classes, data->classname);
        delete *it;
        descriptorsMetadata.clear();

        cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainSamples,cv::ml::ROW_SAMPLE, trainResponses);
        std::cout << "Training neural network again" << std::endl;
        //cv::Ptr<cv::ml::ANN_MLP> mlp2 = mlp;
        mlp->train(trainData, cv::ml::ANN_MLP::UPDATE_WEIGHTS);

        vocabulary.push_back(vocabulary2);

        trainSamples.release();
        trainResponses.release();

        saveModels(mlp, vocabulary, classes);
    }



    return 0;
}