#include "ANNtrainer.hpp"


std::vector<std::string> getFilesInDirectory(const char* directory)
{
    std::vector<std::string> files;
    DIR * dir_p = opendir(directory);
    dirent * ptr;
    if (!dir_p){
        std::cerr << "can't open dir" << std::endl;
        exit(1);
    }
    while ((ptr = readdir(dir_p)) != nullptr){
        if (ptr->d_type == DT_REG) {
            std::string path = directory;
            std::string name = path + ptr->d_name;
            files.push_back(name);
            std::cout << name << std::endl;
        }
    }
    closedir(dir_p);
    return files;
}


inline std::string getClassName(const std::string& filename)
{

    std::size_t pos1 = filename.find_last_of('/');
    std::string first = filename.substr(pos1+1, filename.find_last_of(EOF) - pos1);
    return first.substr(0, first.find_first_of('.'));
}


cv::Mat getDescriptors(const cv::Mat& img)
{
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    return descriptors;
}


void readImages(vec_iter begin, vec_iter end, std::function<void (const std::string&, const cv::Mat&)> callback)
{
    for (auto it = begin; it != end; ++it)
    {
        std::string filename = *it;
        std::cout << "Reading image " << filename << "..." << std::endl;
        cv::Mat img = cv::imread(filename, 0);
        if (img.empty())
        {
            std::cerr << "WARNING: Could not read image." << std::endl;
            continue;
        }
        std::string classname = getClassName(filename);
        cv::Mat descriptors = getDescriptors(img);
        callback(classname, descriptors);
    }
}


int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
    int index = 0;
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        if (*it == classname) break;
        ++index;
    }
    return index;
}


cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
    cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
    int index = getClassId(classes, classname);
    code.at<float>(index) = 1;
    return code;
}


cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
                       int vocabularySize)
{
    cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
    std::vector<cv::DMatch> matches;
    flann.match(descriptors, matches);
    for (size_t j = 0; j < matches.size(); j++)
    {
        int visualWord = matches[j].trainIdx;
        outputArray.at<float>(visualWord)++;
    }
    return outputArray;
}


cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples,
                                                 const cv::Mat& trainResponses)
{
    int networkInputSize = trainSamples.cols;
    int networkOutputSize = trainResponses.cols;
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
    std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
                                    networkOutputSize };
    mlp->setLayerSizes(layerSizes);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
    return mlp;
}


int getPredictedClass(const cv::Mat& predictions)
{
    float maxPrediction = predictions.at<float>(0);
    float maxPredictionIndex = 0;
    const float* ptrPredictions = predictions.ptr<float>(0);
    for (int i = 0; i < predictions.cols; i++)
    {
        float prediction = *ptrPredictions++;
        if (prediction > maxPrediction)
        {
            maxPrediction = prediction;
            maxPredictionIndex = i;
        }
    }
    return maxPredictionIndex;
}


void saveModels(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary,
                const std::set<std::string>& classes)
{
    mlp->save("mlp.yaml");
    cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();
    std::ofstream classesOutput("classes.txt");
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        classesOutput << getClassId(classes, *it) << "\t" << *it << std::endl;
    }
    classesOutput.close();
}


double Accuracy(cv::Ptr<cv::ml::ANN_MLP> mlp,
                const cv::Mat& testSamples, const std::vector<int>& testOutputExpected)
{
    cv::Mat testOutput;
    mlp->predict(testSamples, testOutput);
    //std::vector<std::vector<int> > confusionMatrix(2, std::vector<int>(2));
    int i = 0;
    int right = 0;

    for (i = 0; i < testOutput.rows; i++)
    {
        int predictedClass = getPredictedClass(testOutput.row(i));
        int expectedClass = testOutputExpected.at(i);
        if (predictedClass == expectedClass) {
            ++right;
        }
    }
    double result = (double)right / i;
    return result;
}


void readImage(std::string& filename,std::function<void (const std::string&, const cv::Mat&)> callback)
{
    std::cout << "Reading image " << filename << "..." << std::endl;
    cv::Mat img = cv::imread(filename, 0);

    std::string classname = getClassName(filename);
    cv::Mat descriptors = getDescriptors(img);
    callback(classname, descriptors);

}


std::set<std::string> readclasses(std::set<std::string>& list, std::ifstream& input){
    //int k = 0;
    while (!input.eof()){
        int d;
        std::string name;
        input >> d;
        input >> name;
        if (name != "") {
            list.insert(name);
        }
    }
    return list;
}