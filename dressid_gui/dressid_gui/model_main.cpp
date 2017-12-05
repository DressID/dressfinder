#include "model_main.h"
#include "ui_model_main.h"
#include "ANNtrainer.hpp"

#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QInputDialog>
#include <QDir>
#include <QListWidgetItem>

Model_main::Model_main(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Model_main)
{
    ui->setupUi(this);
    std::string mlp_path = "model/mlp.yaml";
    std::string vocabulary_path = "model/vocabulary.yaml";
    std::string classes_path = "model/classes.txt";
    qDebug() << "loading resources";
    mlp=cv::ml::ANN_MLP::load(mlp_path);
    cv::FileStorage fs(vocabulary_path, cv::FileStorage::READ);
    fs["vocabulary"] >> vocabulary;
    fs.release();
    std::ifstream input(classes_path);
    readclasses(classes, input);
    input.close();
}

Model_main::~Model_main()
{
    delete ui;
}




QString convert_id(int id)
{
    QString classname;
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
void Model_main::on_btnOnePic_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "Choose file", QDir::homePath());

    cv::FlannBasedMatcher flann;
    flann.add(vocabulary);
    flann.train();
    cv::Mat output;
    cv::Mat testSamples;
    int networkInputSize = 512;
    std::string filename = file_name.toStdString();
    if (filename.empty()){
        return;
    }
    readImage(filename,
              [&](const std::string& classname, const cv::Mat& descriptors) {

                  cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
                  cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
                  testSamples.push_back(bowFeatures);
              });
    mlp->predict(testSamples, output);
    int predictedClass;
    predictedClass = getPredictedClass(output);
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this,"predicted class","I think this is "+ convert_id(predictedClass)+".Am I right?", QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::No) {
        bool ok;
        QString qname = QInputDialog::getText(this, tr("Enter right classnem"), tr("What is it?"), QLineEdit::Normal, "", &ok);
        std::string name = qname.toStdString();
        cv::Mat descriptorsSet;
        std::vector<ImageData*> descriptorsMetadata;

        readImage(filename,
                  [&](const std::string& classname, const cv::Mat& descriptors) {
            descriptorsSet.push_back(descriptors);
            ImageData* data = new ImageData;
            data->classname = name;
            data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
            for (int j = 0; j < descriptors.rows; j++)
            {
                descriptorsMetadata.push_back(data);
            }
        });

        //std::cout << "Creating new vocabulary" << std::endl;
        cv::Mat labels;
        cv::Mat vocabulary2;

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


        //std::cout << "Preparing neural network" << std::endl;
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
        //std::cout << "Training neural network again" << std::endl;
        cv::Ptr<cv::ml::ANN_MLP> mlp2 = mlp;
        mlp2->train(trainData, cv::ml::ANN_MLP::UPDATE_WEIGHTS);

        //vocabulary.push_back(vocabulary2);

        trainSamples.release();
        trainResponses.release();

        saveModels(mlp2, vocabulary2, classes);
        QMessageBox::information(this,"Yeee","Trained!");
    }
}






void Model_main::on_btnSort_clicked()
{
    QString path = ui->edt1->text();
    //QMessageBox::information(this,"Yeee",path);
    std::vector<std::string> files = getFilesInDirectory(path.toStdString().c_str());
    cv::FlannBasedMatcher flann;
    flann.add(vocabulary);
    flann.train();
    cv::Mat output;
    cv::Mat testSamples;
    int networkInputSize = 512;
    readImages(files.begin(), files.end(),
               [&](const std::string& classname, const cv::Mat& descriptors) {

        cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
        cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
        testSamples.push_back(bowFeatures);
    });



    cv::Mat testOutput;
    mlp->predict(testSamples, testOutput);
    int i = 0;
    vec_iter it = files.begin();
    for (i = 0; i < testOutput.rows; i++, it++)
    {
        std::size_t pos1 = (*it).find_last_of('/');
        std::string first = (*it).substr(pos1+1, (*it).find_last_of(EOF) - pos1);
        QString name = QString::fromStdString(first);
        QListWidgetItem *item = new QListWidgetItem(name);
        int predictedClass = getPredictedClass(testOutput.row(i));
        switch (predictedClass){
        case 0:
            ui->lst1->addItem(item);
            break;
        case 1:
            ui->lst2->addItem(item);
            break;
        case 2:
            ui->lst3->addItem(item);
            break;
        case 3:
            ui->lst4->addItem(item);
            break;
        case 4:
            ui->lst5->addItem(item);
            break;
        default:
            break;
        }
    }


}
