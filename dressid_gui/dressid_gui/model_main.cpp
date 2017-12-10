#include "model_main.h"
#include "ui_model_main.h"
#include "ANNtrainer.hpp"
#include <opencv2/opencv.hpp>

#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QInputDialog>
#include <QDir>
#include <QListWidgetItem>
#include <QPixmap>

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
    LoadFromFile();
//    QLabel *newlbl = new QLabel;
//    QPixmap pc("/home/danil/Изображения/карл.jpg");
//    newlbl->setPixmap(pc.scaledToHeight(100));
//    ui->gridLayout->addWidget(newlbl);
//    QLabel *newlbl2 = new QLabel;
//    QPixmap pc2("/home/danil/Изображения/wolf.jpg");
//    newlbl2->setPixmap(pc2.scaledToHeight(100));
//    ui->gridLayout->addWidget(newlbl2);

}

Model_main::~Model_main()
{
    outerwear.clear();
    dress.clear();
    pants.clear();
    tshirt.clear();
    pullover.clear();
    delete ui;
}

QString convert_id(int id);

void Model_main::LoadFromFile(/*QString filepath*/)
{
    QFile file("model/save.txt");
    if (!file.open(QFile::ReadOnly | QFile::Text)){
        qDebug() << "cant open file" ;
        //qDebug() << " Removing from file";
        return;
    }
    QString text;
    while (!file.atEnd()){
        text = file.readLine();
        QTextStream lineIn(&text);
        lineIn << text;
        int type;
        QString path;
        lineIn >> path;
        lineIn >> type;
        qDebug() << path << type;
        if (QFile::exists(path)){
            QLabel *newlbl = new QLabel;
            QPixmap pc(path);
            newlbl->setPixmap(pc.scaled(90,150,Qt::KeepAspectRatioByExpanding));
            newlbl->setFrameShape(QFrame::Panel);
            newlbl->setFrameShadow(QFrame::Raised);
            newlbl->setFixedSize(90,150);
            ui->gridLayout->addWidget(newlbl);
            switch (type){
            case 0:
                outerwear.push_back(qDress(path,type));
                break;
            case 1:
                dress.push_back(qDress(path,type));
                break;
            case 2:
                pants.push_back(qDress(path,type));
                break;
            case 3:
                pullover.push_back(qDress(path,type));
                break;
            case 4:
                tshirt.push_back(qDress(path,type));
                break;
            }
        } else {
            qDebug() << "no such file: " << path;
        }
    }
    file.close();
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
            classname = "blouse";
            break;
        case 4:
            classname = "tshirt";
            break;
        default:
            break;
    }
    return classname;
}
void Model_main::on_btnOnePic_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, tr("Choose file"), QDir::homePath(), tr("Image Files (*.png *.jpg *.bmp)"),nullptr, QFileDialog::DontUseNativeDialog);
    QFileDialog::getOpenFileName();
    //qDebug() << "file name: " <<file_name;
    cv::FlannBasedMatcher flann;
    flann.add(vocabulary);
    flann.train();
    cv::Mat output;
    cv::Mat testSamples;
    int networkInputSize = 1024;
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
    //qDebug() << testSamples.cols << testSamples.rows;
    mlp->predict(testSamples, output);
    int predictedClass;
    predictedClass = getPredictedClass(output);
    output.release();
    flann.clear();
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this,"predicted class","I think this is "+ convert_id(predictedClass)+". Am I right?", QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::No) {
        bool ok;
        QStringList dressTypes;
        dressTypes << tr("Outerwear") << tr("dress") << tr("pants") << tr("pullover") << tr("tshirt");
        QString qname = QInputDialog::getItem(this,"Choose right", "this is:", dressTypes, 0, false, &ok);
        int i = dressTypes.indexOf(qname);
       // QString qname = QInputDialog::getText(this, tr("Enter right classnem"), tr("What is it?"), QLineEdit::Normal, "", &ok);
        if (ok){
            switch (i){
            case 0:
                outerwear.push_back(qDress(file_name,i));
                break;
            case 1:
                dress.push_back(qDress(file_name,i));
                break;
            case 2:
                pants.push_back(qDress(file_name,i));
                break;
            case 3:
                pullover.push_back(qDress(file_name,i));
                break;
            case 4:
                tshirt.push_back(qDress(file_name,i));
                break;
            }
            predictedClass = i;
//            std::string name = qname.toStdString();
//            cv::Mat descriptorsSet;
//            std::vector<ImageData*> descriptorsMetadata;

//            readImage(filename,
//                      [&](const std::string& classname, const cv::Mat& descriptors) {
//                descriptorsSet.push_back(descriptors);
//                ImageData* data = new ImageData;
//                data->classname = name;
//                data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
//                for (int j = 0; j < descriptors.rows; j++)
//                {
//                    descriptorsMetadata.push_back(data);
//                }
//            });

//            //std::cout << "Creating new vocabulary" << std::endl;
//            cv::Mat labels;
//            cv::Mat vocabulary2;

//            cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
//                                                                                  cv::TermCriteria::MAX_ITER, 10, 0.01), 1,
//                       cv::KMEANS_PP_CENTERS, vocabulary2);

//            descriptorsSet.release();


//            int* ptrLabels = (int*)(labels.data);
//            int size = labels.rows * labels.cols;
//            for (int i = 0; i < size; i++)
//            {
//                int label = *ptrLabels++;
//                ImageData* data = descriptorsMetadata[i];
//                data->bowFeatures.at<float>(label)++;
//            }


//            //std::cout << "Preparing neural network" << std::endl;
//            cv::Mat trainSamples;
//            cv::Mat trainResponses;
//            std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
//            auto it = uniqueMetadata.begin();

//            ImageData* data = *it;
//            cv::Mat normalizedHist;
//            cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
//            trainSamples = normalizedHist;
//            trainResponses = getClassCode(classes, data->classname);
//            delete *it;
//            descriptorsMetadata.clear();
            //cv::Mat trainResponses = getClassCode(classes, name);
            //cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(testSamples,cv::ml::ROW_SAMPLE, trainResponses);
            //std::cout << "Training neural network again" << std::endl;
            //cv::Ptr<cv::ml::ANN_MLP> mlp2 = mlp;
            //mlp->train(trainData, cv::ml::ANN_MLP::UPDATE_WEIGHTS);

            //vocabulary.push_back(vocabulary2);

            //trainSamples.release();
            testSamples.release();
            //trainResponses.release();

            //saveModels(mlp, vocabulary, classes);
            QMessageBox::information(this,"Yeee","Done");
        } else {
            testSamples.release();
            return;
        }
    } else {
        switch (predictedClass){
        case 0:
            outerwear.push_back(qDress(file_name,predictedClass));
            break;
        case 1:
            dress.push_back(qDress(file_name,predictedClass));
            break;
        case 2:
            pants.push_back(qDress(file_name,predictedClass));
            break;
        case 3:
            pullover.push_back(qDress(file_name,predictedClass));
            break;
        case 4:
            tshirt.push_back(qDress(file_name,predictedClass));
            break;
        }
        testSamples.release();
    }

    QFile file("model/save.txt");
    if (file.open(QFile::Append)){
        QByteArray data;
        data.append(file_name);
        data.append(" ");
        data.append(QString::number(predictedClass));
        data.append("\n");
        file.write(data);
    }
    file.close();
    QLabel *newlbl = new QLabel;
    QPixmap pc(file_name);
    newlbl->setPixmap(pc.scaledToHeight(100));
    newlbl->setFrameShape(QFrame::Panel);
    newlbl->setFrameShadow(QFrame::Raised);
    ui->gridLayout->addWidget(newlbl);
    switch (predictedClass){
    case 0:
        outerwear.push_back(qDress(file_name,predictedClass));
        break;
    case 1:
        dress.push_back(qDress(file_name,predictedClass));
        break;
    case 2:
        pants.push_back(qDress(file_name,predictedClass));
        break;
    case 3:
        pullover.push_back(qDress(file_name,predictedClass));
        break;
    case 4:
        tshirt.push_back(qDress(file_name,predictedClass));
        break;
    }
    //this->repaint();
}






//void Model_main::on_btnSort_clicked()
//{
//    QString path = ui->edt1->text();
//    //QMessageBox::information(this,"Yeee",path);
//    std::vector<std::string> files = getFilesInDirectory(path.toStdString().c_str());
//    cv::FlannBasedMatcher flann;
//    flann.add(vocabulary);
//    flann.train();
//    cv::Mat output;
//    cv::Mat testSamples;
//    int networkInputSize = 1024;
//    readImages(files.begin(), files.end(),
//               [&](const std::string& classname, const cv::Mat& descriptors) {

//        cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
//        cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
//        testSamples.push_back(bowFeatures);
//    });



//    cv::Mat testOutput;
//    mlp->predict(testSamples, testOutput);
//    int i = 0;
//    vec_iter it = files.begin();
//    for (i = 0; i < testOutput.rows; i++, it++)
//    {
//        std::size_t pos1 = (*it).find_last_of('/');
//        std::string first = (*it).substr(pos1+1, (*it).find_last_of(EOF) - pos1);
//        QString name = QString::fromStdString(first);
//        QListWidgetItem *item = new QListWidgetItem(name);
//        int predictedClass = getPredictedClass(testOutput.row(i));
//        switch (predictedClass){
//        case 0:
//            ui->lst1->addItem(item);
//            break;
//        case 1:
//            ui->lst2->addItem(item);
//            break;
//        case 2:
//            ui->lst3->addItem(item);
//            break;
//        case 3:
//            ui->lst4->addItem(item);
//            break;
//        case 4:
//            ui->lst5->addItem(item);
//            break;
//        default:
//            break;
//        }
//    }


//}

cv::Mat resizeImg(const cv::Mat& img, double scale ){
    cv::Mat out;
    double k = scale / img.cols;
    cv::resize(img, out, cv::Size(), k,k);
    return out;
}

void Model_main::on_pushButton_clicked()
{
    cv::destroyAllWindows();
    int num;
    std::string ou = "outerwear";
    std::string dr = "dress";
    std::string pt = "pants";
    std:: string bl = "blouse";
    std::string ts = "tshirt";
    int i = ui->comboBox->currentIndex();
    switch (i){
    case 0:
        srand(time(NULL));
        num = rand()%(outerwear.size());
        cv::namedWindow("outerwear");
        cv::moveWindow("outerwear",100,0);
        cv::imshow("outerwear", resizeImg(outerwear[num].img, 350));
        //ui->label_high->setPixmap(QPixmap(outerwear[num].path).scaledToHeight(150));
        if ( (dress.size() != 0) && (pants.size() !=0 ) && (pullover.size() != 0) ){
            srand(time(NULL));
            switch(rand()%2){
            case 0:
                cv::namedWindow("dress");
                cv::moveWindow(dr, 460, 0);
                srand(time(NULL));
                num = rand()%(dress.size());
                cv::imshow("dress", resizeImg(dress[num].img,350));
                break;
            case 1:
                cv::namedWindow("blouse");
                cv::moveWindow(bl, 460, 0);
                srand(time(NULL));
                num = rand()%(pullover.size());
                cv::imshow("blouse", resizeImg(pullover[num].img, 350));
                cv::namedWindow("pants");
                cv::moveWindow(pt, 820, 0);
                srand(time(NULL));
                num = rand()%(pants.size());
                cv::imshow("pants", resizeImg(pants[num].img, 350));
            }
        } else if ((pants.size() ==0 ) || (pullover.size() == 0)){
            cv::namedWindow("dress");
            cv::moveWindow(dr, 460, 0);
            srand(time(NULL));
            num = rand()%(dress.size());
            cv::imshow("dress", resizeImg(dress[num].img,350));
        } else if ((dress.size() == 0)){
            cv::namedWindow("blouse");
            cv::moveWindow(bl, 460, 0);
            srand(time(NULL));
            num = rand()%(pullover.size());
            cv::imshow("blouse", resizeImg(resizeImg(pullover[num].img, 350), 350));
            cv::namedWindow("pants");
            cv::moveWindow(pt, 820, 0);
            srand(time(NULL));
            num = rand()%(pants.size());
            cv::imshow("pants", resizeImg(pants[num].img, 350));
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
        break;
    case 1:
        if ( (dress.size() != 0) && (pants.size() !=0 ) && (pullover.size() != 0) ){
            switch(rand()%2){
            case 0:
                cv::namedWindow("dress");
                cv::moveWindow(dr, 100, 0);
                srand(time(NULL));
                num = rand()%(dress.size());
                cv::imshow("dress", resizeImg(dress[num].img,350));
                break;
            case 1:
                cv::namedWindow("blouse");
                cv::moveWindow(bl, 100, 0);
                srand(time(NULL));
                num = rand()%(pullover.size());
                cv::imshow("blouse", resizeImg(resizeImg(pullover[num].img, 350), 350));
                cv::namedWindow("pants");
                cv::moveWindow(pt, 460, 0);
                srand(time(NULL));
                num = rand()%(pants.size());
                cv::imshow("pants", resizeImg(pants[num].img, 350));
            }
        } else if ((pants.size() ==0 ) || (pullover.size() == 0)){
            cv::namedWindow("dress");
            cv::moveWindow(dr, 100, 0);
            srand(time(NULL));
            num = rand()%(dress.size());
            cv::imshow("dress", resizeImg(dress[num].img,350));
        } else if ((dress.size() == 0)){
            cv::namedWindow("blouse");
            cv::moveWindow(bl, 100, 0);
            srand(time(NULL));
            num = rand()%(pullover.size());
            cv::imshow("blouse", resizeImg(resizeImg(pullover[num].img, 350), 350));
            cv::namedWindow("pants");
            cv::moveWindow(pt, 460, 0);
            srand(time(NULL));
            num = rand()%(pants.size());
            cv::imshow("pants", resizeImg(pants[num].img, 350));
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
        break;
    case 2:
        if ( (dress.size() != 0) && (pants.size() !=0 ) && (tshirt.size() != 0) ){
            switch(rand()%2){
            case 0:
                cv::namedWindow("dress");
                cv::moveWindow(dr, 100, 0);
                //cv::moveWindow("dress", 100, 0);
                srand(time(NULL));
                num = rand()%(dress.size());
                cv::imshow("dress", resizeImg(dress[num].img,350));
                break;
            case 1:
                cv::namedWindow("tshirt");
                cv::moveWindow(ts, 100, 0);
                srand(time(NULL));
                num = rand()%(tshirt.size());
                cv::imshow("tshirt", resizeImg(tshirt[num].img, 350));
                cv::namedWindow("pants");
                cv::moveWindow(pt, 460, 0);
                srand(time(NULL));
                num = rand()%(pants.size());
                cv::imshow("pants", resizeImg(pants[num].img, 350));
            }
        } else if ((pants.size() ==0 ) || (tshirt.size() == 0)){
            cv::namedWindow("dress");
            cv::moveWindow(dr, 100, 0);
            //cv::moveWindow("dress", 100, 0);
            srand(time(NULL));
            num = rand()%(dress.size());
            cv::imshow("dress", resizeImg(dress[num].img,350));
        } else if ((dress.size() == 0)){
            cv::namedWindow("tshirt");
            cv::moveWindow(ts, 100, 0);
            srand(time(NULL));
            num = rand()%(tshirt.size());
            cv::imshow("tshirt", resizeImg(tshirt[num].img, 350));
            cv::namedWindow("pants");
            cv::moveWindow(pt, 460, 0);
            srand(time(NULL));
            num = rand()%(pants.size());
            cv::imshow("pants", resizeImg(pants[num].img, 350));
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
        break;
    }
}
