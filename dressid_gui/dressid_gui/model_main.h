#ifndef MODEL_MAIN_H
#define MODEL_MAIN_H

#include <QMainWindow>
#include <QString>
#include "ANNtrainer.hpp"
#include <opencv2/opencv.hpp>
#include <QFile>
#include <QTextStream>
#include "qdress.h"
#include <stdlib.h>
#include <ctime>

namespace Ui {
class Model_main;
}

class Model_main : public QMainWindow
{
    Q_OBJECT

public:
    explicit Model_main(QWidget *parent = 0);
    ~Model_main();
    void LoadFromFile(/*QString filepath*/);

    cv::Ptr<cv::ml::ANN_MLP> mlp;
    cv::Mat vocabulary;
    std::set<std::string> classes;
private slots:
    void on_btnOnePic_clicked();

    //void on_btnSort_clicked();

    void on_pushButton_clicked();

private:
    Ui::Model_main *ui;
    QList<qDress> outerwear;
    QList<qDress> dress;
    QList<qDress> pants;
    QList<qDress> pullover;
    QList<qDress> tshirt;
};

#endif // MODEL_MAIN_H
