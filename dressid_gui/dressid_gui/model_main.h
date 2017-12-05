#ifndef MODEL_MAIN_H
#define MODEL_MAIN_H

#include <QMainWindow>
#include <QString>
#include "ANNtrainer.hpp"

namespace Ui {
class Model_main;
}

class Model_main : public QMainWindow
{
    Q_OBJECT

public:
    explicit Model_main(QWidget *parent = 0);
    ~Model_main();


    cv::Ptr<cv::ml::ANN_MLP> mlp;
    cv::Mat vocabulary;
    std::set<std::string> classes;
private slots:
    void on_btnOnePic_clicked();

    void on_btnSort_clicked();

private:
    Ui::Model_main *ui;
};

#endif // MODEL_MAIN_H
