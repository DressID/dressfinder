#ifndef QDRESS_H
#define QDRESS_H
#include <QString>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>






class qDress
{
public:
    qDress(QString filepath, int typ);
    ~qDress();
    friend bool operator ==(const qDress& q1, const qDress& q2);
    friend bool operator !=(const qDress& q1, const qDress& q2);
    QString path;
    int type;
    cv::Mat img;
};




#endif // QDRESS_H
