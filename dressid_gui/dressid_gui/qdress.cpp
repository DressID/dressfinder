#include "qdress.h"

qDress::qDress(QString filepath, int typ)
{
    path = filepath;
    type = typ;
    img = imread(path.toStdString(), cv::IMREAD_COLOR);
}



qDress::~qDress()
{
    path.clear();
    img.release();
}

bool operator ==(const qDress &q1, const qDress &q2)
{
    return (q1.path == q2.path ? true : false);
}

bool operator !=(const qDress &q1, const qDress &q2)
{
    return !(q1==q2);
}
