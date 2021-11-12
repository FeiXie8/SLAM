#include"iostream"
#include"cmath"
#include<Eigen/Core>
#include<Eigen/Geometry>
#include"sophus/se3.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char const *argv[])
{
    Matrix3d R=AngleAxisd(M_PI/2,Vector3d(0,0,1)).toRotationMatrix();
    Quaterniond q(R);
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);
    cout<<"SO(3) from matrix:\n"<<SO3_R.matrix()<<endl;
    cout<<"SO(3) from quaterniond:\n"<<SO3_q.matrix()<<endl;
    Vector3d so3=SO3_R.log();//log直接获得向量形式的李代数
    cout<<"so3=\n"<<so3.transpose()<<endl;
    cout<<"so3 hat=\n"<<Sophus::SO3d::hat(so3)<<endl;//由向量获得反对称矩阵，为运算符号^
    cout<<"so3 hat vee=\n"<<Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose()<<endl;
    
    //增量扰动
    Vector3d update_so3(1e-4,0,0);
    Sophus::SO3d SO3_update=Sophus::SO3d::exp(update_so3)*SO3_R;
    cout<<"*************************************"<<endl;
    Vector3d t(1,0,0);
    Sophus::SE3d SE3_Rt(R,t); //此时构造的是SE3了
    Sophus::SE3d SE3_qt(q,t);//SE3的李代数是6维
    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d se3=SE3_Rt.log();
    //se3的更新和so3一样，省略
    
    return 0;
}
