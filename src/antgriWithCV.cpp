#include<iostream>
#include<Eigen/Dense>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include"ctime"
#include"opencv2/opencv.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;

Mat befor;
Mat picture=Mat(480,480,CV_8UC3,Scalar(255,255,255));
Point node0=Point(60,60);
Point node1=Point(420,60);
Point node2=Point(420,420);
Point node3=Point(60,420);

#define CLOCK_PER_SEC ((clock_t)1000)

const int nodeNum = 4;     //节点数量
const int pathNum = 5;    //路径数量
const int antNum = 40;     //蚂蚁数量
const double pVol = 0.3;   //信息素挥发系数 0.2~0.5
const int pQ = 10;         //信息素强度 10~1000
const double pImp = 1;     //信息素相对重要性 1~4
const double qImp = 0;     //启发信息相对重要性 3~4.5
const int gen =20;       //迭代次数 100~500

struct ant                 //蚂蚁结构体
{
	int loc;               //位置
	int tabu[nodeNum];     //禁忌表
	int antPath[pathNum];  //走过的路
	bool flag;             //是否到达终点7
};
struct ant ants[antNum];   //蚁群

typedef Matrix<double, 4, 4> Matrix4d;
Matrix4d dist;             //距离矩阵
Matrix4d pher;             //信息素矩阵
Matrix4d nextPher;         //下一代信息素矩阵
Matrix4d insp;             //启发信息矩阵


struct node                //节点结构体
{
	int num;               //编号
	double prob;           //选择概率
};
struct node nodeProb[nodeNum];    //可到达节点组
double lineNodeProb[nodeNum];     //线性化 可到达节点的选择概率

clock_t start, finish;     
double duration;

void initAnts();                  
void initNodeProb();              
void initMarix();   
bool ifNodeInTabu(int, int);
int nodeSelect(int, int);
void updateAnt(int, int);
double nodeSelProb(int, int);
int getAntLen(ant);
int getBestPath();
void printBestPath(int, int);
void updatePher();
void evolution();
void drawNode();
void updateMatrix();
Mat mergePicture(Mat& src1,Mat& src2);


int main()
{
    unsigned seed;
    seed=time(0);
    srand(seed);
	srand((unsigned)time(NULL));
    drawNode();
    befor=picture.clone();
    

	evolution();
    //updateMatrix();
    // imshow("picture",picture);
    // waitKey(0);
}

//蚁群初始化
void initAnts()
{
	//初始化禁忌表与行走路线
	for (int i = 0; i < antNum; i++)
	{
		for (int j = 0; j < nodeNum; j++)
		{
			ants[i].tabu[j] = -1;
		}
		for (int j = 0; j < pathNum; j++)
		{
			ants[i].antPath[j] = -1;
		}
	}
	//将蚂蚁放入节点
	for (int i = 0; i < antNum; i++)
	{
		//ants[i].loc = rand() % 8;
		ants[i].loc = 0;//出发点都在起点
		ants[i].tabu[0] = ants[i].loc;
		ants[i].antPath[0] = ants[i].loc;
		ants[i].flag = 0;
	}
}

//初始化节点选择概率数组
void initNodeProb()
{

	for (int i = 0; i < nodeNum; i++)
	{
		nodeProb[i].num = -1;
		nodeProb[i].prob = 0;
		lineNodeProb[i] = 0;
	}
}

//初始化距离、信息素、启发信息矩阵
void initMarix()
{
	dist = Matrix4d::Constant(4, 4, -1);
	dist(0, 1) = 5;
	dist(0, 2) = 15;
	dist(0, 3) = 9;
	dist(1, 2) = 10;
	dist(3, 2) = 3;

	putText(befor,to_string((int)dist(0, 1)),Point(240,55),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);
    putText(befor,to_string((int)dist(1, 2)),Point(425,240),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);
    putText(befor,to_string((int)dist(3, 2)),Point(240,425),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);
    putText(befor,to_string((int)dist(0, 3)),Point(55,240),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);
    putText(befor,to_string((int)dist(0, 2)),Point(240,240),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);

	pher = Matrix4d::Zero();
	nextPher = Matrix4d::Zero();
	insp = Matrix4d::Zero();
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (dist(i, j) != -1)
			{
				insp(i, j) = 1 / dist(i, j);//启发信息为距离的倒数
				pher(i, j) = 1;             //信息素浓度初始值为1
			}

		}
	}
}

//轮盘赌选择下一步行进节点
int nodeSelect(int k, int f)
{
	int c = 0;//记录蚂蚁可行进的节点个数
	//1、计算可行进的各节点 选择概率
	for (int m = 0; m < nodeNum; m++)
	{
		//若节点（i,j）之间有路且j不在蚂蚁k的禁忌表中，则计算概率
		if (dist(ants[k].loc, m) != -1 && !ifNodeInTabu(m, k))
		{
			nodeProb[c].num = m;
			nodeProb[c].prob = nodeSelProb(k, m);
			c++;
		}
	}

	//2、线性化选择概率
	for (int m = 0; m < c; m++)
	{
		for (int n = m; n >= 0; n--)
		{
			lineNodeProb[m] += nodeProb[n].prob;
		}
	}

	//3、产生随机数选择节点
	double r = rand() / double(RAND_MAX);
	int j = 0;   //选取的目标节点
	for (int m = 0; m < nodeNum; m++)
	{
		if (r <= lineNodeProb[m])
		{
			j = nodeProb[m].num;
			updateAnt(k, j);
			if (j == f)
				ants[k].flag = 1;  //若蚂蚁k下一步节点为目的节点，则修改标志
			return j;
		}

	}
}

//更新蚂蚁信息
void updateAnt(int k, int l)
{
	ants[k].loc = l;
	for (int i = 0; i < nodeNum; i++)
		if (ants[k].tabu[i] == -1)
		{
			ants[k].tabu[i] = l;
			break;
		}
	for (int i = 0; i < pathNum; i++)
		if (ants[k].antPath[i] == -1)
		{
			ants[k].antPath[i] = l;
			break;
		}
}

//蚂蚁k从当前节点i选择下一步行进节点为节点j的概率
double nodeSelProb(int k, int j)
{
	double a, b, c, prob;
	a = b = c = prob = 0;
	int i = ants[k].loc;

	a = pow(pher(i, j), pImp) + pow(insp(i, j), qImp);
	for (int m = 0; m < nodeNum; m++)
	{
		if (dist(i, m) != -1 && !ifNodeInTabu(m, k))
		{
			b = pow(pher(i, m), pImp) + pow(insp(i, m), qImp);
			c += b;
		}
	}

	prob = a / c;
	return prob;
}

//判断节点j是否在蚂蚁k的禁忌表中
bool ifNodeInTabu(int j, int k)
{
	for (int i = 0; i < nodeNum; i++)
	{
		if (j == ants[k].tabu[i])
		{
			return 1;
			//break;
		}
	}
	return 0;
}

//计算路径长度
int getAntLen(struct ant a)
{
	int len = 0;
	for (int j = 0; j < pathNum; j++)
	{
		if (a.antPath[j] == -1 || a.antPath[j + 1] == -1)
			break;
		else
			len += dist(a.antPath[j], a.antPath[j + 1]);

	}
	return len;
}

//计算最优路径对应的蚂蚁编号
int getBestPath()
{
	int d[antNum];
	int min;
	int k;  //蚂蚁k的路线到达目的地节点最短
	for (int i = 0; i < antNum; i++)
	{
		d[i] = -1;
	}
	for (int i = 0; i < antNum; i++)
	{
		
		d[i] = getAntLen(ants[i]);
	}

	min = d[0];
	k = 0;
	for (int i = 1; i < antNum; i++)
	{
		if (d[i] < min && ants[i].flag == 1)  // 最优路径只从到达目标点的蚂蚁中筛选
		{
			min = d[i];
			k = i;
		}
	}
	return k;
}

//打印最优路径、最短距离
void printBestPath(int k, int f)
{
	cout << "  最短路径为：";
	for (int i = 0; i < pathNum; i++)
	{
		if (ants[k].antPath[i] == -1)
			break;

		cout << ants[k].antPath[i];
		if (ants[k].antPath[i+1] != -1)
			cout << "->";
	}
	cout << endl;
	cout << "  对应距离为：" << getAntLen(ants[k]) << endl;
}

//更新信息素矩阵
void updatePher()
{
	for (int i = 0; i < antNum; i++)
	{
		if(ants[i].flag == 1)  //只对到达目的点的蚂蚁 所走过路径 更新信息素
			for (int j = 0; j < pathNum; j++)
			{
				if (ants[i].antPath[j] == -1 || ants[i].antPath[j + 1] == -1)
					break;
				else
					nextPher(ants[i].antPath[j], ants[i].antPath[j + 1])
					+= pQ / (getAntLen(ants[i])*1.3);
			}
		
	}
	nextPher = pVol * pher + nextPher;
}


//迭代
void evolution()
{
    Mat result;
	int f=2;
	cout << "【从源点0到定点" << f << "】" << endl;
	cout << "开始迭代........." << endl;

	//初始化参数
	initAnts();
	initMarix();

	int g = 0; //当前代数
	start = clock();

	while (g < gen)
	{
		//1、蚁群内所有蚂蚁都到达目的地
		int p = 0; //蚁群前进步数
		while (p < pathNum)
		{
			for (int i = 0; i < antNum; i++)
			{
				if (ants[i].flag == 1)//到达目的地
					continue;
				nodeSelect(i, f);
				initNodeProb();
			}
			p++;
		}

		if (g == gen - 1)
		{
			cout << "达到最高迭代次数!" << endl;
			printBestPath(getBestPath(), f);
		}
				

		//3、更新信息素矩阵
		updatePher();

		//4、初始化蚁群；更新信息素矩阵
		initAnts();
		pher = nextPher;
		nextPher = Matrix4d::Zero();
        updateMatrix();
		putText(picture,"generation="+to_string(g+1),Point(80,460),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2);
        result=mergePicture(befor,picture);
        imshow("picture",result);
        waitKey(1000);
		g++;
		
	}
    destroyAllWindows();
	finish = clock();
	duration = ((double)finish - start) / CLOCK_PER_SEC/1000;
	cout << "  耗时：" << duration << "秒" << endl;
    imshow("picture",result);
    waitKey(0);
}

void drawNode(){

    circle(picture,node0,10,Scalar(169,169,169),-1);
    circle(picture,node1,10,Scalar(169,169,169),-1);
    circle(picture,node2,10,Scalar(169,169,169),-1);
    circle(picture,node3,10,Scalar(169,169,169),-1);
    line(picture,node0,node1,Scalar(0,0,0),1);
    line(picture,node1,node2,Scalar(0,0,0),1);
    line(picture,node2,node3,Scalar(0,0,0),1);
    line(picture,node3,node0,Scalar(0,0,0),1);
    line(picture,node0,node2,Scalar(0,0,0),1);
    putText(picture,"node0",Point(node0.x-40,node0.y-15),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);
    putText(picture,"node1",Point(node1.x-40,node1.y-15),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);
    putText(picture,"node2",Point(node2.x-40,node2.y+30),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);
    putText(picture,"node3",Point(node3.x-40,node3.y+30),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,0),2);

}

void updateMatrix(){
    picture=Mat(480,480,CV_8UC3,Scalar(255,255,255));
    line(picture,node0,node1,Scalar(0,0,0),ceil(pher(0,1)/2));
    line(picture,node1,node2,Scalar(0,0,0),ceil(pher(1,2)/2));
    line(picture,node2,node3,Scalar(0,0,0),ceil(pher(3,2)/2));
    line(picture,node3,node0,Scalar(0,0,0),ceil(pher(0,3)/2));
    line(picture,node0,node2,Scalar(0,0,0),ceil(pher(0,2)/2));
    
}

Mat mergePicture(Mat& src1,Mat& src2){
    Mat result;
    result.create(src1.rows,src1.rows+src2.rows,CV_8UC3);
    src1.copyTo(result(Rect(0,0,src1.rows,src1.cols)));
    src2.copyTo(result(Rect(src1.cols,0,src1.rows,src1.cols)));
    return result;
}