#include<bits/stdc++.h>
#include "Matrix.hpp"
using namespace std;
int main()
{
    Matrix A(3,2);
    double u=0.0;
    for(int i=0;i<A.row();i++)
    {
        for(int j=0;j<A.cols();j++)
        {
            A(i,j)=u++;
        }
    }
    Matrix B=A.transpose();
    for(int i=0;i<B.row();i++)
    {
        for(int j=0;j<B.cols();j++)
        {
            std::cout<<B(i,j)<<" ";
        }
        std::cout<<std::endl;
    }

}