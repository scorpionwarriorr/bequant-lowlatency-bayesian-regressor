#include "Matrix.hpp"
#include <iostream>
#include <cassert>
Matrix Matrix::transpose() const
{
    Matrix T(this->cols(),this->row());
    for(int i=0;i<T.row();i++)
    {
        for(int j=0;j<T.cols();j++)
        {
            T(i,j)=(*this)(j,i);
        }
    }
    
    return T;

}

Matrix Matrix::multiply(const Matrix& other) const
{
    assert(this->row()==other.cols());
    Matrix T(this->row(),other.cols());
    for(int i=0;i<T.row();i++)
    {
        for(int j=0;j<T.cols();j++)
        {
            double sum=0.0;
            for(int k=0;k<this->cols();k++)
            {
                sum+=(*this)(i,k)*other(k,j);
            }
            T(i,j)=sum;
        }
    }
    return T;
}

Matrix Matrix::inverse() const
{
    
}