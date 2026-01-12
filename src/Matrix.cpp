#include "Matrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
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
    assert(this->row()==this->cols());//inverse only for square matrices
    size_t n = this->row();

    //Now the augmented matrix [A|I]
    Matrix aug(n,2*n);

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            aug(i,j)=(*this)(i,j);
        }
    }//left hand side of the aug matrix

    for(int i=0;i<n;i++)
    {
        for(int j=n;j<2*n;j++)
        {
            if(i==j-n)
            {
                aug(i,j)=1.0;
            }
            else
            {
                aug(i,j)=0.0;
            }
        }
    }//putting the identity matrix on the right hand side

    //Now we use "gaussian elimination"
    for(int i=0;i<n;i++)
    {
        double ele=aug(i,i);
        assert(std::abs(ele) > 1e-12);// to ensure no division by 0
        for(int j=0;j<2*n;j++)
        {
            aug(i,j)=aug(i,j)/ele;
        }
        for(int r=0;r<n;r++)
        {
            if(r==i)
            {
                continue;
            }
            double diff=aug(r,i);
            for(int c=0;c<2*n;c++)
            {
                aug(r,c)=aug(r,c)-diff*aug(i,c);
            }
        }
    }
    Matrix inv(n,n);
        for(int i=0;i<n;i++)
        {
            for(int j=n;j<2*n;j++)
            {
                inv(i,j-n)=aug(i,j);
            }
        }
    return inv;

}


