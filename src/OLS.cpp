#include "OLS.hpp"
#include <cassert>

// beta = (XTX)inv XT y

void OLS::fit(const Matrix&X,const Matrix& y)
{
    assert(X.row()==y.row());
    assert(y.cols()==1);
    Matrix XT=X.transpose();
    Matrix temp=XT.multiply(X);
    temp=temp.inverse();
    temp=temp.multiply(XT);
    this->beta=temp.multiply(y);//The OLS object calls the function so we need to set for that model beta 
    fitted=true;
}

Matrix OLS::predict(const Matrix&X) const
{
    assert(fitted);
    Matrix T=X.multiply(this->beta);
    return T;
}

const Matrix& OLS::coefficients() const
{
    return this->beta;
    //just for access and see it, read only cause const
}