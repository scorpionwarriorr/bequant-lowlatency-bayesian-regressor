#include "OLS.hpp"
#include <cassert>

// beta = (XTX)inv XT y

void OLS::fit(const Matrix&X,const Matrix& y)
{
    assert(X.row()==y.row());
    Matrix XT=X.transpose();
    Matrix temp=XT.multiply(X);
    temp=temp.inverse();
    temp=temp.multiply(XT);
    this->beta=temp.multiply(y);//The OLS object calls the function so we need to set for that model beta 
    fitted=true;
}