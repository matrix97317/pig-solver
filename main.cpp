//
// Created by jinyuanfeng.
//
//#define USE_SSE
#include "pig_tensor.hpp"

int main(){

    //Testing matrix multiplication
    NMatrix<float> a;
    a.create({20,20,200});
    a.set_value(1);
    NMatrix<float> b;
    b.create({20,20,200});
    b.set_value(2);
    auto s = chrono::system_clock::now();
    NMatrix<float> c = a.dot(b);
    auto e = chrono::system_clock::now();
    cout<<"NMatrix Execution time: "<<chrono::duration<double>{e-s}.count()*1000<<" ms"<<endl;
    c.shape();
    // END

    return 0;
}