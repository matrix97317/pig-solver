//
// Created by jinyuanfeng.
//
#include "tiny_solver.h"
void test(initializer_list<int> t){
    cout<<t.begin()<<endl;
}
int main(){
//      initializer_list<size_t> t = {1,2,3};
////      test(t);
//      NShape a(t);
//      a.show_dims();
    NMatrix<int> a({{{1,2,3},{4,5,6}},{{9,7,8},{11,12,14}}});
//        a.reshape({6});
    a.chg_axis({2,1,0},true);
//        a.chg_axis({0,1},false);
    a.shape();
    a.map_index();
    a.axis_weight();
    a.reshape({2,2,3});
    a.shape();
    a.map_index();
    a.axis_weight();
    a.reshape({1,12});
    a.shape();
    a.map_index();
    a.axis_weight();

//         a.shape();
//        a.reshape({6});
//        a.shape();

//        a.reshape({3,2});
//        a.reshape({6});
//        a.reshape({6,1});
//        a.reshape({2,3});
//      a.shape();
//      a.chg_axis({1,0}, false);
//      a.mask_index();
//      a.shape();
//      cout<<"re"<<endl;
//      a.reshape({6});
//      a.mask_index();
//      a.shape();
//      cout<<"end re"<<endl;
//
//      cout<<"re1"<<endl;
//      a.reshape({3,2});
//      a.mask_index();
//      a.shape();
//      cout<<"end re1"<<endl;
//
//      a.chg_axis({1,0},false);
//      a.shape();
//      a.mask_index();
//    cout<<"re2"<<endl;
//      a.reshape({6});
//      a.shape();
//      a.mask_index();
//    cout<<"end re2"<<endl;
//    cout<<"re4"<<endl;
//    a.reshape({1,6});
//    a.shape();
//    a.mask_index();
//    cout<<"end re4"<<endl;

    cout<<endl;
    for(unsigned long i=0; i<1;i++){
        for(unsigned long j=0;j<12;j++) {
            cout << a.get({i,j}) << endl;
        }
    }

//      a.set({0,0},1000);
//      for(unsigned long i=0; i<2;i++){
//          for(unsigned long j=0;j<2;j++){
//              for(unsigned long k=0;k<3;k++) {
//                  cout << a.get({i,j,k}) << endl;
//              }
//          }
//      }

//    return 0;

//    test<int,3>({{{1,2,3,4},{2,3,4,6}},{{1,2,3,4},{2,3,1000,6}},{{1,2,3,4},{2,3,4,6}}});
//    test<int,5>({{{{{1,2,4},{4,5,4}},{{6,7,6},{8,9,100}}},{{{1,2,2},{4,5,2}},{{6,7,2},{8,2,2}}}},{{{{1,5,6},{4,5,6}},{{6,7,6},{8,9,6}}},{{{1,2,6},{4,6,5}},{{6,7,6},{8,9,11}}}}});
}

