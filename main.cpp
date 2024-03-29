//
// Created by jinyuanfeng.
//
#define OPENBLAS
#define USE_SSE
#include "pig_solver.hpp"

int main(){
//     //NMatrix 初始化
//     // case 1：直接数据初始化
//     PS::NMatrix<float> a({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}}); //这是一个[2,2,3]维度的数据
//     // case 2: 指定维度信息初始化
//     PS::NMatrix<float> b;
//     b.create({2,2,3});   //根据维度信息创建一个[2,2,3]维度的数据
//     b.set_value(10);  //设置NMatrix的初值
//     b.set_random();         //使用随机数初始化NMatrix
//     // case 3: 加载数据文件初始化
//     PS::NMatrix<float> c;
//     c.load_data("FILE.bin",100,{2,5,5,2}); //指定数据文件路径，数据长度，数据维度信息

//     //NMatrix 逻辑视角的改变
//     PS::NMatrix<float> a({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}}); //这是一个[2,2,3]维度的数据
//     a.reshape({2,6}); //[2,2,3]->[2,6] 底层内存数据布局不变
//     PS::NMatrix<float> b({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}}); //这是一个[2,2,3]维度的数据
//     b.chg_axis({0,2,1},false); //[2,2,3]->[2,3,2]  改变维度轴，false时不改变内存布局
//     b.chg_axis({0,2,1},true);  //[2,2,3]->[2,3,2]  改变维度轴, true时改变内存布局

//     //NMatrix 数据读写
//     PS::NMatrix<float> a({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}});//这是一个[2,2,3]维度的数据
//     float out = a.get({1,1,2}); //读取 [1,1,2]位置的数据
//     a.set({0,1,1},out);         //写入数据到 [0,1,1] 位置

//       //NMatrix 计算操作
//        PS::NMatrix<float> a({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}});//这是一个[2,2,3]维度的数据
//        PS::NMatrix<float> b({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}});//这是一个[2,2,3]维度的数据
//        auto c = a+b; //加法运算 输出[2,2,3]维数据
//        auto d = a+1; //加常数运算 输出[2,2,3]维数据
//        auto e = a-b; //减法运算 输出[2,2,3]维数据
//        auto f = a-1; //减常数运算 输出[2,2,3]维数据
//        auto g = a*b; //乘法运算 输出[2,2,3]维数据
//        auto h = a*2; //乘常数运算 输出[2,2,3]维数据
//        auto i = a/b; //除法运算 输出[2,2,3]维数据
//        auto j = a/2; //除常数运算 输出[2,2,3]维数据
//        auto k = a.exp();  //自然指数函数运算 exp(a) 输出[2,2,3]维数据
//        auto l = a.log();  //对数函数运算 log(a) 输出[2,2,3]维数据
//        a.add_inplace(b); //自加运算 a=a+b
//        auto m = a.inverse(); //倒数运算 1/a 输出[2,2,3]维数据
//        auto n = a.inverse_square(); //平方倒数运算  -1/a*a 输出[2,2,3]维数据
//        auto o = a.pow(2); //指数运算 a^2 输出[2,2,3]维数据
//        auto p = a.nabs();    //绝对值运算 ｜a｜ 输出[2,2,3]维数据
//        auto q = a.inflate(1,2); //膨胀运算, 沿轴1 膨胀2倍 [2,2,3]->[2,4,3]
//        auto r = a.reduce(1);       //压缩运算, 沿轴1 压缩 [2,2,3]->[2,1,3]
//        auto s = a.padding({0,2,0}); //padding 运算， 沿轴1，两边补0, [2,2,3]->[2,4,3]
//        auto t = s.unpadding({0,2,0}); //padding 逆运算，沿轴1，压缩两边 [2,4,3]->[2,2,3]
//        auto u = a.dot(b);   //矩阵乘法运算 [...,H1,W]x[...,H2,W]=[...,H1,H2]

//          //NMatrix 其他操作
//          PS::NMatrix<float> a({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}}); //这是一个[2,2,3]维度的数据
//          a.shape(); //显示维度信息
//          a.show();  //显示部分数据
//          a.clear(); //释放内存数据

//            //NTensor
//            PS::NTensor<float> a({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}}); //这是一个[2,2,3]维度的数据
//            //NTensor 比 NMatrix只是多了grad数据,parent_op,还有bp反向传播函数
//            //a.grad;
//            //a.parent_op;
//            //a.bp();
    //training flow
    //initialize training dataloader
    PS::NImageData<float> train_loader("train.txt",8,{32,32,3});
    vector<PS::NMatrix<float>> data_label;
    PS::NTensor<float>* input;
    PS::NTensor<float>* lables;
    int epoch_num = 1;
    Model<float> model;
    CrossEntropyLoss<float> loss;
    PS::NOptimizer<float> optimizer(model.model_params,-0.00001);
    optimizer.show_model_params_info();

    for(int e=0;e<epoch_num;e++){
        vector<vector<int>> batch_id = train_loader.get_batch_id_generator();
        for(int i=0; i<150;i++){
            int j = i%10;
            cout<<"----------------> Processing :"<<i<<endl;
            data_label = train_loader.get_batch_data(batch_id[j]);
//                     cout<<"labels";
//                     data_label[1].show();
            input = new PS::NTensor<float>(data_label[0]);
            lables = new PS::NTensor<float>(data_label[1]);
            auto out  = model.forward(input);
//                     cout<<"pred";
//                     out->show();
            auto lv  =  loss.forward(out,lables);
            cout<<"loss: ";
            lv->show();
            lv->bp();
            optimizer.step();
            optimizer.set_zero_grad();
            cout<<PS::global_mem_size<<endl;
            PS::clean_tensor<PS::NTensor<float>>();
            cout<<PS::global_mem_size<<endl;
        }
    }
    return 0;
}
