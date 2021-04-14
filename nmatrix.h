//
// Created by jinyuanfeng.
//
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>

#define DIM0_ADDR(a)    (a.begin())  //N
#define DIM1_ADDR(a)    (*DIM0_ADDR(a)).begin() //C
#define DIM2_ADDR(a)    (*DIM1_ADDR(a)).begin() //T
#define DIM3_ADDR(a)    (*DIM2_ADDR(a)).begin() //H
#define DIM4_ADDR(a)    (*DIM3_ADDR(a)).begin() //W

#define DIM0_SIZE(a) (a.size())
#define DIM1_SIZE(a) (*DIM0_ADDR(a)).size()
#define DIM2_SIZE(a) (*DIM1_ADDR(a)).size()
#define DIM3_SIZE(a) (*DIM2_ADDR(a)).size()
#define DIM4_SIZE(a) (*DIM3_ADDR(a)).size()

#define DIM1_R(a,index) (*(a.begin()+index))
#define DIM2_R(a,index0,index1)  (*(DIM1_R(a,index0).begin()+index1))
#define DIM3_R(a,index0,index1,index2)  (*(DIM2_R(a,index0,index1).begin()+index2))
#define DIM4_R(a,index0,index1,index2,index3)  (*(DIM3_R(a,index0,index1,index2).begin()+index3))
#define DIM5_R(a,index0,index1,index2,index3,index4)  (*(DIM4_R(a,index0,index1,index2,index3).begin()+index4))

using namespace std;

template <class T, std::size_t I>
struct new_initializer_list
{
    using type = std::initializer_list<typename new_initializer_list<T, I - 1>::type>;
};

template <class T>
struct new_initializer_list<T, 0>
{
    using type = T;
};

template <class T, std::size_t I>
using new_initializer_list_t = typename new_initializer_list<T, I>::type;

template <typename T>
class NStorage{
private:
    T* handle;
    unsigned int mem_size;
public:
    NStorage()=default;
    ~NStorage()=default;
    void alloc(unsigned int size){
        mem_size = size;
        handle = (T*)malloc(sizeof(T)*size);
    }
    T read(unsigned int pos){
        return *(handle+pos);
    }
    int write(unsigned int pos,T value){
        *(handle+pos)=value;
        return 0;
    }
    int continuous_copy_5(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
        T* new_handle = (T*)malloc(sizeof(T)*mem_size);
        long index = 0;
        long dst_index=0;
        long dims_size = dims.size();
        for(int i=0;i<dims[map_index[0]];i++){
            for(int j=0;j<dims[map_index[1]];j++){
                for(int k=0;k<dims[map_index[2]];k++){
                    for(int l=0;l<dims[map_index[3]];l++){
                        for(int m=0;m<dims[map_index[4]];m++){
                            index = i*axis_weight[dims_size-map_index[0]-1]+
                                    j*axis_weight[dims_size-map_index[1]-1]+
                                    k*axis_weight[dims_size-map_index[2]-1]+
                                    l*axis_weight[dims_size-map_index[3]-1]+
                                    m*axis_weight[dims_size-map_index[4]-1];
                            *(new_handle+dst_index) = read(index);
                            dst_index++;
                        }
                    }
                }
            }
        }
        free(handle);
        handle=new_handle;
        return 0;
    }
    int continuous_copy_4(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
        T* new_handle = (T*)malloc(sizeof(T)*mem_size);
        long index = 0;
        long dst_index=0;
        long dims_size = dims.size();
        for(int i=0;i<dims[map_index[0]];i++){
            for(int j=0;j<dims[map_index[1]];j++){
                for(int k=0;k<dims[map_index[2]];k++){
                    for(int l=0;l<dims[map_index[3]];l++){
                        index = i*axis_weight[dims_size-map_index[0]-1]+
                                j*axis_weight[dims_size-map_index[1]-1]+
                                k*axis_weight[dims_size-map_index[2]-1]+
                                l*axis_weight[dims_size-map_index[3]-1];
                        *(new_handle+dst_index) = read(index);
                        dst_index++;
                    }
                }
            }
        }
        free(handle);
        handle=new_handle;
        return 0;
    }
    int continuous_copy_3(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
        T* new_handle = (T*)malloc(sizeof(T)*mem_size);
        memset(new_handle,0,sizeof(T)*mem_size);
        long index = 0;
        long dst_index=0;
        long dims_size = dims.size();
        for(int i=0;i<dims[map_index[0]];i++){
            for(int j=0;j<dims[map_index[1]];j++){
                for(int k=0;k<dims[map_index[2]];k++){
                    index = i*axis_weight[dims_size-map_index[0]-1]+
                            j*axis_weight[dims_size-map_index[1]-1]+
                            k*axis_weight[dims_size-map_index[2]-1];
                    cout<<read(index)<<" ";
                    *(new_handle+dst_index) = read(index);
                    dst_index++;
                }
            }
        }
        cout<<endl;
        free(handle);
        handle=new_handle;
        return 0;
    }
    int continuous_copy_2(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
        T* new_handle = (T*)malloc(sizeof(T)*mem_size);
        long index = 0;
        long dst_index=0;
        long dims_size = dims.size();
        for(int i=0;i<dims[map_index[0]];i++){
            for(int j=0;j<dims[map_index[1]];j++){
                index = i*axis_weight[dims_size-map_index[0]-1]+
                        j*axis_weight[dims_size-map_index[1]-1];
                *(new_handle+dst_index) = read(index);
                dst_index++;
            }
        }
        free(handle);
        handle=new_handle;
        return 0;
    }
    int continuous_copy_1(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
        T* new_handle = (T*)malloc(sizeof(T)*mem_size);
        long index = 0;
        long dst_index=0;
        long dims_size = dims.size();
        for(int i=0;i<dims[map_index[0]];i++){
            index = i*axis_weight[dims_size-map_index[0]-1];
            *(new_handle+dst_index) = read(index);
            dst_index++;
        }
        free(handle);
        handle=new_handle;
        return 0;
    }

    void release(){
        free(handle);
    }

};
class NShape{
private:
    vector<size_t> axis_weight;
    vector<size_t> map_index;
    vector<size_t> dims;
    long dims_product=1;
public:
    bool is_continuous;
    NShape()=default;
    ~NShape()=default;
    NShape(initializer_list<size_t> params){
        is_continuous = true;
        dims.assign(params); //dims={axis0_size,axis1_size,...}
        init_axis_weight(params);
        init_map_index();
        for(auto v:params){
            dims_product*=v;
        }
    }
    void init_map_index(){
        for(int i=0;i<dims.size();i++){
            map_index.push_back(i); //0,1,2,3,4
        }
    }
    void init_axis_weight(const initializer_list<size_t> &params){
        if(params.size()!=1){ // >=2
            initializer_list<size_t>::iterator  iter=params.begin();
            axis_weight.push_back(1);
            int tmp = iter[params.size()-1];
            axis_weight.push_back(tmp);
            for(int i=params.size()-2;i>0;i--){
                tmp*=iter[i];
                axis_weight.push_back(tmp);
            }
        }else{//==1
            axis_weight.push_back(1);
        }
    }
    void change_axis(const initializer_list<size_t> &params){
        is_continuous = false;
        ostringstream oss;
        try {
            //check params size
            if (params.size() != map_index.size()) {
                oss << "params size != map_index size " <<"info: map_index size("<< map_index.size() << ") params size(" << params.size() <<")"<<endl;
                throw oss.str();
            }
            //check param repeat value
            set<size_t> s(params.begin(), params.end());
            if (s.size() != params.size()) {
                oss << "exist repeat value" << endl;
                throw oss.str();
            }
            //check value limit
            for (auto v:params) {
                if (v >= params.size()) {
                    oss << "value exceeds limits " << "info: max value("<<params.size() - 1 <<")"<<endl;
                    throw oss.str();
                }
            }
        }catch (string e){
            cout<<"[CLASS:NShape FUNC:change_axis]=> "<<e<<endl;
            exit(-1);
        }
        map_index.assign(params.begin(),params.end());
    }
    void refresh_attribute(){
        vector<size_t> new_dims;
        for(int i=0;i<map_index.size();i++){
            new_dims.push_back(dims[map_index[i]]);
        }
        dims=new_dims;
        cout<<"refresh_dims"<<endl;
        for_each(dims.begin(),dims.end(), p_vector);
        cout<<endl;
        refresh_map_index();
        refresh_axis_weight(dims);

    }
    void refresh_map_index(){
        for(int i=0;i<dims.size();i++){
            map_index[i]=i; //0,1,2,3,4
        }
        cout<<"refresh_map_index"<<endl;
        for_each(map_index.begin(),map_index.end(), p_vector);
        cout<<endl;
    }
    void refresh_axis_weight(const vector<size_t> &params){
        axis_weight.clear();
        if(params.size()!=1){ // >=2
            axis_weight.push_back(1);
            int tmp = params[params.size()-1];
            axis_weight.push_back(tmp);
            for(int i=params.size()-2;i>0;i--){
                tmp*=params[i];
                axis_weight.push_back(tmp);
            }
        }else{//==1
            axis_weight.push_back(1);
        }
        cout<<"refresh_axis_weight"<<endl;
        for_each(axis_weight.begin(),axis_weight.end(), p_vector);
        cout<<endl;
    }

    long get_index(const initializer_list<size_t> &params){
        ostringstream oss;
        try{
            if (params.size()!=axis_weight.size()) {
                oss<<"params list size != axis_weight size"<<endl;
                throw oss.str();
            }else{
                long ret = 0;
                initializer_list<size_t>::iterator  iter=params.begin();
                vector<size_t>::iterator iter_axis_weight = axis_weight.begin();
                for(int i = 0; i<params.size();i++){
                    if (iter[i]>=dims[map_index[i]]){
                        oss<<"axis exceeds limit !!! "<<"info: axis("<<i<<") "<<"input("<<iter[i]<<") exceeds limit of ("<<dims[map_index[i]]<<")"<<endl;
                        throw oss.str();
                    }
                    ret += iter[i]*iter_axis_weight[params.size()-map_index[i]-1];
                }
                return ret;
            }
        }catch (string e){
            cout<<"[CLASS:NShape FUNC:get_index]=> "<<e<<endl;
            exit(-1);
        }
        return 0;
    }
    int reshape(const initializer_list<size_t> &params){
        ostringstream oss;
        //check dims
        try {
            int new_dims = 1;
            for (auto v:params) {
                new_dims *= v;
            }
            if (new_dims != dims_product) {
                oss<<"don't equal dims_prodcut!!! "<<"info: dims_product("<<dims_product<<")"<<endl;
                throw oss.str();
            }
        }catch (string e){
            cout<<"[CLASS:NShape FUNC:reshape]=> "<<e<<endl;
            exit(-1);
        }
        //强制连续
        //重置shape
        axis_weight.clear();
        dims.clear();
        map_index.clear();
        dims.assign(params);
        init_axis_weight(params);
        init_map_index();
        return 0;
    }
    vector<size_t> get_axis_weight(){
        return axis_weight;
    }
    vector<size_t> get_map_index(){
        return map_index;
    }
    vector<size_t> get_dims(){
        return dims;
    }
    long get_dims_product(){
        return dims_product;
    }
    void show_dims(){
        cout<<"dims: [";
        for(int i=0;i<dims.size()-1;i++){
            cout<<dims[map_index[i]]<<",";
        }
        cout<<dims[map_index[dims.size()-1]]<<"]"<<endl;

    }
    void show_map_index(){
        cout<<"map_index: [";
        for(int i=0;i<map_index.size()-1;i++){
            cout<<map_index[i]<<",";
        }
        cout<<map_index[map_index.size()-1]<<"]"<<endl;
    }
    void show_axis_weight(){
        cout<<"axis_weight: [";
        for(int i=0;i<axis_weight.size()-1;i++){
            cout<<axis_weight[i]<<",";
        }
        cout<<axis_weight[axis_weight.size()-1]<<"]"<<endl;
    }
    static void p_vector(size_t v){
        cout<<v<<" ";
    }

};

template <typename T>
class NMatrix{
private:
    NStorage<T> storage;
    NShape s;
public:
    NMatrix()=default;
    ~NMatrix()=default;
    NMatrix(const NMatrix<T> &t){
        storage = t.storage;
        s = t.s;
    }
    NMatrix(const new_initializer_list_t<T,1> &t){
        initializer_list<size_t> dims = {DIM0_SIZE(t)};
        initializer_list<size_t>::iterator iter_dims = dims.begin();
        create(dims);
        long dst_index=0;
        for(int i=0;i<iter_dims[0];i++){
            storage.write(dst_index, DIM1_R(t,i));
            dst_index++;
        }
    }
    NMatrix(const new_initializer_list_t<T,2> &t){
        initializer_list<size_t> dims = {DIM0_SIZE(t),DIM1_SIZE(t)};
        initializer_list<size_t>::iterator iter_dims = dims.begin();
        create(dims);
        long dst_index=0;
        for(int i=0;i<iter_dims[0];i++){
            for(int j=0;j<iter_dims[1];j++){
                storage.write(dst_index, DIM2_R(t,i,j));
                dst_index++;
            }
        }

    }
    NMatrix(const new_initializer_list_t<T,3> &t){
        initializer_list<size_t> dims = {DIM0_SIZE(t),DIM1_SIZE(t),DIM2_SIZE(t)};
        initializer_list<size_t>::iterator iter_dims = dims.begin();
        create(dims);
        long dst_index=0;
        for(int i=0;i<iter_dims[0];i++){
            for(int j=0;j<iter_dims[1];j++){
                for(int k=0;k<iter_dims[2];k++){
                    storage.write(dst_index, DIM3_R(t,i,j,k));
                    dst_index++;
                }
            }
        }
    }
    NMatrix(const new_initializer_list_t<T,4> &t){
        initializer_list<size_t> dims = {DIM0_SIZE(t),DIM1_SIZE(t),DIM2_SIZE(t),DIM3_SIZE(t)};
        initializer_list<size_t>::iterator iter_dims = dims.begin();
        create(dims);
        long dst_index=0;
        for(int i=0;i<iter_dims[0];i++){
            for(int j=0;j<iter_dims[1];j++){
                for(int k=0;k<iter_dims[2];k++){
                    for(int l=0;l<iter_dims[3];l++){
                        storage.write(dst_index, DIM4_R(t,i,j,k,l));
                        dst_index++;
                    }
                }
            }
        }
    }
    NMatrix(const new_initializer_list_t<T,5> &t){
        initializer_list<size_t> dims = {DIM0_SIZE(t),DIM1_SIZE(t),DIM2_SIZE(t),DIM3_SIZE(t),DIM4_SIZE(t)};
        initializer_list<size_t>::iterator iter_dims = dims.begin();
        create(dims);
        long dst_index=0;
        for(int i=0;i<iter_dims[0];i++){
            for(int j=0;j<iter_dims[1];j++){
                for(int k=0;k<iter_dims[2];k++){
                    for(int l=0;l<iter_dims[3];l++){
                        for(int m=0;m<iter_dims[4];m++){
                            storage.write(dst_index, DIM5_R(t,i,j,k,l,m));
                            dst_index++;
                        }
                    }
                }
            }
        }
    }

    void create(const initializer_list<size_t> &t){
        ostringstream oss;
        try {
            if (t.size() > 5) {
                oss<<"dims width is limited as 5"<<endl;
                throw oss.str();
            }
        }catch (string e){
            cout<<"[CLASS:NMatrix FUNC:create]=> "<<e<<endl;
            exit(-1);
        }
        s=NShape(t);
        storage.alloc(s.get_dims_product());
    }
    T get(const initializer_list<size_t> &query_list){
        long index = s.get_index(query_list);
        return storage.read(index);
    }
    void set(const initializer_list<size_t> &query_list,T value){
        long index = s.get_index(query_list);
        storage.write(index,value);
    }
    void shape(){
        s.show_dims();
    }
    void map_index(){
        s.show_map_index();
    }
    void axis_weight(){
        s.show_axis_weight();
    }
    void enable_continuous(){
        switch(s.get_dims().size()){
            case 5: storage.continuous_copy_5(s.get_axis_weight(),s.get_map_index(),s.get_dims());
                s.refresh_attribute();
                break;
            case 4: storage.continuous_copy_4(s.get_axis_weight(),s.get_map_index(),s.get_dims());
                s.refresh_attribute();
                break;
            case 3: storage.continuous_copy_3(s.get_axis_weight(),s.get_map_index(),s.get_dims());
                s.refresh_attribute();
                break;
            case 2: storage.continuous_copy_2(s.get_axis_weight(),s.get_map_index(),s.get_dims());
                s.refresh_attribute();
                break;
            case 1: storage.continuous_copy_1(s.get_axis_weight(),s.get_map_index(),s.get_dims());
                s.refresh_attribute();
                break;
        }
    }
    void chg_axis(const initializer_list<size_t> &query_list,bool en_continuous=false){
        s.change_axis(query_list);
        if(en_continuous){
            enable_continuous();
            s.is_continuous=true;
        }
    }

    void reshape(const initializer_list<size_t> &query_list){
        if (!s.is_continuous) {
            cout<<"enbale continuous"<<endl;
            enable_continuous();
        }
        s.reshape(query_list);
    }

};
namespace TS{ //Tiny Solver (TS)
    ;
};

// Operator Interface

