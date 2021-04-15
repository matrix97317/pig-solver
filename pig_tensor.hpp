//
// Created by jinyuanfeng.
//

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
//#define USE_SSE
#ifdef USE_SSE
#include <x86intrin.h>
#endif
#define ALIGN_WIDTH (8)
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
namespace __range_to_initializer_list {

    constexpr size_t DEFAULT_MAX_LENGTH = 128;

    template <typename V> struct backingValue { static V value; };
    template <typename V> V backingValue<V>::value;

    template <typename V, typename... Vcount> struct backingList { static std::initializer_list<V> list; };
    template <typename V, typename... Vcount>
    std::initializer_list<V> backingList<V, Vcount...>::list = {(Vcount)backingValue<V>::value...};

    template <size_t maxLength, typename It, typename V = typename It::value_type, typename... Vcount>
    static typename std::enable_if< sizeof...(Vcount) >= maxLength,
            std::initializer_list<V> >::type generate_n(It begin, It end, It current)
    {
        throw std::length_error("More than maxLength elements in range.");
    }

    template <size_t maxLength = DEFAULT_MAX_LENGTH, typename It, typename V = typename It::value_type, typename... Vcount>
    static typename std::enable_if< sizeof...(Vcount) < maxLength,
            std::initializer_list<V> >::type generate_n(It begin, It end, It current)
    {
        if (current != end)
            return generate_n<maxLength, It, V, V, Vcount...>(begin, end, ++current);

        current = begin;
        for (auto it = backingList<V,Vcount...>::list.begin();
             it != backingList<V,Vcount...>::list.end();
             ++current, ++it)
            *const_cast<V*>(&*it) = *current;

        return backingList<V,Vcount...>::list;
    }

}

template <typename It>
std::initializer_list<typename It::value_type> range_to_initializer_list(It begin, It end)
{
    return __range_to_initializer_list::generate_n(begin, end, begin);
}
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
    unsigned long mem_size;
public:
    NStorage()=default;
    NStorage(const NStorage<T> &t);
    ~NStorage()=default;
    void alloc(unsigned int size);
    void exalloc(unsigned int size);
    T read(unsigned int pos);
    int write(unsigned int pos,T value);
    void set(T value);
    int continuous_copy_5(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims);
    int continuous_copy_4(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims);
    int continuous_copy_3(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims);
    int continuous_copy_2(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims);
    int continuous_copy_1(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims);
    void release();
    T* get_handle();
};

class NShape{
private:
    vector<size_t> axis_weight;
    vector<size_t> map_index;
    vector<size_t> dims;
    unsigned long dims_product=1;
    unsigned long sub_dims_product=1;
public:
    bool is_continuous;
    NShape()=default;
    NShape(const NShape&t);
    ~NShape()=default;
    NShape(initializer_list<size_t> params);
    void init_map_index();
    void init_axis_weight(const initializer_list<size_t> &params);
    void change_axis(const initializer_list<size_t> &params);
    void refresh_attribute();
    void refresh_map_index();
    void refresh_axis_weight(const vector<size_t> &params);
    long get_index(const initializer_list<size_t> &params);
    int reshape(const initializer_list<size_t> &params);
    vector<size_t> get_axis_weight();
    vector<size_t> get_map_index();
    vector<size_t> get_dims();
    size_t get_dims(int axis);
    unsigned long get_dims_product();
    unsigned long get_sub_dims_product();
    void show_dims();
    void show_map_index();
    void show_axis_weight();
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
    NMatrix(const NMatrix<T> &t);
    NMatrix(const new_initializer_list_t<T,1> &t);
    NMatrix(const new_initializer_list_t<T,2> &t);
    NMatrix(const new_initializer_list_t<T,3> &t);
    NMatrix(const new_initializer_list_t<T,4> &t);
    NMatrix(const new_initializer_list_t<T,5> &t);

    void create(const initializer_list<size_t> &t);
    T get(const initializer_list<size_t> &query_list);
    void set(const initializer_list<size_t> &query_list,T value);
    void shape();
    void map_index();
    void axis_weight();
    void enable_continuous();
    void chg_axis(const initializer_list<size_t> &query_list,bool en_continuous=false);

    void reshape(const initializer_list<size_t> &query_list);
    bool check_dims_consistency(const vector<size_t> & a,const vector<size_t> & b);
    bool check_dims_consistency_dot(const vector<size_t> & a,const vector<size_t> & b);
    //viusal data
    void basic_dim1(T* addr,size_t w);
    void basic_dim2(T* addr,size_t h, size_t w);
    void basic_dim3(T* addr,size_t t,size_t h, size_t w);
    void basic_dimN(T* addr,const vector<size_t> &dims);
    void show();
    //define calculate
    void basic_dot(T* op1,T* op2,T* out,size_t rows,size_t cols, size_t K);
    void set_value(T value);
    //c = a+b  mem_block 3
    NMatrix<T> operator+(NMatrix<T> &a);
    NMatrix<T> dot(NMatrix<T> &a);
    //a = a+b mem_block 2
//    void add(T a);
//    void add_constant(T a);
};

//Implemention
template <typename T>
NStorage<T>::NStorage(const NStorage<T> &t){
    handle = t.handle;
    mem_size = t.mem_size;
}
template <typename T>
void NStorage<T>::alloc(unsigned int size){
    mem_size = size;
    handle = (T*)malloc(sizeof(T)*size);
    memset(handle,0,sizeof(T)*mem_size);
}

template <typename T>
void NStorage<T>::exalloc(unsigned int size){
    handle = (T*) realloc(handle,sizeof(T)*size);
}

template <typename T>
T NStorage<T>::read(unsigned int pos){
    return *(handle+pos);
}
template <typename T>
int NStorage<T>::write(unsigned int pos,T value){
    *(handle+pos)=value;
    return 0;
}
template <typename T>
void NStorage<T>::set(T value){
    fill(handle,handle+mem_size,value);
}
template <typename T>
int NStorage<T>::continuous_copy_5(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
    T* new_handle = (T*)malloc(sizeof(T)*mem_size);
    memset(new_handle,0,sizeof(T)*mem_size);
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
template <typename T>
int NStorage<T>::continuous_copy_4(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
    T* new_handle = (T*)malloc(sizeof(T)*mem_size);
    memset(new_handle,0,sizeof(T)*mem_size);
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
template <typename T>
int NStorage<T>::continuous_copy_3(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
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
//                            cout<<read(index)<<" ";
                *(new_handle+dst_index) = read(index);
                dst_index++;
            }
        }
    }
//        cout<<endl;
    free(handle);
    handle=new_handle;
    return 0;
}
template <typename T>
int NStorage<T>::continuous_copy_2(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
    T* new_handle = (T*)malloc(sizeof(T)*mem_size);
    memset(new_handle,0,sizeof(T)*mem_size);
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
template <typename T>
int NStorage<T>::continuous_copy_1(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
    T* new_handle = (T*)malloc(sizeof(T)*mem_size);
    memset(new_handle,0,sizeof(T)*mem_size);
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
template <typename T>
void NStorage<T>::release(){
    free(handle);
}
template <typename T>
T* NStorage<T>::get_handle(){
    return handle;
}


NShape::NShape(const NShape&t){
    axis_weight = t.axis_weight;
    map_index = t.map_index;
    dims = t.dims;
    dims_product = t.dims_product;
    is_continuous = t.is_continuous;
}

NShape::NShape(initializer_list<size_t> params){
    is_continuous = true;
    dims.assign(params); //dims={axis0_size,axis1_size,...}
    init_axis_weight(params);
    init_map_index();
    initializer_list<size_t>::iterator iter = params.begin();
    for(int i=0;i<params.size()-1;i++){
        dims_product*=iter[i];
        sub_dims_product*=iter[i];
    }
    dims_product*=iter[params.size()-1];
}
void NShape::init_map_index(){
    for(int i=0;i<dims.size();i++){
        map_index.push_back(i); //0,1,2,3,4
    }
}
void NShape::init_axis_weight(const initializer_list<size_t> &params){
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
void NShape::change_axis(const initializer_list<size_t> &params){
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
void NShape::refresh_attribute(){
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
void NShape::refresh_map_index(){
    for(int i=0;i<dims.size();i++){
        map_index[i]=i; //0,1,2,3,4
    }
    cout<<"refresh_map_index"<<endl;
    for_each(map_index.begin(),map_index.end(), p_vector);
    cout<<endl;
}
void NShape::refresh_axis_weight(const vector<size_t> &params){
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
long NShape::get_index(const initializer_list<size_t> &params){
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
int NShape::reshape(const initializer_list<size_t> &params){
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
vector<size_t> NShape::get_axis_weight(){
    return axis_weight;
}
vector<size_t> NShape::get_map_index(){
    return map_index;
}
vector<size_t> NShape::get_dims(){
    return dims;
}
size_t NShape::get_dims(int axis){
    return dims[axis];
}
unsigned long NShape::get_sub_dims_product(){
    return sub_dims_product;
}
unsigned long NShape::get_dims_product(){
    return dims_product;
}
void NShape::show_dims(){
    cout<<"dims: [";
    for(int i=0;i<dims.size()-1;i++){
        cout<<dims[map_index[i]]<<",";
    }
    cout<<dims[map_index[dims.size()-1]]<<"]"<<endl;

}
void NShape::show_map_index(){
    cout<<"map_index: [";
    for(int i=0;i<map_index.size()-1;i++){
        cout<<map_index[i]<<",";
    }
    cout<<map_index[map_index.size()-1]<<"]"<<endl;
}
void NShape::show_axis_weight(){
    cout<<"axis_weight: [";
    for(int i=0;i<axis_weight.size()-1;i++){
        cout<<axis_weight[i]<<",";
    }
    cout<<axis_weight[axis_weight.size()-1]<<"]"<<endl;
}

template <typename T>
NMatrix<T>::NMatrix(const NMatrix<T> &t){
    storage = t.storage;
    s = t.s;
}
template <typename T>
NMatrix<T>::NMatrix(const new_initializer_list_t<T,1> &t){
    initializer_list<size_t> dims = {DIM0_SIZE(t)};
    initializer_list<size_t>::iterator iter_dims = dims.begin();
    create(dims);
    long dst_index=0;
    for(int i=0;i<iter_dims[0];i++){
        storage.write(dst_index, DIM1_R(t,i));
        dst_index++;
    }
}
template <typename T>
NMatrix<T>::NMatrix(const new_initializer_list_t<T,2> &t){
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

template <typename T>
NMatrix<T>::NMatrix(const new_initializer_list_t<T,3> &t){
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

template <typename T>
NMatrix<T>::NMatrix(const new_initializer_list_t<T,4> &t){
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

template <typename T>
NMatrix<T>::NMatrix(const new_initializer_list_t<T,5> &t){
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

template <typename T>
void NMatrix<T>::create(const initializer_list<size_t> &t){
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

template <typename T>
T NMatrix<T>::get(const initializer_list<size_t> &query_list){
    long index = s.get_index(query_list);
    return storage.read(index);
}

template <typename T>
void NMatrix<T>::set(const initializer_list<size_t> &query_list,T value){
    long index = s.get_index(query_list);
    storage.write(index,value);
}

template <typename T>
void NMatrix<T>::shape(){
    s.show_dims();
}

template <typename T>
void NMatrix<T>::map_index(){
    s.show_map_index();
}

template <typename T>
void NMatrix<T>::axis_weight(){
    s.show_axis_weight();
}

template <typename T>
void NMatrix<T>::enable_continuous(){
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

template <typename T>
void NMatrix<T>::chg_axis(const initializer_list<size_t> &query_list,bool en_continuous){
    if(!s.is_continuous){
        enable_continuous();
        s.is_continuous=true;
    }
    s.change_axis(query_list);
    if(en_continuous){
        enable_continuous();
        s.is_continuous=true;
    }
}

template <typename T>
void NMatrix<T>::reshape(const initializer_list<size_t> &query_list){
    if (!s.is_continuous) {
        cout<<"enbale continuous"<<endl;
        enable_continuous();
        s.is_continuous = true;
    }
    s.reshape(query_list);
}

template <typename T>
bool NMatrix<T>::check_dims_consistency(const vector<size_t> & a,const vector<size_t> & b){
    if (a.size()!=b.size())return false;
    for(int i=0;i<a.size();i++){
        if(a[i]!=b[i]){
            return false;
        }
    }
    return true;
}

template <typename T>
bool NMatrix<T>::check_dims_consistency_dot(const vector<size_t> & a,const vector<size_t> & b){
    if (a.size()<2) return false;
    if (a.size()!=b.size())return false;
    for(int i=0;i<a.size()-2;i++){
        if(a[i]!=b[i]){
            return false;
        }
    }
    if(!(a[a.size()-1]==b[b.size()-1])){
        return false;
    }
    return true;
}

template <typename T>
//viusal data
void NMatrix<T>::basic_dim1(T* addr,size_t w){
    cout<<"[";
    for(int i=0;i<w-1;i++){
        cout<<*(addr+i)<<" ";
    }
    cout<<*(addr+w-1)<<"]";
}

template <typename T>
void NMatrix<T>::basic_dim2(T* addr,size_t h, size_t w){
    cout<<"[";
    for(int i=0;i<h-1;i++){
        basic_dim1(addr+w*i,w);
        cout<<endl;
    }
    basic_dim1(addr+w*(h-1),w);
    cout<<"]";
}

template <typename T>
void NMatrix<T>::basic_dim3(T* addr,size_t t,size_t h, size_t w){
//        cout<<"t "<<t<<"h "<<h<<"w "<<w<<endl;

    cout<<"[";
    for(int i=0;i<t-1;i++){
        basic_dim2(addr+i*h*w,h,w);
        cout<<endl;
    }
    basic_dim2(addr+(t-1)*h*w,h,w);
    cout<<"]";
}

template <typename T>
void NMatrix<T>::basic_dimN(T* addr,const vector<size_t> &dims){
    for(int i=0;i<dims.size()-3;i++){
        cout<<"[ ";
    }
    basic_dim3(addr,dims[dims.size()-3],dims[dims.size()-2], dims[dims.size()-1]);
    for(int i=0;i<dims.size()-3;i++){
        cout<<"] ";
    }
    cout<<endl;
}

template <typename T>
void NMatrix<T>::show(){
    if(!s.is_continuous){
        enable_continuous();
        s.is_continuous=true;
    }
    vector<size_t> dims = s.get_dims();
    T* addr = storage.get_handle();

    switch(dims.size()){
        case 1:basic_dim1(addr,dims[dims.size()-1]);cout<<endl;break;
        case 2:basic_dim2(addr,dims[dims.size()-2],dims[dims.size()-1]);cout<<endl;break;
        case 3:basic_dim3(addr,dims[dims.size()-3],dims[dims.size()-2],dims[dims.size()-1]);cout<<endl;break;
        default:basic_dimN(addr,dims);break;
    }
}
//define calculate
template <typename T>
void NMatrix<T>::set_value(T value){
    storage.set(value);
}
//c = a+b  mem_block 3
template <typename T>
NMatrix<T> NMatrix<T>::operator+(NMatrix<T> &a){
    //enable continuous
    if(!s.is_continuous){
        enable_continuous();
        s.is_continuous=true;
    }
    if(!a.s.is_continuous){
        a.enable_continuous();
        a.s.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency(s.get_dims(),a.s.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:operator+]=> "<<e<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    vector<size_t> dims=s.get_dims();
    initializer_list<size_t> init_dims=range_to_initializer_list(dims.begin(),dims.end());
    out.create(init_dims);
    unsigned long size = s.get_dims_product();
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
    T* op3 = out.storage.get_handle();
#ifdef USE_SSE
    unsigned long i;
        for(i=0;i<size;i+=8){
            __m256 a;
            __m256 b;
            __m256 c;
            //load data
            a = _mm256_load_ps(op1+i);
            b = _mm256_load_ps(op2+i);
            c = _mm256_add_ps(a,b);
            _mm256_store_ps(op3+i,c);
        }
        int tail = size%8;
        if (tail!=0){
            for(int j=0; j<tail;j++){
                op3[i+j]=op1[i+j]+op2[i+j];
            }
        }
#else
    for(int i=0;i<size;i++){
        op3[i]=op1[i]+op2[i];
    }
#endif
    return out;
}

template <typename T>
void NMatrix<T>::basic_dot(T* op1,T* op2,T* out,size_t rows,size_t cols, size_t K){
#ifdef USE_SSE
    T* temp = (T*)malloc(sizeof(T)*ALIGN_WIDTH);
    memset(temp,0,sizeof(T)*ALIGN_WIDTH);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            unsigned long l;
            __m256 acc = _mm256_setzero_ps();
            for(l=0; l< (K/ALIGN_WIDTH); l+=ALIGN_WIDTH){
                __m256 a = _mm256_loadu_ps(op1+i*K+l);
                __m256 b =_mm256_loadu_ps(op2+j*K+l);
                __m256 c = _mm256_mul_ps(a, b);
                acc = _mm256_add_ps(acc, c);
            }
            _mm256_storeu_ps(temp, acc);
            out[i*cols+j] = temp[0] +temp[1] +temp[2] +temp[3] +
                                        temp[4] +temp[5] +temp[6] +temp[7];
            //process K's tail data
            for (; l < K; l++) {
                out[i*cols+j] += op1[i*K+l] * op2[j*K+l];
            }
        }
    }
    free(temp);
#else
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            for (int l=0; l < K; l++) {
                out[i*cols+j] += op1[i*K+l] * op2[j*K+l];
            }
        }
    }
#endif

}

template <typename T>
NMatrix<T> NMatrix<T>::dot(NMatrix<T> &a){
    //enable continuous
    if(!s.is_continuous){
        enable_continuous();
        s.is_continuous=true;
    }
    if(!a.s.is_continuous){
        a.enable_continuous();
        a.s.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency_dot(s.get_dims(),a.s.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<"should like: [a,b,c,H,W] [a,b,c,Q,W]"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:dot]=> "<<e<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    vector<size_t> dims=s.get_dims();
    size_t  dims_size = dims.size();
    dims[dims_size-1]=a.s.get_dims(dims_size-2); //H,H
    initializer_list<size_t> init_dims=range_to_initializer_list(dims.begin(),dims.end());
    out.create(init_dims);
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
    T* op3 = out.storage.get_handle();

    size_t K = s.get_dims(dims_size-1);
    size_t out_row_size = s.get_dims(dims_size-2);
    size_t out_col_size = a.s.get_dims(dims_size-2);

    if(dims_size>2){
        size_t matrix_num = 1;
        for(int i=0;i<dims_size-2;i++){
            matrix_num*=dims[i];
        }

        for(int i=0;i<matrix_num;i++){
            T* nop1 = op1+i*out_row_size*K;
            T* nop2 = op2+i*out_col_size*K;
            T* nop3 = op3+i*out_row_size*out_col_size;

            basic_dot(nop1,nop2,nop3,out_row_size,out_col_size,K);
        }
    }else{
        basic_dot(op1,op2,op3,out_row_size,out_col_size,K);
    }
    return out;
}

//a = a+b mem_block 2
//    void add(T a){
//
//    }
//    void add_constant(T a){
//
//    }




//namespace TS{ //Tiny Solver (TS)
//    ;
//};
// Operator Interface

