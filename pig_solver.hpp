//
// Created by jinyuanfeng.
//

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <iostream>
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
//namespace __range_to_initializer_list {
//
//    constexpr size_t DEFAULT_MAX_LENGTH = 128;
//
//    template <typename V> struct backingValue { static V value; };
//    template <typename V> V backingValue<V>::value;
//
//    template <typename V, typename... Vcount> struct backingList { static std::initializer_list<V> list; };
//    template <typename V, typename... Vcount>
//    std::initializer_list<V> backingList<V, Vcount...>::list = {(Vcount)backingValue<V>::value...};
//
//    template <size_t maxLength, typename It, typename V = typename It::value_type, typename... Vcount>
//    static typename std::enable_if< sizeof...(Vcount) >= maxLength,
//            std::initializer_list<V> >::type generate_n(It begin, It end, It current)
//    {
//        throw std::length_error("More than maxLength elements in range.");
//    }
//
//    template <size_t maxLength = DEFAULT_MAX_LENGTH, typename It, typename V = typename It::value_type, typename... Vcount>
//    static typename std::enable_if< sizeof...(Vcount) < maxLength,
//            std::initializer_list<V> >::type generate_n(It begin, It end, It current)
//    {
//        if (current != end)
//            return generate_n<maxLength, It, V, V, Vcount...>(begin, end, ++current);
//
//        current = begin;
//        for (auto it = backingList<V,Vcount...>::list.begin();
//             it != backingList<V,Vcount...>::list.end();
//             ++current, ++it)
//            *const_cast<V*>(&*it) = *current;
//
//        return backingList<V,Vcount...>::list;
//    }
//
//}

//template <typename It>
//std::initializer_list<typename It::value_type> range_to_initializer_list(It begin, It end)
//{
//    return __range_to_initializer_list::generate_n(begin, end, begin);
//}

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
class OP;
template <typename T>
class Add;

namespace PS {
    //Global Info
    unsigned long global_mem_size=0;
    unsigned long node_count=0;
    void seed(size_t value) {
        srand(value);
    }
    double generateGaussianNoise(double mu, double sigma)
    {
//        const double epsilon = std::numeric_limits<double>::min();
//        const double two_pi = 2.0*3.14159265358979323846;
//        static double z0, z1;
//        static bool generate;
//        generate = !generate;
//        if (!generate)
//            return z1 * sigma + mu;
//        double u1, u2;
//        do
//        {
//            u1 = rand() * (1.0 / RAND_MAX);
//            u2 = rand() * (1.0 / RAND_MAX);
//        }
//        while ( u1 <= epsilon );
//        z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
//        z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
//        return z0 * sigma + mu;
        return rand() % (1000) / (float)(1000);
    }

    template<typename T>
    class NStorage {
    private:
        T *handle;
        unsigned long mem_size;
    public:
        NStorage() = default;

        NStorage(const NStorage<T> &t);
//        void operator=(const NStorage<T> &t);
        void set_handle(T* new_handle);
        void set_mem_size(unsigned long new_mem_size);
        unsigned long get_mem_size();
        ~NStorage() = default;
        T* copy();
        void alloc(unsigned int size);
        void exalloc(unsigned int size);
        T read(unsigned int pos);
        int write(unsigned int pos, T value);
        int addself(unsigned int pos, T value);
        void set(T value);
        void set_random();
        int continuous_copy_5(const vector<size_t> &axis_weight, const vector<size_t> &map_index,
                              const vector<size_t> &dims);

        int continuous_copy_4(const vector<size_t> &axis_weight, const vector<size_t> &map_index,
                              const vector<size_t> &dims);

        int continuous_copy_3(const vector<size_t> &axis_weight, const vector<size_t> &map_index,
                              const vector<size_t> &dims);

        int continuous_copy_2(const vector<size_t> &axis_weight, const vector<size_t> &map_index,
                              const vector<size_t> &dims);

        int continuous_copy_1(const vector<size_t> &axis_weight, const vector<size_t> &map_index,
                              const vector<size_t> &dims);
        void release();
        T *get_handle();
    };

    class NShape {
    private:
        vector<size_t> axis_weight;
        vector<size_t> map_index;
        vector<size_t> dims;
        unsigned long dims_product = 1;
        unsigned long sub_dims_product = 1;
    public:
        bool is_continuous;

        NShape() = default;

        NShape(const NShape &t);

        ~NShape() = default;

        NShape(vector<size_t> params);

        void init_map_index();

        void init_axis_weight(const vector<size_t> &params);

        void change_axis(const vector<size_t> &params);

        void refresh_attribute();

        void refresh_map_index();

        void refresh_axis_weight(const vector<size_t> &params);

        long get_index(const vector<size_t> &params);

        int reshape(const vector<size_t> &params);

        vector<size_t> get_axis_weight();

        vector<size_t> get_map_index();

        vector<size_t> get_dims();

        size_t get_dims(int axis);

        unsigned long get_dims_product();

        unsigned long get_sub_dims_product();

        void show_dims();

        void show_map_index();

        void show_axis_weight();

        void reset();

        static void p_vector(size_t v) {
            cout << v << " ";
        }
    };

    template<typename T>
    class NMatrix {
    private:
        NStorage<T> storage;
        NShape visitor;
    public:
        NMatrix() = default;

        ~NMatrix() = default;

        NMatrix(const NMatrix<T> &t);

        NMatrix(const new_initializer_list_t<T, 1> &t);

        NMatrix(const new_initializer_list_t<T, 2> &t);

        NMatrix(const new_initializer_list_t<T, 3> &t);

        NMatrix(const new_initializer_list_t<T, 4> &t);

        NMatrix(const new_initializer_list_t<T, 5> &t);

        void clear();

        void create(const vector<size_t> &t);
        void save_data(string file_path);
        void load_data(string file_path, size_t data_length,const vector<size_t> &t);
        NMatrix<T> copy();

        bool is_empty();

        T get(const vector<size_t> &query_list);

        void set(const vector<size_t> &query_list, T value);

        void addself(const vector<size_t> &query_list, T value);

        void shape();

        void map_index();

        void axis_weight();

        void enable_continuous();

        void chg_axis(const vector<size_t> &query_list, bool en_continuous = false);

        void reshape(const vector<size_t> &query_list);

        bool check_dims_consistency(const vector<size_t> &a, const vector<size_t> &b);

        bool check_dims_consistency_dot(const vector<size_t> &a, const vector<size_t> &b);

        size_t get_dims(int axis);
        vector<size_t> get_dims();

        T get_local_max_2D(size_t n_index, size_t c_index, size_t s_h,size_t s_w,size_t h_size,size_t w_size,size_t *pos_w_h);

        //viusal data
        void basic_dim1(T *addr, size_t w);

        void basic_dim2(T *addr, size_t h, size_t w);

        void basic_dim3(T *addr, size_t t, size_t h, size_t w);

        void basic_dimN(T *addr, const vector<size_t> &dims);

        void show();

        //define calculate
        void basic_dot_omp(T *op1, T *op2, T *out, size_t rows, size_t cols, size_t K);
        void basic_dot(T *op1, T *op2, T *out, size_t rows, size_t cols, size_t K);

        void set_value(T value);
        void set_random();
        NMatrix<T> padding(vector<size_t> pad_list);
        NMatrix<T> unpadding(vector<size_t> pad_list);
        //add c = a+b
        NMatrix<T> operator+(NMatrix<T> &a);
        NMatrix<T> operator+(T a);
        //sub
        NMatrix<T> operator-(NMatrix<T> &a);
        NMatrix<T> operator-(T a);
        //mul
        NMatrix<T> operator*(NMatrix<T> &a);
        NMatrix<T> operator*(T a);
        //div
        NMatrix<T> operator/(NMatrix<T> &a);
        NMatrix<T> operator/(T a);
        //exp
        NMatrix<T> exp();
        //log
        NMatrix<T> log();
        //add_inplace a=a+b
        void add_inplace(NMatrix<T> &a);
        // a.inverse == 1/a
        NMatrix<T> inverse();
        // a.inverse_square == -1/a^2
        NMatrix<T> inverse_square();
        // a.pow(size_t n) == a^n
        NMatrix<T> pow(size_t n);
        // a.abs() == |a|
        NMatrix<T> nabs();
        // inflate [N,1,H,W]->[N,C,H,W]
        NMatrix<T> inflate(size_t axis,size_t n);
        // reduce [N,C,H,W]->[N,1,H,W]
        NMatrix<T> reduce(size_t axis);

        NMatrix<T> dot(NMatrix<T> &a);

        NMatrix<T> img2col(vector<size_t> khw_size,int c_in,int stride_h,int stride_w,bool padding,unsigned long *newhw);

        void col2img(vector<size_t> khw_size,int c_in,int stride_h,int stride_w,bool padding,unsigned long *newhw,NMatrix<T> &img2col_nmatrix);

    };

    template <typename T>
    class NTensor:public NMatrix<T>{
    private:
        string id;
    public:
        NMatrix<T> grad;
        OP<T> *parent_op;
        void init_tensor() {
            PS::node_count++;
            id = "tensor"+to_string(PS::node_count);
            parent_op= nullptr;
        }
        NTensor(){
            init_tensor();
        };
        ~NTensor()=default;
        NTensor(const new_initializer_list_t<T, 1> &t):NMatrix<T>(t){
            init_tensor();
        };
        NTensor(const new_initializer_list_t<T, 2> &t):NMatrix<T>(t){
            init_tensor();
        };
        NTensor(const new_initializer_list_t<T, 3> &t):NMatrix<T>(t){
            init_tensor();
        };
        NTensor(const new_initializer_list_t<T, 4> &t):NMatrix<T>(t){
            init_tensor();
        };
        NTensor(const new_initializer_list_t<T, 5> &t):NMatrix<T>(t){
            init_tensor();
        };
        NTensor(const NMatrix<T> &t):NMatrix<T>(t){
            init_tensor();
        }

        NTensor<T> dcopy(){
            NTensor<T> out(this->copy());
            return out;
        }
        string get_id(){
            return id;
        }

        //BP
        void bp(NMatrix<T> from_grad=NMatrix<T> ()){
            if(from_grad.is_empty()){
                from_grad = this->copy();
                from_grad.set_value(1);
                grad=from_grad.copy();
            }
            if (parent_op!= nullptr){
                vector<NMatrix<T>> next_grad = parent_op->backward(from_grad);
                vector<NTensor<T>*> pt = parent_op->get_context();
                for(int i=0; i<next_grad.size();i++){
                    if(pt[i]->grad.is_empty()){
                        pt[i]->grad = pt[i]->copy();
                        pt[i]->grad.set_value(0);
                        pt[i]->grad.add_inplace(next_grad[i]);
                    }else {
                        pt[i]->grad.add_inplace(next_grad[i]);
                    }
                    pt[i]->bp(next_grad[i]);
                }
            }else{
                from_grad.clear();
                return;
            }
        }
    };
    template <typename T>
    class NOptimizer {
        float lr;
        vector<PS::NTensor<T>*> *m_params;
    public:
        NOptimizer(  vector<PS::NTensor<T>*> *model_params, float learning_rate) {
            m_params = model_params;
            lr = learning_rate;
        }
        void set_zero_grad() {
            for(PS::NTensor<T>* v:(*m_params)) {
                v->grad.set_value(0);
            }
        }
        void show_model_params_info() {
            for(PS::NTensor<T>* v:(*m_params)) {
                cout<<"NAME: "<<v->get_id()<<" ";
                cout<<"DATA Shape: ";
                v->shape();
                cout<<"Grad Shape: ";
                v->grad.shape();
            }
        }
        void step() {
            for (PS::NTensor<T> *v:(*m_params)) {
                NMatrix<T> tmp = (v->grad) * lr;
                v->add_inplace(tmp);
                tmp.clear();
            }
        }
    };
};

//Implemention  of  NStorage
template <typename T>
PS::NStorage<T>::NStorage(const NStorage<T> &t){
    handle = t.handle;
    mem_size = t.mem_size;
}
template <typename T>
T* PS::NStorage<T>::copy(){
    T * new_handle = (T*)malloc(sizeof(T)*(mem_size));
    PS::global_mem_size +=sizeof(T)*(mem_size);
    memcpy(new_handle,handle,sizeof(T)*(mem_size));
    return new_handle;
}
template <typename T>
void PS::NStorage<T>::set_handle(T* new_handle){
    handle=new_handle;
}
template <typename T>
void PS::NStorage<T>::set_mem_size(unsigned long new_mem_size){
    mem_size=new_mem_size;
}
template <typename T>
unsigned long PS::NStorage<T>::get_mem_size(){
    return mem_size;
}
//template <typename T>
//void PS::NStorage<T>::operator=(const NStorage<T> &t){
////    cout<<"come NStorage"<<endl;
//    T * new_handle = (T*)malloc(sizeof(T)*(t.mem_size));
//    PS::global_mem_size +=sizeof(T)*(t.mem_size);
//    memcpy(new_handle,t.handle,sizeof(T)*(t.mem_size));
//    handle = new_handle;
//    mem_size = t.mem_size;
//}
template <typename T>
void PS::NStorage<T>::alloc(unsigned int size){
    mem_size = size;
    handle = (T*)malloc(sizeof(T)*size);
    PS::global_mem_size +=sizeof(T)*size;
    memset(handle,0,sizeof(T)*mem_size);
}
template <typename T>
void PS::NStorage<T>::exalloc(unsigned int size){
    PS::global_mem_size-=mem_size*sizeof(T);
    PS::global_mem_size+=size*sizeof(T);
    mem_size = size;
    handle = (T*) realloc(handle,sizeof(T)*size);
}
template <typename T>
T PS::NStorage<T>::read(unsigned int pos){
    return *(handle+pos);
}
template <typename T>
int PS::NStorage<T>::write(unsigned int pos,T value){
    *(handle+pos)=value;
    return 0;
}
template <typename T>
int PS::NStorage<T>::addself(unsigned int pos, T value) {
    *(handle + pos) = *(handle + pos) + value;
    return 0;
}
template <typename T>
void PS::NStorage<T>::set(T value){
    fill(handle,handle+mem_size,value);
}

template <typename T>
void PS::NStorage<T>::set_random(){
    for(size_t i= 0;i<mem_size;i++) {
        T v = (T)PS::generateGaussianNoise(0,1.0);
        *(handle + i)=v;
    }
}

template <typename T>
int PS::NStorage<T>::continuous_copy_5(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
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
int PS::NStorage<T>::continuous_copy_4(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
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
int PS::NStorage<T>::continuous_copy_3(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
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
int PS::NStorage<T>::continuous_copy_2(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
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
int PS::NStorage<T>::continuous_copy_1(const vector<size_t> &axis_weight,const vector<size_t> &map_index,const vector<size_t> &dims){
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
void PS::NStorage<T>::release(){
    PS::global_mem_size-=sizeof(T)*mem_size;
    free(handle);
    mem_size=0;
}
template <typename T>
T* PS::NStorage<T>::get_handle(){
    return handle;
}

//Implemention  of  NShape
PS::NShape::NShape(const NShape&t){
    axis_weight = t.axis_weight;
    map_index = t.map_index;
    dims = t.dims;
    dims_product = t.dims_product;
    is_continuous = t.is_continuous;
    sub_dims_product = t.sub_dims_product;
}
PS::NShape::NShape(vector<size_t> params){
    is_continuous = true;
    dims=params; //dims={axis0_size,axis1_size,...}
    init_axis_weight(params);
    init_map_index();
    for(int i=0;i<params.size()-1;i++){
        dims_product*=params[i];
        sub_dims_product*=params[i];
    }
    dims_product*=params[params.size()-1];
}
void PS::NShape::init_map_index(){
    for(int i=0;i<dims.size();i++){
        map_index.push_back(i); //0,1,2,3,4
    }
}
void PS::NShape::init_axis_weight(const vector<size_t> &params){
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
}
void PS::NShape::change_axis(const vector<size_t> &params){
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
void PS::NShape::refresh_attribute(){
    vector<size_t> new_dims;
    for(int i=0;i<map_index.size();i++){
        new_dims.push_back(dims[map_index[i]]);
    }
    dims=new_dims;
//    cout<<"refresh_dims"<<endl;
//    for_each(dims.begin(),dims.end(), p_vector);
//    cout<<endl;
    refresh_map_index();
    refresh_axis_weight(dims);

}
void PS::NShape::refresh_map_index(){
    for(int i=0;i<dims.size();i++){
        map_index[i]=i; //0,1,2,3,4
    }
//    cout<<"refresh_map_index"<<endl;
//    for_each(map_index.begin(),map_index.end(), p_vector);
//    cout<<endl;
}
void PS::NShape::refresh_axis_weight(const vector<size_t> &params){
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
//    cout<<"refresh_axis_weight"<<endl;
//    for_each(axis_weight.begin(),axis_weight.end(), p_vector);
//    cout<<endl;
}
long PS::NShape::get_index(const vector<size_t> &params){
    ostringstream oss;
    try{
        if (params.size()!=axis_weight.size()) {
            oss<<"params list size != axis_weight size"<<endl;
            throw oss.str();
        }else{
            long ret = 0;
            vector<size_t>::iterator iter_axis_weight = axis_weight.begin();
            for(int i = 0; i<params.size();i++){
                if (params[i]>=dims[map_index[i]]){
                    oss<<"axis exceeds limit !!! "<<"info: axis("<<i<<") "<<"input("<<params[i]<<") exceeds limit of ("<<dims[map_index[i]]<<")"<<endl;
                    throw oss.str();
                }
                ret += params[i]*iter_axis_weight[params.size()-map_index[i]-1];
            }
            return ret;
        }
    }catch (string e){
        cout<<"[CLASS:NShape FUNC:get_index]=> "<<e<<endl;
        exit(-1);
    }
    return 0;
}
int PS::NShape::reshape(const vector<size_t> &params){
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
    dims=params;
    init_axis_weight(params);
    init_map_index();
    return 0;
}
vector<size_t> PS::NShape::get_axis_weight(){
    return axis_weight;
}
vector<size_t> PS::NShape::get_map_index(){
    return map_index;
}
vector<size_t> PS::NShape::get_dims(){
    return dims;
}
size_t PS::NShape::get_dims(int axis){
    return dims[axis];
}
unsigned long PS::NShape::get_sub_dims_product(){
    return sub_dims_product;
}
unsigned long PS::NShape::get_dims_product(){
    return dims_product;
}
void PS::NShape::show_dims(){
    cout<<"dims: [";
    for(int i=0;i<dims.size()-1;i++){
        cout<<dims[map_index[i]]<<",";
    }
    cout<<dims[map_index[dims.size()-1]]<<"]"<<endl;

}
void PS::NShape::show_map_index(){
    cout<<"map_index: [";
    for(int i=0;i<map_index.size()-1;i++){
        cout<<map_index[i]<<",";
    }
    cout<<map_index[map_index.size()-1]<<"]"<<endl;
}
void PS::NShape::show_axis_weight(){
    cout<<"axis_weight: [";
    for(int i=0;i<axis_weight.size()-1;i++){
        cout<<axis_weight[i]<<",";
    }
    cout<<axis_weight[axis_weight.size()-1]<<"]"<<endl;
}
void PS::NShape::reset() {
    map_index.clear();
    dims.clear();
    axis_weight.clear();
    dims_product = 1;
    sub_dims_product = 1;
    is_continuous=true;
}

//Implemention  of  NMatrix
template <typename T>
PS::NMatrix<T>::NMatrix(const NMatrix<T> &t){
    storage = t.storage;
    visitor = t.visitor;
}

template <typename T>
void PS::NMatrix<T>::save_data(string file_path) {
    unsigned long data_length = storage.get_mem_size();
    T * handle=storage.get_handle();
    std::ofstream  ofs(file_path, std::ios::binary | std::ios::out);
    ofs.write((const char*)handle, sizeof(T) * data_length);
    ofs.close();
}

template <typename T>
void PS::NMatrix<T>::load_data(string file_path, size_t data_length,const vector<size_t> &t) {
    size_t  dimsproduct = 1;
    for(auto v: t) {
        dimsproduct*=v;
    }
    if(dimsproduct!=data_length) {
        cout<<"data length != dims product "<<"info: "<<data_length<<" dimsproduct: "<<dimsproduct<<endl;
        exit(-1);
    }
    T* data_ptr = new T[data_length];
    std::ifstream ifs(file_path, std::ios::binary | std::ios::in);
    ifs.read((char*)data_ptr, sizeof(T) * data_length);
    ifs.close();
    create(t);
    unsigned long i;
    T * handle = storage.get_handle();
    for(i=0;i<data_length;i++) {
        *(handle+i)=data_ptr[i];
    }
}

template <typename T>
void PS::NMatrix<T>::clear() {
    storage.release();
    visitor.reset();
}
template <typename T>
bool PS::NMatrix<T>::is_empty() {
    if(visitor.get_dims().size()==0){
        return true;
    }else{
        return false;
    }
}
template <typename T>
PS:: NMatrix<T> PS::NMatrix<T>::copy(){
    NMatrix<T> out;
    out.storage.set_handle(storage.copy());
    out.storage.set_mem_size(storage.get_mem_size());
    out.visitor = visitor;
    return out;
}
template <typename T>
PS::NMatrix<T>::NMatrix(const new_initializer_list_t<T,1> &t){
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
PS::NMatrix<T>::NMatrix(const new_initializer_list_t<T,2> &t){
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
PS::NMatrix<T>::NMatrix(const new_initializer_list_t<T,3> &t){
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
PS::NMatrix<T>::NMatrix(const new_initializer_list_t<T,4> &t){
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
PS::NMatrix<T>::NMatrix(const new_initializer_list_t<T,5> &t){
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
void PS::NMatrix<T>::create(const vector<size_t> &t){
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
    visitor=NShape(t);
    storage.alloc(visitor.get_dims_product());
}
template <typename T>
size_t PS::NMatrix<T>::get_dims(int axis) {
    return visitor.get_dims(axis);
}
template <typename T>
vector<size_t> PS::NMatrix<T>::get_dims() {
    return visitor.get_dims();
}
template <typename T>
T PS::NMatrix<T>::get_local_max_2D(size_t n_index, size_t c_index, size_t s_h,size_t s_w,size_t h_size, size_t w_size,size_t *pos_h_w) {
    if (!visitor.is_continuous) {
        cout<<"enbale continuous"<<endl;
        enable_continuous();
        visitor.is_continuous = true;
    }
    vector<size_t> dims = visitor.get_dims();
    if(dims.size()!=4) {
        cout<<"get_local_max_2D just support [N,H,W,C] data!!!"<<endl;
        exit(-1);
    }
    T value=get({n_index,s_h,s_w,c_index});
    pos_h_w[0]=s_h;
    pos_h_w[1]=s_w;
    unsigned long i,j;
    for(i=0;i<h_size;i++) {
        for(j=0;j<w_size;j++) {
            if(get({n_index,s_h+i,s_w+j,c_index})>value) {
                value = get({n_index,s_h+i,s_w+j,c_index});
                pos_h_w[0]=s_h+i;
                pos_h_w[1]=s_w+j;
            }
        }
    }
    return value;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::padding(vector<size_t> pad_list) {
    if (!visitor.is_continuous) {
        cout<<"enbale continuous"<<endl;
        enable_continuous();
        visitor.is_continuous = true;
    }
    vector<size_t> dims = visitor.get_dims();
    if (dims.size()!=pad_list.size()) {
        cout<<"pad_list size != dims size"<<endl;
        exit(-1);
    }
    for(auto v:pad_list) {
        if (v%2!=0) {
            cout<<"pad stride must even!!!"<<endl;
            exit(-1);
        }
    }
    vector<size_t> new_dims;
    for(int i=0;i<dims.size();i++) {
        new_dims.push_back(dims[i]+pad_list[i]);
    }
    NMatrix<T> out;
    out.create(new_dims);
    switch(new_dims.size()) {
        case 5:
            unsigned long i,j,k,l,m;
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    for(k=0;k<new_dims[2];k++){
                        for(l=0;l<new_dims[3];l++){
                            for(m=0;m<new_dims[4];m++){
                                if(     (i-pad_list[0]/2)<0||
                                        (i-pad_list[0]/2)>=dims[0]||
                                        (j-pad_list[1]/2)<0||
                                        (j-pad_list[1]/2)>=dims[1]||
                                        (k-pad_list[2]/2)<0||
                                        (k-pad_list[2]/2)>=dims[2]||
                                        (l-pad_list[3]/2)<0||
                                        (l-pad_list[3]/2)>=dims[3]||
                                        (m-pad_list[4]/2)<0||
                                        (m-pad_list[4]/2)>=dims[4]
                                        )
                                {
                                    out.set({i,j,k,l,m},0);
                                }else {
                                    out.set({i,j,k,l,m}, get({(i-pad_list[0]/2),
                                                              (j-pad_list[1]/2),
                                                              (k-pad_list[2]/2),
                                                              (l-pad_list[3]/2),
                                                              (m-pad_list[4]/2)}));
                                }
                            }
                        }
                    }
                }
            }
            break;
        case 4:
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    for(k=0;k<new_dims[2];k++){
                        for(l=0;l<new_dims[3];l++){
                            if(     (i-pad_list[0]/2)<0||
                                    (i-pad_list[0]/2)>=dims[0]||
                                    (j-pad_list[1]/2)<0||
                                    (j-pad_list[1]/2)>=dims[1]||
                                    (k-pad_list[2]/2)<0||
                                    (k-pad_list[2]/2)>=dims[2]||
                                    (l-pad_list[3]/2)<0||
                                    (l-pad_list[3]/2)>=dims[3]
                                    )
                            {
                                out.set({i,j,k,l},0);
                            }else {
                                out.set({i,j,k,l}, get({(i-pad_list[0]/2),
                                                        (j-pad_list[1]/2),
                                                        (k-pad_list[2]/2),
                                                        (l-pad_list[3]/2)}));
                            }
                        }
                    }
                }
            }
            break;
        case 3:
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    for(k=0;k<new_dims[2];k++){
                        if(     (i-pad_list[0]/2)<0||
                                (i-pad_list[0]/2)>=dims[0]||
                                (j-pad_list[1]/2)<0||
                                (j-pad_list[1]/2)>=dims[1]||
                                (k-pad_list[2]/2)<0||
                                (k-pad_list[2]/2)>=dims[2]
                                )
                        {
                            out.set({i,j,k},0);
                        }else {
                            out.set({i,j,k}, get({(i-pad_list[0]/2),
                                                  (j-pad_list[1]/2),
                                                  (k-pad_list[2]/2)}));
                        }
                    }
                }
            }
            break;
        case 2:
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    if(     (i-pad_list[0]/2)<0||
                            (i-pad_list[0]/2)>=dims[0]||
                            (j-pad_list[1]/2)<0||
                            (j-pad_list[1]/2)>=dims[1]
                            )
                    {
                        out.set({i,j},0);
                    }else {
                        out.set({i,j}, get({(i-pad_list[0]/2),
                                            (j-pad_list[1]/2)}));
                    }
                }
            }
            break;
        case 1:
            for( i=0;i<new_dims[0];i++) {
                if(     (i-pad_list[0]/2)<0||
                        (i-pad_list[0]/2)>=dims[0]
                        )
                {
                    out.set({i},0);
                }else {
                    out.set({i}, get({(i-pad_list[0]/2)}));
                }
            }
            break;
    }
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::unpadding(vector<size_t> pad_list) {
    if (!visitor.is_continuous) {
        cout<<"enbale continuous"<<endl;
        enable_continuous();
        visitor.is_continuous = true;
    }
    vector<size_t> new_dims = visitor.get_dims();
    if (new_dims.size()!=pad_list.size()) {
        cout<<"pad_list size != dims size"<<endl;
        exit(-1);
    }
    for(auto v:pad_list) {
        if (v%2!=0) {
            cout<<"pad stride must even!!!"<<endl;
            exit(-1);
        }
    }
    vector<size_t> dims;
    for(int i=0;i<new_dims.size();i++) {
        dims.push_back(new_dims[i]-pad_list[i]);
    }
    NMatrix<T> out;
    out.create(dims);
    switch(new_dims.size()) {
        case 5:
            unsigned long i,j,k,l,m;
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    for(k=0;k<new_dims[2];k++){
                        for(l=0;l<new_dims[3];l++){
                            for(m=0;m<new_dims[4];m++){
                                if(     (i-pad_list[0]/2)<0||
                                        (i-pad_list[0]/2)>=dims[0]||
                                        (j-pad_list[1]/2)<0||
                                        (j-pad_list[1]/2)>=dims[1]||
                                        (k-pad_list[2]/2)<0||
                                        (k-pad_list[2]/2)>=dims[2]||
                                        (l-pad_list[3]/2)<0||
                                        (l-pad_list[3]/2)>=dims[3]||
                                        (m-pad_list[4]/2)<0||
                                        (m-pad_list[4]/2)>=dims[4]
                                        )
                                {
                                    continue;
                                }else {
                                    out.set({(i-pad_list[0]/2),
                                             (j-pad_list[1]/2),
                                             (k-pad_list[2]/2),
                                             (l-pad_list[3]/2),
                                             (m-pad_list[4]/2)}, get({i,j,k,l,m}));
                                }
                            }
                        }
                    }
                }
            }
            break;
        case 4:
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    for(k=0;k<new_dims[2];k++){
                        for(l=0;l<new_dims[3];l++){
                            if(     (i-pad_list[0]/2)<0||
                                    (i-pad_list[0]/2)>=dims[0]||
                                    (j-pad_list[1]/2)<0||
                                    (j-pad_list[1]/2)>=dims[1]||
                                    (k-pad_list[2]/2)<0||
                                    (k-pad_list[2]/2)>=dims[2]||
                                    (l-pad_list[3]/2)<0||
                                    (l-pad_list[3]/2)>=dims[3]
                                    )
                            {
                                continue;
                            }else {
                                out.set({(i-pad_list[0]/2),
                                         (j-pad_list[1]/2),
                                         (k-pad_list[2]/2),
                                         (l-pad_list[3]/2)}, get({i,j,k,l}));
                            }
                        }
                    }
                }
            }
            break;
        case 3:
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    for(k=0;k<new_dims[2];k++){
                        if(     (i-pad_list[0]/2)<0||
                                (i-pad_list[0]/2)>=dims[0]||
                                (j-pad_list[1]/2)<0||
                                (j-pad_list[1]/2)>=dims[1]||
                                (k-pad_list[2]/2)<0||
                                (k-pad_list[2]/2)>=dims[2]
                                )
                        {
                            continue;
                        }else {
                            out.set({(i-pad_list[0]/2),
                                     (j-pad_list[1]/2),
                                     (k-pad_list[2]/2)}, get({i,j,k}));
                        }
                    }
                }
            }
            break;
        case 2:
            for( i=0;i<new_dims[0];i++) {
                for(j=0;j<new_dims[1];j++){
                    if(     (i-pad_list[0]/2)<0||
                            (i-pad_list[0]/2)>=dims[0]||
                            (j-pad_list[1]/2)<0||
                            (j-pad_list[1]/2)>=dims[1]
                            )
                    {
                        continue;
                    }else {
                        out.set({(i-pad_list[0]/2),
                                 (j-pad_list[1]/2)}, get({i,j}));
                    }
                }
            }
            break;
        case 1:
            for( i=0;i<new_dims[0];i++) {
                if(     (i-pad_list[0]/2)<0||
                        (i-pad_list[0]/2)>=dims[0]
                        )
                {
                    continue;
                }else {
                    out.set({(i-pad_list[0]/2)}, get({i}));
                }
            }
            break;
    }
    return out;
}
template <typename T>
T PS::NMatrix<T>::get(const vector<size_t> &query_list){
    long index = visitor.get_index(query_list);
    return storage.read(index);
}
template <typename T>
void PS::NMatrix<T>::set(const vector<size_t> &query_list,T value){
    long index = visitor.get_index(query_list);
    storage.write(index,value);
}
template <typename T>
void PS::NMatrix<T>::addself(const vector<size_t> &query_list, T value) {
    long index = visitor.get_index(query_list);
    storage.addself(index, value);
}
template <typename T>
void PS::NMatrix<T>::shape(){
    visitor.show_dims();
}
template <typename T>
void PS::NMatrix<T>::map_index(){
    visitor.show_map_index();
}
template <typename T>
void PS::NMatrix<T>::axis_weight(){
    visitor.show_axis_weight();
}
template <typename T>
void PS::NMatrix<T>::enable_continuous(){
    switch(visitor.get_dims().size()){
        case 5: storage.continuous_copy_5(visitor.get_axis_weight(),visitor.get_map_index(),visitor.get_dims());
            visitor.refresh_attribute();
            break;
        case 4: storage.continuous_copy_4(visitor.get_axis_weight(),visitor.get_map_index(),visitor.get_dims());
            visitor.refresh_attribute();
            break;
        case 3: storage.continuous_copy_3(visitor.get_axis_weight(),visitor.get_map_index(),visitor.get_dims());
            visitor.refresh_attribute();
            break;
        case 2: storage.continuous_copy_2(visitor.get_axis_weight(),visitor.get_map_index(),visitor.get_dims());
            visitor.refresh_attribute();
            break;
        case 1: storage.continuous_copy_1(visitor.get_axis_weight(),visitor.get_map_index(),visitor.get_dims());
            visitor.refresh_attribute();
            break;
    }
}
template <typename T>
void PS::NMatrix<T>::chg_axis(const vector<size_t> &query_list,bool en_continuous){
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    visitor.change_axis(query_list);
    if(en_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
}
template <typename T>
void PS::NMatrix<T>::reshape(const vector<size_t> &query_list){
    if (!visitor.is_continuous) {
        cout<<"enbale continuous"<<endl;
        enable_continuous();
        visitor.is_continuous = true;
    }
    visitor.reshape(query_list);
}
template <typename T>
bool PS::NMatrix<T>::check_dims_consistency(const vector<size_t> & a,const vector<size_t> & b){
    if (a.size()!=b.size())return false;
    for(int i=0;i<a.size();i++){
        if(a[i]!=b[i]){
            return false;
        }
    }
    return true;
}
template <typename T>
bool PS::NMatrix<T>::check_dims_consistency_dot(const vector<size_t> & a,const vector<size_t> & b){
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
void PS::NMatrix<T>::basic_dim1(T* addr,size_t w){
    cout<<"[";
    for(int i=0;i<w-1;i++){
        cout<<*(addr+i)<<" ";
    }
    cout<<*(addr+w-1)<<"]";
}
template <typename T>
void PS::NMatrix<T>::basic_dim2(T* addr,size_t h, size_t w){
    cout<<"[";
    for(int i=0;i<h-1;i++){
        basic_dim1(addr+w*i,w);
        cout<<endl;
    }
    basic_dim1(addr+w*(h-1),w);
    cout<<"]";
}
template <typename T>
void PS::NMatrix<T>::basic_dim3(T* addr,size_t t,size_t h, size_t w){
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
void PS::NMatrix<T>::basic_dimN(T* addr,const vector<size_t> &dims){
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
void PS::NMatrix<T>::show(){
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    vector<size_t> dims = visitor.get_dims();
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
void PS::NMatrix<T>::set_value(T value){
    storage.set(value);
}
template <typename T>
void PS::NMatrix<T>::set_random() {
    storage.set_random();
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator+(NMatrix<T> &a){
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    if(!a.visitor.is_continuous){
        a.enable_continuous();
        a.visitor.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency(visitor.get_dims(),a.visitor.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:operator+]=> "<<e<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
    T* op3 = out.storage.get_handle();
#ifdef USE_SSE
    unsigned long i;
    for(i=0;i<(size/ALIGN_WIDTH);i+=ALIGN_WIDTH){
        __m256 a;
        __m256 b;
        __m256 c;
        //load data
        a = _mm256_loadu_ps(op1+i);
        b = _mm256_loadu_ps(op2+i);
        c = _mm256_add_ps(a,b);
        _mm256_storeu_ps(op3+i,c);
    }
    //process K's tail data
    for (; i < size; i++) {
        op3[i]=op1[i]+op2[i];
    }
#else
    for(int i=0;i<size;i++){
        op3[i]=op1[i]+op2[i];
    }
#endif
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator+(T a) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]=op1[i]+a;
    }
    return out;
}
template <typename T>
void PS::NMatrix<T>::add_inplace(NMatrix<T> &a){
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    if(!a.visitor.is_continuous){
        a.enable_continuous();
        a.visitor.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency(visitor.get_dims(),a.visitor.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:operator+]=> "<<e<<endl;
        exit(-1);
    }
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
#ifdef USE_SSE
    unsigned long i;
    for(i=0;i<(size/ALIGN_WIDTH);i+=ALIGN_WIDTH){
        __m256 a;
        __m256 b;
        __m256 c;
        //load data
        a = _mm256_loadu_ps(op1+i);
        b = _mm256_loadu_ps(op2+i);
        c = _mm256_add_ps(a,b);
        _mm256_storeu_ps(op1+i,c);
    }
    for(; i<size;i++){
        op1[i]=op1[i]+op2[i];
    }

#else
    for(int i=0;i<size;i++){
        op1[i]=op1[i]+op2[i];
    }
#endif
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator-(NMatrix<T> &a) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    if(!a.visitor.is_continuous){
        a.enable_continuous();
        a.visitor.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency(visitor.get_dims(),a.visitor.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:operator-]=> "<<e<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
    T* op3 = out.storage.get_handle();
#ifdef USE_SSE
    unsigned long i;
    for(i=0;i<(size/ALIGN_WIDTH);i+=ALIGN_WIDTH){
        __m256 a;
        __m256 b;
        __m256 c;
        //load data
        a = _mm256_loadu_ps(op1+i);
        b = _mm256_loadu_ps(op2+i);
        c = _mm256_sub_ps(a,b);
        _mm256_storeu_ps(op3+i,c);
    }
    for(; i<size;i++){
            op3[i]=op1[i]-op2[i];
    }

#else
    for(int i=0;i<size;i++){
        op3[i]=op1[i]-op2[i];
    }
#endif
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator-(T a) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]=op1[i]-a;
    }
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator*(NMatrix<T> &a) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    if(!a.visitor.is_continuous){
        a.enable_continuous();
        a.visitor.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency(visitor.get_dims(),a.visitor.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:operator*]=> "<<e<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
    T* op3 = out.storage.get_handle();
#ifdef USE_SSE
    unsigned long i;
    for(i=0;i<(size/ALIGN_WIDTH);i+=ALIGN_WIDTH){
        __m256 a;
        __m256 b;
        __m256 c;
        //load data
        a = _mm256_loadu_ps(op1+i);
        b = _mm256_loadu_ps(op2+i);
        c = _mm256_mul_ps(a,b);
        _mm256_storeu_ps(op3+i,c);
    }
    //process K's tail data
    for (; i < size; i++) {
        op3[i]=op1[i]*op2[i];
    }
#else
    for(int i=0;i<size;i++){
        op3[i]=op1[i]*op2[i];
    }
#endif
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator*(T a) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]=op1[i]*a;
    }
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator/(NMatrix<T> &a) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    if(!a.visitor.is_continuous){
        a.enable_continuous();
        a.visitor.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency(visitor.get_dims(),a.visitor.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:operator*]=> "<<e<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
    T* op3 = out.storage.get_handle();
#ifdef USE_SSE
    unsigned long i;
    for(i=0;i<(size/ALIGN_WIDTH);i+=ALIGN_WIDTH){
        __m256 a;
        __m256 b;
        __m256 c;
        //load data
        a = _mm256_load_ps(op1+i);
        b = _mm256_load_ps(op2+i);
        c = _mm256_div_ps(a,b);
        _mm256_store_ps(op3+i,c);
    }
    for(; i<size;i++){
            op3[i]=op1[i]/op2[i];
    }

#else
    for(int i=0;i<size;i++){
        op3[i]=op1[i]/op2[i];
    }
#endif
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::operator/(T a) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]=op1[i]/a;
    }
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::exp(){
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]= expf(op1[i]);
    }
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::log() {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]= logf(op1[i]);
    }
    return out;
}
// a.inverse == 1/a
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::inverse() {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]= 1.0/op1[i];
    }
    return out;
}
// a.inverse_square == -1/a^2
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::inverse_square() {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]= -1.0/(op1[i]*op1[i]);
    }
    return out;
}
// a.pow(size_t n) == a^n
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::pow(size_t n) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]= powf(op1[i],(float)n);

    }
    return out;
}
// a.abs() == |a|
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::nabs() {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    out.create(dims);
    unsigned long size = visitor.get_dims_product();
    T* op1 = storage.get_handle();
    T* op3 = out.storage.get_handle();
    for(int i=0;i<size;i++){
        op3[i]= abs(op1[i]);
    }
    return out;
}
// inflate [N,1,H,W]->[N,C,H,W]
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::inflate(size_t axis,size_t n) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();

    if (axis>=dims.size()) {
        cout<<"axis exceeds limit!!!,"<<"current nmatrxi width: "<<dims.size()<<endl;
        exit(-1);
    }
    dims[axis]=dims[axis]*n;
    out.create(dims);
    switch(dims.size()) {
        case 5:
            unsigned i,j,k,l,m;
            for(i=0;i<dims[0];i++) {
                for(j=0;j<dims[1];j++) {
                    for (k = 0; k < dims[2]; k++) {
                        for(l=0;l<dims[3];l++) {
                            for(m=0;m<dims[4];m++) {
                                switch (axis) {
                                    case 0:out.addself({i,j,k,l,m}, get({i%visitor.get_dims(0),j,k,l,m}));break;
                                    case 1:out.addself({i,j,k,l,m}, get({i,j%visitor.get_dims(1),k,l,m}));break;
                                    case 2:out.addself({i,j,k,l,m}, get({i,j,k%visitor.get_dims(2),l,m}));break;
                                    case 3:out.addself({i,j,k,l,m}, get({i,j,k,l%visitor.get_dims(3),m}));break;
                                    case 4:out.addself({i,j,k,l,m}, get({i,j,k,l,m%visitor.get_dims(4)}));break;
                                }
                            }
                        }
                    }
                }
            }
            break;
        case 4:
            for(i=0;i<dims[0];i++) {
                for(j=0;j<dims[1];j++) {
                    for (k = 0; k < dims[2]; k++) {
                        for(l=0;l<dims[3];l++) {
                            switch (axis) {
                                case 0:out.addself({i,j,k,l}, get({i%visitor.get_dims(0),j,k,l}));break;
                                case 1:out.addself({i,j,k,l}, get({i,j%visitor.get_dims(1),k,l}));break;
                                case 2:out.addself({i,j,k,l}, get({i,j,k%visitor.get_dims(2),l}));break;
                                case 3:out.addself({i,j,k,l}, get({i,j,k,l%visitor.get_dims(3)}));break;
                            }

                        }
                    }
                }
            }
            break;
        case 3:
            for(i=0;i<dims[0];i++) {
                for(j=0;j<dims[1];j++) {
                    for (k = 0; k < dims[2]; k++) {
                        switch (axis) {
                            case 0:out.addself({i,j,k}, get({i%visitor.get_dims(0),j,k}));break;
                            case 1:out.addself({i,j,k}, get({i,j%visitor.get_dims(1),k}));break;
                            case 2:out.addself({i,j,k}, get({i,j,k%visitor.get_dims(2)}));break;
                        }
                    }
                }
            }
            break;
        case 2:
            for(i=0;i<dims[0];i++) {
                for(j=0;j<dims[1];j++) {
                    switch (axis) {
                        case 0:out.addself({i,j}, get({i%visitor.get_dims(0),j}));break;
                        case 1:out.addself({i,j}, get({i,j%visitor.get_dims(1)}));break;
                    }
                }
            }
            break;
        case 1:
            for(i=0;i<dims[0];i++) {
                switch (axis) {
                    case 0:out.addself({i}, get({i%visitor.get_dims(0)}));break;
                }
            }
            break;
    }
    return out;
}
// reduce [N,C,H,W]->[N,1,H,W]
template <typename T>
PS::NMatrix<T>  PS::NMatrix<T>::reduce(size_t axis) {
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    //check dims
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();

    if (axis>=dims.size()) {
        cout<<"axis exceeds limit!!!,"<<"current nmatrxi width: "<<dims.size()<<endl;
        exit(-1);
    }
    dims[axis]=1;

    out.create(dims);
    switch(dims.size()) {
        case 5:
            unsigned i,j,k,l,m;
            for(i=0;i<visitor.get_dims(0);i++) {
                for(j=0;j<visitor.get_dims(1);j++) {
                    for (k = 0; k < visitor.get_dims(2); k++) {
                        for(l=0;l<visitor.get_dims(3);l++) {
                            for(m=0;m<visitor.get_dims(4);m++) {
                                switch (axis) {
                                    case 0:out.addself({0,j,k,l,m}, get({i,j,k,l,m}));break;
                                    case 1:out.addself({i,0,k,l,m}, get({i,j,k,l,m}));break;
                                    case 2:out.addself({i,j,0,l,m}, get({i,j,k,l,m}));break;
                                    case 3:out.addself({i,j,k,0,m}, get({i,j,k,l,m}));break;
                                    case 4:out.addself({i,j,k,l,0}, get({i,j,k,l,m}));break;
                                }
                            }
                        }
                    }
                }
            }
            break;
        case 4:
            for(i=0;i<visitor.get_dims(0);i++) {
                for(j=0;j<visitor.get_dims(1);j++) {
                    for (k = 0; k < visitor.get_dims(2); k++) {
                        for(l=0;l<visitor.get_dims(3);l++) {
                            switch (axis) {
                                case 0:out.addself({0,j,k,l}, get({i,j,k,l}));break;
                                case 1:out.addself({i,0,k,l}, get({i,j,k,l}));break;
                                case 2:out.addself({i,j,0,l}, get({i,j,k,l}));break;
                                case 3:out.addself({i,j,k,0}, get({i,j,k,l}));break;
                            }
                        }
                    }
                }
            }
            break;
        case 3:
            for(i=0;i<visitor.get_dims(0);i++) {
                for(j=0;j<visitor.get_dims(1);j++) {
                    for (k = 0; k < visitor.get_dims(2); k++) {
                        switch (axis) {
                            case 0:out.addself({0,j,k}, get({i,j,k}));break;
                            case 1:out.addself({i,0,k}, get({i,j,k}));break;
                            case 2:out.addself({i,j,0}, get({i,j,k}));break;
                        }
                    }
                }
            }
            break;
        case 2:
            for(i=0;i<visitor.get_dims(0);i++) {
                for(j=0;j<visitor.get_dims(1);j++) {
                    switch (axis) {
                        case 0:out.addself({0,j}, get({i,j}));break;
                        case 1:out.addself({i,0}, get({i,j}));break;
                    }
                }
            }
            break;
        case 1:
            for(i=0;i<visitor.get_dims(0);i++) {
                switch (axis) {
                    case 0:out.addself({0}, get({i}));break;
                }
            }
            break;
    }
    return out;
}

template <typename T>
void PS::NMatrix<T>::basic_dot_omp(T* op1,T* op2,T* out,size_t rows,size_t cols, size_t K){
#ifdef USE_SSE
    omp_set_num_threads(16);
     int i,j;
     unsigned long l;
     T temp[ALIGN_WIDTH];
     #pragma omp parallel shared(op1,op2,out) private(i,j,l)
    {
        #pragma omp for schedule(dynamic)
        for(i=0;i<rows;i++){
            for(j=0;j<cols;j++){
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
    }

#else
    omp_set_num_threads(16);
    int i,j,l;
#pragma omp parallel shared(op1,op2,out) private(i,j,l)
    {
#pragma omp for schedule(dynamic)
        for(i=0;i<rows;i++){
            for(j=0;j<cols;j++){
                for (l=0; l < K; l++) {
                    out[i*cols+j] += op1[i*K+l] * op2[j*K+l];
                }
            }
        }
    }
#endif
}
template <typename T>
void PS::NMatrix<T>::basic_dot(T* op1,T* op2,T* out,size_t rows,size_t cols, size_t K){
#ifdef USE_SSE
    int i,j;
     unsigned long l;
     T temp[ALIGN_WIDTH];
        for(i=0;i<rows;i++){
            for(j=0;j<cols;j++){
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

#else

    int i,j,l;
    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            for (l=0; l < K; l++) {
                out[i*cols+j] += op1[i*K+l] * op2[j*K+l];
            }
        }
    }
#endif
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::dot(NMatrix<T> &a){
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    if(!a.visitor.is_continuous){
        a.enable_continuous();
        a.visitor.is_continuous=true;
    }
    //check dims
    try{
        if(!check_dims_consistency_dot(visitor.get_dims(),a.visitor.get_dims())){
            ostringstream oss;
            oss<<"dims have not consistency"<<"should like: [a,b,c,H,W] [a,b,c,Q,W]"<<endl;
            throw oss.str();
        }
    }catch (string e){
        cout<<"[CLASS:NMatrix FUNC:dot]=> "<<e<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    vector<size_t> dims=visitor.get_dims();
    size_t  dims_size = dims.size();
    dims[dims_size-1]=a.visitor.get_dims(dims_size-2); //H,H
    out.create(dims);
    T* op1 = storage.get_handle();
    T* op2 = a.storage.get_handle();
    T* op3 = out.storage.get_handle();

    size_t K = visitor.get_dims(dims_size-1);
    size_t out_row_size = visitor.get_dims(dims_size-2);
    size_t out_col_size = a.visitor.get_dims(dims_size-2);
//    size_t op1_slice_size = out_row_size * K;
//    size_t op2_slice_size = out_col_size * K;
    size_t op3_slice_size = out_row_size * out_col_size;
    size_t threshold_value = K*out_row_size*out_col_size;
    if(dims_size>2){
        size_t matrix_num = 1;
        for(int i=0;i<dims_size-2;i++){
            matrix_num*=dims[i];
        }
        int i;
        for (i = 0; i < matrix_num; i++) {
            T *nop1 = op1 + i * out_row_size * K;
            T *nop2 = op2 + i * out_col_size * K;
            T *nop3 = op3 + i * op3_slice_size;
            if(threshold_value>3000){
                basic_dot_omp(nop1, nop2, nop3, out_row_size, out_col_size, K);
            }else{
                basic_dot(nop1, nop2, nop3, out_row_size, out_col_size, K);
            }

        }
    }else{
        if(threshold_value>3000){
            basic_dot_omp(op1,op2,op3,out_row_size,out_col_size,K);
        }else{
            basic_dot(op1,op2,op3,out_row_size,out_col_size,K);
        }

    }
    return out;
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::img2col(vector<size_t> khw_size,int c_in,int stride_h,int stride_w,bool padding,unsigned long *newhw){
    //N,H,W,C
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    unsigned long N = visitor.get_dims(0);
    unsigned long H = visitor.get_dims(1);
    unsigned long W = visitor.get_dims(2);
    unsigned long C = visitor.get_dims(3);
    NMatrix<T> out; //N*new_H*new_W,K*K*C
    vector<size_t> khw;
    khw=khw_size;
    unsigned  long Kh = khw[0];
    unsigned  long Kw = khw[1];

    unsigned long padding_w_size;
    unsigned long padding_h_size;
    if (padding){
        padding_w_size =  (W-1)*stride_w+Kw-W;
        padding_h_size =  (H-1)*stride_h+Kh-H;
    }else{
        padding_w_size=0;
        padding_w_size=0;
    }
    if(padding_w_size%2!=0){
        cout<<"padding_w_size is Odd！！！"<<endl;
        exit(-1);
    }

    if(padding_h_size%2!=0){
        cout<<"padding_w_size is Odd！！！"<<endl;
        exit(-1);
    }
    unsigned  long new_h = padding==true?(H-khw[0]+padding_h_size)/stride_h+1:(H-khw[0])/stride_h+1;
    unsigned  long new_w = padding==true?(W-khw[1]+padding_w_size)/stride_w+1:(W-khw[1])/stride_w+1;
//    cout<<"new_h: "<<new_h<<endl;
//    cout<<"new_w: "<<new_w<<endl;
    newhw[0]=new_h;
    newhw[1]=new_w;
    vector<size_t> dims = {N*new_h*new_w,C*khw[0]*khw[1]};
    out.create(dims);
//    out.shape();
    T* op = out.storage.get_handle();
//    unsigned long i,j,n,c,kh,kw;
    unsigned long index_in_CKhKwHW;
    unsigned long index_in_N;
    unsigned long index_in_C;
    unsigned long index_in_KhKwHW;
    unsigned long index_in_Kw;
    unsigned long index_in_KhHW;
    unsigned long index_in_Kh;
    unsigned long index_in_HW;
    unsigned long index_in_W;
    unsigned long index_in_H;

    unsigned long NHWKhKwC = N*new_h*new_w*C*Kh*Kw;
    unsigned long HWKhKwC = new_h*new_w*C*Kh*Kw;
    for(int i = 0;i<NHWKhKwC;i++){
        index_in_CKhKwHW = i % (HWKhKwC);//[0,....,C*Kh*Kw*H*W-1]
        index_in_N = i / (HWKhKwC); //[0,...,N-1]
        index_in_C = index_in_CKhKwHW  % C;   //[0,...C-1]
        index_in_KhKwHW = index_in_CKhKwHW / C; //[0,...,KhKwHW-1]
        index_in_Kw = index_in_KhKwHW % (Kw); //[0,...,Kw-1]
        index_in_KhHW = index_in_KhKwHW /(Kw); //[0,...,KhHW-1]
        index_in_Kh = index_in_KhHW % (Kh); //[0,...,Kh-1]
        index_in_HW = index_in_KhKwHW / (Kh*Kw);//[0,...HW-1]
        index_in_W = index_in_HW % new_w;//[0,...W-1]
        index_in_H = index_in_HW / new_w;//[0,...H-1]
//        cout<<"index_in_CKhKwHW: "<<index_in_CKhKwHW<<endl;
//        cout<<"index_in_KhKwHW: "<<index_in_KhKwHW<<endl;
//        cout<<"N H W C Kh Kw:"<<index_in_N<<" "<<index_in_H<<" "<<index_in_W<<" "<<index_in_C<<" "<<index_in_Kh<<" "<<index_in_Kw<<endl;
        if(((index_in_H*stride_h+index_in_Kh)-(padding_h_size/2))<0
           || ((index_in_H*stride_h+index_in_Kh)-(padding_h_size/2))>=H
           || (index_in_W*stride_w+index_in_Kw)-(padding_w_size/2)<0
           || (index_in_W*stride_w+index_in_Kw)-(padding_w_size/2)>=W) {
            op[i]=0;
        }else{
            op[i]=this->get({index_in_N,((index_in_H*stride_h+index_in_Kh)-(padding_h_size/2)),(index_in_W*stride_w+index_in_Kw)-(padding_w_size/2),index_in_C});
        }
    }
    return out;
}
template <typename T>
void PS::NMatrix<T>::col2img(vector<size_t> khw_size,int c_in,int stride_h,int stride_w,bool padding,unsigned long *newhw,NMatrix<T> &img2col_nmatrix){
    //N,H,W,C
    //enable continuous
    if(!visitor.is_continuous){
        enable_continuous();
        visitor.is_continuous=true;
    }
    unsigned long N = visitor.get_dims(0);
    unsigned long H = visitor.get_dims(1);
    unsigned long W = visitor.get_dims(2);
    unsigned long C = visitor.get_dims(3);
    NMatrix<T> out; //N*new_H*new_W,K*K*C
    vector<size_t> khw;
    khw=khw_size;
    unsigned  long Kh = khw[0];
    unsigned  long Kw = khw[1];

    unsigned long padding_w_size;
    unsigned long padding_h_size;
    if (padding){
        padding_w_size =  (W-1)*stride_w+Kw-W;
        padding_h_size =  (H-1)*stride_h+Kh-H;
    }else{
        padding_w_size=0;
        padding_w_size=0;
    }
    if(padding_w_size%2!=0){
        cout<<"padding_w_size is Odd！！！"<<endl;
        exit(-1);
    }

    if(padding_h_size%2!=0){
        cout<<"padding_w_size is Odd！！！"<<endl;
        exit(-1);
    }
    unsigned  long new_h = padding==true?(H-khw[0]+padding_h_size)/stride_h+1:(H-khw[0])/stride_h+1;
    unsigned  long new_w = padding==true?(W-khw[1]+padding_w_size)/stride_w+1:(W-khw[1])/stride_w+1;
    newhw[0]=new_h;
    newhw[1]=new_w;
    T* op = img2col_nmatrix.storage.get_handle();
//    unsigned long i,j,n,c,kh,kw;
    unsigned long index_in_CKhKwHW;
    unsigned long index_in_N;
    unsigned long index_in_C;
    unsigned long index_in_KhKwHW;
    unsigned long index_in_Kw;
    unsigned long index_in_KhHW;
    unsigned long index_in_Kh;
    unsigned long index_in_HW;
    unsigned long index_in_W;
    unsigned long index_in_H;

    unsigned long NHWKhKwC = N*new_h*new_w*C*Kh*Kw;
    unsigned long HWKhKwC = new_h*new_w*C*Kh*Kw;
    for(int i = 0;i<NHWKhKwC;i++){
        index_in_CKhKwHW = i % (HWKhKwC);//[0,....,C*Kh*Kw*H*W-1]
        index_in_N = i / (HWKhKwC); //[0,...,N-1]
        index_in_C = index_in_CKhKwHW  % C;   //[0,...C-1]
        index_in_KhKwHW = index_in_CKhKwHW / C; //[0,...,KhKwHW-1]
        index_in_Kw = index_in_KhKwHW % (Kw); //[0,...,Kw-1]
        index_in_KhHW = index_in_KhKwHW /(Kw); //[0,...,KhHW-1]
        index_in_Kh = index_in_KhHW % (Kh); //[0,...,Kh-1]
        index_in_HW = index_in_KhKwHW / (Kh*Kw);//[0,...HW-1]
        index_in_W = index_in_HW % new_w;//[0,...W-1]
        index_in_H = index_in_HW / new_w;//[0,...H-1]
//        cout<<"index_in_CKhKwHW: "<<index_in_CKhKwHW<<endl;
//        cout<<"index_in_KhKwHW: "<<index_in_KhKwHW<<endl;
//        cout<<"N H W C Kh Kw:"<<index_in_N<<" "<<index_in_H<<" "<<index_in_W<<" "<<index_in_C<<" "<<index_in_Kh<<" "<<index_in_Kw<<endl;
        if(((index_in_H*stride_h+index_in_Kh)-(padding_h_size/2))<0
           || ((index_in_H*stride_h+index_in_Kh)-(padding_h_size/2))>=H
           || (index_in_W*stride_w+index_in_Kw)-(padding_w_size/2)<0
           || (index_in_W*stride_w+index_in_Kw)-(padding_w_size/2)>=W) {
            continue;
        }else{
            this->addself({index_in_N,((index_in_H*stride_h+index_in_Kh)-(padding_h_size/2)),(index_in_W*stride_w+index_in_Kw)-(padding_w_size/2),index_in_C},op[i]);
        }
    }
}




// Operator Interface
template <typename T>
class OP{
public:
    virtual vector<PS::NTensor<T>*> get_context();
    virtual void forward();
    virtual vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> OP<T>::get_context(){
    cout<<"ERROR"<<endl;exit(-1);};
template <typename T>
void OP<T>::forward(){;}
template <typename T>
vector<PS::NMatrix<T>> OP<T>:: backward(PS::NMatrix<T> grad){
    cout<<"ERROR"<<endl;exit(-1);}
// Conv2D Operator Of NN

// FC Operator Of NN

// MaxPooling Operator Of NN

// ReLU Operator Of NN

// Sum Operator

// Add Operator
template <typename T>
class Add:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a, PS::NTensor<T> *b);

    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Add<T>::get_context(){
    return context;
}
template <typename T>
PS::NTensor<T>* Add<T>::forward(PS::NTensor<T> *a, PS::NTensor<T> *b){
    context.push_back(a);
    context.push_back(b);
    PS::NTensor<T>*  ret =  new PS::NTensor<T>((*a)+(*b));
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Add<T>::backward(PS::NMatrix<T> grad){
    PS::NMatrix<T> grad_a = grad*1;
    PS::NMatrix<T> grad_b = grad*1;
    grad.clear();
    return {grad_a,grad_b};
}
// Sub Operator

// Mul Operator
template <typename T>
class Mul:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a, PS::NTensor<T> *b);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Mul<T>::get_context(){
    return context;
}
template <typename T>
PS::NTensor<T>* Mul<T>::forward(PS::NTensor<T> *a, PS::NTensor<T> *b){
    context.push_back(a);
    context.push_back(b);
    PS::NTensor<T>*  ret =  new PS::NTensor<T>((*a)*(*b));
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Mul<T>::backward(PS::NMatrix<T> grad){
    PS::NMatrix<T> grad_a = grad*(*context[1]);
    PS::NMatrix<T> grad_b = grad*(*context[0]);
    grad.clear();
    return {grad_a,grad_b};
}
//Div Operator

//define Model








