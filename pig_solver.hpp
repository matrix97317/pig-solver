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
#include <algorithm>
#include <random>
#include <chrono>
#include <string.h>
using namespace std;

#ifdef OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_SSE
#include <x86intrin.h>
#endif

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
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



inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
    __m128 row1 = _mm_loadu_ps(&A[0*lda]);
    __m128 row2 = _mm_loadu_ps(&A[1*lda]);
    __m128 row3 = _mm_loadu_ps(&A[2*lda]);
    __m128 row4 = _mm_loadu_ps(&A[3*lda]);
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_storeu_ps(&B[0*ldb], row1);
    _mm_storeu_ps(&B[1*ldb], row2);
    _mm_storeu_ps(&B[2*ldb], row3);
    _mm_storeu_ps(&B[3*ldb], row4);
}

inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
#pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
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
class OP;

namespace PS {
    //Global Info
    unsigned long global_mem_size=0;
    unsigned long node_count=0;
    unsigned long random_seed=666;
    vector<void*> tensor_collector;
    template <typename T>
    void clean_tensor() {
        //cout<<"clear extra tensor...."<<endl;
        for(unsigned long i=0;i<tensor_collector.size();i++) {
            T* tmp = (T*)tensor_collector[i];
            if(tmp->get_id().substr(0,6) == "tensor") {
                delete tmp;
            }
        }
        tensor_collector.clear();
    }
    void seed(size_t value) {
        srand(value);
    }
    void split(const string& s, vector<string>& tokens, char delim = ' ') {
        tokens.clear();
        auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
            for (size_t i = pos; i < s.size(); i++) {
                if (s[i] != delim) return i;
            }
            return string::npos;
        };
        size_t lastPos = string_find_first_not(0);
        size_t pos = s.find(delim, lastPos);
        while (lastPos != string::npos) {
            tokens.emplace_back(s.substr(lastPos, pos - lastPos));
            lastPos = string_find_first_not(pos);
            pos = s.find(delim, lastPos);
        }
    }
    double generateRandomNoise() {
        return rand() % (1000) / (float)(1000);
    }
    double generateGaussianNoise(double mu, double sigma)
    {
        const double epsilon = std::numeric_limits<double>::min();
        const double two_pi = 2.0*3.14159265358979323846;
        static double z0, z1;
        static bool generate;
        generate = !generate;
        if (!generate)
            return z1 * sigma + mu;
        double u1, u2;
        do
        {
            u1 = rand() * (1.0 / RAND_MAX);
            u2 = rand() * (1.0 / RAND_MAX);
        }
        while ( u1 <= epsilon );
        z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
        z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
        return z0 * sigma + mu;
    }

    template<typename T>
    class NStorage {
    private:
        T *handle;
        unsigned long mem_size;
    public:
        NStorage();

        NStorage(const NStorage<T> &t);

        ~NStorage() = default;

        void set_handle(T* new_handle);

        void set_mem_size(unsigned long new_mem_size);

        T *get_handle();

        unsigned long get_mem_size();

        T* copy();

        void alloc(unsigned int size);

        void exalloc(unsigned int size);

        T read(unsigned int pos);

        int write(unsigned int pos, T value);

        int addself(unsigned int pos, T value);

        void set(T value);

        void set_random();

        void release();

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

        NShape();

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
        NMatrix();

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

        T get(size_t pos);

        void set(size_t pos, T value);

        void set(const vector<size_t> &query_list, T value);

        void set_value(T value);

        void set_random();

        void kaiming_normal_init(); //fout,relu

        void normal_init(double mu,double sigma);

        void addself(const vector<size_t> &query_list, T value);

        void addself(size_t pos, T value);

        void shape();

        void map_index();

        void axis_weight();

        void enable_continuous();

        void chg_axis(const vector<size_t> &query_list, bool en_continuous = false);

        void reshape(const vector<size_t> &query_list);

        bool check_dims_consistency(const vector<size_t> &a, const vector<size_t> &b);

        bool check_dims_consistency_dot(const vector<size_t> &a, const vector<size_t> &b);

        void fill_data(vector<size_t> dim_index, NMatrix<T> &a);

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
        NMatrix<T> transpose();

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
        bool requires_grad;
        void init_tensor(string name) {
            PS::node_count++;
            if (name!="") {
                id="params_"+to_string(PS::node_count)+"_"+name;
            }else {
                id = "tensor" + to_string(PS::node_count);
            }
            parent_op= nullptr;
            requires_grad= false;
        }
        NTensor(string name=""){
            init_tensor(name);
        };
        ~NTensor()=default;
        NTensor(const new_initializer_list_t<T, 1> &t):NMatrix<T>(t){
            init_tensor("");
        };
        NTensor(const new_initializer_list_t<T, 2> &t):NMatrix<T>(t){
            init_tensor("");
        };
        NTensor(const new_initializer_list_t<T, 3> &t):NMatrix<T>(t){
            init_tensor("");
        };
        NTensor(const new_initializer_list_t<T, 4> &t):NMatrix<T>(t){
            init_tensor("");
        };
        NTensor(const new_initializer_list_t<T, 5> &t):NMatrix<T>(t){
            init_tensor("");
        };
        NTensor(const NMatrix<T> &t):NMatrix<T>(t){
            init_tensor("");
        }

        NTensor<T> dcopy(){
            NTensor<T> out(this->copy());
            return out;
        }
        string get_id(){
            return id;
        }
        void * operator new(size_t size)
        {
            void * p = ::new NTensor<T>();
            tensor_collector.push_back(p);
            return p;
        }

        void operator delete(void * p)
        {
            NTensor<T> * tmp = (NTensor<T>*)p;
            tmp->parent_op= nullptr;
            tmp->id="";
            if(tmp->grad.is_empty()) {
                ;
            }else {
                tmp->grad.clear();
            }
            tmp->clear();
            free(tmp);
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
                    if(pt[i]->requires_grad) {
                        if (isnan(next_grad[i].get(0))) {
                            cout<< pt[i]->get_id()<<endl;
                        }
                        if (pt[i]->grad.is_empty()) {
                            pt[i]->grad = pt[i]->copy();
                            pt[i]->grad.set_value(0);
                            pt[i]->grad.add_inplace(next_grad[i]);
                        } else {
                            pt[i]->grad.add_inplace(next_grad[i]);
                        }
                    }
                    pt[i]->bp(next_grad[i]);
                }
                parent_op->clear_context();
            }else{
                from_grad.clear();
                return;
            }
        }
    };
    template <typename T>
    class NImageData {
    private:
        vector<vector<string>> dataset;
    public:
        size_t dataset_size;
        vector<size_t> image_shape;
        vector<int> data_index;
        size_t img_size=1;
        int b_size;
        NImageData(string meta_file_path,int batch_size,vector<size_t> img_shape) {
            //load meta_file
            ifstream ifs(meta_file_path,ios::in);
            if(!ifs.good()) {
                cout<<"file not exists"<<endl;
                exit(-1);
            }
            string str;
            int cnt=0;
            while(getline(ifs,str)){
                vector<string> items;
                PS::split(str,items,' ');
                //cout<<items[0].size()<<" "<<items[1].size()<<endl;
                dataset.push_back(items);
                data_index.push_back(cnt);
                cnt++;
            }
            ifs.close();
            image_shape = img_shape;
            b_size = batch_size;
            dataset_size = dataset.size();
            for(auto v:image_shape) {
                img_size*=v;
            }
            if(data_index.size()==dataset_size) {
                cout << "Load Dataset Size: " << dataset_size << endl;
                cout << "Dataset Tail Index: "<<data_index[dataset_size-1]<<endl;
            }
        }
        ~NImageData()=default;
        vector<vector<int>> get_batch_id_generator() {
            vector<vector<int>> batch_id;
            shuffle(data_index.begin(),data_index.end(),default_random_engine(PS::random_seed));
            int seg_num = dataset_size/b_size;
            for(int i=0;i<seg_num;i++) {
                batch_id.push_back(vector<int>(data_index.begin()+i*b_size,data_index.begin()+i*b_size+b_size));
            }
            if (dataset_size%b_size !=0) {
                batch_id.push_back(vector<int>(data_index.begin()+seg_num*b_size,data_index.end()));
            }
            return batch_id;
        }
        vector<NMatrix<T>> get_batch_data(vector<int>& batch_id) {//N,H,W,C
            size_t N = batch_id.size();
            vector<size_t> out_shape({N});
            for(auto v: image_shape) {
                out_shape.push_back(v);
            }
            NMatrix<T> data;
            data.create(out_shape);
            NMatrix<T> label;
            label.create({N,1});
            for(size_t i=0;i<N;i++) {
                //cout<<dataset[batch_id[i]][0]<<" "<<stof(dataset[batch_id[i]][1])<<endl;
                NMatrix<T> tmp;
                tmp.load_data(dataset[batch_id[i]][0],img_size,image_shape); //HWC
                data.fill_data({i},tmp);
                float label_value = stof(dataset[batch_id[i]][1]);
                label.set({i,0},label_value);
                tmp.clear();
            }
            return {data,label};
        };
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
                if(!(v->grad.is_empty())) {
                    cout<<"Grad Shape: ";
                    v->grad.shape();
                }
            }
        }
        void step() {
            for(int i=0;i<(*m_params).size();i++) {
                PS::NTensor<T> *v = (*m_params)[i];
                //cout<<v->get_id()<<endl;
                auto tmp = (v->grad) * (lr);
                v->add_inplace(tmp);
                tmp.clear();
            }
        }
    };
};

//Implemention  of  NStorage

template <typename T>
PS::NStorage<T>::NStorage() {
    handle= nullptr;
    mem_size=0;
}
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
        T v = (T)PS::generateRandomNoise();
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
                *(new_handle+dst_index) = read(index);
                dst_index++;
            }
        }
    }
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
    unsigned long i_size = dims[map_index[0]]; // dims[1]
    unsigned long j_size = dims[map_index[1]]; // dims[0]
    unsigned long i_prefix = axis_weight[dims_size-map_index[0]-1];
    unsigned long j_prefix = axis_weight[dims_size-map_index[1]-1];
    int i,j;
    for (i = 0; i < i_size; i++) {
        for (j = 0; j < j_size; j++) {
            index = i * i_prefix + j * j_prefix;
            *(new_handle + dst_index) = *(handle + index);
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
PS::NShape::NShape() {
    dims_product = 1;
    is_continuous = true;
    sub_dims_product = 1;
}
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
PS::NMatrix<T>::NMatrix() {
    storage = PS::NStorage<T>();
    visitor = PS::NShape();
}
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
    if(ifs.good()) {
        ifs.read((char *) data_ptr, sizeof(T) * data_length);
        ifs.close();
    }else {
        cout<<"file not exists!!!"<<endl;
        exit(-1);
    }
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
    if(storage.get_mem_size()==0){
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
void PS::NMatrix<T>::fill_data(vector<size_t> dim_index, NMatrix<T> &a) {
    if (!visitor.is_continuous) {
        cout<<"enbale continuous"<<endl;
        enable_continuous();
        visitor.is_continuous = true;
    }
    if (!a.visitor.is_continuous) {
        cout<<"enbale continuous"<<endl;
        a.enable_continuous();
        a.visitor.is_continuous = true;
    }
    vector<size_t> dims = visitor.get_dims();
    if((dims.size()-dim_index.size())!= a.get_dims().size()) {
        cout<<"input data dims don't match"<<" info: "<<(dims.size()-dim_index.size())<<" "<<a.get_dims().size()<<endl;
        exit(-1);
    }
    vector<size_t> axis_w =  visitor.get_axis_weight(); //[1,W,HW,HWC,HWCT]
    reverse(axis_w.begin(), axis_w.end()); //[TCHW,CHW,HW,W,1]
    size_t dim_index_size = dim_index.size();
    size_t dimsproduct = visitor.get_dims_product();
    for(int i=0;i<dim_index_size;i++) {
        dimsproduct /= visitor.get_dims(i);
    }
    if(dimsproduct != a.visitor.get_dims_product()) {
        cout<<"input data length don't match"<<" info: "<<dimsproduct<<" "<<a.visitor.get_dims_product()<<endl;
        exit(-1);
    }
    T *dst = storage.get_handle();
    T *src = a.storage.get_handle();
    size_t data_size = a.storage.get_mem_size();
    size_t addr_offset = 0;
    for(int i=0;i<dim_index_size;i++) {
        addr_offset+=dim_index[i]*axis_w[i];
    }
    dst = dst+(addr_offset);
    memcpy(dst,src,sizeof(T)*data_size);
}
template <typename T>
PS::NMatrix<T> PS::NMatrix<T>::transpose() {
    vector<size_t> dims = visitor.get_dims();
    //H,W
    if(dims.size()!=2) {
        cout<<"this NMatrix don't support transpose, because dims size too big."<<" info: "<<dims.size()<<endl;
        exit(-1);
    }
    NMatrix<T> out;
    out.create({dims[1],dims[0]}); //W,H
    unsigned  long crows= dims[0]; //B[0]
    unsigned  long ccols= dims[1]; //B[1]
    float calpha = 1.0;
    T * opA = storage.get_handle();
    unsigned  long clda = dims[1];//A[0]
    T  * opB = out.storage.get_handle();
    unsigned  long cldb = dims[0];//B[0]
    //transpose_block_SSE4x4(opA,opB,Arows,Acols,Acols,Bcols,16);
    if(crows%4==0 && ccols%4==0) {
        //cout<<"Come in"<<endl;
        transpose_block_SSE4x4(opA,opB,crows,ccols,ccols,crows,16);
        //cblas_somatcopy(CblasRowMajor, CblasTrans, crows, ccols, 1.0, opA , clda, opB, cldb);
    }else {
        // cout<<"Come out"<<endl;
#ifdef OPENBLAS
        cblas_somatcopy(CblasRowMajor, CblasTrans, crows, ccols, 1.0, opA , clda, opB, cldb);
#else
        unsigned long i_size = dims[1]; // dims[1]
        unsigned long j_size = dims[0]; // dims[0]
        unsigned long i_prefix = 1;
        unsigned long j_prefix = dims[1];
        int i,j,dst_index=0,index=0;
        for (i = 0; i < i_size; i++) {
            for (j = 0; j < j_size; j++) {
                index = i * i_prefix + j * j_prefix;
                *(opB + dst_index) = *(opA + index);
                dst_index++;
            }
        }
#endif
    }
    //cblas_simatcopy(CblasRowMajor, CblasTrans, crows, ccols, 1.0, opA , clda, cldb);
    //out.reshape({dims[1],dims[0]});
    return out;
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
    T tmp_value=0;
    pos_h_w[0]=s_h;
    pos_h_w[1]=s_w;
    unsigned long n_prefix = dims[1]*dims[2]*dims[3];
    unsigned long h_prefix = dims[2]*dims[3];
    unsigned long w_prefix = dims[3];
    unsigned long i,j,pos;

    for(i=0;i<h_size;i++) {
        for(j=0;j<w_size;j++) {
            pos = n_index*n_prefix+(s_h+i)*h_prefix+(s_w+j)*w_prefix+c_index;
            tmp_value = get(pos);
            if(tmp_value>value) {
                value = tmp_value;
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
T PS::NMatrix<T>::get(size_t pos){
    return storage.read(pos);
}
template <typename T>
void PS::NMatrix<T>::set(size_t pos,T value){
    storage.write(pos,value);
}
template <typename T>
void PS::NMatrix<T>::addself(const vector<size_t> &query_list, T value) {
    long index = visitor.get_index(query_list);
    storage.addself(index, value);
}
template <typename T>
void PS::NMatrix<T>::addself(size_t pos,T value) {
    storage.addself(pos, value);
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
void PS::NMatrix<T>::kaiming_normal_init() {
    //[out_dim,kh,kw,in_dim]
    // fout: out_dim
    // ReLU: sqrt(2)
    float gain = sqrtf(2);
    float fan = get_dims(0)* get_dims(1);
    float std = gain / (sqrtf(fan));
    unsigned long mem_size = storage.get_mem_size();
    unsigned long i;
    double v;
    for(i=0;i<mem_size;i++) {
        v = generateGaussianNoise( 0,  std);
        set(i,(T)v);
    }
}
template <typename T>
void PS::NMatrix<T>::normal_init(double mu,double sigma) {
    unsigned long mem_size = storage.get_mem_size();
    unsigned long i;
    double v;
    for(i=0;i<mem_size;i++) {
        v = generateGaussianNoise( mu,  sigma);
        set(i,(T)v);
    }
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
        a = _mm256_loadu_ps(op1+i);
        b = _mm256_loadu_ps(op2+i);
        c = _mm256_div_ps(a,b);
        _mm256_storeu_ps(op3+i,c);
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

#ifdef OPENBLAS
        //op1 [M,K]
        //op2 [N,K]
        //op3 [M,N]
        const int M=out_row_size;
        const int N=out_col_size;
        const int mK=K;
        const float alpha=1;
        const float beta=0;
        const int lda=mK;
        const int ldb=mK;
        const int ldc=N;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, mK, alpha, op1, lda, op2, ldb, beta, op3, ldc);
#else
        if(threshold_value>3000){
            basic_dot_omp(op1,op2,op3,out_row_size,out_col_size,K);
        }else{
            basic_dot(op1,op2,op3,out_row_size,out_col_size,K);
        }
#endif

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
    unsigned long i,nh,nw,l,fh,fw,g_i=0;
    //C,H,W,N,fh,fw
    //INPUT [N,H,W,C]
    unsigned long n_prefix = H*W*C;
    unsigned long h_prefix = W*C;
    unsigned long w_prefix = C;

    for (i = 0; i < N; i++) {
        for (nh = 0; nh < new_h; nh++) {
            for (nw = 0; nw < new_w; nw++) {
                for (l = 0; l < C; l++) {
                    for (fh = 0; fh < Kh; fh++) {
                        for (fw = 0; fw < Kw; fw++) {
                            unsigned cur_h = ((nh * stride_h + fh) - (padding_h_size / 2));
                            unsigned cur_w = ((nw * stride_w + fw) - (padding_w_size / 2));
                            if ((cur_h) < 0 || (cur_h) >= H || (cur_w) < 0 || (cur_w) >= W) {
                                op[g_i] = 0;
                            } else {
                                op[g_i] = get(i*n_prefix+cur_h*h_prefix+cur_w*w_prefix+l);//this->get({i, cur_h, cur_w, l});
                            }
                            g_i++;
                        }
                    }
                }
            }
        }
    }
    // once for method
//    unsigned long index_in_CKhKwHW;
//    unsigned long index_in_N;
//    unsigned long index_in_C;
//    unsigned long index_in_KhKwHW;
//    unsigned long index_in_Kw;
//    unsigned long index_in_KhHW;
//    unsigned long index_in_Kh;
//    unsigned long index_in_HW;
//    unsigned long index_in_W;
//    unsigned long index_in_H;
//
//    unsigned long NHWKhKwC = N*new_h*new_w*C*Kh*Kw;
//    unsigned long HWKhKwC = new_h*new_w*C*Kh*Kw;
//    int i;
//    for (i = 0; i < NHWKhKwC; i++) {
//            index_in_CKhKwHW = i % (HWKhKwC);//[0,....,C*Kh*Kw*H*W-1]
//            index_in_N = i / (HWKhKwC); //[0,...,N-1]
//            index_in_C = index_in_CKhKwHW % C;   //[0,...C-1]
//            index_in_KhKwHW = index_in_CKhKwHW / C; //[0,...,KhKwHW-1]
//            index_in_Kw = index_in_KhKwHW % (Kw); //[0,...,Kw-1]
//            index_in_KhHW = index_in_KhKwHW / (Kw); //[0,...,KhHW-1]
//            index_in_Kh = index_in_KhHW % (Kh); //[0,...,Kh-1]
//            index_in_HW = index_in_KhKwHW / (Kh * Kw);//[0,...HW-1]
//            index_in_W = index_in_HW % new_w;//[0,...W-1]
//            index_in_H = index_in_HW / new_w;//[0,...H-1]
////        cout<<"index_in_CKhKwHW: "<<index_in_CKhKwHW<<endl;
////        cout<<"index_in_KhKwHW: "<<index_in_KhKwHW<<endl;
////        cout<<"N H W C Kh Kw:"<<index_in_N<<" "<<index_in_H<<" "<<index_in_W<<" "<<index_in_C<<" "<<index_in_Kh<<" "<<index_in_Kw<<endl;
//            if (((index_in_H * stride_h + index_in_Kh) - (padding_h_size / 2)) < 0
//                || ((index_in_H * stride_h + index_in_Kh) - (padding_h_size / 2)) >= H
//                || (index_in_W * stride_w + index_in_Kw) - (padding_w_size / 2) < 0
//                || (index_in_W * stride_w + index_in_Kw) - (padding_w_size / 2) >= W) {
//                op[i] = 0;
//            } else {
//                op[i] = this->get({index_in_N, ((index_in_H * stride_h + index_in_Kh) - (padding_h_size / 2)),
//                                   (index_in_W * stride_w + index_in_Kw) - (padding_w_size / 2), index_in_C});
//            }
//    }
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
    unsigned long i,nh,nw,l,fh,fw,g_i=0;
    //C,H,W,N,fh,fw
    //INPUT [N,H,W,C]
    unsigned long n_prefix = H*W*C;
    unsigned long h_prefix = W*C;
    unsigned long w_prefix = C;

    for (i = 0; i < N; i++) {
        for (nh = 0; nh < new_h; nh++) {
            for (nw = 0; nw < new_w; nw++) {
                for (l = 0; l < C; l++) {
                    for (fh = 0; fh < Kh; fh++) {
                        for (fw = 0; fw < Kw; fw++) {
                            unsigned cur_h = ((nh * stride_h + fh) - (padding_h_size / 2));
                            unsigned cur_w = ((nw * stride_w + fw) - (padding_w_size / 2));
                            if ((cur_h) < 0 || (cur_h) >= H || (cur_w) < 0 || (cur_w) >= W) {
                                continue;
                            } else {
                                addself(i*n_prefix+cur_h*h_prefix+cur_w*w_prefix+l,op[g_i]);
                            }
                            g_i++;
                        }
                    }
                }
            }
        }
    }

    //once for method
//    unsigned long index_in_CKhKwHW;
//    unsigned long index_in_N;
//    unsigned long index_in_C;
//    unsigned long index_in_KhKwHW;
//    unsigned long index_in_Kw;
//    unsigned long index_in_KhHW;
//    unsigned long index_in_Kh;
//    unsigned long index_in_HW;
//    unsigned long index_in_W;
//    unsigned long index_in_H;
//
//    unsigned long NHWKhKwC = N*new_h*new_w*C*Kh*Kw;
//    unsigned long HWKhKwC = new_h*new_w*C*Kh*Kw;
//    omp_set_num_threads(16);
//    int i;
//    for ( i = 0; i < NHWKhKwC; i++) {
//            index_in_CKhKwHW = i % (HWKhKwC);//[0,....,C*Kh*Kw*H*W-1]
//            index_in_N = i / (HWKhKwC); //[0,...,N-1]
//            index_in_C = index_in_CKhKwHW % C;   //[0,...C-1]
//            index_in_KhKwHW = index_in_CKhKwHW / C; //[0,...,KhKwHW-1]
//            index_in_Kw = index_in_KhKwHW % (Kw); //[0,...,Kw-1]
//            index_in_KhHW = index_in_KhKwHW / (Kw); //[0,...,KhHW-1]
//            index_in_Kh = index_in_KhHW % (Kh); //[0,...,Kh-1]
//            index_in_HW = index_in_KhKwHW / (Kh * Kw);//[0,...HW-1]
//            index_in_W = index_in_HW % new_w;//[0,...W-1]
//            index_in_H = index_in_HW / new_w;//[0,...H-1]
////        cout<<"index_in_CKhKwHW: "<<index_in_CKhKwHW<<endl;
////        cout<<"index_in_KhKwHW: "<<index_in_KhKwHW<<endl;
////        cout<<"N H W C Kh Kw:"<<index_in_N<<" "<<index_in_H<<" "<<index_in_W<<" "<<index_in_C<<" "<<index_in_Kh<<" "<<index_in_Kw<<endl;
//            if (((index_in_H * stride_h + index_in_Kh) - (padding_h_size / 2)) < 0
//                || ((index_in_H * stride_h + index_in_Kh) - (padding_h_size / 2)) >= H
//                || (index_in_W * stride_w + index_in_Kw) - (padding_w_size / 2) < 0
//                || (index_in_W * stride_w + index_in_Kw) - (padding_w_size / 2) >= W) {
//                continue;
//            } else {
//                this->addself({index_in_N, ((index_in_H * stride_h + index_in_Kh) - (padding_h_size / 2)),
//                               (index_in_W * stride_w + index_in_Kw) - (padding_w_size / 2), index_in_C}, op[i]);
//            }
//    }
//
}



// Operator Interface
template <typename T>
class OP{
public:
    virtual vector<PS::NTensor<T>*> get_context();
    virtual void clear_context();
    virtual void forward();
    virtual vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> OP<T>::get_context(){
    cout<<"ERROR"<<endl;exit(-1);};
template <typename T>
void OP<T>::forward(){;}
template <typename T>
void OP<T>::clear_context(){;}
template <typename T>
vector<PS::NMatrix<T>> OP<T>:: backward(PS::NMatrix<T> grad){
    cout<<"ERROR"<<endl;exit(-1);}
// Conv2D Operator Of NN
template <typename T>
class Conv2D:public OP<T>{
private:
    unsigned long kh;
    unsigned long kw;
    unsigned long channel_out;
    unsigned long channel_in;
    unsigned long strideh;
    unsigned long stridew;
    bool pad;
public:
    PS::NTensor<T> *K;
    Conv2D(int kernel_w,int kernel_h,int c_out,int c_in,int stride_h,int stride_w,bool padding);
    ~Conv2D()=default;
    vector<PS::NTensor<T>*> context;
    void clear_context();
    vector<PS::NTensor<T>*> get_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};

template <typename T>
Conv2D<T>::Conv2D(int kernel_h,int kernel_w,int c_out,int c_in,int stride_h,int stride_w,bool padding){
    //Create Conv2D kernel
    kh=kernel_h;
    kw=kernel_w;
    channel_out=c_out;
    channel_in=c_in;
    strideh = stride_h;
    stridew = stride_w;
    pad=padding;
    K = new PS::NTensor<T>("Conv2DWieght");
    K->create({channel_out,kh,kw,channel_in});
    K->kaiming_normal_init();
    cout<<"Kernel Weight Shape: ";
    K->shape();
    K->requires_grad=true;
}
template <typename T>
vector<PS::NTensor<T>*> Conv2D<T>::get_context(){
    return context;
}
template <typename T>
void Conv2D<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* Conv2D<T>::forward(PS::NTensor<T> *a){
    context.push_back(a);
    context.push_back(K);
    //img2col
    unsigned long newhw[2]={0,0};
//    auto s = chrono::system_clock::now();
    PS::NMatrix<T> tmp_img2col = (*a).img2col({kh,kw},channel_in,strideh,stridew,pad,newhw); //NHW,khkw*c_in
    //tmp_img2col.shape();
    (*K).reshape({channel_out,kh*kw*channel_in});//c_out,khkw*c_in
    //dot
    PS::NTensor<T>*  ret =  new PS::NTensor<T>(tmp_img2col.dot((*K))); //NHW,c_out
//    auto e = chrono::system_clock::now();
//    cout<<"forward time: "<<chrono::duration<double>{e-s}.count()*1000<<"ms"<<endl;
    //reshape [N,H,W,c_out]
    unsigned long N = (*ret).get_dims(0)/(newhw[0]*newhw[1]);
    (*ret).reshape({N,newhw[0],newhw[1],(*ret).get_dims(1)});
    //release img2col
    tmp_img2col.clear();
    (*K).reshape({channel_out,kh,kw,channel_in});//c_out,kh,kw,c_in
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Conv2D<T>::backward(PS::NMatrix<T> grad){
    //grad shape [N,H,W,c_out]
    //calculate  K's grad
    grad.reshape({grad.get_dims(0)*grad.get_dims(1)*grad.get_dims(2),grad.get_dims(3)}); //NHW,c_out
    grad.chg_axis({1,0},true);//c_out,NHW
    unsigned long newhw[2]={0,0};
    PS::NMatrix<T> tmp_img2col = (*context[0]).img2col({kh,kw},channel_in,strideh,stridew,pad,newhw); //NHW,khkw*c_in
    tmp_img2col.chg_axis({1,0},true); //khkw*c_in,NHW
    PS::NMatrix<T> K_grad = grad.dot(tmp_img2col); //cout,khkw*c_in
    tmp_img2col.clear();
    K_grad.reshape({channel_out,kh,kw,channel_in});
    //K_grad.show();
    //calculate  Input's grad
    grad.chg_axis({1,0},true);//NHW,c_out
    (*K).reshape({channel_out,kh*kw*channel_in});//c_out,khkw*c_in
    (*K).chg_axis({1,0},true);//khkw*c_in,c_out
    PS::NMatrix<T> input_grad = grad.dot((*K)); //NHW,khkw*c_in
    //input_grad.show();
    PS::NMatrix<float> input_grad_col2img = (*context[0]).copy();
    input_grad_col2img.set_value(0);
    input_grad_col2img.col2img({kh,kw},channel_in,strideh,stridew,pad,newhw,input_grad);//N,H,W,c_in
    //input_grad_col2img.show();
    (*K).chg_axis({1,0},true);//c_out,khkw*c_in
    (*K).reshape({channel_out,kh,kw,channel_in});//c_out,kh,kw,c_in
    input_grad.clear();
    //input_grad_col2img.shape();
    grad.clear();
    return {input_grad_col2img,K_grad};
}
// FC Operator Of NN
template <typename T>
class FC:public OP<T>{
private:
    size_t i_dims;
    size_t o_dims;
    size_t bs;
    vector<size_t> input_dims;
public:
    PS::NTensor<T> * W;
    PS::NTensor<T> * B;
    FC(int in_dims,int out_dims,int batch_size);
    ~FC()=default;
    vector<PS::NTensor<T>*> context;
    void clear_context();
    vector<PS::NTensor<T>*> get_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
FC<T>::FC(int in_dims,int out_dims,int batch_size){
    i_dims = in_dims;
    o_dims = out_dims;
    bs = batch_size;
    W = new PS::NTensor<T>("FCWeight");
    W->create({o_dims,i_dims});
    W->normal_init(0,0.01);
    B = new PS::NTensor<T>("FCBais");
    B->create({bs,o_dims});
    B->set_value(0);
    cout<<"FC Weight Shape: ";
    W->shape();
    cout<<"FC Bais Shape: ";
    B->shape();
    W->requires_grad=true;
    B->requires_grad=true;
}
template <typename T>
vector<PS::NTensor<T>*> FC<T>::get_context(){
    return context;
}
template <typename T>
void FC<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* FC<T>::forward(PS::NTensor<T> *a){
    vector<size_t> dims=a->get_dims();
    input_dims = dims;//N,xxx
    context.push_back(a); //N,xxx
    context.push_back(W); //out_dims, in_dims
    context.push_back(B); //N,out_dims
    a->reshape({bs,i_dims}); //N,in_dims
    auto wx = (*a).dot((*W));//N,out_dims
    auto wx_b = wx+(*B);
    PS::NTensor<T>*  ret =  new PS::NTensor<T>(wx_b);
    ret->parent_op = this;
    a->reshape(input_dims); //N,xxx
    wx.clear();
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> FC<T>::backward(PS::NMatrix<T> grad){

    //grad [N,out_dims]
    auto W_T = W->transpose(); //in_dims,out_dims
    PS::NTensor<T> grad_a = grad.dot(W_T);//N,in_dims;
    grad_a.reshape(input_dims);
    W_T.clear();//out_dims,in_dims


//    cout<<"grad_a";
//    grad_a.shape();
    //W'grad

    context[0]->reshape({bs,i_dims});//N,i_dims
    auto a_T = context[0]->transpose();//i_dims,N
    auto grad_T = grad.transpose();//out_dims,N
    PS::NTensor<T> grad_W = grad_T.dot(a_T);//out_dims,in_dims;
    grad_T.clear();
    a_T.clear();
    context[0]->reshape(input_dims);//N,xxx
//    cout<<"grad_W";
//    grad_W.shape();

    PS::NTensor<T> grad_B = grad*1;
//    cout<<"grad_B";
//    grad_B.shape();

    grad.clear();
    return {grad_a,grad_W,grad_B};
}
// MaxPooling Operator Of NN
//
template <typename T>
class MaxPooling:public OP<T>{
private:
    unsigned long kh;
    unsigned long kw;
    unsigned long strideh;
    unsigned long stridew;
    unsigned long padding_w_size;
    unsigned long padding_h_size;
    bool pad;
    vector<size_t> input_dims;
    PS::NMatrix<size_t> max_pos_h;
    PS::NMatrix<size_t> max_pos_w;
    unsigned long new_h;
    unsigned long new_w;
public:
    MaxPooling(int kernel_h,int kernel_w,int stride_h,int stride_w,bool padding);
    ~MaxPooling()=default;
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};

template <typename T>
MaxPooling<T>::MaxPooling(int kernel_h,int kernel_w,int stride_h,int stride_w,bool padding){
    kh=kernel_h;
    kw=kernel_w;
    strideh=stride_h;
    stridew=stride_w;
    pad=padding;
}

template <typename T>
vector<PS::NTensor<T>*> MaxPooling<T>::get_context(){
    return context;
}
template <typename T>
void MaxPooling<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* MaxPooling<T>::forward(PS::NTensor<T> *a){
    context.push_back(a);
    //a [N,H,W,C]
    unsigned long N = a->get_dims(0);
    unsigned long H = a->get_dims(1);
    unsigned long W = a->get_dims(2);
    unsigned long C = a->get_dims(3);

    if (pad){
        padding_w_size =  (W-1)*stridew+kw-W;
        padding_h_size =  (H-1)*strideh+kh-H;
    }else{
        padding_w_size=0;
        padding_h_size=0;
    }
    if(padding_w_size%2!=0){
        cout<<"padding_w_size is Odd！！！"<<endl;
        exit(-1);
    }
    if(padding_h_size%2!=0){
        cout<<"padding_w_size is Odd！！！"<<endl;
        exit(-1);
    }
    new_h = pad==true?(H-kh+padding_h_size)/strideh+1:(H-kh)/strideh+1;
    new_w = pad==true?(W-kw+padding_w_size)/stridew+1:(W-kw)/stridew+1;
    PS::NMatrix<T> out;
    out.create({N,new_h,new_w,C});
    if(pad){
        PS::NMatrix<T> padded_a = a->padding({0,padding_h_size,padding_w_size,0});
        input_dims = {N,H+padding_h_size,W+padding_w_size,C};
        max_pos_h.create({N,new_h,new_w,C});
        max_pos_w.create({N,new_h,new_w,C});
        unsigned i,j,k,l;
        size_t poshw[2];
        T v;

        unsigned long n_prefix = new_h*new_w*C;
        unsigned long h_prefix = new_w*C;
        unsigned long w_prefix = C;

        for(l=0;l<C;l++){
            for(j=0;j<new_h;j++){
                for(k=0;k<new_w;k++){
                    for(i=0;i<N;i++){
                        unsigned long pos = i*n_prefix+j*h_prefix+k*w_prefix+l;
                        v = padded_a.get_local_max_2D(i,l,j*strideh,k*stridew,kh,kw,poshw);
                        max_pos_h.set(pos,poshw[0]);
                        max_pos_w.set(pos,poshw[1]);
                        out.set(pos,v);
                    }
                }
            }
        }
        padded_a.clear();
    }else{
        input_dims = {N,H+padding_h_size,W+padding_w_size,C};
        max_pos_h.create({N,new_h,new_w,C});
        max_pos_w.create({N,new_h,new_w,C});
        unsigned i,j,k,l;
        size_t poshw[2];
        T v;

        unsigned long n_prefix = new_h*new_w*C;
        unsigned long h_prefix = new_w*C;
        unsigned long w_prefix = C;
//        auto s = chrono::system_clock::now();
        for (l = 0; l < C; l++) {
            for (j = 0; j < new_h; j++) {
                for (k = 0; k < new_w; k++) {
                    for (i = 0; i < N; i++) {
                        unsigned long pos = i*n_prefix+j*h_prefix+k*w_prefix+l;
                        v = a->get_local_max_2D(i, l, j * strideh, k * stridew, kh, kw, poshw);
//                        if(l==0 && i==0){
//                            cout<<"HW:"<<poshw[0]<<" "<<poshw[1]<<endl;
//                        }
                        max_pos_h.set(pos,poshw[0]);
                        max_pos_w.set(pos,poshw[1]);
                        out.set(pos,v);
                    }
                }
            }
        }
//        auto e = chrono::system_clock::now();
//        cout<<"forward time: "<<chrono::duration<double>{e-s}.count()*1000<<"ms"<<endl;
    }
    PS::NTensor<T>*  ret =  new PS::NTensor<T>(out);
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> MaxPooling<T>::backward(PS::NMatrix<T> grad){
    //grad N,H,W,C
    unsigned long aN = input_dims[0];
    unsigned long aH = input_dims[1];
    unsigned long aW = input_dims[2];
    unsigned long aC = input_dims[3];
    unsigned long nH = new_h;
    unsigned long nW = new_w;
    PS::NMatrix<T> grad_a;
    if(pad){
        PS::NMatrix<T> grad_padded_a;
        grad_padded_a.create({aN,aH,aW,aC});
        unsigned i,j,k,l,posh,posw;
        T v;
        unsigned long n_prefix = new_h*new_w*aC;
        unsigned long h_prefix = new_w*aC;
        unsigned long w_prefix = aC;

        unsigned long n_prefix2 = aH*aW*aC;
        unsigned long h_prefix2 = aW*aC;
        unsigned long w_prefix2 = aC;
        //auto s = chrono::system_clock::now();
        for(l=0;l<aC;l++){
            for(j=0;j<nH;j++){
                for(k=0;k<nW;k++){
                    for(i=0;i<aN;i++){
                        unsigned long pos = i*n_prefix+j*h_prefix+k*w_prefix+l;
                        v = grad.get(pos);
                        posh = max_pos_h.get(pos);
                        posw = max_pos_w.get(pos);
                        unsigned long npos = i*n_prefix2+posh*h_prefix2+posw*w_prefix2+l;
                        grad_padded_a.addself(npos,v);
                    }
                }
            }
        }
        grad_a = grad_padded_a.unpadding({0,padding_h_size,padding_w_size,0});
        grad_padded_a.clear();
    }else{
        grad_a.create({aN,aH,aW,aC});
        unsigned i,j,k,l,posh,posw;
        T v;
        unsigned long n_prefix = new_h*new_w*aC;
        unsigned long h_prefix = new_w*aC;
        unsigned long w_prefix = aC;

        unsigned long n_prefix2 = aH*aW*aC;
        unsigned long h_prefix2 = aW*aC;
        unsigned long w_prefix2 = aC;

        for (l = 0; l < aC; l++) {
            for (j = 0; j < nH; j++) {
                for (k = 0; k < nW; k++) {
                    for (i = 0; i < aN; i++) {
                        unsigned long pos = i*n_prefix+j*h_prefix+k*w_prefix+l;
                        v = grad.get(pos);
                        posh = max_pos_h.get(pos);
                        posw = max_pos_w.get(pos);
                        unsigned long npos = i*n_prefix2+posh*h_prefix2+posw*w_prefix2+l;
                        grad_a.addself(npos,v);
                    }
                }
            }
        }

    }
    max_pos_h.clear();
    max_pos_w.clear();
    grad.clear();
    return {grad_a};
}
// ReLU Operator Of NN
template <typename T>
class ReLU:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> ReLU<T>::get_context(){
    return context;
}
template <typename T>
void ReLU<T>::clear_context(){
    context.clear();
}
template <typename T>
PS::NTensor<T>* ReLU<T>::forward(PS::NTensor<T> *a){
    //a [N,H,W,C]
    context.push_back(a);
    vector<size_t> dims = a->get_dims();
    unsigned long dimsproduct=1;
    for(auto v:dims){
        dimsproduct*=v;
    }
    PS::NMatrix<T> out;
    out.create(dims);
    for(unsigned long i=0;i<dimsproduct;i++){
        if(a->get(i)>0){
            out.set(i,a->get(i));
        }
    }
    PS::NTensor<T>*  ret =  new PS::NTensor<T>(out);
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> ReLU<T>::backward(PS::NMatrix<T> grad){
    vector<size_t> dims = context[0]->get_dims();
    unsigned long dimsproduct=1;
    for(auto v:dims){
        dimsproduct*=v;
    }
    PS::NMatrix<T> out;
    out.create(dims);
    for(unsigned long i=0;i<dimsproduct;i++){
        if(context[0]->get(i)>0){
            out.set(i,1);
        }
    }
    PS::NMatrix<T> grad_a = grad*out;
    out.clear();
    grad.clear();
    return {grad_a};
}

// Softmax Operator Of NN
template <typename T>
class Softmax:public OP<T>{
    PS::NMatrix<T> out_exp;
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Softmax<T>::get_context(){
    return context;
}
template <typename T>
void Softmax<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* Softmax<T>::forward(PS::NTensor<T> *a){
    //a [N,C]
    context.push_back(a);
    size_t N = a->get_dims(0);
    size_t C = a->get_dims(1);
    auto a_exp = a->exp(); //exp(a)
    auto a_exp_sum = a_exp.reduce(1); //[N,1]
    auto a_exp_inflate = a_exp_sum.inflate(1,C); //[N,C]
    auto out = a_exp/a_exp_inflate;
    a_exp.clear();
    a_exp_sum.clear();
    a_exp_inflate.clear();
    out_exp = out;
    PS::NTensor<T>*  ret =  new PS::NTensor<T>(out);
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Softmax<T>::backward(PS::NMatrix<T> grad){
    //grad [N,C]
    size_t N = context[0]->get_dims(0);
    size_t C = context[0]->get_dims(1);
    PS::NMatrix<T> exp_ncc;
    exp_ncc.create({N,C,C});
    unsigned n,i,j;
    for(n=0;n<N;n++){
        for(i=0;i<C;i++){
            for(j=0;j<C;j++){
                T y_i = out_exp.get({n,i});
                T y_j = out_exp.get({n,j});
                if(i==j){
                    exp_ncc.set({n,i,j},y_i*(1-y_i));
                }else{
                    exp_ncc.set({n,i,j},y_i*(0-y_j));
                }
            }
        }
    }
    grad.reshape({N,1,C});
    PS::NMatrix<T> grad_a = exp_ncc.dot(grad); //[N,C,1]
    grad_a.reshape({N,C});
    exp_ncc.clear();
    grad.clear();
    return {grad_a};
}
// NNLLoss Operator Of NN
template <typename T>
class NNLLoss:public OP<T>{
private:
    PS::NMatrix<T> one_hot;
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a, PS::NTensor<T> *b);

    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> NNLLoss<T>::get_context(){
    return context;
}
template <typename T>
void NNLLoss<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* NNLLoss<T>::forward(PS::NTensor<T> *a, PS::NTensor<T> *b){
    //a [N,C]
    //b [N,1]
    size_t N = a->get_dims(0);
    size_t C = a->get_dims(1);
    context.push_back(a);
    one_hot.create({N,C});
    for(unsigned long i=0;i<N;i++){
        unsigned long pos = (unsigned long) (b->get({i,0}));
        one_hot.set({i,pos},1);
    }
    PS::NMatrix<T>  mask_v = (*a)*one_hot; //N,C
    PS::NMatrix<T>  mask_v_reduce = mask_v.reduce(1); //N,1
    PS::NMatrix<T>  mask_v_reduce2 =  mask_v_reduce.reduce(0); //1,1
    PS::NMatrix<T>  out = mask_v_reduce2*(-1.0/N); //1,1
    mask_v_reduce2.clear();
    mask_v_reduce.clear();
    mask_v.clear();
    PS::NTensor<T>*  ret =  new PS::NTensor<T>(out);
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> NNLLoss<T>::backward(PS::NMatrix<T> grad){
    //grad [1,1]
    size_t N = one_hot.get_dims(0);
    size_t C = one_hot.get_dims(1);
    PS::NMatrix<T> inflate_v0 = grad.inflate(0,N);
    PS::NMatrix<T> inflate_v1 = inflate_v0.inflate(1,C);
    PS::NMatrix<T> mask_v = inflate_v1*one_hot;
    PS::NMatrix<T> grad_a = mask_v*(-1.0/N);
    one_hot.clear();
    inflate_v0.clear();
    inflate_v1.clear();
    mask_v.clear();
    grad.clear();
    return {grad_a};
}

// Sum Operator
template <typename T>
class Sum:public OP<T>{
private:
    size_t axis_size;
    size_t _axis;
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a,size_t axis);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Sum<T>::get_context(){
    return context;
}
template <typename T>
void Sum<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* Sum<T>::forward(PS::NTensor<T> *a,size_t axis){
    //[N,XXX]
    _axis=axis;
    axis_size = a->get_dims(axis);
    context.push_back(a);
    PS::NTensor<T>*  ret =  new PS::NTensor<T>((*a).reduce(axis)); //[N,1]
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Sum<T>::backward(PS::NMatrix<T> grad){
    //grad [N,1]
    PS::NTensor<T> grad_a = grad.inflate(_axis,axis_size);
    grad.clear();
    return {grad_a};
}
// Add Operator
template <typename T>
class Add:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a, PS::NTensor<T> *b);

    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Add<T>::get_context(){
    return context;
}
template <typename T>
void Add<T>::clear_context() {
    context.clear();
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
template <typename T>
class Sub:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a, PS::NTensor<T> *b);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Sub<T>::get_context(){
    return context;
}
template <typename T>
void Sub<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* Sub<T>::forward(PS::NTensor<T> *a, PS::NTensor<T> *b){
    context.push_back(a);
    context.push_back(b);
    PS::NTensor<T>*  ret =  new PS::NTensor<T>((*a)-(*b));
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Sub<T>::backward(PS::NMatrix<T> grad){
    PS::NMatrix<T> grad_a = grad*1;
    PS::NMatrix<T> grad_b = grad*(-1);
    grad.clear();
    return {grad_a,grad_b};
}
// Mul Operator
template <typename T>
class Mul:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a, PS::NTensor<T> *b);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Mul<T>::get_context(){
    return context;
}
template <typename T>
void Mul<T>::clear_context() {
    context.clear();
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
template <typename T>
class Div:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a, PS::NTensor<T> *b);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Div<T>::get_context(){
    return context;
}
template <typename T>
void Div<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* Div<T>::forward(PS::NTensor<T> *a, PS::NTensor<T> *b){
    context.push_back(a);
    context.push_back(b);
    PS::NTensor<T>*  ret =  new PS::NTensor<T>((*a)/(*b));
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Div<T>::backward(PS::NMatrix<T> grad){
    auto tmp = (*context[1]).inverse();
    PS::NMatrix<T> grad_a = grad*tmp;
    tmp.clear();
    auto tmp2 = (*context[1]).inverse_square();
    auto tmp3 = (*context[0])*tmp2;
    PS::NMatrix<T> grad_b = grad*(tmp3);
    tmp3.clear();
    tmp2.clear();
    grad.clear();
    return {grad_a,grad_b};
}
//Log  Operator
template <typename T>
class Log:public OP<T>{
public:
    vector<PS::NTensor<T>*> context;
    vector<PS::NTensor<T>*> get_context();
    void clear_context();
    PS::NTensor<T>* forward(PS::NTensor<T> *a);
    vector<PS::NMatrix<T>> backward(PS::NMatrix<T> grad);
};
template <typename T>
vector<PS::NTensor<T>*> Log<T>::get_context(){
    return context;
}
template <typename T>
void Log<T>::clear_context() {
    context.clear();
}
template <typename T>
PS::NTensor<T>* Log<T>::forward(PS::NTensor<T> *a){
    context.push_back(a);
    PS::NTensor<T>*  ret =  new PS::NTensor<T>((*a).log());
    ret->parent_op = this;
    return ret;
}
template <typename T>
vector<PS::NMatrix<T>> Log<T>::backward(PS::NMatrix<T> grad){
    PS::NMatrix<T> a_inverse = context[0]->inverse();
    PS::NMatrix<T> grad_a = grad*(a_inverse);
    a_inverse.clear();
    grad.clear();
    return {grad_a};
}

//define function
template <typename T>
class CrossEntropyLoss{
private:
public:
    Softmax<T>* op_softmax;
    Log<T>*  op_log;
    NNLLoss<T>* op_nllloss;
    CrossEntropyLoss(){
        op_softmax = new Softmax<T>();
        op_log = new Log<T>();
        op_nllloss = new NNLLoss<T>();
    }
    ~CrossEntropyLoss()=default;
    PS::NTensor<T>* forward(PS::NTensor<T> *a,PS::NTensor<T> *b){
        auto tmp1 = op_softmax->forward(a);
        auto tmp2 = op_log->forward(tmp1);
        auto tmp3 = op_nllloss->forward(tmp2,b);
        return tmp3;
    }
};

//define Model
template <typename T>
class Model{
private:
public:
    Conv2D<T>* conv32_64_0;
    Conv2D<T>* conv32_64_1;
    MaxPooling<T>* pool15;
    Conv2D<T>* conv15_128_0;
    Conv2D<T>* conv15_128_1;
    MaxPooling<T>* pool7;
    Conv2D<T>* conv7_256_0;
    Conv2D<T>* conv7_256_1;
    Conv2D<T>* conv7_256_2;
    MaxPooling<T>* pool3;
    Conv2D<T>* conv3_512_0;
    Conv2D<T>* conv3_512_1;
    Conv2D<T>* conv3_512_2;
    MaxPooling<T>* pool1; //N,1,1,512
    FC<T>* fc4096_0;  //N,4096
    ReLU<T>* relu0;
    FC<T>* fc4096_1; //N,4096
    ReLU<T>* relu1;
    FC<T>* fc10_0;   //4096,10

    vector<PS::NTensor<T>*> *model_params = new  vector<PS::NTensor<T>*>();
    Model(){
        conv32_64_0 = new Conv2D<T>(3,3,64,3,1,1,true);
        model_params->push_back(conv32_64_0->K);
        conv32_64_1 = new Conv2D<T>(3,3,64,64,1,1,true);
        model_params->push_back(conv32_64_1->K);
        pool15 = new MaxPooling<T>(3,3,2,2,false);
        conv15_128_0 = new Conv2D<T>(3,3,128,64,1,1,true);
        model_params->push_back(conv15_128_0->K);
        conv15_128_1 = new Conv2D<T>(3,3,128,128,1,1,true);
        model_params->push_back(conv15_128_1->K);
        pool7 = new MaxPooling<T>(3,3,2,2,false);
        conv7_256_0 = new Conv2D<T>(3,3,256,128,1,1,true);
        model_params->push_back(conv7_256_0->K);
        conv7_256_1 = new Conv2D<T>(3,3,256,256,1,1,true);
        model_params->push_back(conv7_256_1->K);
        conv7_256_2 = new Conv2D<T>(3,3,256,256,1,1,true);
        model_params->push_back(conv7_256_2->K);
        pool3 = new MaxPooling<T>(3,3,2,2,false);
        conv3_512_0= new Conv2D<T>(3,3,512,256,1,1,true);
        model_params->push_back(conv3_512_0->K);
        conv3_512_1= new Conv2D<T>(3,3,512,512,1,1,true);
        model_params->push_back(conv3_512_1->K);
        conv3_512_2= new Conv2D<T>(3,3,512,512,1,1,true);
        model_params->push_back(conv3_512_2->K);
        pool1 = new MaxPooling<T>(3,3,2,2,false);
        fc4096_0 = new FC<T>(512,4096,8);
        model_params->push_back(fc4096_0->W);
        model_params->push_back(fc4096_0->B);
        relu0 = new ReLU<T>();
        fc4096_1 = new FC<T>(4096,4096,8);
        model_params->push_back(fc4096_1->W);
        model_params->push_back(fc4096_1->B);
        relu1 = new ReLU<T>();
        fc10_0 = new FC<T>(4096,10,8);
        model_params->push_back(fc10_0->W);
        model_params->push_back(fc10_0->B);
    }
    ~Model()=default;
    PS::NTensor<T>* forward(PS::NTensor<T> *a){
        auto out1 =  conv32_64_0->forward(a);
        auto out2 =  conv32_64_1->forward(out1);
        auto out3 =  pool15->forward(out2);
        auto out4 =  conv15_128_0->forward(out3);
        auto out5 =  conv15_128_1->forward(out4);
        auto out6 =  pool7->forward(out5);
        auto out7 =  conv7_256_0->forward(out6);
        auto out8 =  conv7_256_1->forward(out7);
        auto out9 =  conv7_256_2->forward(out8);
        auto out10 =  pool3->forward(out9);
        auto out11 =  conv3_512_0->forward(out10);
        auto out12 =  conv3_512_1->forward(out11);
        auto out13 =  conv3_512_2->forward(out12);
        auto out14 =  pool1->forward(out13);
        auto out15 = fc4096_0->forward(out14);
        auto out16 = relu0->forward(out15);
        auto out17 = fc4096_1->forward(out16);
        auto out18 = relu1->forward(out17);
        auto out19 = fc10_0->forward(out18);
        return out19;
    }
};







