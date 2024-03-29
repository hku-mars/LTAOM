#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Geometry>
#include <pcl/point_types.h>
#include <pthread.h>
#include <stdio.h>
#include <queue>
#include <time.h>
#include <chrono>
#include <unistd.h>
#include <math.h>
#include <algorithm>
#include <memory.h>

#define EPSS 1e-6
#define Minimal_Unbalanced_Tree_Size 10
#define Multi_Thread_Rebuild_Point_Num 1500
#define ForceRebuildPercentage 0.2
#define prime 107323
#define Max_Update_Time 100
#define Q_LEN 10000
#define MAX_HASH_LEN 12582917 // 1e7

using namespace std;

// struct PointType
// {
//     float x,y,z;
// };

typedef pcl::PointXYZINormal PointType;


typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;

struct KD_TREE_NODE
{
    PointType point,box_center;
    int update_counter;
    int division_axis;  
    int TreeSize = 1;
    int invalid_point_num = 0;
    bool point_deleted = false;
    bool tree_deleted = false; 
    bool is_root_node = false;
    float node_range_x[2], node_range_y[2], node_range_z[2];   
    KD_TREE_NODE *left_son_ptr = nullptr;
    KD_TREE_NODE *right_son_ptr = nullptr;
    KD_TREE_NODE *father_ptr = nullptr;
    // For paper data record
    double timestamp;
    double max_time;
    float alpha_del;
    float alpha_bal;
};

struct PointType_CMP{
    PointType point;
    float dist;
    PointType_CMP (PointType p, float d){
        this->point = p;
        this->dist = d;
    };
    bool operator < (const PointType_CMP &a)const{
        return dist < a.dist;
    }    
};

struct BoxPointType{
    float vertex_min[3];
    float vertex_max[3];
};

struct PointCubeIndexType{
    PointType point,center;
    int CubeIndex, Update_Counter;
    double time;
    PointCubeIndexType (PointType p, PointType center, int CubeIndex, int Counter, double timestamp){
        this->point = p;
        this->center = center;
        this->CubeIndex = CubeIndex;
        this->Update_Counter = Counter;
        this->time = timestamp;
    }
    bool operator < (const PointCubeIndexType &a) const{
        return CubeIndex < a.CubeIndex;
    }
};


enum operation_set {ADD_POINT, DELETE_POINT};

enum delete_point_storage_set {NOT_RECORD, DELETE_POINTS_REC, MULTI_THREAD_REC, DOWNSAMPLE_REC};

struct Operation_Logger_Type{
    PointType point,center;
    int CubeIndex;
    double time;
    operation_set op;
};

class MANUAL_QUEUE{
    private:
        int head = 0,tail = 0, counter = 0;
        Operation_Logger_Type q[Q_LEN];
        bool is_empty;
    public:
        void pop();
        Operation_Logger_Type front();
        Operation_Logger_Type back();
        void clear();
        void push(Operation_Logger_Type op);
        bool empty();
};

class KD_FOREST
{
private:
    // KD-FOREST 
    KD_TREE_NODE ** static_roots;
    float x_min, y_min, z_min;
    void sort_by_cubeindex(PointVector &Points, vector<PointCubeIndexType> &sort_points, double timestamp);
    int get_cube_index(PointType point);
    PointType get_box_center(PointType point);
    void Initial_Downsample(PointVector &points, PointVector &downsampled_points, PointType min_center, PointType max_center);    
    // Multi-thread Tree Rebuild
    int max_rebuild_num = 0;
    int max_need_rebuild_num = 0;
    bool termination_flag = false;
    bool rebuild_flag = false;
    bool copy_flag = false;
    pthread_t rebuild_thread;
    pthread_mutex_t termination_flag_mutex_lock, rebuild_ptr_mutex_lock, working_flag_mutex, search_flag_mutex;
    pthread_mutex_t rebuild_logger_mutex_lock, points_deleted_rebuild_mutex_lock;
    int * Treesize_tmp;
    int * Validnum_tmp;
    float * alpha_bal_tmp;
    float * alpha_del_tmp;
    // queue<Operation_Logger_Type> Rebuild_Logger;
    MANUAL_QUEUE Rebuild_Logger;
    vector<PointCubeIndexType> Rebuild_PCL_Storage;
    KD_TREE_NODE ** Rebuild_Ptr;
    int search_mutex_counter = 0;
    static void * multi_thread_ptr(void *arg);
    void multi_thread_rebuild();
    void start_thread();
    void stop_thread();
    void run_operation(KD_TREE_NODE ** root, Operation_Logger_Type operation);
    // KD Tree Functions and augmented variables
    float delete_criterion_param = 0.5f;
    float balance_criterion_param = 0.7f;
    float downsample_size = 0.2f;
    bool Drop_MultiThread_Rebuild = false;
    bool Delete_Storage_Disabled = false;
    KD_TREE_NODE * STATIC_ROOT_NODE = nullptr;
    PointVector Points_deleted;
    PointVector Downsample_Storage;
    PointVector Multithread_Points_deleted;
    void InitTreeNode(KD_TREE_NODE * root);
    void Test_Lock_States(KD_TREE_NODE *root);
    void BuildTree(KD_TREE_NODE ** root, int l, int r, vector<PointCubeIndexType> & Storage);
    void Rebuild(KD_TREE_NODE ** root);
    void NearestSearchOnTree(KD_TREE_NODE *root, PointType point, int k_nearest, priority_queue<PointType_CMP> &q, double time_range, double max_dist);
    void Delete_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild);
    void Add_by_point(KD_TREE_NODE ** root, PointCubeIndexType point, bool allow_rebuild);
    void Search(KD_TREE_NODE * root, int k_nearest, PointType point, priority_queue<PointType_CMP> &q, double time_range, double max_dist);
    void Search_by_range(KD_TREE_NODE *root, BoxPointType boxpoint, vector<PointCubeIndexType> &Storage);
    bool Criterion_Check(KD_TREE_NODE * root);
    void Push_Down(KD_TREE_NODE * root);
    void Update(KD_TREE_NODE * root); 
    void delete_tree_nodes(KD_TREE_NODE ** root, delete_point_storage_set storage_type);
    void downsample(KD_TREE_NODE ** root);
    bool same_point(PointType a, PointType b);
    bool point_in_box(PointType point, PointType box_center);
    float calc_dist(PointType a, PointType b);
    float calc_box_dist(KD_TREE_NODE * node, PointType point);    
    static bool point_cmp_x(PointCubeIndexType a, PointCubeIndexType b); 
    static bool point_cmp_y(PointCubeIndexType a, PointCubeIndexType b); 
    static bool point_cmp_z(PointCubeIndexType a, PointCubeIndexType b); 
    void print_treenode(KD_TREE_NODE * root, int index, FILE *fp, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);
    int SearchSteps[26][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
                              {1,1,0},{1,-1,0},{1,0,1},{1,0,-1},{0,1,1},{0,1,-1},{-1,1,0},{-1,-1,0},{-1,0,1},{-1,0,-1},{0,-1,1},{0,-1,-1},
                              {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},{-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}};
public:
    KD_FOREST(float delete_param = 0.5, float balance_param = 0.6 , float box_length = 0.2, int env_length_param = 48, int env_width_param = 48, int env_height_param = 48, float cube_param = 50);
    ~KD_FOREST();
    void Set_delete_criterion_param(float delete_param);
    void Set_balance_criterion_param(float balance_param);
    void Set_downsample_param(float box_length);
    void Set_environment(int env_length_param, int env_width_param, int env_height_param, float cube_param);
    void InitializeKDTree(float delete_param = 0.5, float balance_param = 0.7, float box_length = 0.2, int env_length_param = 48, int env_width_param = 48, int env_height_param = 48, float cube_param = 50); 
    int size(int cube_index);
    int Validnum(int cube_index);
    void root_alpha(float &alpha_bal, float &alpha_del, int cube_index);
    void Build(PointVector point_cloud, bool need_downsample, double timestamp);
    uint8_t Nearest_Search(PointType point, int k_nearest, PointVector &Nearest_Points, vector<float> & Point_Distance, double time_range, double max_dist = INFINITY);
    void Add_Points(PointVector & PointToAdd, double timestamp);
    void Delete_Points(PointVector & PointToDel);
    void Box_Search(BoxPointType boxpoint, vector<PointCubeIndexType> & Storage);
    void flatten(KD_TREE_NODE * root, vector<PointCubeIndexType> &Storage);
    void acquire_removed_points(PointVector & removed_points);
    void print_tree(int index, FILE *fp, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);
    BoxPointType tree_range(int cube_index);
    vector<PointCubeIndexType> PCL_Storage;     
    KD_TREE_NODE * Root_Node = nullptr;  
    vector<float> add_rec,delete_rec;
    int Env_Length = 0, Env_Width = 0, Env_Height = 0;
    float Cube_length = 0;
    vector<int>   add_counter_rec, delete_counter_rec;
    bool initialized = false;
    int total_size = 0;
    KD_TREE_NODE ** roots;
    vector<int> root_index;
    int max_counter = 0;
    long long Max_Index = 0;
    bool HashOn = false;
};


