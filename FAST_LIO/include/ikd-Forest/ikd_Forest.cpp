#include "ikd_Forest.h"

/*
Description: ikd-Tree: an incremental k-d tree for robotic applications 
Author: Yixi Cai
email: yixicai@connect.hku.hk
*/

KD_FOREST::KD_FOREST(float delete_param, float balance_param, float box_length,int env_length_param, int env_width_param, int env_height_param, float cube_param) {   
    Set_delete_criterion_param(delete_param);
    Set_balance_criterion_param(balance_param);
    Set_downsample_param(box_length);
    Set_environment(env_length_param, env_width_param, env_height_param, cube_param);
    root_index.clear();
    termination_flag = false;
    start_thread(); 
}

KD_FOREST::~KD_FOREST()
{
    stop_thread();
    Delete_Storage_Disabled = true;
    for (int i = 0; i < Env_Length * Env_Width * Env_Height; i++){
        delete_tree_nodes(&(roots[i]), NOT_RECORD);
    }
    vector<PointCubeIndexType> ().swap(PCL_Storage);
    // queue<Operation_Logger_Type> ().swap(Rebuild_Logger); 
}

void KD_FOREST::Set_delete_criterion_param(float delete_param){
    delete_criterion_param = delete_param;
}

void KD_FOREST::Set_balance_criterion_param(float balance_param){
    balance_criterion_param = balance_param;
}

void KD_FOREST::Set_downsample_param(float downsample_param){
    downsample_size = downsample_param;
}

void KD_FOREST::Set_environment(int env_length_param, int env_width_param, int env_height_param, float cube_param){
    Env_Length = env_length_param;
    Env_Width = env_width_param;
    Env_Height = env_height_param;
    Cube_length = cube_param;
    Max_Index = env_length_param * env_width_param * env_height_param;
    x_min = -Env_Length/2*cube_param - cube_param/2.0;
    y_min = -Env_Width/2*cube_param - cube_param/2.0;
    z_min = -Env_Height/2*cube_param - cube_param/2.0;   
    if (Max_Index > MAX_HASH_LEN){
        HashOn = true;
        Max_Index = MAX_HASH_LEN;
    }
    roots = new KD_TREE_NODE * [Max_Index];
    static_roots = new KD_TREE_NODE * [Max_Index];
    Treesize_tmp = new int [Max_Index];
    Validnum_tmp = new int [Max_Index];
    alpha_bal_tmp = new float [Max_Index];
    alpha_del_tmp = new float [Max_Index];
    memset(roots,0,sizeof(roots));
    memset(roots,0,sizeof(static_roots));
}

void KD_FOREST::InitializeKDTree(float delete_param, float balance_param, float box_length, int env_length_param, int env_width_param, int env_height_param, float cube_param){
    Set_delete_criterion_param(delete_param);
    Set_balance_criterion_param(balance_param);
    Set_downsample_param(box_length);
    Set_environment(env_length_param, env_width_param, env_height_param, cube_param);
}

void KD_FOREST::InitTreeNode(KD_TREE_NODE * root){
    root->point.x = 0.0f;
    root->point.y = 0.0f;
    root->point.z = 0.0f;
    root->box_center.x = 0.0f;
    root->box_center.y = 0.0f;
    root->box_center.z = 0.0f;
    root->node_range_x[0] = 0.0f;
    root->node_range_x[1] = 0.0f;
    root->node_range_y[0] = 0.0f;
    root->node_range_y[1] = 0.0f;    
    root->node_range_z[0] = 0.0f;
    root->node_range_z[1] = 0.0f;
    root->update_counter = 0;
    root->division_axis = 0;
    root->father_ptr = nullptr;
    root->left_son_ptr = nullptr;
    root->right_son_ptr = nullptr;
    root->TreeSize = 0;
    root->invalid_point_num = 0;
    root->point_deleted = false;
    root->tree_deleted = false;
    root->alpha_bal = 0.5;
    root->alpha_del = 0.0;
}   

int KD_FOREST::size(int cube_index){
    int s = 0;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != roots[cube_index]){
        if (roots[cube_index] != nullptr) {
            return roots[cube_index]->TreeSize;
        } else {
            return 0;
        }
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            s = roots[cube_index]->TreeSize;
            pthread_mutex_unlock(&working_flag_mutex);
            return s;
        } else {
            return Treesize_tmp[cube_index];
        }
    }
}

BoxPointType KD_FOREST::tree_range(int cube_index){
    BoxPointType range;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != roots[cube_index]){
        if (Root_Node != nullptr) {
            range.vertex_min[0] = roots[cube_index]->node_range_x[0];
            range.vertex_min[1] = roots[cube_index]->node_range_y[0];
            range.vertex_min[2] = roots[cube_index]->node_range_z[0];
            range.vertex_max[0] = roots[cube_index]->node_range_x[1];
            range.vertex_max[1] = roots[cube_index]->node_range_y[1];
            range.vertex_max[2] = roots[cube_index]->node_range_z[1];
        } else {
            memset(&range, 0, sizeof(range));
        }
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            range.vertex_min[0] = roots[cube_index]->node_range_x[0];
            range.vertex_min[1] = roots[cube_index]->node_range_y[0];
            range.vertex_min[2] = roots[cube_index]->node_range_z[0];
            range.vertex_max[0] = roots[cube_index]->node_range_x[1];
            range.vertex_max[1] = roots[cube_index]->node_range_y[1];
            range.vertex_max[2] = roots[cube_index]->node_range_z[1];
            pthread_mutex_unlock(&working_flag_mutex);
        } else {
            memset(&range, 0, sizeof(range));
        }
    }
    return range;
}

int KD_FOREST::Validnum(int cube_index){
    int s = 0;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != roots[cube_index]){
        return (roots[cube_index]->TreeSize - roots[cube_index]->invalid_point_num);
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            s = roots[cube_index]->TreeSize-roots[cube_index]->invalid_point_num;
            pthread_mutex_unlock(&working_flag_mutex);
            return s;
        } else {
            return Validnum_tmp[cube_index];
        }
    }
}

void KD_FOREST::root_alpha(float &alpha_bal, float &alpha_del, int cube_index){
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != roots[cube_index]){
        alpha_bal = roots[cube_index]->alpha_bal;
        alpha_del = roots[cube_index]->alpha_del;
        return;
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            alpha_bal = roots[cube_index]->alpha_bal;
            alpha_del = roots[cube_index]->alpha_del;
            pthread_mutex_unlock(&working_flag_mutex);
            return;
        } else {
            alpha_bal = alpha_bal_tmp[cube_index];
            alpha_del = alpha_del_tmp[cube_index];      
            return;
        }
    }    
}

void KD_FOREST::start_thread(){
    pthread_mutex_init(&termination_flag_mutex_lock, NULL);   
    pthread_mutex_init(&rebuild_ptr_mutex_lock, NULL);     
    pthread_mutex_init(&rebuild_logger_mutex_lock, NULL);
    pthread_mutex_init(&points_deleted_rebuild_mutex_lock, NULL); 
    pthread_mutex_init(&working_flag_mutex, NULL);
    pthread_mutex_init(&search_flag_mutex, NULL);
    pthread_create(&rebuild_thread, NULL, multi_thread_ptr, (void*) this);
    printf("[ ikd-Forest ]  Multi thread started \n");    
}

void KD_FOREST::stop_thread(){

    pthread_mutex_lock(&termination_flag_mutex_lock);
    termination_flag = true;
    pthread_mutex_unlock(&termination_flag_mutex_lock);
    if (rebuild_thread) pthread_join(rebuild_thread, NULL);
    pthread_mutex_destroy(&termination_flag_mutex_lock);
    pthread_mutex_destroy(&rebuild_logger_mutex_lock);
    pthread_mutex_destroy(&rebuild_ptr_mutex_lock);
    pthread_mutex_destroy(&points_deleted_rebuild_mutex_lock);
    pthread_mutex_destroy(&working_flag_mutex);
    pthread_mutex_destroy(&search_flag_mutex);     
}

void * KD_FOREST::multi_thread_ptr(void * arg){
    KD_FOREST * handle = (KD_FOREST*) arg;
    handle->multi_thread_rebuild();
}    

void KD_FOREST::multi_thread_rebuild(){
    bool terminated = false;
    KD_TREE_NODE * father_ptr, ** new_node_ptr;
    pthread_mutex_lock(&termination_flag_mutex_lock);
    terminated = termination_flag;
    pthread_mutex_unlock(&termination_flag_mutex_lock);
    // Not sure whether we need a flag to notice this thread to finish and stop
    while (!terminated){
        pthread_mutex_lock(&rebuild_ptr_mutex_lock);
        pthread_mutex_lock(&working_flag_mutex);
        if (Rebuild_Ptr != nullptr ){                    
            /* Flatten and copy */
            rebuild_flag = true;
            if (!Rebuild_Logger.empty()){
                printf("\n\n[ T2 ]: ERROR, Rebuild Logger force cleaned \n\n");
                Rebuild_Logger.clear();
            }
            int cube_index = get_cube_index((*Rebuild_Ptr)->point);
            max_rebuild_num = max(max_rebuild_num, (*Rebuild_Ptr)->TreeSize);
            if ((*Rebuild_Ptr)->is_root_node) {
                Treesize_tmp[cube_index] = roots[cube_index]->TreeSize;
                Validnum_tmp[cube_index] = roots[cube_index]->TreeSize - roots[cube_index]->invalid_point_num;
                alpha_bal_tmp[cube_index] = roots[cube_index]->alpha_bal;
                alpha_del_tmp[cube_index] = roots[cube_index]->alpha_del;
            }
            KD_TREE_NODE * old_root_node = (*Rebuild_Ptr);                            
            father_ptr = (*Rebuild_Ptr)->father_ptr;  
            Rebuild_PCL_Storage.clear();
            flatten(*Rebuild_Ptr, Rebuild_PCL_Storage);
            pthread_mutex_unlock(&working_flag_mutex);
            /* Rebuild and update missed operations*/
            Operation_Logger_Type Operation;
            KD_TREE_NODE * new_root_node = nullptr;
            if (int(Rebuild_PCL_Storage.size()) > 0){
                BuildTree(&new_root_node, 0, Rebuild_PCL_Storage.size()-1, Rebuild_PCL_Storage);
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                while (!Rebuild_Logger.empty()){
                    Operation = Rebuild_Logger.front();
                    Rebuild_Logger.pop();
                    pthread_mutex_unlock(&rebuild_logger_mutex_lock);
                    pthread_mutex_unlock(&working_flag_mutex);
                    run_operation(&new_root_node, Operation);
                    pthread_mutex_lock(&working_flag_mutex);
                    pthread_mutex_lock(&rebuild_logger_mutex_lock);
                }   
               pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }
            /* Replace to original tree*/
            if (Rebuild_Ptr == nullptr){
                delete_tree_nodes(&new_root_node, NOT_RECORD);
                rebuild_flag = false;
                pthread_mutex_unlock(&working_flag_mutex);
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter != 0){
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(10);             
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter = -1;
                pthread_mutex_unlock(&search_flag_mutex);
                if (father_ptr->left_son_ptr == *Rebuild_Ptr) {
                    father_ptr->left_son_ptr = new_root_node;
                } else if (father_ptr->right_son_ptr == *Rebuild_Ptr){             
                    father_ptr->right_son_ptr = new_root_node;
                } else {
                    throw "[ ikd-Forest ]: Error: Father ptr incompatible with current node\n";
                }
                if (new_root_node != nullptr) new_root_node->father_ptr = father_ptr;
                (*Rebuild_Ptr) = new_root_node;
                if (father_ptr == static_roots[cube_index]) {
                    total_size -= roots[cube_index]->TreeSize;
                    roots[cube_index] = static_roots[cube_index]->left_son_ptr;
                    static_roots[cube_index]->left_son_ptr->is_root_node = true;
                    total_size += roots[cube_index]->TreeSize;
                }
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter = 0;
                pthread_mutex_unlock(&search_flag_mutex);
                Rebuild_Ptr = nullptr;
                pthread_mutex_unlock(&working_flag_mutex);
                rebuild_flag = false;                     
                /* Delete discarded tree nodes */  
                delete_tree_nodes(&old_root_node, NOT_RECORD);
            }
        } else {
            pthread_mutex_unlock(&working_flag_mutex);             
        }
        pthread_mutex_unlock(&rebuild_ptr_mutex_lock);         
        pthread_mutex_lock(&termination_flag_mutex_lock);
        terminated = termination_flag;
        pthread_mutex_unlock(&termination_flag_mutex_lock);
        usleep(100); 
    }
    printf("[ ikd-Forest ]: Rebuild thread terminated normally\n");    
}

void KD_FOREST::run_operation(KD_TREE_NODE ** root, Operation_Logger_Type operation){
    switch (operation.op)
    {
    case ADD_POINT:
        Add_by_point(root,  PointCubeIndexType(operation.point,operation.center,operation.CubeIndex,1,operation.time), false);
        break;
    case DELETE_POINT:
        Delete_by_point(root, operation.point, false);
        break;
    default:
        break;
    }
}

void KD_FOREST::Build(PointVector point_cloud, bool need_downsample, double timestamp){
    int i, index;
    vector<PointCubeIndexType> sort_points,points;    
    if (need_downsample){
        PointType min_value, max_value;
        PointVector downsampled_points;
        min_value = point_cloud[0];
        max_value = point_cloud[0];
        for (i = 0;i<point_cloud.size();i++){
            min_value.x = min(point_cloud[i].x,min_value.x);
            min_value.y = min(point_cloud[i].y,min_value.y);
            min_value.z = min(point_cloud[i].z,min_value.z);
            max_value.x = max(point_cloud[i].x,max_value.x);
            max_value.y = max(point_cloud[i].y,max_value.y);
            max_value.z = max(point_cloud[i].z,max_value.z);       
        }
        PointType min_center, max_center;
        min_center = get_box_center(min_value);
        max_center = get_box_center(max_value);
        Initial_Downsample(point_cloud, downsampled_points, min_center, max_center);
        sort_by_cubeindex(downsampled_points, sort_points, timestamp);
    } else {
        sort_by_cubeindex(point_cloud, sort_points, timestamp);
    }
    for (i=0;i<sort_points.size();i++){
        if (i == int(sort_points.size())-1 || sort_points[i].CubeIndex != sort_points[i+1].CubeIndex){
            points.push_back(sort_points[i]);
            index = sort_points[i].CubeIndex;
            if (roots[index] != nullptr){
                delete_tree_nodes(&roots[index], NOT_RECORD);
            }
            root_index.push_back(index);
            static_roots[index] = new KD_TREE_NODE;
            InitTreeNode(static_roots[index]);
            BuildTree(&(static_roots[index]->left_son_ptr),0, points.size()-1, points);
            static_roots[index]->left_son_ptr->is_root_node = true;
            Update(static_roots[index]);
            static_roots[index]->TreeSize = 0;
            roots[index] = static_roots[index]->left_son_ptr;
            points.clear();
            total_size += roots[index]->TreeSize;
            if (!initialized) initialized = true;
        } else {
            points.push_back(sort_points[i]);
        }
    }
}

uint8_t KD_FOREST::Nearest_Search(PointType point, int k_nearest, PointVector& Nearest_Points, vector<float> & Point_Distance, double time_range, double max_dist){   
    priority_queue<PointType_CMP> q; 
    int index_x, index_y, index_z, cube_index, i;
    PointType neighbor_point;
    cube_index = get_cube_index(point);
    NearestSearchOnTree(roots[cube_index], point, k_nearest, q, time_range, max_dist);
    if (k_nearest == 1) return 0;
    if (2 * max_dist <= Cube_length){
        for (i = 0; i < 26; i++){
            neighbor_point.x = point.x + SearchSteps[i][0] * Cube_length;
            neighbor_point.y = point.y + SearchSteps[i][1] * Cube_length;
            neighbor_point.z = point.z + SearchSteps[i][2] * Cube_length;
            cube_index = get_cube_index(neighbor_point);
            if (cube_index < 0 || cube_index > Max_Index) continue;
            NearestSearchOnTree(roots[cube_index], point, k_nearest, q, time_range, max_dist);
        }
    } else {
        int j,k;
        int search_N = int(ceil(max_dist/Cube_length));
        for (i = -search_N; i<=search_N; i++){
            for (j = -search_N; j<=search_N; j++){
                for (k = -search_N; k<=search_N; k++){
                    if (i == 0 && j == 0 && k == 0) continue;
                    neighbor_point.x = point.x + i * Cube_length;
                    neighbor_point.y = point.y + j * Cube_length;
                    neighbor_point.z = point.z + k * Cube_length;
                    cube_index = get_cube_index(neighbor_point);
                    if (cube_index < 0 || cube_index > Max_Index) continue;
                    NearestSearchOnTree(roots[cube_index], point, k_nearest, q, time_range, max_dist);
                }
            }
        }
    }

    uint8_t flag = 0;
    int k_found;
    if (q.size()<k_nearest){
        k_found = q.size();
        flag = 1;
    } else {
        k_found = k_nearest;
    }
    Nearest_Points.clear();
    Point_Distance.clear();
    for (i=0;i < k_found;i++){
        Nearest_Points.insert(Nearest_Points.begin(), q.top().point);
        Point_Distance.insert(Point_Distance.begin(), q.top().dist);
        q.pop();
    }
    return flag;
}

void KD_FOREST::NearestSearchOnTree(KD_TREE_NODE *root, PointType point, int k_nearest, priority_queue<PointType_CMP> &q, double time_range, double max_dist){
    if (root == nullptr) return;
    float box_dist;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root){
        box_dist = calc_box_dist(root, point);
        if (q.size() < k_nearest || box_dist <= q.top().dist + EPSS){
            Search(root, k_nearest, point, q, time_range, max_dist);
        }
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        while (search_mutex_counter == -1)
        {
            pthread_mutex_unlock(&search_flag_mutex);
            usleep(1);
            pthread_mutex_lock(&search_flag_mutex);
        }
        search_mutex_counter += 1;
        pthread_mutex_unlock(&search_flag_mutex);  
        box_dist = calc_box_dist(root, point);
        if (q.size() < k_nearest || box_dist <= q.top().dist + EPSS){
            Search(root, k_nearest, point, q, time_range, max_dist);
        } 
        pthread_mutex_lock(&search_flag_mutex);
        search_mutex_counter -= 1;
        pthread_mutex_unlock(&search_flag_mutex);      
    }
    return;
}

void KD_FOREST::Add_Points(PointVector & PointToAdd, double timestamp){
    vector<PointCubeIndexType> sort_points,points;
    sort_by_cubeindex(PointToAdd, sort_points, timestamp);
    int i,j, index, NewPointSize, ori_tree_size;
    for (i=0;i<sort_points.size();i++){
        if (i == int(sort_points.size())-1 || sort_points[i].CubeIndex != sort_points[i+1].CubeIndex){ 
            index = sort_points[i].CubeIndex;
            points.push_back(sort_points[i]);
            if (roots[index] == nullptr){ 
                static_roots[index] = new KD_TREE_NODE;
                InitTreeNode(static_roots[index]);
                for (j=0;j<points.size();j++){
                    Add_by_point(&(static_roots[index]->left_son_ptr), points[j], true);
                }
                static_roots[index]->left_son_ptr->is_root_node = true;
                Update(static_roots[index]);
                static_roots[index]->TreeSize = 0;
                roots[index] = static_roots[index]->left_son_ptr;
                points.clear();
                total_size += roots[index]->TreeSize;
                root_index.push_back(index);
                if (!initialized) initialized = true;
            } else {
                for (j=0; j<points.size();j++){
                    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != roots[index]){
                        PointType center = points[j].center;
                        ori_tree_size = roots[index]->TreeSize;
                        Add_by_point(&roots[index], points[j], true);
                        total_size = total_size - ori_tree_size + roots[index]->TreeSize;
                    } else {
                        Operation_Logger_Type operation;
                        operation.op = ADD_POINT;                        
                        operation.point = points[j].point;
                        operation.center = points[j].center;
                        operation.CubeIndex = index;
                        operation.time = timestamp;
                        pthread_mutex_lock(&working_flag_mutex);
                        ori_tree_size = roots[index]->TreeSize;
                        Add_by_point(&roots[index], points[j], false);
                        if (rebuild_flag){
                            pthread_mutex_lock(&rebuild_logger_mutex_lock);
                            Rebuild_Logger.push(operation);
                            pthread_mutex_unlock(&rebuild_logger_mutex_lock);
                        }
                        total_size = total_size - ori_tree_size + roots[index]->TreeSize;
                        pthread_mutex_unlock(&working_flag_mutex);                        
                    }
                }
            }
            points.clear();
        } else {
            points.push_back(sort_points[i]);
        }
    }
    return;
}


void KD_FOREST::Delete_Points(PointVector & PointToDel){        
    int index;
    for (int i=0;i<PointToDel.size();i++){
        index = get_cube_index(PointToDel[i]);
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != roots[index]){               
            Delete_by_point(&roots[index], PointToDel[i], true);
        } else {
            Operation_Logger_Type operation;
            operation.point = PointToDel[i];
            operation.op = DELETE_POINT;
            pthread_mutex_lock(&working_flag_mutex);        
            Delete_by_point(&roots[index], PointToDel[i], false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }      
    }      
    return;
}


void KD_FOREST::acquire_removed_points(PointVector & removed_points){
    pthread_mutex_lock(&points_deleted_rebuild_mutex_lock); 
    for (int i = 0; i < Points_deleted.size();i++){
        removed_points.push_back(Points_deleted[i]);
    }
    for (int i = 0; i < Multithread_Points_deleted.size();i++){
        removed_points.push_back(Multithread_Points_deleted[i]);
    }
    Points_deleted.clear();
    Multithread_Points_deleted.clear();
    pthread_mutex_unlock(&points_deleted_rebuild_mutex_lock);   
    return;
}

void KD_FOREST::BuildTree(KD_TREE_NODE ** root, int l, int r, vector<PointCubeIndexType> & Storage){
    if (l>r) return;
    *root = new KD_TREE_NODE;
    InitTreeNode(*root);
    int mid = (l+r)>>1; 
    // Find the best division Axis
    int i;
    float average[3] = {0,0,0};
    float covariance[3] = {0,0,0};
    for (i=l;i<=r;i++){
        average[0] += Storage[i].center.x;
        average[1] += Storage[i].center.y;
        average[2] += Storage[i].center.z;
    }
    for (i=0;i<3;i++) average[i] = average[i]/(r-l+1);
    for (i=l;i<=r;i++){
        covariance[0] += (Storage[i].center.x - average[0]) * (Storage[i].center.x - average[0]);
        covariance[1] += (Storage[i].center.y - average[1]) * (Storage[i].center.y - average[1]);  
        covariance[2] += (Storage[i].center.z - average[2]) * (Storage[i].center.z - average[2]);              
    }
    for (i=0;i<3;i++) covariance[i] = covariance[i]/(r-l+1);    
    int div_axis = 0;
    for (i = 1;i<3;i++){
        if (covariance[i] > covariance[div_axis]) div_axis = i;
    }
    (*root)->division_axis = div_axis;
    switch (div_axis)
    {
    case 0:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_x);
        break;
    case 1:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_y);
        break;
    case 2:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_z);
        break;
    default:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_x);
        break;
    }
    (*root)->point = Storage[mid].point; 
    (*root)->box_center = Storage[mid].center;
    (*root)->update_counter = Storage[mid].Update_Counter;
    (*root)->timestamp = Storage[mid].time;
    KD_TREE_NODE * left_son = nullptr, * right_son = nullptr;    
    BuildTree(&left_son, l, mid-1, Storage);
    BuildTree(&right_son, mid+1, r, Storage);  
    (*root)->left_son_ptr = left_son;
    (*root)->right_son_ptr = right_son;
    Update((*root));   
    return;
}

void KD_FOREST::Rebuild(KD_TREE_NODE ** root){    
    KD_TREE_NODE * father_ptr;
    if ((*root)->TreeSize >= Multi_Thread_Rebuild_Point_Num) { 
        max_need_rebuild_num = max((*root)->TreeSize,max_need_rebuild_num);
        if (!pthread_mutex_trylock(&rebuild_ptr_mutex_lock)){     
            if (Rebuild_Ptr== nullptr || ((*root)->TreeSize > (*Rebuild_Ptr)->TreeSize)) {
                Rebuild_Ptr = root;
            }
            pthread_mutex_unlock(&rebuild_ptr_mutex_lock);
        }
    } else {
        father_ptr = (*root)->father_ptr;
        bool root_node_flag = false;
        if ((*root)->is_root_node) root_node_flag = true;
        int cube_index = get_cube_index((*root)->point);
        PCL_Storage.clear();
        flatten(*root, PCL_Storage);
        delete_tree_nodes(root, NOT_RECORD);
        BuildTree(root, 0, PCL_Storage.size()-1, PCL_Storage);
        if (*root != nullptr) (*root)->father_ptr = father_ptr;
        if (root_node_flag) {
            static_roots[cube_index]->left_son_ptr = *root;
            (*root)->is_root_node = true;
        }
    } 
    return;
}

void KD_FOREST::Delete_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild){   
    if ((*root) == nullptr || (*root)->tree_deleted) return;
    if (point_in_box(point,(*root)->box_center)){
        if (!(*root)->point_deleted) {
            (*root)->point_deleted = true;
            (*root)->invalid_point_num += 1;
            if ((*root)->invalid_point_num == (*root)->TreeSize) (*root)->tree_deleted = true;    
        } 
        return;
    }
    Operation_Logger_Type delete_log;
    struct timespec Timeout;    
    delete_log.op = DELETE_POINT;
    delete_log.point = point;
    delete_log.center = get_box_center(point);
    delete_log.CubeIndex = get_cube_index(point);
    if (((*root)->division_axis == 0 && point.x < (*root)->box_center.x) || ((*root)->division_axis == 1 && point.y < (*root)->box_center.y) || ((*root)->division_axis == 2 && point.z < (*root)->box_center.z)){           
        if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){          
            Delete_by_point(&(*root)->left_son_ptr, point, allow_rebuild);         
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Delete_by_point(&(*root)->left_son_ptr, point,false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(delete_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }
    } else {       
        if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){         
            Delete_by_point(&(*root)->right_son_ptr, point, allow_rebuild);         
        } else {
            pthread_mutex_lock(&working_flag_mutex); 
            Delete_by_point(&(*root)->right_son_ptr, point, false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(delete_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }        
    }
    Update(*root);
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);
    return;
}


void KD_FOREST::Add_by_point(KD_TREE_NODE ** root, PointCubeIndexType PointCenter, bool allow_rebuild){    
    if (*root == nullptr){
        // if (thread_id == 2){
        //     printf("        !!!!Point Box is: (%0.4f,%0.4f,%0.4f)\n",PointCenter.center.x,PointCenter.center.y,PointCenter.center.z);
        //     printf("        New Point: (%0.4f,%0.4f,%0.4f)\n\n",PointCenter.point.x,PointCenter.point.y,PointCenter.point.z);        
        // }
        *root = new KD_TREE_NODE;
        InitTreeNode(*root);
        (*root)->point = PointCenter.point;
        (*root)->box_center = PointCenter.center;
        (*root)->update_counter = 1;
        (*root)->division_axis = rand() % 3;
        (*root)->timestamp = PointCenter.time;
        Update(*root);
        return;
    }

    if (point_in_box(PointCenter.point, (*root)->box_center)){
        if ((*root)->point_deleted || (*root)->update_counter <= Max_Update_Time && calc_dist(PointCenter.point, (*root)->box_center) <= calc_dist((*root)->point, (*root)->box_center)){
            (*root)->point_deleted = false;
            (*root)->tree_deleted = false;
            (*root)->point = PointCenter.point;
            Update(*root);
        }
        if ((*root)->update_counter <= Max_Update_Time) (*root)->update_counter ++;
        max_counter = max((*root)->update_counter, max_counter);
        (*root)->timestamp = PointCenter.time;
        return;
    }
    Operation_Logger_Type add_log;
    struct timespec Timeout;    
    add_log.op = ADD_POINT;
    add_log.point = PointCenter.point;
    add_log.center = PointCenter.center;
    add_log.CubeIndex = PointCenter.CubeIndex; 
    PointCubeIndexType node_pointcenter((*root)->point,(*root)->box_center,get_cube_index((*root)->point),(*root)->update_counter, (*root)->timestamp);
    if (((*root)->division_axis == 0 && point_cmp_x(PointCenter, node_pointcenter)) || ((*root)->division_axis == 1 && point_cmp_y(PointCenter, node_pointcenter)) || ((*root)->division_axis == 2 && point_cmp_z(PointCenter, node_pointcenter))){
        if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){          
            Add_by_point(&(*root)->left_son_ptr, PointCenter, allow_rebuild);
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_point(&(*root)->left_son_ptr, PointCenter, false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(add_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);            
        }
    } else {  
        if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){         
            Add_by_point(&(*root)->right_son_ptr, PointCenter, allow_rebuild);
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_point(&(*root)->right_son_ptr, PointCenter, false);       
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(add_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex); 
        }
    }
    Update(*root);
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);
    return;
}

void KD_FOREST::Search(KD_TREE_NODE * root, int k_nearest, PointType point, priority_queue<PointType_CMP> &q, double time_range, double max_dist){
    if (root == nullptr || root->tree_deleted) return;
    double cur_dist = calc_box_dist(root, point);
    if (cur_dist > max_dist) return;
    if (time_range > 0 && root->max_time < time_range) return;
    if (!root->point_deleted){
        float dist = calc_dist(point, root->point);
        if ((time_range < 0.0 || root->timestamp >= time_range) && (q.size() < k_nearest || dist < q.top().dist)){
            if (q.size() >= k_nearest) q.pop();
            PointType_CMP current_point{root->point, dist};                    
            q.push(current_point);            
        }
    } 
    float dist_left_node = calc_box_dist(root->left_son_ptr, point);
    float dist_right_node = calc_box_dist(root->right_son_ptr, point);
    if (q.size()< k_nearest || dist_left_node < q.top().dist && dist_right_node < q.top().dist){
        if (dist_left_node <= dist_right_node) {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                Search(root->left_son_ptr, k_nearest, point, q, time_range, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);
                Search(root->left_son_ptr, k_nearest, point, q, time_range, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
            if (q.size() < k_nearest || dist_right_node < q.top().dist) {
                if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                    Search(root->right_son_ptr, k_nearest, point, q, time_range, max_dist);                       
                } else {
                    pthread_mutex_lock(&search_flag_mutex);
                    while (search_mutex_counter == -1)
                    {
                        pthread_mutex_unlock(&search_flag_mutex);
                        usleep(1);
                        pthread_mutex_lock(&search_flag_mutex);
                    }
                    search_mutex_counter += 1;
                    pthread_mutex_unlock(&search_flag_mutex);                    
                    Search(root->right_son_ptr, k_nearest, point, q, time_range, max_dist);  
                    pthread_mutex_lock(&search_flag_mutex);
                    search_mutex_counter -= 1;
                    pthread_mutex_unlock(&search_flag_mutex);
                }                
            }
        } else {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                Search(root->right_son_ptr, k_nearest, point, q, time_range, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);                   
                Search(root->right_son_ptr, k_nearest, point, q, time_range, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
            if (q.size() < k_nearest || dist_left_node < q.top().dist) {            
                if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                    Search(root->left_son_ptr, k_nearest, point, q, time_range, max_dist);                       
                } else {
                    pthread_mutex_lock(&search_flag_mutex);
                    while (search_mutex_counter == -1)
                    {
                        pthread_mutex_unlock(&search_flag_mutex);
                        usleep(1);
                        pthread_mutex_lock(&search_flag_mutex);
                    }
                    search_mutex_counter += 1;
                    pthread_mutex_unlock(&search_flag_mutex);  
                    Search(root->left_son_ptr, k_nearest, point, q, time_range, max_dist);  
                    pthread_mutex_lock(&search_flag_mutex);
                    search_mutex_counter -= 1;
                    pthread_mutex_unlock(&search_flag_mutex);
                }
            }
        }
    } else {
        if (dist_left_node < q.top().dist) {        
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                Search(root->left_son_ptr, k_nearest, point, q, time_range, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);  
                Search(root->left_son_ptr, k_nearest, point, q, time_range, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
        }
        if (dist_right_node < q.top().dist) {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                Search(root->right_son_ptr, k_nearest, point, q, time_range, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);  
                Search(root->right_son_ptr, k_nearest, point, q, time_range, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
        }
    }
    return;
}

void KD_FOREST::Box_Search(BoxPointType boxpoint, vector<PointCubeIndexType> & Storage){
    int Search_Step[3],i,j,k,cube_index;
    int ori_index;
    PointType tmp_point;
    for (i = 0; i < 3; i++) Search_Step[i] = int(ceil((boxpoint.vertex_max[i] - boxpoint.vertex_min[i])/Cube_length+EPSS))+1;
    tmp_point.x = boxpoint.vertex_min[0];
    tmp_point.y = boxpoint.vertex_min[1];
    tmp_point.z = boxpoint.vertex_min[2];
    ori_index = get_cube_index(tmp_point);
    if (ori_index >= 0 || ori_index < Max_Index) Search_by_range(roots[ori_index],boxpoint, Storage);    
    for (i = 0; i < Search_Step[0]; i++){
        for (j = 0; j < Search_Step[1]; j++){
            for (k = 0; k < Search_Step[2]; k++){
                if (i == 0 && j == 0 && k == 0) continue;
                tmp_point.x = boxpoint.vertex_min[0] + Cube_length * i;
                tmp_point.y = boxpoint.vertex_min[1] + Cube_length * j;
                tmp_point.z = boxpoint.vertex_min[2] + Cube_length * k;
                cube_index = get_cube_index(tmp_point);
                if (cube_index < 0 || cube_index > Max_Index || cube_index == ori_index) continue;
                Search_by_range(roots[cube_index],boxpoint, Storage);
            }
        }       
    }
    return;
}

void KD_FOREST::Search_by_range(KD_TREE_NODE *root, BoxPointType boxpoint, vector<PointCubeIndexType> & Storage){
    if (root == nullptr) return;
    if (boxpoint.vertex_max[0] + EPSS < root->node_range_x[0] || boxpoint.vertex_min[0] - EPSS > root->node_range_x[1]) return;
    if (boxpoint.vertex_max[1] + EPSS < root->node_range_y[0] || boxpoint.vertex_min[1] - EPSS > root->node_range_y[1]) return;
    if (boxpoint.vertex_max[2] + EPSS < root->node_range_z[0] || boxpoint.vertex_min[2] - EPSS > root->node_range_z[1]) return;
    if (boxpoint.vertex_min[0] - EPSS < root->node_range_x[0] && boxpoint.vertex_max[0]+EPSS > root->node_range_x[1] && boxpoint.vertex_min[1]-EPSS < root->node_range_y[0] && boxpoint.vertex_max[1]+EPSS > root->node_range_y[1] && boxpoint.vertex_min[2]-EPSS < root->node_range_z[0] && boxpoint.vertex_max[2]+EPSS > root->node_range_z[1]){
        flatten(root, Storage);
        return;
    }
    if (boxpoint.vertex_min[0]-EPSS < root->point.x && boxpoint.vertex_max[0]+EPSS > root->point.x && boxpoint.vertex_min[1]-EPSS < root->point.y && boxpoint.vertex_max[1]+EPSS > root->point.y && boxpoint.vertex_min[2]-EPSS < root->point.z && boxpoint.vertex_max[2]+EPSS > root->point.z){
        if (!root->point_deleted) Storage.push_back(PointCubeIndexType(root->point,root->box_center,get_cube_index(root->point),root->update_counter,root->timestamp));
    }
    if ((Rebuild_Ptr == nullptr) || root->left_son_ptr != *Rebuild_Ptr){
        Search_by_range(root->left_son_ptr, boxpoint, Storage);
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_range(root->left_son_ptr, boxpoint, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    if ((Rebuild_Ptr == nullptr) || root->right_son_ptr != *Rebuild_Ptr){
        Search_by_range(root->right_son_ptr, boxpoint, Storage);
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_range(root->right_son_ptr, boxpoint, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    return;    
}

bool KD_FOREST::Criterion_Check(KD_TREE_NODE * root){
    if (root->TreeSize <= Minimal_Unbalanced_Tree_Size){
        return false;
    }
    float balance_evaluation = 0.0f;
    float delete_evaluation = 0.0f;
    KD_TREE_NODE * son_ptr = root->left_son_ptr;
    if (son_ptr == nullptr) son_ptr = root->right_son_ptr;
    delete_evaluation = float(root->invalid_point_num)/ root->TreeSize;
    balance_evaluation = float(son_ptr->TreeSize) / (root->TreeSize-1);  
    if (delete_evaluation > delete_criterion_param){
        return true;
    }
    if (balance_evaluation > balance_criterion_param || balance_evaluation < 1-balance_criterion_param){
        return true;
    } 
    return false;
}

void KD_FOREST::Update(KD_TREE_NODE * root){
    KD_TREE_NODE * left_son_ptr = root->left_son_ptr;
    KD_TREE_NODE * right_son_ptr = root->right_son_ptr;
    // Update Tree Size
    if (left_son_ptr != nullptr && right_son_ptr != nullptr){
        root->TreeSize = left_son_ptr->TreeSize + right_son_ptr->TreeSize + 1;
        root->invalid_point_num = left_son_ptr->invalid_point_num + right_son_ptr->invalid_point_num + (root->point_deleted? 1:0);
        root->tree_deleted = left_son_ptr->tree_deleted && right_son_ptr->tree_deleted && root->point_deleted;
        root->node_range_x[0] = min(min(left_son_ptr->node_range_x[0],right_son_ptr->node_range_x[0]),root->point.x);
        root->node_range_x[1] = max(max(left_son_ptr->node_range_x[1],right_son_ptr->node_range_x[1]),root->point.x);
        root->node_range_y[0] = min(min(left_son_ptr->node_range_y[0],right_son_ptr->node_range_y[0]),root->point.y);
        root->node_range_y[1] = max(max(left_son_ptr->node_range_y[1],right_son_ptr->node_range_y[1]),root->point.y);        
        root->node_range_z[0] = min(min(left_son_ptr->node_range_z[0],right_son_ptr->node_range_z[0]),root->point.z);
        root->node_range_z[1] = max(max(left_son_ptr->node_range_z[1],right_son_ptr->node_range_z[1]),root->point.z);         
        root->max_time = max(max(left_son_ptr->max_time,right_son_ptr->max_time),root->timestamp);
    } else if (left_son_ptr != nullptr){
        root->TreeSize = left_son_ptr->TreeSize + 1;
        root->invalid_point_num = left_son_ptr->invalid_point_num + (root->point_deleted?1:0);
        root->tree_deleted = left_son_ptr->tree_deleted && root->point_deleted;
        root->node_range_x[0] = min(left_son_ptr->node_range_x[0],root->point.x);
        root->node_range_x[1] = max(left_son_ptr->node_range_x[1],root->point.x);
        root->node_range_y[0] = min(left_son_ptr->node_range_y[0],root->point.y);
        root->node_range_y[1] = max(left_son_ptr->node_range_y[1],root->point.y); 
        root->node_range_z[0] = min(left_son_ptr->node_range_z[0],root->point.z);
        root->node_range_z[1] = max(left_son_ptr->node_range_z[1],root->point.z); 
        root->max_time = max(left_son_ptr->max_time,root->timestamp);              
    } else if (right_son_ptr != nullptr){
        root->TreeSize = right_son_ptr->TreeSize + 1;
        root->invalid_point_num = right_son_ptr->invalid_point_num + (root->point_deleted? 1:0);
        root->tree_deleted = right_son_ptr->tree_deleted && root->point_deleted;        
        root->node_range_x[0] = min(right_son_ptr->node_range_x[0],root->point.x);
        root->node_range_x[1] = max(right_son_ptr->node_range_x[1],root->point.x);
        root->node_range_y[0] = min(right_son_ptr->node_range_y[0],root->point.y);
        root->node_range_y[1] = max(right_son_ptr->node_range_y[1],root->point.y); 
        root->node_range_z[0] = min(right_son_ptr->node_range_z[0],root->point.z);
        root->node_range_z[1] = max(right_son_ptr->node_range_z[1],root->point.z);
        root->max_time = max(right_son_ptr->max_time,root->timestamp);
    } else {
        root->TreeSize = 1;
        root->invalid_point_num = (root->point_deleted? 1:0);
        root->tree_deleted = root->point_deleted;
        root->node_range_x[0] = root->point.x;
        root->node_range_x[1] = root->point.x;       
        root->node_range_y[0] = root->point.y;
        root->node_range_y[1] = root->point.y; 
        root->node_range_z[0] = root->point.z;
        root->node_range_z[1] = root->point.z;    
        root->max_time = root->timestamp;             
    }
    if (left_son_ptr != nullptr) left_son_ptr -> father_ptr = root;
    if (right_son_ptr != nullptr) right_son_ptr -> father_ptr = root;
    int cube_index = get_cube_index(root->point);
    if (root == roots[cube_index]){
        if (root->TreeSize > 3){
            KD_TREE_NODE * son_ptr = root->left_son_ptr;
            if (son_ptr == nullptr) son_ptr = root->right_son_ptr;
            float tmp_bal = float(son_ptr->TreeSize) / (root->TreeSize-1);
            root->alpha_del = float(root->invalid_point_num)/ root->TreeSize;
            root->alpha_bal = (tmp_bal>=0.5-EPSS)?tmp_bal:1-tmp_bal;
        }
    }
    return;
}

void KD_FOREST::flatten(KD_TREE_NODE * root, vector<PointCubeIndexType> &Storage){
    if (root == nullptr || root->tree_deleted) return;
    if (!root->point_deleted) {
        Storage.push_back(PointCubeIndexType(root->point,root->box_center,get_cube_index(root->point),root->update_counter,root->timestamp));
    }
    flatten(root->left_son_ptr, Storage);
    flatten(root->right_son_ptr, Storage);
    return;
}

void KD_FOREST::delete_tree_nodes(KD_TREE_NODE ** root, delete_point_storage_set storage_type){ 
    if (*root == nullptr) return;
    delete_tree_nodes(&(*root)->left_son_ptr, storage_type);
    delete_tree_nodes(&(*root)->right_son_ptr, storage_type);  
    switch (storage_type)
    {
    case NOT_RECORD:
        break;
    case DELETE_POINTS_REC:
        if ((*root)->point_deleted) {
            Points_deleted.push_back((*root)->point);
        }       
        break;
    case MULTI_THREAD_REC:
        pthread_mutex_lock(&points_deleted_rebuild_mutex_lock);    
        if ((*root)->point_deleted) {
            Multithread_Points_deleted.push_back((*root)->point);
        }
        pthread_mutex_unlock(&points_deleted_rebuild_mutex_lock);     
        break;
    case DOWNSAMPLE_REC:
        if (!(*root)->point_deleted) Downsample_Storage.push_back((*root)->point);
        break;
    default:
        break;
    }               
    delete *root;
    *root = nullptr;                    

    return;
}

bool KD_FOREST::same_point(PointType a, PointType b){
    return (fabs(a.x-b.x) < EPSS && fabs(a.y-b.y) < EPSS && fabs(a.z-b.z) < EPSS );
}

bool KD_FOREST::point_in_box(PointType point, PointType box_center){
    if (fabs(point.x - box_center.x) <= downsample_size/2.0 + 0.001 && fabs(point.y - box_center.y) <= downsample_size/2.0 + 0.001 && fabs(point.z - box_center.z) <= downsample_size/2.0 + 0.001){
        return true;
    }
    return false;
}

float KD_FOREST::calc_dist(PointType a, PointType b){
    float dist = 0.0f;
    dist = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z);
    return dist;
}

float KD_FOREST::calc_box_dist(KD_TREE_NODE * node, PointType point){
    if (node == nullptr) return INFINITY;
    float min_dist = 0.0;
    if (point.x < node->node_range_x[0]) min_dist += (point.x - node->node_range_x[0])*(point.x - node->node_range_x[0]);
    if (point.x > node->node_range_x[1]) min_dist += (point.x - node->node_range_x[1])*(point.x - node->node_range_x[1]);
    if (point.y < node->node_range_y[0]) min_dist += (point.y - node->node_range_y[0])*(point.y - node->node_range_y[0]);
    if (point.y > node->node_range_y[1]) min_dist += (point.y - node->node_range_y[1])*(point.y - node->node_range_y[1]);
    if (point.z < node->node_range_z[0]) min_dist += (point.z - node->node_range_z[0])*(point.z - node->node_range_z[0]);
    if (point.z > node->node_range_z[1]) min_dist += (point.z - node->node_range_z[1])*(point.z - node->node_range_z[1]);
    return min_dist;
}

bool KD_FOREST::point_cmp_x(PointCubeIndexType a, PointCubeIndexType b) { 
    if (fabs(a.center.x - b.center.x) > EPSS) return (a.center.x < b.center.x);  
    if (fabs(a.center.y - b.center.y) > EPSS) return (a.center.y < b.center.y);  
    if (fabs(a.center.z - b.center.z) > EPSS) return (a.center.z < b.center.z);  
}
bool KD_FOREST::point_cmp_y(PointCubeIndexType a, PointCubeIndexType b) {
    if (fabs(a.center.y - b.center.y) > EPSS) return (a.center.y < b.center.y);  
    if (fabs(a.center.x - b.center.x) > EPSS) return (a.center.x < b.center.x);  
    if (fabs(a.center.z - b.center.z) > EPSS) return (a.center.z < b.center.z);  
}
bool KD_FOREST::point_cmp_z(PointCubeIndexType a, PointCubeIndexType b){ 
    if (fabs(a.center.z - b.center.z) > EPSS) return (a.center.z < b.center.z);
    if (fabs(a.center.x - b.center.x) > EPSS) return (a.center.x < b.center.x);  
    if (fabs(a.center.y - b.center.y) > EPSS) return (a.center.y < b.center.y);  
}
void KD_FOREST::print_tree(int index, FILE *fp, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max){
    pthread_mutex_lock(&working_flag_mutex);
    print_treenode(Root_Node, index, fp, x_min,x_max,y_min,y_max,z_min,z_max);
    pthread_mutex_unlock(&working_flag_mutex);       
}

void KD_FOREST::print_treenode(KD_TREE_NODE * root, int index, FILE *fp, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max){
    if (root == nullptr) return;
    fprintf(fp,"%d,%0.3f,%0.3f,%0.3f",index,root->point.x,root->point.y,root->point.z);
    fprintf(fp,",%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f\n",x_min,x_max,y_min,y_max,z_min,z_max);
    switch (root->division_axis)
    {
    case 0:
        print_treenode(root->left_son_ptr, index, fp, x_min,root->point.x,y_min,y_max,z_min,z_max);
        print_treenode(root->right_son_ptr,index, fp, root->point.x,x_max,y_min,y_max,z_min,z_max);  
        break;
    case 1:
        print_treenode(root->left_son_ptr, index, fp, x_min,x_max,y_min,root->point.y,z_min,z_max);
        print_treenode(root->right_son_ptr,index, fp, x_min,x_max,root->point.y,y_max,z_min,z_max);   
        break;
    case 2:
        print_treenode(root->left_son_ptr, index, fp, x_min,x_max,y_min,y_max,z_min,root->point.z);
        print_treenode(root->right_son_ptr,index, fp, x_min,x_max,y_min,y_max,root->point.z,z_max);   
        break;             
    default:
        break;
    }
    return;    
}

void KD_FOREST::sort_by_cubeindex(PointVector &Points, vector<PointCubeIndexType> &sort_points, double timestamp){
    sort_points.clear();
    for (int i=0; i < Points.size();i++){
        sort_points.push_back(PointCubeIndexType(Points[i], get_box_center(Points[i]),get_cube_index(Points[i]),1, timestamp));
    }   
    sort(sort_points.begin(),sort_points.end());
}

int KD_FOREST::get_cube_index(PointType point){
    long long index_x, index_y, index_z, cube_index;
    index_x = int(round(floor((point.x - x_min)/Cube_length + EPSS)));
    index_y = int(round(floor((point.y - y_min)/Cube_length + EPSS)));
    index_z = int(round(floor((point.z - z_min)/Cube_length + EPSS)));
    if (!HashOn){
        cube_index = index_x + index_y * Env_Length + index_z * Env_Length * Env_Width;
    } else {
        cube_index = (((((index_z * prime) % MAX_HASH_LEN + index_y) * prime) % MAX_HASH_LEN) + index_x) % MAX_HASH_LEN;
    }
    return cube_index;
}

PointType KD_FOREST::get_box_center(PointType point){
    PointType box_center;
    box_center.x = floor((point.x - x_min)/downsample_size + EPSS) * downsample_size + x_min + downsample_size/2;
    box_center.y = floor((point.y - y_min)/downsample_size + EPSS) * downsample_size + y_min + downsample_size/2;
    box_center.z = floor((point.z - z_min)/downsample_size + EPSS) * downsample_size + z_min + downsample_size/2;
    return box_center;
}

void KD_FOREST::Initial_Downsample(PointVector &points, PointVector &downsampled_points, PointType min_center, PointType max_center){
    int len_x,len_y,len_z,i;
    long long max_index, ind_x, ind_y, ind_z, box_index;
    int * point_index;
    PointType center;
    bool hash_on = false;
    len_x = int(round((max_center.x - min_center.x)/downsample_size) + 1);
    len_y = int(round((max_center.y - min_center.y)/downsample_size) + 1);
    len_z = int(round((max_center.z - min_center.z)/downsample_size) + 1);
    max_index = len_x*len_y*len_z;
    if (max_index > 1e6){
        hash_on = true;
        max_index = int(1e6);
    }
    point_index = new int[max_index];
    for (i = 0; i<max_index; i++) point_index[i] = -1;
    downsampled_points.clear();
    for (i = 0;i<points.size();i++){
        center = get_box_center(points[i]);
        ind_x = static_cast<long long>(round(floor((center.x - min_center.x  + EPSS)/downsample_size)));
        ind_y = static_cast<long long>(round(floor((center.y - min_center.y  + EPSS)/downsample_size)));
        ind_z = static_cast<long long>(round(floor((center.z - min_center.z  + EPSS)/downsample_size)));
        if (hash_on){
            box_index = ((((ind_z * prime) % max_index + ind_y) * prime) % max_index + ind_x) % max_index;
        } else {
            box_index = ind_x + ind_y * len_x + ind_z * len_x * len_y;
        }
        if (point_index[box_index] == -1){
            point_index[box_index] = downsampled_points.size();
            downsampled_points.push_back(points[i]);
        } else {
            if (calc_dist(center, downsampled_points[point_index[box_index]]) > calc_dist(center, points[i])){
                downsampled_points[point_index[box_index]] = points[i];
            }
        }
    }
}

// manual queue
void MANUAL_QUEUE::clear(){
    head = 0;
    tail = 0;
    counter = 0;
    is_empty = true;
    return;
}

void MANUAL_QUEUE::pop(){
    if (counter == 0) return;
    head ++;
    head %= Q_LEN;
    counter --;
    if (counter == 0) is_empty = true;
    return;
}

Operation_Logger_Type MANUAL_QUEUE::front(){
    return q[head];
}

Operation_Logger_Type MANUAL_QUEUE::back(){
    return q[tail];
}

void MANUAL_QUEUE::push(Operation_Logger_Type op){
    q[tail] = op;
    counter ++;
    if (is_empty) is_empty = false;
    tail ++;
    tail %= Q_LEN;
}

bool MANUAL_QUEUE::empty(){
    return is_empty;
}



