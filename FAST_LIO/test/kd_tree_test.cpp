#include <pcl/kdtree/kdtree_flann.h>
#include <flann/flann.hpp>
#include <ikd-Tree/ikd_Tree.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <common_lib.h>


#define X_MAX 5.0
#define X_MIN -5.0
#define Y_MAX 5.0
#define Y_MIN -5.0
#define Z_MAX 5.0
#define Z_MIN -5.0

#define Point_Num 50000
#define New_Point_Num 200
#define Delete_Point_Num 0
#define Nearest_Num 5
#define Test_Time 1000
#define Search_Time 200
#define Box_Length 1.0
#define Box_Num 4
#define Delete_Box_Switch false
#define Add_Box_Switch false
#define MAXN 1000000
#define DOWNSAMPLE_LEN 0.2

// PointVector point_cloud;
PointVector point_cloud;
PointVector cloud_increment;
PointVector cloud_decrement;
PointVector cloud_deleted;
PointVector search_result;
PointVector raw_cmp_result;
PointVector DeletePoints;
PointVector removed_points;


int X_LEN, Y_LEN, Z_LEN;
PointType point_cloud_arr[MAXN];
bool box_occupy[MAXN];

KD_TREE scapegoat_kd_tree(0.3,0.6,DOWNSAMPLE_LEN);

float rand_float(float x_min, float x_max){
    float rand_ratio = rand()/(float)RAND_MAX;
    return (x_min + rand_ratio * (x_max - x_min));
}

float calc_dist(PointType p1, PointType p2){
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

void insert_point_cloud(PointType point){
    int index_x, index_y, index_z, index;
    index_x = int(floor((point.x - X_MIN)/DOWNSAMPLE_LEN));
    index_y = int(floor((point.y - Y_MIN)/DOWNSAMPLE_LEN));
    index_z = int(floor((point.z - Z_MIN)/DOWNSAMPLE_LEN));
    index = index_x + index_y * X_LEN + index_z* X_LEN * Y_LEN;
    if (!box_occupy[index]){
        point_cloud_arr[index] = point;
        box_occupy[index] = true;
    } else {
        PointType center;
        center.x = index_x * DOWNSAMPLE_LEN + DOWNSAMPLE_LEN/2 - X_LEN/2*DOWNSAMPLE_LEN;
        center.y = index_y * DOWNSAMPLE_LEN + DOWNSAMPLE_LEN/2 - Y_LEN/2*DOWNSAMPLE_LEN;
        center.z = index_z * DOWNSAMPLE_LEN + DOWNSAMPLE_LEN/2 - Z_LEN/2*DOWNSAMPLE_LEN;
        if (calc_dist(center, point) < calc_dist(center, point_cloud_arr[index])){
            point_cloud_arr[index] = point;
        }
    }
}

bool check_in_box(BoxPointType box, PointType point){
    if (point.x >= box.vertex_min[0] && point.x <= box.vertex_max[0] && point.y >= box.vertex_min[1] && point.y <= box.vertex_max[1] && point.z >= box.vertex_min[2] && point.z <= box.vertex_max[2]){
        return true;
    }
    return false;
}

void delete_points_in_box(BoxPointType box){
    int box_min[3], box_max[3];
    int i,j,k;
    int index;   

    for (i = 0; i<3;i++){
        box_min[i] = floor((box.vertex_min[i]+5)/DOWNSAMPLE_LEN);
        box_max[i] = ceil((box.vertex_max[i]+5)/DOWNSAMPLE_LEN);
    }
    for (i = box_min[0]; i<box_max[0]; i++){
        for (j = box_min[1]; j<box_max[1]; j++){
            for (k = box_min[2]; k<box_max[2]; k++){
                index = i + j * X_LEN + k* X_LEN * Y_LEN;
                if (check_in_box(box, point_cloud_arr[index])){
                    box_occupy[index] = false;
                }
            }
        }
    }
}

void generate_initial_point_cloud(int num){
    PointVector ().swap(point_cloud);
    PointType new_point;
    int index;
    for (int i=0;i<num;i++){
        new_point.x = rand_float(X_MIN, X_MAX);
        new_point.y = rand_float(Y_MIN, Y_MAX);
        new_point.z = rand_float(Z_MIN, Z_MAX);
        insert_point_cloud(new_point);
    }
    int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;
    int i;
    point_cloud.clear();
    for (i=0;i<LEN*LEN*LEN;i++){
        if (box_occupy[i]){
            point_cloud.push_back(point_cloud_arr[i]);
        }
    }
    return;
}

void generate_increment_point_cloud(int num){
    PointVector ().swap(cloud_increment);
    PointType new_point;
    int index;
    for (int i=0;i<num;i++){
        new_point.x = rand_float(X_MIN, X_MAX);
        new_point.y = rand_float(Y_MIN, Y_MAX);
        new_point.z = rand_float(Z_MIN, Z_MAX);
        insert_point_cloud(new_point);        
        cloud_increment.push_back(new_point);        
    }
    int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;
    int i;
    point_cloud.clear();
    for (i=0;i<LEN*LEN*LEN;i++){
        if (box_occupy[i]){
            point_cloud.push_back(point_cloud_arr[i]);
        }
    }    
    return;
}

void generate_box_decrement(vector<BoxPointType> & Delete_Boxes, float box_length, int box_num){
    vector<BoxPointType> ().swap(Delete_Boxes);
    float d = box_length/2;
    float x_p, y_p, z_p;
    BoxPointType boxpoint;
    for (int k=0;k < box_num; k++){
        x_p = rand_float(X_MIN, X_MAX);
        y_p = rand_float(Y_MIN, Y_MAX);
        z_p = rand_float(Z_MIN, Z_MAX);        
        boxpoint.vertex_min[0] = x_p - d;
        boxpoint.vertex_max[0] = x_p + d;
        boxpoint.vertex_min[1] = y_p - d;
        boxpoint.vertex_max[1] = y_p + d;  
        boxpoint.vertex_min[2] = z_p - d;
        boxpoint.vertex_max[2] = z_p + d;
        Delete_Boxes.push_back(boxpoint);
        delete_points_in_box(boxpoint);
        printf("Deleted box: x:(%0.3f %0.3f) y:(%0.3f %0.3f) z:(%0.3f %0.3f)\n",boxpoint.vertex_min[0],boxpoint.vertex_max[0],boxpoint.vertex_min[1],boxpoint.vertex_max[1], boxpoint.vertex_min[2],boxpoint.vertex_max[2]); 
    }
    int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;
    int i;
    point_cloud.clear();
    for (i=0;i<LEN*LEN*LEN;i++){
        if (box_occupy[i]){
            point_cloud.push_back(point_cloud_arr[i]);
        }
    }    
}

void generate_box_increment(vector<BoxPointType> & Add_Boxes, float box_length, int box_num){
    vector<BoxPointType> ().swap(Add_Boxes);
    float d = box_length/2;
    float x_p, y_p, z_p;
    BoxPointType boxpoint;
    for (int k=0;k < box_num; k++){
        x_p = rand_float(X_MIN, X_MAX);
        y_p = rand_float(Y_MIN, Y_MAX);
        z_p = rand_float(Z_MIN, Z_MAX);        
        boxpoint.vertex_min[0] = x_p - d;
        boxpoint.vertex_max[0] = x_p + d;
        boxpoint.vertex_min[1] = y_p - d;
        boxpoint.vertex_max[1] = y_p + d;  
        boxpoint.vertex_min[2] = z_p - d;
        boxpoint.vertex_max[2] = z_p + d;
        Add_Boxes.push_back(boxpoint);
        int n = cloud_deleted.size();
        int counter = 0;
        while (counter < n){
            PointType tmp = cloud_deleted[cloud_deleted.size()-1];
            cloud_deleted.pop_back();
        
            if (tmp.x +EPSS < boxpoint.vertex_min[0] || tmp.x - EPSS > boxpoint.vertex_max[0] || tmp.y + EPSS < boxpoint.vertex_min[1] || tmp.y - EPSS > boxpoint.vertex_max[1] || tmp.z + EPSS < boxpoint.vertex_min[2] || tmp.z - EPSS > boxpoint.vertex_max[2]){
                cloud_deleted.insert(cloud_deleted.begin(),tmp);
            } else {
                // printf("        Incremental Push back: (%0.3f, %0.3f, %0.3f)\n",tmp.x,tmp.y,tmp.z);                    
                point_cloud.push_back(tmp);
            }
            counter += 1;
        }
        printf("Add boxes: x:(%0.3f %0.3f) y:(%0.3f %0.3f) z:(%0.3f %0.3f)\n",boxpoint.vertex_min[0],boxpoint.vertex_max[0],boxpoint.vertex_min[1],boxpoint.vertex_max[1], boxpoint.vertex_min[2],boxpoint.vertex_max[2]); 
    }
}

PointType generate_target_point(){
    PointType point;
    point.x = rand_float(X_MIN, X_MAX);
    point.y = rand_float(Y_MIN, Y_MAX);
    point.z = rand_float(Z_MIN, Z_MAX);
    return point;
}

void generate_decrement_point_cloud(int num){
    PointVector ().swap(cloud_decrement);
    auto rng = default_random_engine();
    shuffle(point_cloud.begin(), point_cloud.end(), rng);    
    for (int i=0;i<num;i++){
        cloud_decrement.push_back(point_cloud[point_cloud.size()-1]);
        point_cloud.pop_back();
    }
    return;
}

void print_point_vec(PointVector vec){
    printf("Size is %d\n", int(vec.size()));
    for (int i=0;i<vec.size();i++){
        printf("(%0.3f, %0.3f, %0.3f)\n",vec[i].x,vec[i].y,vec[i].z);
    }
    return;
}

void raw_cmp(PointType target, int k_nearest){
    vector<PointType_CMP> points;
    PointVector ().swap(raw_cmp_result);
    float dist;
    for (int i=0;i<point_cloud.size();i++){
        dist = (point_cloud[i].x-target.x)*(point_cloud[i].x-target.x) + (point_cloud[i].y-target.y)*(point_cloud[i].y-target.y) + (point_cloud[i].z-target.z)*(point_cloud[i].z-target.z);
        PointType_CMP tmp{point_cloud[i],dist};
        points.push_back(tmp);
    }
    sort(points.begin(),points.end());
    for (int i=0;i<k_nearest;i++){
        raw_cmp_result.push_back(points[i].point);
    }
    return;
}

bool cmp_point_vec(PointVector a, PointVector b){
    if (a.size() != b.size()) return false;
    for (int i =0;i<a.size();i++){
        if (fabs(a[i].x-b[i].x)>EPSS || fabs(a[i].y-b[i].y)>EPSS || fabs(a[i].y-b[i].y)>EPSS) return false;
    }
    return true;
}

void output_clouds(){
    int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;
    int i, ct = 0;
    // printf("Original points\n");
    for (i=0;i<LEN*LEN*LEN;i++){
        if (box_occupy[i]){
            // printf("%0.3f,%0.3f,%0.3f\n",point_cloud_arr[i].x,point_cloud_arr[i].y,point_cloud_arr[i].z);
            ct ++;
        }
    }
    // printf("ikd-Tree \n");
    PointVector tmp_point;
    scapegoat_kd_tree.flatten(scapegoat_kd_tree.Root_Node, tmp_point);
    // print_point_vec(tmp_point);
    printf("\n Sizes are: %d %d\n",ct,int(tmp_point.size()));
}

flann::Matrix<float> query;

int main(int argc, char** argv){
    srand((unsigned) time(NULL));
    char c;

    int counter = 0;
    bool flag = true;
    vector<BoxPointType> Delete_Boxes;
    vector<BoxPointType> Add_Boxes;
    vector<float> PointDist;
    float max_total_time = 0.0;
    float box_delete_time = 0.0;
    float box_add_time = 0.0;
    float add_time = 0.0;
    float delete_time = 0.0;
    float search_time = 0.0;
    float ori_build_time = 0.0;
    float ori_search_time = 0.0;
    int max_point_num = 0;
    int point_num_start = 0;
    float average_total_time = 0.0f;
    int add_rebuild_record = 0, add_tmp_rebuild_counter = 0;
    int delete_rebuild_record = 0, delete_tmp_rebuild_counter = 0;    
    int delete_box_rebuild_record = 0, delete_box_tmp_rebuild_counter = 0;
    X_LEN = int(round((X_MAX - X_MIN)/DOWNSAMPLE_LEN));
    Y_LEN = int(round((Y_MAX - Y_MIN)/DOWNSAMPLE_LEN));
    Z_LEN = int(round((Z_MAX - Z_MIN)/DOWNSAMPLE_LEN));
    int wa_rec = 0;
    PointType target; 
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
    PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
    // Initialize k-d tree
    FILE *fp_log, *fp_kd_space;
    fp_kd_space = fopen("kd_tree_plot_log.csv","w");
    fprintf(fp_kd_space,"Operation Index, x, y, z\n");
    fclose(fp_kd_space);
    fp_log = fopen("kd_tree_test_log.csv","w");
    fprintf(fp_log,"Add, Delete Points, Delete Boxes, Add Boxes, Search, Total, Ori_Build, Ori_Search, Treesize, ValidNum\n");
    fclose(fp_log);
    fp_log = fopen("kd_tree_test_param.csv","w");
    fprintf(fp_log,"Add Num, Delete Num, Boxes Length, Boxes Num, Search Num, Total Num\n");  
    fprintf(fp_log,"%d,%d,%0.3f,%d,%d,%d\n", New_Point_Num, Delete_Point_Num, Box_Length, Box_Num, Search_Time, Point_Num);
    fclose(fp_log);
    generate_initial_point_cloud(Point_Num);

    auto t1 = chrono::high_resolution_clock::now();
    scapegoat_kd_tree.Build(point_cloud);    
    auto t2 = chrono::high_resolution_clock::now();    
    auto build_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
    printf("Build tree time cost is: %0.3f\n",build_duration/1e3);
    fp_kd_space = fopen("kd_tree_plot_log.csv","a");
    scapegoat_kd_tree.print_tree(0,fp_kd_space,X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX);
    int counter_div = 0;
    printf("Testing ...\n");    
    while (counter < Test_Time){
        // counter_div = int(counter/100 + EPSS);
        // generate_initial_point_cloud(Point_Num * (counter_div + 1));
        // scapegoat_kd_tree.Build(point_cloud);            
        printf("Test %d:\n",counter+1);      
        // Incremental Operation
        generate_increment_point_cloud(New_Point_Num);

        // generate_increment_point_cloud(New_Point_Num);
        printf("Start add\n");
        // printf("New Points are\n");
        // print_point_vec(cloud_increment);
        t1 = chrono::high_resolution_clock::now();
        scapegoat_kd_tree.Add_Points(cloud_increment, true);
        t2 = chrono::high_resolution_clock::now();
        auto add_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();      
        auto total_duration = add_duration;
        // add_rebuild_record = scapegoat_kd_tree.rebuild_counter;
        printf("Add point time cost is %0.3f ms\n",float(add_duration)/1e3);
        // Decremental Operation
        generate_decrement_point_cloud(Delete_Point_Num);     
        printf("Start delete\n");        
        t1 = chrono::high_resolution_clock::now();
        scapegoat_kd_tree.Delete_Points(cloud_decrement);
        t2 = chrono::high_resolution_clock::now();
        auto delete_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();       
        total_duration += delete_duration;
        printf("Delete point time cost is %0.3f ms\n",float(delete_duration)/1e3);      
        auto box_delete_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();
        // delete_tmp_rebuild_counter = scapegoat_kd_tree.rebuild_counter;        
        // Box Decremental Operation
        if (Delete_Box_Switch && (counter+1) % 50  == 0){ 
            printf("Wrong answer with counter %d\n", wa_rec);               
            generate_box_decrement(Delete_Boxes, Box_Length, Box_Num);
            t1 = chrono::high_resolution_clock::now();
            scapegoat_kd_tree.Delete_Point_Boxes(Delete_Boxes);
            t2 = chrono::high_resolution_clock::now();            
            box_delete_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            printf("Delete box points time cost is %0.3f ms\n",float(box_delete_duration)/1e3); 
            // delete_box_tmp_rebuild_counter = scapegoat_kd_tree.rebuild_counter;
        }
        total_duration += box_delete_duration;  
        auto box_add_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();        
        if (Add_Box_Switch && (counter+1) % 100  == 0){ 
            printf("Wrong answer with counter %d\n", wa_rec);               
            generate_box_increment(Add_Boxes, Box_Length, Box_Num);
            t1 = chrono::high_resolution_clock::now();
            scapegoat_kd_tree.Add_Point_Boxes(Add_Boxes);
            t2 = chrono::high_resolution_clock::now();            
            box_add_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            printf("Add box points time cost is %0.3f ms\n",float(box_add_duration)/1e3); 
            // delete_box_tmp_rebuild_counter = scapegoat_kd_tree.rebuild_counter;
        }
        total_duration += box_add_duration;               
        // Search Operation
        printf("Original K-D Tree Reconstruction:\n"); 
        int rows = point_cloud.size();
        int cols = 3;
        flann::Matrix<float> dataset(new float[rows*cols], rows, cols);
        for (int i = 0; i<rows; i++){
            dataset[i][0] = point_cloud[i].x;
            dataset[i][1] = point_cloud[i].y;
            dataset[i][2] = point_cloud[i].z;
        }
        // featsFromMap->points = point_cloud;   
        t1 = chrono::high_resolution_clock::now();
        flann::Index<flann::L2<float>> index(dataset, flann::KDTreeSingleIndexParams(1));
        index.buildIndex();
        // kdtreeSurfFromMap->setInputCloud(nullptr);
        t2 = chrono::high_resolution_clock::now();
        // scanf("%d",&c);        
        auto ori_build_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();        
        printf("Reconstruction time cost is %0.3f ms\n", float(ori_build_duration)/1e3);
        auto search_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();
        auto ori_search_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();
        printf("Start Search\n"); 
        usleep(1000);                      
        for (int k=0;k<Search_Time;k++){
            PointVector ().swap(search_result);             
            target = generate_target_point();    
            t1 = chrono::high_resolution_clock::now();
            scapegoat_kd_tree.Nearest_Search(target, Nearest_Num, search_result, PointDist);
            t2 = chrono::high_resolution_clock::now();          
            search_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            flann::Matrix<float> query(new float[cols],1,cols);
            query[0][0] = target.x;
            query[0][1] = target.y;
            query[0][2] = target.z;
            flann::Matrix<int> indices(new int[query.rows * 5], query.rows, 5);
            flann::Matrix<float> dists(new float[query.rows * 5], query.rows, 5);
            t1 = chrono::high_resolution_clock::now();
            // std::vector<int> tmptmp;
            // kdtreeSurfFromMap->nearestKSearch(target, Nearest_Num, tmptmp, PointDist);
            index.knnSearch(query, indices, dists, 5, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED,0,true));
            t2 = chrono::high_resolution_clock::now();            
            for (int i=0;i<5;i++){
                // printf("ikd-Tree %0.3f FLANN %0.3f\n", PointDist[i], dists[0][i]);
                if (fabs(PointDist[i] - dists[0][i])>0.0001){
                    printf("\n\n\n\n ERROR!\n\n\n\n\n\n\n");
                    flag = false;
                }
            }

            // for (int g = 0; g<5; g++){
            //     printf("%0.3f,",dists[0][g]);
            // }
            // printf("\n");
            ori_search_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
        }
        printf("Search nearest point time cost is %0.3f ms\n",float(search_duration)/1e3);
        printf("Original Search nearest point time cost is %0.3f ms\n",float(ori_search_duration)/1e3);
        total_duration += search_duration;
        printf("Total time is %0.3f ms\n\n",total_duration/1e3);
        // if (float(total_duration) > max_total_time){
        //     max_total_time = float(total_duration);
        //     box_delete_time = box_delete_duration;
        //     box_add_time = box_add_duration;
        //     add_time = add_duration;
        //     delete_time = delete_duration;
        //     search_time = search_duration;            
        //     max_point_num = point_num_start;
        //     add_rebuild_record = add_tmp_rebuild_counter;
        //     delete_rebuild_record = delete_tmp_rebuild_counter;
        //     delete_box_rebuild_record = delete_box_tmp_rebuild_counter;
        // }
        // Calculate total running time
        average_total_time += float(total_duration)/1e3;
        box_delete_time += float(box_delete_duration)/1e3;
        box_add_time += float(box_add_duration)/1e3;
        add_time += float(add_duration)/1e3;
        delete_time += float(delete_duration)/1e3;
        search_time += float(search_duration)/1e3;         
        max_total_time = max(max_total_time, float(total_duration));
        max_point_num = max(max_point_num,point_num_start);
        // raw_cmp(target, Nearest_Num);    
        // flag = cmp_point_vec(search_result, raw_cmp_result);    
  
        if (!flag) {
            wa_rec += 1;
            printf("Wrong answer with counter %d\n", wa_rec);
        }
        counter += 1;    
        fp_log = fopen("kd_tree_test_log.csv","a");
        fprintf(fp_log,"%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n",float(add_duration)/1e3,float(delete_duration)/1e3,float(box_delete_duration)/1e3,float(box_add_duration)/1e3,float(search_duration)/1e3,float(total_duration)/1e3, float(ori_build_duration)/1e3, float(ori_search_duration)/1e3, scapegoat_kd_tree.size(), scapegoat_kd_tree.validnum());
        fclose(fp_log);        
        printf("Treesize: %d\n", scapegoat_kd_tree.size());
        PointVector ().swap(removed_points);
        scapegoat_kd_tree.acquire_removed_points(removed_points);
        scapegoat_kd_tree.print_tree(counter, fp_kd_space, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX);
        // print_point_vec(removed_points);
        output_clouds();
    }
    fclose(fp_kd_space);
    printf("Test time is : %d\n",Test_Time);
    usleep(1e5);
    // printf("Point Cloud Points:\n");
    // printf("Target Point is : (%0.3f, %0.3f, %0.3f)\n", target.x, target.y, target.z);
    FILE *fp;
    if (!flag & (scapegoat_kd_tree.Root_Node==nullptr || scapegoat_kd_tree.Root_Node->TreeSize >=5)){
        printf("Find Dataset for debug!\n");
        fp = freopen("Data_for_fault.txt","w",stdout);
        for (int i=0;i<point_cloud.size();i++){
            fprintf(fp,"%0.6f,%0.6f,%0.6f\n",point_cloud[i].x,point_cloud[i].y,point_cloud[i].z);
        }
        printf("Raw cmp:\n");
        print_point_vec(raw_cmp_result);
        printf("k d tree\n");
        print_point_vec(search_result);
        printf("Points in kd_tree\n");
        scapegoat_kd_tree.flatten(scapegoat_kd_tree.Root_Node, scapegoat_kd_tree.PCL_Storage);
        print_point_vec(scapegoat_kd_tree.PCL_Storage);
        fclose(stdout);        
    } else {
        printf("Finished %d times test\n",counter);
        printf("Average Total Time is: %0.3fms\n",average_total_time);
        printf("Add Time is: %0.3fms\n",add_time);        
        printf("Delete Time is: %0.3fms\n",delete_time);
        printf("Box delete Time is: %0.3fms\n",box_delete_time);    
        printf("Box Add Time is: %0.3fms\n",box_add_time);          
        printf("Search Time is: %0.3fms\n",search_time);           
        printf("Corresponding point number is: %d\n",max_point_num);
        PointVector ().swap(scapegoat_kd_tree.PCL_Storage);
        // t1 = chrono::high_resolution_clock::now(); 
        // if (scapegoat_kd_tree.Root_Node != nullptr) scapegoat_kd_tree.flatten(scapegoat_kd_tree.Root_Node, scapegoat_kd_tree.PCL_Storage);
        // t2 = chrono::high_resolution_clock::now();               
        // auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
        // printf("Traverse time is %0.3f ms\n",duration/1e3);
        // print_point_vec(scapegoat_kd_tree.PCL_Storage);           
    }
    printf("WA counter is %d\n",wa_rec);
    return 0;
}