#include "ikd_Forest.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <flann/flann.hpp>

#define ENV_X 3
#define ENV_Y 3
#define ENV_Z 3
#define DOWNSAMPLE_LEN 0.2
#define Cube_Len 5.0f
#define X_MIN (-ENV_X/2*Cube_Len - Cube_Len/2.0)
#define Y_MIN (-ENV_Y/2*Cube_Len - Cube_Len/2.0)
#define Z_MIN (-ENV_Z/2*Cube_Len - Cube_Len/2.0)
#define X_MAX (X_MIN + Cube_Len * ENV_X)
#define Y_MAX (Y_MIN + Cube_Len * ENV_Y)
#define Z_MAX (Z_MIN + Cube_Len * ENV_Z)

#define Point_Num 10000
#define New_Point_Num 0
#define Delete_Point_Num 0
#define Nearest_Num 5
#define Test_Time 1
#define Search_Counter 5
#define Box_Length 1.5
#define Box_Num 4
#define Delete_Box_Switch false
#define Add_Box_Switch false
#define MAXN 1000000


PointVector point_cloud;
PointVector cloud_increment;
PointVector cloud_decrement;
PointVector cloud_deleted;
PointVector search_result;
PointVector raw_cmp_result;
PointVector DeletePoints;
PointVector removed_points;

KD_FOREST ikd_Tree(0.3,0.6,0.2,ENV_X, ENV_Y,ENV_Z,Cube_Len);

int X_LEN, Y_LEN, Z_LEN;
PointType point_cloud_arr[MAXN];
bool box_occupy[MAXN];
int box_update_counter[MAXN];
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
    PointType center;
    center.x = index_x * DOWNSAMPLE_LEN + DOWNSAMPLE_LEN/2 - X_LEN/2.0*DOWNSAMPLE_LEN;
    center.y = index_y * DOWNSAMPLE_LEN + DOWNSAMPLE_LEN/2 - Y_LEN/2.0*DOWNSAMPLE_LEN;
    center.z = index_z * DOWNSAMPLE_LEN + DOWNSAMPLE_LEN/2 - Z_LEN/2.0*DOWNSAMPLE_LEN;
    if (!box_occupy[index]){
        point_cloud_arr[index] = point;
        box_occupy[index] = true;
        box_update_counter[index] = 1;
    } else {
        if (box_update_counter[index] < Max_Update_Time && calc_dist(center, point) < calc_dist(center, point_cloud_arr[index])){
            point_cloud_arr[index] = point;
        }
        if (box_update_counter[index] < Max_Update_Time) box_update_counter[index] ++;
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

/*
   Generate the points to initialize an incremental k-d tree
*/


void generate_initial_point_cloud(int num, bool use_random){
    if (use_random){
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
    } else {
        PointVector ().swap(point_cloud);
        FILE * fp = fopen("tree.in","r");
        int number;
        fscanf(fp, "%d\n", &number);
        PointType tmp_point;
        point_cloud.clear();
        for (int i=0;i<number;i++){
            fscanf(fp, "%f,%f,%f\n",&tmp_point.x,&tmp_point.y,&tmp_point.z);
            insert_point_cloud(tmp_point);
        }
        point_cloud.clear();
        int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;        
        for (int i=0;i<LEN*LEN*LEN;i++){
            if (box_occupy[i]){
                point_cloud.push_back(point_cloud_arr[i]);
            }
        }        
    }

    return;
}

/*
    Generate random new points for point-wise insertion to the incremental k-d tree
*/

void generate_increment_point_cloud(int num, bool use_random){
    if (use_random){
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
    } else {
        FILE * fp = fopen("add.in","r");
        int number;
        fscanf(fp, "%d\n", &number);
        PointType tmp_point;
        PointVector ().swap(cloud_increment);
        for (int i=0;i<number;i++){
            fscanf(fp, "%f,%f,%f\n",&tmp_point.x,&tmp_point.y,&tmp_point.z);
            insert_point_cloud(tmp_point);        
            cloud_increment.push_back(tmp_point);
        }      
        point_cloud.clear();
        int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;        
        for (int i=0;i<LEN*LEN*LEN;i++){
            if (box_occupy[i]){
                point_cloud.push_back(point_cloud_arr[i]);
            }
        }           
    }
    
    return;
}


/*
    Generate random points for point-wise delete on the incremental k-d tree
*/

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

/*
    Generate random boxes for box-wise re-insertion on the incremental k-d tree
*/

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
                point_cloud.push_back(tmp);
            }
            counter += 1;
        }
    }
}

/*
    Generate random boxes for box-wise delete on the incremental k-d tree
*/

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


/*
    Generate target point for nearest search on the incremental k-d tree
*/

PointType generate_target_point(bool use_random){
    PointType point;
    if (use_random){
        point.x = rand_float(0, X_MAX/5);
        point.y = rand_float(0, Y_MAX/5);
        point.z = rand_float(0, Z_MAX/5);
    } else {
        point.x =  1.3019f;
        point.y = 0.9463f;
        point.z = 0.0323f;
    }
    return point;
}

void print_point_vec(PointVector vec){
    printf("Size is %d\n", int(vec.size()));
    for (int i=0;i<vec.size();i++){
        printf("(%0.3f, %0.3f, %0.3f)\n",vec[i].x,vec[i].y,vec[i].z);
    }
    return;
}


void output_clouds(){
    int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;
    int i, ct = 0;
    // printf("Original points\n");
    for (i=0;i<point_cloud.size();i++){
        // printf("%0.3f,%0.3f,%0.3f\n",point_cloud[i].x,point_cloud[i].y,point_cloud[i].z);
        ct ++;
    }
    // printf("ikd-Tree \n");
    vector<PointCubeIndexType> tmp_points;
    for (int i=0;i<ENV_X * ENV_Y * ENV_Z;i++){
        ikd_Tree.flatten(ikd_Tree.roots[i],tmp_points);
    }
    PointVector tmp;
    for (int i=0;i<tmp_points.size();i++){
        tmp.push_back(tmp_points[i].point);
    }
    // print_point_vec(tmp);
    printf("\n Sizes are: %d %d\n",ct,int(tmp_points.size()));
}

flann::Matrix<float> query;

void get_cube_index(PointType point, int index[]){
    index[0] = int(round(floor((point.x - X_MIN)/Cube_Len + EPSS)));
    index[1] = int(round(floor((point.y - Y_MIN)/Cube_Len + EPSS)));
    index[2] = int(round(floor((point.z - Z_MIN)/Cube_Len + EPSS)));
    return;
}

int main(int argc, char** argv){
    srand((unsigned) time(NULL));
    memset(box_occupy,0,sizeof(box_occupy));
    memset(box_update_counter,0,sizeof(box_update_counter));
    printf("Testing ...\n");
    int counter = 0;
    bool flag = true;
    vector<BoxPointType> Delete_Boxes;
    vector<BoxPointType> Add_Boxes;
    vector<float> PointDist;
    float average_total_time = 0.0;
    float box_delete_time = 0.0;
    float box_add_time = 0.0;
    float add_time = 0.0;
    float delete_time = 0.0;
    float search_time = 0.0;
    int box_delete_counter = 0;
    int box_add_counter = 0;
    PointType target; 
    X_LEN = int(round((X_MAX - X_MIN)/DOWNSAMPLE_LEN));
    Y_LEN = int(round((Y_MAX - Y_MIN)/DOWNSAMPLE_LEN));
    Z_LEN = int(round((Z_MAX - Z_MIN)/DOWNSAMPLE_LEN));
    int wa_rec = 0;
    // Initialize k-d tree
    generate_initial_point_cloud(Point_Num, true);
    auto t1 = chrono::high_resolution_clock::now();
    ikd_Tree.Build(point_cloud, true);    
    auto t2 = chrono::high_resolution_clock::now(); 
    auto build_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
    while (counter < Test_Time){           
        printf("Test %d:\n",counter+1);
        // Point-wise Insertion
        generate_increment_point_cloud(New_Point_Num,true);
        t1 = chrono::high_resolution_clock::now();
        ikd_Tree.Add_Points(cloud_increment);
        t2 = chrono::high_resolution_clock::now();
        auto add_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();      
        auto total_duration = add_duration;
        printf("Add point time cost is %0.3f ms\n",float(add_duration)/1e3);
        // Point-wise Delete
        generate_decrement_point_cloud(Delete_Point_Num);            
        t1 = chrono::high_resolution_clock::now();
        ikd_Tree.Delete_Points(cloud_decrement);
        t2 = chrono::high_resolution_clock::now();
        auto delete_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();       
        total_duration += delete_duration;
        printf("Delete point time cost is %0.3f ms\n",float(delete_duration)/1e3);      
        // Box-wise Delete
        auto box_delete_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();
        if (Delete_Box_Switch && (counter+1) % 50  == 0){ 
            generate_box_decrement(Delete_Boxes, Box_Length, Box_Num);
            t1 = chrono::high_resolution_clock::now();
            ikd_Tree.Delete_Point_Boxes(Delete_Boxes);
            t2 = chrono::high_resolution_clock::now();            
            box_delete_counter ++;
            box_delete_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            printf("Delete box points time cost is %0.3f ms\n",float(box_delete_duration)/1e3); 
        }
        total_duration += box_delete_duration;  
        // Box-wise Re-insertion 
        auto box_add_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();        
        if (Add_Box_Switch && (counter+1) % 100  == 0){ 
            generate_box_increment(Add_Boxes, Box_Length, Box_Num);
            t1 = chrono::high_resolution_clock::now();
            ikd_Tree.Add_Point_Boxes(Add_Boxes);
            t2 = chrono::high_resolution_clock::now();            
            box_add_counter ++;
            box_add_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            printf("Add box points time cost is %0.3f ms\n",float(box_add_duration)/1e3); 
        }
        total_duration += box_add_duration;               
        // Nearest Search
        int rows = point_cloud.size();
        int cols = 3;
        flann::Matrix<float> dataset(new float[rows*cols], rows, cols);
        for (int i = 0; i<rows; i++){
            dataset[i][0] = point_cloud[i].x;
            dataset[i][1] = point_cloud[i].y;
            dataset[i][2] = point_cloud[i].z;
        }         
        flann::Index<flann::L2<float>> index(dataset, flann::KDTreeSingleIndexParams(1));
        index.buildIndex();        
        auto search_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();
        auto ori_search_duration = chrono::duration_cast<chrono::microseconds>(t2-t2).count();
        for (int k=0;k<Search_Counter;k++){
            PointVector ().swap(search_result);             
            target = generate_target_point(true);    
            t1 = chrono::high_resolution_clock::now();
            uint8_t ret = ikd_Tree.Nearest_Search(target, Nearest_Num, search_result, PointDist);
            t2 = chrono::high_resolution_clock::now();
            search_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            flann::Matrix<float> query(new float[cols],1,cols);
            query[0][0] = target.x;
            query[0][1] = target.y;
            query[0][2] = target.z;
            flann::Matrix<int> indices(new int[query.rows * 5], query.rows, 5);
            flann::Matrix<float> dists(new float[query.rows * 5], query.rows, 5);
            t1 = chrono::high_resolution_clock::now();
            index.knnSearch(query, indices, dists, 5, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED,0,true));
            t2 = chrono::high_resolution_clock::now();
            ori_search_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            int target_index[3], ikd_index[3], flann_index[3];
            get_cube_index(target, target_index);
            for (int i=0;i<5;i++){
                get_cube_index(search_result[i],ikd_index);

                PointType tmp_point;
                tmp_point.x = dataset[indices[0][i]][0];
                tmp_point.y = dataset[indices[0][i]][1];
                tmp_point.z = dataset[indices[0][i]][2];
                get_cube_index(tmp_point, flann_index);
                if (ret > 0 || fabs(PointDist[i] - dists[0][i])>0.0001){
                    printf("\n\n\n\n ERROR!\n");
                    printf("ikd-Tree %0.3f FLANN %0.3f\n", PointDist[i], dists[0][i]);
                    printf("Target Point (%0.4f,%0.4f,%0.4f)\n", target.x, target.y, target.z);
                    printf("Target Index (%d,%d,%d)\n",target_index[0], target_index[1], target_index[2]);                    
                    printf("    ikd Tree point (%0.4f,%0.4f,%0.4f)\n",search_result[i].x,search_result[i].y,search_result[i].z);
                    printf("    ikd Tree Index (%d,%d,%d)\n",ikd_index[0], ikd_index[1], ikd_index[2]);                    
                    printf("    flann point (%0.4f,%0.4f,%0.4f)\n",tmp_point.x, tmp_point.y, tmp_point.z);
                    printf("    flann Index (%d,%d,%d)\n", flann_index[0],flann_index[1],flann_index[2]);                    
                    flag = false;
                }
            }                   
            if (!flag) break;    
        }
        if (!flag) {
            // output_clouds();            
            break;
        }
        printf("Search nearest point time cost is %0.3f ms\n",float(search_duration)/1e3);
        printf("    Average Search Trees is %0.3f\n",ikd_Tree.total_search_counter/float(ikd_Tree.search_time_counter));
        printf("Original Search nearest point time cost is %0.3f ms\n",float(ori_search_duration)/1e3);

        total_duration += search_duration;
        printf("Total time is %0.3f ms\n",total_duration/1e3);
        printf("Cloud size is %d \n", point_cloud.size());
        // printf("Tree size is: %d\n\n", ikd_Tree.size());
        // If necessary, the removed points can be collected.
        PointVector ().swap(removed_points);
        ikd_Tree.acquire_removed_points(removed_points);
        // Calculate total running time
        average_total_time += float(total_duration)/1e3;
        box_delete_time += float(box_delete_duration)/1e3;
        box_add_time += float(box_add_duration)/1e3;
        add_time += float(add_duration)/1e3;
        delete_time += float(delete_duration)/1e3;
        search_time += float(search_duration)/1e3; 
        counter += 1;    
    }

    printf("Finished %d times test\n",counter);
    printf("Average Time:\n");
    printf("Total Time is: %0.3fms\n",average_total_time/1e3);
    printf("Point-wise Insertion (%d points): %0.3fms\n",New_Point_Num,add_time/counter);        
    printf("Point-wise Delete (%d points):    %0.3fms\n", Delete_Point_Num,delete_time/counter);
    printf("Box-wse Delete (%d boxes):        %0.3fms\n",Box_Num,box_delete_time/box_delete_counter);    
    printf("Box-wse Re-insertion (%d boxes):  %0.3fms\n",Box_Num,box_add_time/box_add_counter);          
    printf("Nearest Search (%d points):       %0.3fms\n", Search_Counter,search_time/counter);              
    return 0;
}