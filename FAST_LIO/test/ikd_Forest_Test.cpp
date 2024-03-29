#include <ikd-Forest/ikd_Forest.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <flann/flann.hpp>

#define ENV_X 200
#define ENV_Y 200
#define ENV_Z 200
#define DOWNSAMPLE_LEN 0.5
#define Cube_Len 5.0f
#define X_MIN (-ENV_X/2*Cube_Len - Cube_Len/2.0)
#define Y_MIN (-ENV_Y/2*Cube_Len - Cube_Len/2.0)
#define Z_MIN (-ENV_Z/2*Cube_Len - Cube_Len/2.0)
#define X_MAX (X_MIN + Cube_Len * ENV_X)
#define Y_MAX (Y_MIN + Cube_Len * ENV_Y)
#define Z_MAX (Z_MIN + Cube_Len * ENV_Z)

#define Point_Num 200000
#define New_Point_Num 2000
#define Delete_Point_Num 0
#define Nearest_Num 5
#define Test_Time 2000
#define Search_Counter 2000
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

KD_FOREST ikd_Tree(0.3,0.6,0.5,ENV_X, ENV_Y,ENV_Z,Cube_Len);

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
    long long index_x, index_y, index_z, index;
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
            new_point.x = rand_float(X_MIN, 0);
            new_point.y = rand_float(Y_MIN, 0);
            new_point.z = rand_float(Z_MIN, 0);
            // printf("%f,%f,%f",new_point.x,new_point.y,new_point.z);            
            // insert_point_cloud(new_point);
        }
        // int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;
        // int i;
        // point_cloud.clear();
        // for (i=0;i<LEN*LEN*LEN;i++){
        //     if (box_occupy[i]){
        //         point_cloud.push_back(point_cloud_arr[i]);
        //     }
        // }
        point_cloud.push_back(new_point);        
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
            // new_point.x = rand_float(-DOWNSAMPLE_LEN, DOWNSAMPLE_LEN);
            // new_point.y = rand_float(-DOWNSAMPLE_LEN, DOWNSAMPLE_LEN);
            // new_point.z = rand_float(-DOWNSAMPLE_LEN, DOWNSAMPLE_LEN);            
            // insert_point_cloud(new_point);        
            cloud_increment.push_back(new_point);        
        }
        // int LEN = (X_MAX - X_MIN)/DOWNSAMPLE_LEN;
        // int i;
        // point_cloud.clear();
        // for (i=0;i<LEN*LEN*LEN;i++){
        //     if (box_occupy[i]){
        //         point_cloud.push_back(point_cloud_arr[i]);
        //     }
        // }
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
    Generate target point for nearest search on the incremental k-d tree
*/

PointType generate_target_point(bool use_random){
    PointType point;
    int index = 0;
    if (use_random){
        index = int(rand_float(0,int(point_cloud.size())));
        point.x = rand_float(-Cube_Len/4, Cube_Len/4) + point_cloud[index].x;
        point.y = rand_float(-Cube_Len/4, Cube_Len/4) + point_cloud[index].y;
        point.z = rand_float(-Cube_Len/4, Cube_Len/4) + point_cloud[index].z;
    } else {
        point.x =  -6.2838f;
        point.y = -6.0690f;
        point.z = 2.5977f;
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
        printf("%0.3f,%0.3f,%0.3f\n",point_cloud[i].x,point_cloud[i].y,point_cloud[i].z);
        ct ++;
    }
    printf("ikd-Tree \n");
    vector<PointCubeIndexType> tmp_points;
    for (int i=0;i<ENV_X * ENV_Y * ENV_Z;i++){
        ikd_Tree.flatten(ikd_Tree.roots[i],tmp_points);
    }
    PointVector tmp;
    for (int i=0;i<tmp_points.size();i++){
        tmp.push_back(tmp_points[i].point);
    }
    print_point_vec(tmp);
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
    FILE *log_fp;
    log_fp = fopen("ikd-Forest_testlog.csv","w");
    fprintf(log_fp,"counter,tree size,incremental_time,search_time,valid_search_counter\n");
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
    float ori_search_time = 0.0;
    int box_delete_counter = 0;
    int box_add_counter = 0;
    PointType target; 
    X_LEN = int(round((X_MAX - X_MIN)/DOWNSAMPLE_LEN));
    Y_LEN = int(round((Y_MAX - Y_MIN)/DOWNSAMPLE_LEN));
    Z_LEN = int(round((Z_MAX - Z_MIN)/DOWNSAMPLE_LEN));
    printf("%f %f %f %f %f %f\n",X_MIN,Y_MIN,Z_MIN, X_MAX,Y_MAX,Z_MAX);
    int wa_rec = 0;
    // Initialize k-d tree
    generate_initial_point_cloud(Point_Num, true);
    auto t1 = chrono::high_resolution_clock::now();
    ikd_Tree.Build(point_cloud, true,0);    
    auto t2 = chrono::high_resolution_clock::now(); 
    auto build_duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
    while (counter < Test_Time){           
        printf("Test %d:\n",counter+1);
        // Point-wise Insertion
        generate_increment_point_cloud(New_Point_Num,true);
        t1 = chrono::high_resolution_clock::now();
        ikd_Tree.Add_Points(cloud_increment,counter + 1);
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
        int valid_search_counter = 0;
        for (int k=0;k<Search_Counter;k++){
            PointVector ().swap(search_result);             
            target = generate_target_point(true);    
            t1 = chrono::high_resolution_clock::now();
            uint8_t ret = ikd_Tree.Nearest_Search(target, Nearest_Num, search_result, PointDist, -1);
            t2 = chrono::high_resolution_clock::now();
            if (search_result.size() > 0) {
                valid_search_counter ++;
                search_duration += chrono::duration_cast<chrono::microseconds>(t2-t1).count();
            }
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
            // int target_index[3], ikd_index[3], flann_index[3];
            // get_cube_index(target, target_index);
            // bool report_flag = true;
            // if (ret >0 || fabs(PointDist[4] - dists[0][4])>0.0001){
            //     wa_rec ++;
            // }
            // for (int i=0;i<5;i++){
            //     // get_cube_index(search_result[i],ikd_index);
            //     PointType tmp_point;
            //     tmp_point.x = dataset[indices[0][i]][0];
            //     tmp_point.y = dataset[indices[0][i]][1];
            //     tmp_point.z = dataset[indices[0][i]][2];
            //     // get_cube_index(tmp_point, flann_index);
            //     // if (abs(flann_index[0]-target_index[0])>1 || abs(flann_index[1]-target_index[1])>1 || abs(flann_index[2]-target_index[2])>1){
            //     //     report_flag = false;
            //     // }
            //     if (ret > 0 || (fabs(PointDist[i] - dists[0][i])>0.0001 && report_flag)){
            //         printf("\n\n\n\n ERROR!\n");
            //         printf("Number %d: ikd-Tree %0.3f FLANN %0.3f\n",i, PointDist[i], dists[0][i]);
            //         printf("Target Point (%0.4f,%0.4f,%0.4f)\n", target.x, target.y, target.z);
            //         printf("Target Index (%d,%d,%d)\n",target_index[0], target_index[1], target_index[2]);                    
            //         printf("    ikd Tree point (%0.4f,%0.4f,%0.4f)\n",search_result[i].x,search_result[i].y,search_result[i].z);
            //         printf("    ikd Tree Index (%d,%d,%d)\n",ikd_index[0], ikd_index[1], ikd_index[2]);                    
            //         printf("    flann point (%0.4f,%0.4f,%0.4f)\n",tmp_point.x, tmp_point.y, tmp_point.z);
            //         printf("    flann Index (%d,%d,%d)\n", flann_index[0],flann_index[1],flann_index[2]);                    
            //         flag = false;
            //     }
            // }                      
        }
        // if (!flag) {
        //     output_clouds();   
        //     break;
        // }
        printf("Search nearest point time cost is %0.3f us\n",float(search_duration));
        printf("Original Search nearest point time cost is %0.3f us\n",float(ori_search_duration));

        total_duration += search_duration;
        printf("Total time is %0.3f ms\n",total_duration/1e3);
        printf("Cloud size is %d \n", point_cloud.size());
        printf("Total size is %d\n",ikd_Tree.total_size);

        // printf("Tree size is: %d\n\n", ikd_Tree.size());
        // If necessary, the removed points can be collected.
        // PointVector ().swap(removed_points);
        // ikd_Tree.acquire_removed_points(removed_points);
        // Calculate total running time
        average_total_time += float(total_duration)/1e3;
        add_time += float(add_duration)/1e3;
        delete_time += float(delete_duration)/1e3;
        search_time += float(search_duration)/1e3; 
        ori_search_time += float(ori_search_duration)/1e3;
        counter += 1;
        fprintf(log_fp,"%d, %d, %0.6f,%0.6f,%d\n",counter,ikd_Tree.total_size,float(add_duration),float(search_duration),valid_search_counter);
        usleep(5000);
    }
    fclose(log_fp);
    printf("Finished %d times test\n",counter);
    printf("Average Time:\n");
    printf("Total Time is: %0.3fms\n",average_total_time/1e3);
    printf("Point-wise Insertion (%d points): %0.3fms\n",New_Point_Num,add_time/counter);        
    printf("Point-wise Delete (%d points):    %0.3fms\n", Delete_Point_Num,delete_time/counter);         
    printf("Nearest Search (%d points):       %0.3fms\n", Search_Counter,search_time/counter);
    printf("Ori Nearest Search (%d points):       %0.3fms\n", Search_Counter,ori_search_time/counter);
    printf("Accuracy: %0.3f%%\n",(1-wa_rec/float(Search_Counter * Test_Time)) * 100);           
    return 0;
}