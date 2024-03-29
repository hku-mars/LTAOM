#include "FOV_Checker/FOV_Checker.h"

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>

int main(int argc, char** argv){
    int cube_i, cube_j, cube_k, cube_index;
    Eigen::Vector3d FOV_axis( -0.706281,-0.707018,-0.0359639);
    Eigen::Vector3d FOV_pos(1.56101,-3.29185,0.103002);
    const double theta = 0.750492;
    const double cube_len = 80;
    const double FOV_RANGE = 4;
    double FOV_depth = FOV_RANGE * cube_len;
    FOV_Checker fov_checker;
    BoxPointType env_box;
    env_box.vertex_min[0] = -4800;
    env_box.vertex_min[1] = -4800;
    env_box.vertex_min[2] = -4800;
    env_box.vertex_max[0] = 4800;
    env_box.vertex_max[1] = 4800;
    env_box.vertex_max[2] = 4800;
    fov_checker.Set_Env(env_box);
    fov_checker.Set_BoxLength(cube_len);
    vector<BoxPointType> boxes;
    BoxPointType box;
    box.vertex_min[0] = 0;
    box.vertex_min[1] = -20;
    box.vertex_min[2] = -20;
    box.vertex_max[0] = -20;
    box.vertex_max[1] = 60;
    box.vertex_max[2] = 0;
    Eigen::Vector3d line_p(20,-20,0);
    Eigen::Vector3d line_vec(0,80,0);
    double t1 = 0.0085;
    // BoxPointType tmp;
    // tmp.vertex_min[0] = -40.0f; tmp.vertex_min[1] = -40.0f;     tmp.vertex_min[1] = -40.0f;
    // tmp.vertex_max[0] = 40.0f;  tmp.vertex_max[1] = 40.0f;      tmp.vertex_max[2] = 40.0f;
    // bool s1 = fov_checker.check_box(FOV_pos, FOV_axis, theta, FOV_depth, tmp);
    // printf("Check result is: %d \n", s1);
    fov_checker.check_fov(FOV_pos, FOV_axis, theta, FOV_depth, boxes);
    for (int i = 0; i< boxes.size(); i++){
        printf("Boxes: (%0.1f,%0.1f),(%0.1f,%0.1f),(%0.1f,%0.1f) -- ",boxes[i].vertex_min[0],boxes[i].vertex_max[0],boxes[i].vertex_min[1],boxes[i].vertex_max[1],boxes[i].vertex_min[2],boxes[i].vertex_max[2]);
        cube_i = floor(boxes[i].vertex_min[0] / cube_len  + 0.5 + eps_value) + 24;
        cube_j = floor(boxes[i].vertex_min[1] / cube_len  + 0.5 + eps_value) + 24;
        cube_k = floor(boxes[i].vertex_min[2] / cube_len  + 0.5 + eps_value) + 24;
        cube_index = cube_i + cube_j * 48 + cube_k * 48 * 48;
        printf("(%d,%d,%d), %d ----",cube_i,cube_j,cube_k,cube_index);  
        printf("(%d,%d,%d)\n",cube_index % 48, int((cube_index % (48*48))/48),int(cube_index / (48*48)));
    }
    return 0;
}