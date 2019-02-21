#include "kernel.h"
#include "partitioner.h"

#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include "stdlib.h"
#include "stdio.h"
// Function to generate model parameters for first order flow (xc, yc, D and R)
int gen_model_param2(int x1, int y1, int vx1, int vy1, int x2, int y2, int vx2, int vy2, float *model_param) {
    float temp;
    // xc -> model_param[0], yc -> model_param[1], D -> model_param[2], R -> model_param[3]
    temp = (float)((vx1 * (vx1 - (2 * vx2))) + (vx2 * vx2) + (vy1 * vy1) - (vy2 * ((2 * vy1) - vy2)));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[0] = (((vx1 * ((-vx2 * x1) + (vx1 * x2) - (vx2 * x2) + (vy2 * y1) - (vy2 * y2))) +
                          (vy1 * ((-vy2 * x1) + (vy1 * x2) - (vy2 * x2) - (vx2 * y1) + (vx2 * y2))) +
                          (x1 * ((vy2 * vy2) + (vx2 * vx2)))) /
                      temp);
    model_param[1] = (((vx2 * ((vy1 * x1) - (vy1 * x2) - (vx1 * y1) + (vx2 * y1) - (vx1 * y2))) +
                          (vy2 * ((-vx1 * x1) + (vx1 * x2) - (vy1 * y1) + (vy2 * y1) - (vy1 * y2))) +
                          (y2 * ((vx1 * vx1) + (vy1 * vy1)))) /
                      temp);

    temp = (float)((x1 * (x1 - (2 * x2))) + (x2 * x2) + (y1 * (y1 - (2 * y2))) + (y2 * y2));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[2] = ((((x1 - x2) * (vx1 - vx2)) + ((y1 - y2) * (vy1 - vy2))) / temp);
    model_param[3] = ((((x1 - x2) * (vy1 - vy2)) + ((y2 - y1) * (vx1 - vx2))) / temp);
    /* */
    return (1);
}




// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(Thread_params *p2) {

    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < p2[0].n_threads; k++) {
        cpu_threads.push_back(std::thread([=]() { // [=] 

            Partitioner p = partitioner_create(p2[0].n_tasks, p2[0].alpha , k, p2[0].n_threads);
            int n_thread = p2[0].n_threads;
            flowvector fv[8];
            __m128i      vx_error, vy_error;
            int vx_error2,vy_error2;
//            int vx_error,vy_error;

            // Each thread performs one iteration
            for(int iter = cpu_first(&p); cpu_more(&p); iter = cpu_next(&p)) {
                // Obtain model parameters for First Order Flow - gen_firstOrderFlow_model
                //float *model_param =
                //    &(p2[0].model_param_local)
                //        [4 * iter]; // xc=model_param[0], yc=model_param[1], D=model_param[2], R=model_param[3]

                int   outlier_local_count[4] = {0,0,0,0};
				float model_param[4][4] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


                // Select two random flow vectors
                for(int i = 0;i<4;i++)
                {
                    int rand_num = p2[0].random_numbers[(iter+i) * 2 + 0];
                    fv[2*i]        = p2[0].flowvectors[rand_num];
                    rand_num     = p2[0].random_numbers[(iter+i) * 2 + 1];
                    fv[2*i+1]        = p2[0].flowvectors[rand_num];
                }
                __m256 X,Y,Vx,Vy;
                void *void_fx,*void_fy,*void_fvx,*void_fvy;
                posix_memalign(&void_fx,32,32);
                posix_memalign(&void_fy,32,32);
                posix_memalign(&void_fvx,32,32);
                posix_memalign(&void_fvy,32,32);

                float* fx = (float*)void_fx;
                float* fy = (float*)void_fy;
                float* fvx = (float*)void_fvx;
                float* fvy = (float*)void_fvy;
                
                for(int i = 0;i<8;i++)
                {
                    fx[i] = fv[i].x;
                    fy[i] = fv[i].y;
                    fvx[i] = fv[i].vx;
                    fvy[i] = fv[i].vy;
                }
//                printf("%f\n",fx[0]);
                //Initialization   
                X = _mm256_load_ps(fx);
                Y = _mm256_load_ps(fy);
                Vx = _mm256_load_ps(fvx);
                Vy = _mm256_load_ps(fvy);
                


//                printf("%f \n",a[0]);
                __m256 vx1,vy1;
                int ret = 0;
                /*
                int vx1 = fv[0].vx - fv[0].x;
                int vy1 = fv[0].vy - fv[0].y;
                int vx2 = fv[1].vx - fv[1].x;
                int vy2 = fv[1].vy - fv[1].y;
                */
                vx1 = _mm256_sub_ps(Vx,X);
                vy1 = _mm256_sub_ps(Vy,Y);

                float*a = (float*)&X;
                __m256i X1 = _mm256_set_epi32((int)a[7],(int)a[6],(int)a[5],(int)a[4],(int)a[3],(int)a[2],(int)a[1],(int)a[0]);
                a = (float*)&Y;
                __m256i Y1 = _mm256_set_epi32((int)a[7],(int)a[6],(int)a[5],(int)a[4],(int)a[3],(int)a[2],(int)a[1],(int)a[0]);
                a = (float*)&vx1;
                __m256i Vx1 = _mm256_set_epi32((int)a[7],(int)a[6],(int)a[5],(int)a[4],(int)a[3],(int)a[2],(int)a[1],(int)a[0]);
                a = (float*)&vy1;
                __m256i Vy1 = _mm256_set_epi32((int)a[7],(int)a[6],(int)a[5],(int)a[4],(int)a[3],(int)a[2],(int)a[1],(int)a[0]);
                int* pX,*pY,*pVx,*pVy;
                pX = (int*)&X1;
                pY = (int*)&Y1;
                pVx = (int*)&Vx1;
                pVy = (int*)&Vy1;
//                printf("%d %d %d %d \n",fv[0].x,pX[0],fv[1].x,pX);


                for(int i = 0;i<4;)
                {
                    // Function to generate model parameters according to F-o-F (xc, yc, D and R)
                    ret = gen_model_param2(pX[2*i], pY[2*i], pVx[2*i], pVy[2*i],pX[2*i+1], pY[2*i+1], pVx[2*i+1], pVy[2*i+1], model_param[i]);
                    if(ret == 0)
                        model_param[i][0] = -2011;
                    if(model_param[i][0] == -2011) 
                        outlier_local_count[i] = 10000;
                    i = i+1;                   
                }
//                _mm256_zeroupper();


                // SIMD phase
                // Reset local outlier counter
                void *model_param_col1,*model_param_col2,*model_param_col3,*model_param_col4;
                posix_memalign(&model_param_col1,32,16);
                posix_memalign(&model_param_col2,32,16);
                posix_memalign(&model_param_col3,32,16);
                posix_memalign(&model_param_col4,32,16);
                
                float* m[4] = {(float*)model_param_col1,(float*)model_param_col2,(float*)model_param_col3,(float*)model_param_col4};
                for(int i = 0;i<4;i++)
                {
                    for(int j = 0;j<4;j++)
                    {
                        m[i][j] = model_param[j][i];
                    }
                }
 //               printf("%f %f\n",m[0][0],m[0][1]);
                __m128 model1,model2,model3,model0;

                model0 = _mm_set_ps(m[0][3],m[0][2],m[0][1],m[0][0]);
                model1 = _mm_set_ps(m[1][3],m[1][2],m[1][1],m[1][0]);
                model2 = _mm_set_ps(m[2][3],m[2][2],m[2][1],m[2][0]);
                model3 = _mm_set_ps(m[3][3],m[3][2],m[3][1],m[3][0]);
                float* b = (float*)&model2;
 
//                printf("%f %f %f  \n",model_param[1][2],m[2][1],b[1]);
                // Compute number of outliers

/**/
               for(int i = 0; i < p2[0].flowvector_count; i++) 
               {//5000*20 ,

                    flowvector fvreg = p2[0].flowvectors[i]; // x, y, vx, vy
                    __m128i fvregX = _mm_set1_epi32(fvreg.x);
                    __m128  fvregXf = _mm_set_ps1((float)fvreg.x);

                    __m128i fvregY = _mm_set1_epi32(fvreg.y);
                    __m128  fvregYf = _mm_set_ps1((float)fvreg.y);

                    __m128i fvregVX = _mm_set1_epi32(fvreg.vx);
                    __m128i fvregVY = _mm_set1_epi32(fvreg.vy);

                    __m128 temp1;
                    __m128 temp2;
                    __m128i temp3,temp4;
                    float* d = (float*)&temp2;


                    temp1 = _mm_sub_ps(fvregXf,model0);
                    temp2 = _mm_mul_ps(temp1,model2);
                    temp3 = _mm_set_epi32((int)d[3],(int)d[2],(int)d[1],(int)d[0]);

                    temp1 = _mm_sub_ps(fvregYf,model1);
                    temp2 = _mm_mul_ps(temp1,model3);
                    temp4 = _mm_set_epi32((int)d[3],(int)d[2],(int)d[1],(int)d[0]);

                    temp3 = _mm_add_epi32(fvregX,temp3);
                    temp3 = _mm_sub_epi32(temp3,temp4);
                    temp3 = _mm_sub_epi32(temp3,fvregVX);
                    vx_error = temp3;

                    temp1 = _mm_sub_ps(fvregYf,model1);
                    temp2 = _mm_mul_ps(temp1,model2);
                    temp3 = _mm_set_epi32((int)d[3],(int)d[2],(int)d[1],(int)d[0]);

                    temp1 = _mm_sub_ps(fvregXf,model0);
                    temp2 = _mm_mul_ps(temp1,model3);
                    temp4 = _mm_set_epi32((int)d[3],(int)d[2],(int)d[1],(int)d[0]);                   

                    temp3 = _mm_add_epi32(fvregY,temp3);
                    temp3 = _mm_add_epi32(temp3,temp4);
                    temp3 = _mm_sub_epi32(temp3,fvregVY);
                    vy_error = temp3;
                    

                    __m128i vx_error1 = _mm_abs_epi32(vx_error);
                    __m128i vy_error1 = _mm_abs_epi32(vy_error);

                    int* f = (int*)&vx_error;
                    int* h = (int*)&vy_error;
//                    printf("%d %d \n",f[1],h[1]);

/*                    int ret[4] = {0,0,0,0};
                    for(int j = 0;j<4;j++)
                    {
                        if(f[j] >= p2[0].error_threshold || h[j] >= p2[0].error_threshold) ret[j] = 1;         
                    }
*/                    


                    __m128i error_threshold_x = _mm_set1_epi32(p2[0].error_threshold);
                    __m128i error_threshold_retx = _mm_cmpgt_epi32(error_threshold_x,vx_error1);//1:vxerrori>retx 0:verrori<= retx 0 
                    __m128i error_threshold_y = _mm_set1_epi32(p2[0].error_threshold);
                    __m128i error_threshold_rety = _mm_cmpgt_epi32(error_threshold_y,vy_error1);


                    __m128i ret = _mm_add_epi32(error_threshold_retx,error_threshold_rety);
                    int* c = (int*)&ret;
//                    printf("%d %d %d %d\n",c[0],c[1],c[2],c[3]);
                   for(int i = 0;i<4;i++)
                    {
                        if(c[i] > -2) outlier_local_count[i]++;
                    }

/*                    for(int j = 0;j<4;j++)
                    {
                            vx_error2         = fvreg.x + ((int)((fvreg.x - model_param[j][0]) * model_param[j][2]) -
                                                    (int)((fvreg.y - model_param[j][1]) * model_param[j][3])) -
                                    fvreg.vx;
                            vy_error2 = fvreg.y + ((int)((fvreg.y - model_param[j][1]) * model_param[j][2]) +
                                                    (int)((fvreg.x - model_param[j][0]) * model_param[j][3])) -
                                    fvreg.vy;

                            printf("%d %d %d %d \n",f[j],vx_error2,h[j],vy_error2);

                            if((abs(vx_error2) >= p2[0].error_threshold) || (abs(vy_error2) >= p2[0].error_threshold)) {
                                outlier_local_count[j]++;  

                        }   
                        
                     }*/
//                    printf("%d %d %d %d\n",outlier_local_count[0],outlier_local_count[1],outlier_local_count[2],outlier_local_count[3]);
 

                

                    }
                
 /**/
                // Compare to threshold 
                //printf("%d\n",outlier_local_count[0]);
                for(int i = 0;i<4;i++)
                {
                    if(outlier_local_count[i] < p2[0].flowvector_count * p2[0].convergence_threshold && outlier_local_count[i] < 10000) {
                        int index                 = p2[0].g_out_id->fetch_add(1);
                        p2[0].model_candidate[index]    = iter+i*n_thread;
                        p2[0].outliers_candidate[index] = outlier_local_count[i];/**/ 
                    }                     
                }
                   
                

               
                free(fx);
                free(fy);
                free(fvx);
                free(fvy);
                free(model_param_col1);
                free(model_param_col2);
                free(model_param_col3);
                free(model_param_col4);
                

            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}