#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <hdf5.h>
#include <mkl.h>

int main(int argc,char *argv[])
 {
 if (argc!=4)
   {
   printf("argc=%d \n",argc);
   printf("Erreur d'arguments: start_lbd,Input file,Output file \n");
   fflush(stdout);
   }
 //Decl.&init. variables liées lectures fichier h5 et obtentions de leurs datasets
 hid_t         file,dataset, filespace, memspace;
 hsize_t       dim_gridh5[2] ,dim_lbdh5[1],dim_I[2],dim_Pol[2],dim_order[1],dimh5[1];
 herr_t        status_n,status,status_o,status_i;
 int           rank,k,Ngrid_all,M,i,Nlbd,Ngrid,Nlbd_rest,start_lbd;
 float         Om;
 int           *ord;
 double        *mu_vis,*ds_vis,*wavelgth,*grid_vis,*wave_rest,*v_c_vis;

 start_lbd=atoi(argv[1]);
 Nlbd_rest=atoi(argv[2]);

 printf("Reading h5 files");
 printf("\n");
 fflush(stdout); 

 //Lecture des polynomes de legendre et interpolation
 file      = H5Fopen(argv[3], H5F_ACC_RDONLY, H5P_DEFAULT);
 dataset   = H5Dopen (file, "I_k_i", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank      = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dim_I, NULL);
 memspace  = H5Screate_simple(2,dim_I,NULL);
 printf("Dim I_k_i 2  %d * %d \n",dim_I[0],dim_I[1]);
 fflush(stdout);
 double        *I_k_i = (double *)malloc(dim_I[0] * dim_I[1] * sizeof(double)) ;
 status_i  = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, I_k_i);

 dataset   = H5Dopen (file, "Pol_leg", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank      = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dim_Pol, NULL);
 memspace  = H5Screate_simple(2,dim_Pol,NULL);
 Ngrid     =dim_Pol[0];
 printf("Dim Pol_leg 2  %d * %d \n",dim_Pol[0],dim_Pol[1]);
 fflush(stdout);
 double        *Pol_leg= (double *)malloc(dim_Pol[0] * dim_Pol[1] * sizeof(double)) ;
 status    = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, Pol_leg);
 
 
 double        *MapI=(double *)malloc( (long)dim_Pol[0] * dim_I[1] * sizeof(double)) ;
 cblas_dgemm(CblasRowMajor, CblasNoTrans , CblasNoTrans, dim_Pol[0], dim_I[1], dim_Pol[1], 1. , Pol_leg, dim_Pol[1], I_k_i, dim_I[1], 0, MapI, dim_I[1]);


 dataset   = H5Dopen (file, "wavelength", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank      = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dim_lbdh5, NULL);
 memspace  = H5Screate_simple(1,dim_lbdh5,NULL);
 Nlbd	   = dim_lbdh5[0];
 printf("len(wavelgth) %d \n",dim_lbdh5[0]);
 fflush(stdout);
 wavelgth  = malloc(dim_lbdh5[0]*sizeof(double));
 status    = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, wavelgth);

 dataset   = H5Dopen (file, "order", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank      = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dim_order, NULL);
 memspace  = H5Screate_simple(1,dim_order,NULL);
 Ngrid_all =dim_order[0];
 ord=malloc(dim_order[0]*sizeof(int));
 status_o  = H5Dread(dataset, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, ord);

 //Lecture des fichiers contenant info grille de reconstruction
 dataset   = H5Dopen (file, "mu_vis", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank      = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dimh5, NULL);
 memspace  = H5Screate_simple(1,dimh5,NULL);
 printf("len(mu_vis) %d \n",dimh5[0]);
 fflush(stdout);
 mu_vis    =malloc(dimh5[0]*sizeof(double));
 status    = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, mu_vis);

 dataset = H5Dopen (file, "v_c_vis", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank	   = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dimh5, NULL);
 memspace  = H5Screate_simple(1,dimh5,NULL);
 v_c_vis   =malloc(dimh5[0]*sizeof(double));
 printf("len(v_c_vis) %d \n",dimh5[0]);
 fflush(stdout);
 status    = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, v_c_vis);

 dataset   = H5Dopen (file, "ds_vis", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank      = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dimh5, NULL);
 memspace  = H5Screate_simple(1,dimh5,NULL);
 ds_vis    =malloc(dimh5[0]*sizeof(double));
 printf("len(ds_vis) %d \n",dimh5[0]);
 fflush(stdout);
 status    = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, ds_vis);

 dataset   = H5Dopen (file, "grid_vis", H5P_DEFAULT);
 filespace = H5Dget_space (dataset);
 rank      = H5Sget_simple_extent_ndims (filespace);
 status_n  = H5Sget_simple_extent_dims (filespace, dim_gridh5, NULL);
 memspace  = H5Screate_simple(2,dim_gridh5,NULL);
 grid_vis  = malloc(dim_gridh5[0]*dim_gridh5[1]*sizeof(double));
 status    = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, grid_vis);

 status = H5Dclose(dataset);
 status =H5Fclose(file);


//Initialistation grille reconstruction en lambda [1000A-9000A]
 wave_rest =malloc((Nlbd_rest+1)*sizeof(double));
 for (k=0; k<=Nlbd_rest;k++)
     {
     wave_rest[k]=wavelgth[start_lbd+k];
     }
printf("Wave_rest= %f ",wave_rest);

//Decl.&init. variables liées à la préparation des vecteur d'inputs
 int           l;
 double        *F , F_emergeant[Nlbd_rest];
 F             = malloc(Nlbd_rest*sizeof(long double));
 long double   sum;
 hsize_t       dim_out[1]={Nlbd_rest};
 long int      indk,indl;
 long double   a,b;

 printf("ça va shifter Morty \n");
 printf("len(Nlbd_rest)= %d \n", Nlbd_rest);
 printf("lbd_start= %d \n", start_lbd);
 printf("wave_rest= %d ",sizeof(wave_rest)/sizeof(double));
 printf("\n");
 fflush(stdout);


 #pragma omp parallel for num_threads(18) shared(MapI,wavelgth,wave_rest,Ngrid_all,Nlbd_rest,mu_vis,ds_vis,v_c_vis,ord,F) private(indk,indl,k,a,b,l,sum) schedule(guided)
 for (l=0;l<Nlbd_rest;l++)
     {
     sum=0.0;
     printf("l = %d \n ",l);
     for (k=0;k<Ngrid_all;k++)
         {
         indk=ord[k];
         indl=(long long  int) round(( round(1.0e9/(1+v_c_vis[k])*wave_rest[l]) - round(1.0e9*wavelgth[0]) )/10);
         if (round( (wave_rest[l]*1.0e8/(1+v_c_vis[k])))/1.0e8>=wavelgth[indl])
            {
            a=(MapI[indk*Nlbd+indl+1]*mu_vis[k]*ds_vis[k]-MapI[indk*Nlbd+indl]*mu_vis[k]*ds_vis[k]) /(wavelgth[indl+1]-wavelgth[indl]);
            b=MapI[indk*Nlbd+indl+1]*mu_vis[k]*ds_vis[k]-a*wavelgth[indl+1];
            }else{
            a=(MapI[indk*Nlbd+indl]*mu_vis[k]*ds_vis[k]-MapI[indk*Nlbd+indl-1]*mu_vis[k]*ds_vis[k]) /(wavelgth[indl]-wavelgth[indl-1]);
            b=MapI[indk*Nlbd+indl]*mu_vis[k]*ds_vis[k]-a*wavelgth[indl];
            }
          sum+=a*wave_rest[l]*(1/(1+v_c_vis[k]))+b;          
          }
      F[l]=sum;
      printf("Lambda done at %d \n",(l*100.0)/Nlbd_rest);
      fflush(stdout);
      }
 free(I_k_i);
 free(Pol_leg);
 free(MapI);
 printf("Stop it morty! We have enough Shmeckles \n");
 fflush(stdout); 

 file       = H5Fcreate(argv[4], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
 memspace   = H5Screate_simple(1, dim_out, NULL);
 dataset    = H5Dcreate(file, "I_grid", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
 status     = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,F);

 memspace   = H5Screate_simple(1, dim_out, NULL);
 dataset    = H5Dcreate(file, "wlgth", H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
 status     = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,wave_rest);


 status = H5Dclose(dataset);
 status =H5Fclose(file);

 return 0;
 }
