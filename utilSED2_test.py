#!env python
# -*- coding: utf-8 -*-
"""

####communication phx



Created on Thu Apr  4 16:47:05 2019

@author: kbouchaud-alazzarotto
"""
import os
import numpy as np
from numpy import pi, cos, sin, sqrt
import itertools
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import subprocess as sbp
from scipy.special import legendre
import h5py
from ester import star2d
from multiprocessing import Pool
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import sys


class Star:
    """Class representing the visible grid of the model star"""
    # Important constants, in cgs units
    k    = 1.38064852e-16                  # Boltzmann's constant
    h    = 6.62607015e-27                  # Planck's constant
    c    = 29979245800.                    # Speed of light in vacuum
    pc   = 648000 / np.pi * 14959787070000 # One parsec in centimeters
    dist = 5.13 * pc                       # Distance to the star in centimeters
    Msun = 1.9891e33
    Rsun = 6.95508e10
    Lsun = 3.8396e33
    abs_path=sbp.getoutput('pwd')
    
    """
    Etape: 
        -'ESTER/PHX' : Compute optimal nth ESTER, write PHX scripts
        -'Spectro'   : Creat input files to build a 0.01A step SED
        -'Photo'     : Creat input files to build a 1A step SED
        -'Line'      : Creat input files to build a line profile (raie==A)
    """

    def __init__(self,M,Omega,incl, nth0, nphi0, Etape,raie=0,Xc=1,cluster="Calmip",Abund="solar",sph="False"):
        if Etape=='Line':
            self.raie=raie*1e-8 #Saisi en A
        else:
            self.raie=0
        self.M     = M
        self.Xc    = Xc
        self.nth_opti(M,Omega,Xc)
        self.Omega = Omega
        #self.nth=nth
        self.incl  = incl
        self.inclr = np.deg2rad(incl)
        #self.nth= self.choice_nth(self.inclr,self.mod, nth0)
        self.Etape = Etape
        self.sph   = sph
        self.nth   = nth0
        self.nphi0 = nphi0
        self.Abund=Abund
        self.cluster=cluster
        self.init_grid()
        self.Input_PHX=self.inputs_PHX()
        print("################################################")
        print("Récapitulatif des inputs")
        print('Model      = {}MOm{}_Xc{}'.format(self.M,self.Omega,self.Xc))
        print("Etape      = {}".format(self.Etape))
        print('nth(ESTER) = {}'.format(self.mod.nth))
        print('nth(rbld)  = {}'.format(self.nth))
        print('nphi0      = {}'.format(self.nphi0))
        print("Abond      = {}".format(self.Abund))
        print("Cluster    = {}".format(self.cluster))
        print("Spherical  = {}".format(self.sph))
        print("################################################\n")
        if self.Etape=='ESTER/PHX':
            print("=> Making PHX launcher scripts")
            for k in self.Input_PHX:
                self.Running_PHX(k)
            if self.mod.nth != len(sbp.getoutput('ls -d ./Jobs2send2PHX/{}MOm{}_Xc{}/T*'.format(self.M,self.Omega,self.Xc)).split()) and self.Omega!=0:
                print(sbp.getoutput('ls -d ./Jobs2send2PHX/{}MOm{}_Xc{}/T*'.format(self.M,self.Omega,self.Xc)).split())
                print("==> ERROR : no match between nth(ESTER) = {} and nth(Files) = {}".format(self.mod.nth,len(sbp.getoutput('ls ./Jobs2send2PHX/{}MOm{}_Xc{}/T*'.format(self.M,self.Omega,self.Xc)).split())))
            else:
                print("==> RAS")
            print("=> PHX launcher scripts ready\n")
        print("=> Determining visible surface")
        self.visgrid()
        print("=> Visible surface determined\n")
        
        self.Flux_vis=self.Total_flux()
        if self.Etape!='ESTER/PHX':
            print("=> Making PHX-ESTER scripts")
            self.Ready2rbld()
            print("=> PHX-ESTER scripts ready")
            
        self.concat_launcher()
        print('Curvature radius')
        curve,th=self.Curv_R()
        for k in range(len(curve)):
            print( 'Curv. Rad.= ',(curve[k]*self.mod.Rp) ,'theta= ', th[k] )
        print("dS0 = {}".format(self.ds0))
        print("Nbr de surface = {}".format(len(self.ds_vis)))

    
    def nth_opti(self,M,Omega,Xc):
        print('{}MOm{}_Xc{}.h5'.format(M,Omega,Xc))
        if '{}MOm{}_Xc{}.h5'.format(M,Omega,Xc) not in sbp.getoutput('ls ./Ester_models').split():
            print('{}MOm{}.h5 with Xc={}'.format(M,Omega,Xc))
            #print(sbp.getoutput('ls ./Ester_models').split())
            print('New star in preparation')
            if Omega<0.2:nth_init=3
            elif Omega>=0.2 and Omega<0.5:nth_init=5
            elif Omega>=0.5 and Omega<0.8:nth_init=7
            else :nth_init=9
            nth_test=[nth_init+i for i in range(30)]
            for k in nth_test:
                if '{}M_tmp.h5'.format(M) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp').split():
                    print('1D Model')
                    cmd=os.popen('python ester 1d -M {} -ndomains 30 -npts 30  -Xc {} -noplot -o ./Choice_nbr_PHXmod_opti_tmp/{}M_Xc{}_tmp.h5'.format(M,Xc,M,Xc))
                    time.sleep(5)
                    while '{}M_Xc{}_tmp.h5'.format(M,Xc) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp/').split():
                            time.sleep(1)
                if '{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp'):
                    print('2D Model tmp(k)')
                    if Omega>0.5:
                        cmd=os.popen('python ester 2d -M {} -Omega_bk 0.5 -ndomains 30 -npts 30 -nth {} -Xc {} -i ./Choice_nbr_PHXmod_opti_tmp/{}M_Xc{}_tmp.h5 -o ./Choice_nbr_PHXmod_opti_tmp/{}MOm0.5_Xc{}_nth_tmp{}.h5'.format(M,k,Xc,M,Xc,M,Xc,k))
                        print(cmd)
                        while '{}MOm0.5_Xc{}_nth_tmp{}.h5'.format(M,Xc,k) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp/').split():
                            time.sleep(1)
                        cmd=os.popen('python ester 2d -M {} -Omega_bk {} -ndomains 30 -npts 30 -nth {} -Xc {} -i ./Choice_nbr_PHXmod_opti_tmp/{}MOm0.5_Xc{}_nth_tmp{}.h5 -o ./Choice_nbr_PHXmod_opti_tmp/{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,k,Xc,M,Xc,k,M,Omega,Xc,k))
                        print(cmd)
                        while '{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp/').split():
                            time.sleep(1)
                        cmd=os.popen('rm ./Choice_nbr_PHXmod_opti_tmp/{}MOm0.5_Xc{}_nth_tmp{}.h5 '.format(M,Xc,k))
                        time.sleep(1) 
                    else:
                        cmd=os.popen('python ester 2d -M {} -Omega_bk {} -ndomains 30 -npts 30 -nth {} -Xc {} -i ./Choice_nbr_PHXmod_opti_tmp/{}M_Xc{}_tmp.h5 -o ./Choice_nbr_PHXmod_opti_tmp/{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,k,Xc,M,Xc,M,Omega,Xc,k))
                        print(cmd)
                        while '{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp/').split():
                            time.sleep(1)
                    print('Model {}MOm{}_Xc{}_nth_tmp{} created'.format(M,Omega,Xc,k)) 
                    
                if '{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k+1) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp'):
                    print('2D Model tmp(k+1)')
                    if Omega>0.5:
                        time.sleep(5)
                        cmd=os.popen('python ester 2d -M {} -Omega_bk 0.5 -ndomains 30 -npts 30 -nth {} -Xc {} -i ./Choice_nbr_PHXmod_opti_tmp/{}M_Xc{}_tmp.h5 -o ./Choice_nbr_PHXmod_opti_tmp/{}MOm0.5_Xc{}_nth_tmp{}.h5'.format(M,k+1,Xc,M,Xc,M,Xc,k+1))
                        print(cmd)
                        while '{}MOm0.5_Xc{}_nth_tmp{}.h5'.format(M,Xc,k+1) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp/').split():
                            time.sleep(1)
                        cmd=os.popen('python ester 2d -M {} -Omega_bk {} -ndomains 30 -npts 30 -nth {} -Xc {} -i ./Choice_nbr_PHXmod_opti_tmp/{}MOm0.5_Xc{}_nth_tmp{}.h5 -o ./Choice_nbr_PHXmod_opti_tmp/{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,k+1,Xc,M,Xc,k+1,M,Omega,Xc,k+1))
                        print(cmd)
                        while '{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k+1) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp/').split():
                            time.sleep(1)
                        cmd=os.popen('rm ./Choice_nbr_PHXmod_opti_tmp/{}MOm0.5_Xc{}_nth_tmp{}.h5 '.format(M,Xc,k+1))
#                        time.sleep(5)
                    else:
                        cmd=os.popen('python ester 2d -M {} -Omega_bk {} -ndomains 30 -npts 30 -nth {} -Xc {} -i ./Choice_nbr_PHXmod_opti_tmp/{}M_Xc{}_tmp.h5 -o ./Choice_nbr_PHXmod_opti_tmp/{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,k+1,Xc,M,Xc,M,Omega,Xc,k+1))
                        print(cmd)
                        while '{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k+1) not in sbp.getoutput('ls ./Choice_nbr_PHXmod_opti_tmp/').split():
                            time.sleep(1)
                    print('Model {}MOm{}_Xc{}_nth_tmp{} created'.format(M,Omega,Xc,k+1))
                star1 =star2d('./Choice_nbr_PHXmod_opti_tmp/{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k))
                star2 =star2d('./Choice_nbr_PHXmod_opti_tmp/{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k+1))
                th_all=list(star1.th[0])
                th_all.extend(star2.th[0])
                Teff1 =list(np.dot(star1.Teff[:, 1:-1],[star1.leg_eval_matrix(j) for j in th_all]).ravel())
                Teff2 =list(np.dot(star2.Teff[:, 1:-1],[star2.leg_eval_matrix(j) for j in th_all]).ravel())
                dT=np.abs(np.array(Teff2)-np.array(Teff1))/np.array(Teff2)*100
                print('Ecart max entre les 2 précédentes grilles {}%'.format(max(dT)))
                time.sleep(15)
                if max(dT)<=1e-2:
                    print('Optimal nth determined nth={}'.format(k+1))
                    os.rename('./Choice_nbr_PHXmod_opti_tmp/{}MOm{}_Xc{}_nth_tmp{}.h5'.format(M,Omega,Xc,k+1),'./Ester_models/{}MOm{}_Xc{}.h5'.format(M,Omega,Xc))
                    self.mod=star2
                    self.nth_phx=k+1
                    cmd=os.popen('rm ./Choice_nbr_PHXmod_opti_tmp/*MOm_nth_tmp*')
                    return
                   
        else:
            print('Optimal nth for ester model has already been determined')
            self.mod=star2d('./Ester_models/{}MOm{}_Xc{}.h5'.format(M,Omega,Xc))
            #self.mod_gauss=star2d('./Ester_models/{}MOm{}_gauss_grid.h5'.format(M,Omega))
            return
        
                  
    def init_grid(self):
        """Initiate physical and grid parameters"""
        self.dth                = pi / self.nth
        self.theta              = [self.dth/2. + j * self.dth for j in range(self.nth)]
        #self.theta              = list(np.linspace(0,np.pi,self.nth+1)[1:])
        self.Rs                 = self.mod.Rp
        self.eval_func          = [self.mod.leg_eval_matrix(j) for j in self.theta]
        self.eval_func_antisym  = [self.mod.leg_eval_matrix_antisym(j) for j in self.theta]
        self.w                  = np.dot(self.mod.w[-1, 1:-1], self.eval_func).ravel()
        self.r                  = np.dot(self.mod.r[-1, 1:-1], self.eval_func).ravel()
        self.rt                 = np.dot(self.mod.rt[-1, 1:-1], self.eval_func_antisym).ravel()
        self.Teff               = np.dot(self.mod.Teff[:, 1:-1], self.eval_func).ravel()
        self.logg               = np.array([np.log10(i) for i in np.dot(self.mod.gsup[:, 1:-1],
                                            self.eval_func)[0]]).ravel()
    
        self.ds0                = self.ds_func(0, self.nphi0)
        self.dphi, self.nphi    = self.grid_phi(nphi0=self.nphi0, nth=self.nth, ds0=self.ds0)
        
        self.phi                = [[round(self.dphi[i] / 2. + j * self.dphi[i] -np.pi,15) for j in range(int(self.nphi[i]))] for i in range(self.nth)]
        for j in range(len(self.phi)):
            for k in range(int(len(self.phi[j])/2)):
                self.phi[j][-k-1]=abs(self.phi[j][k])
        self.ds                 = [self.ds_func(i, self.nphi[i]) for i in range(self.nth)]
        self.ngrid              = sum(self.nphi)
        self.init_mu()
        self.flat()
        
        
    def ds_func(self, i, j):
        """
        Function to compute the surface element's area at given theta (theta[i]) and dphi (2*pi/j).
        """
        return self.r[i]**2 * sqrt(1 + (self.rt[i]**2/self.r[i]**2)) * np.sin(self.theta[i])* self.dth * 2 * pi / j


    def grid_phi(self, nphi0, nth, ds0):
        """
        Compute the phi step at every theta so that the surface elements' areas are as
        homogeneous as possible across the star.
        """
        dphi=[2*pi/nphi0]
        nphi = [nphi0]
        for i in range(1, nth):
            temp = float('inf')
            j = 1
            while abs(self.ds_func(i, j) - ds0) <= temp:
                temp = abs(self.ds_func(i, j) - ds0)
                index = j
                j += 1
            dphi.append(2 * pi / index)
            nphi.append(int(index))
        return np.array(dphi), np.array(nphi)
    
    def Ready2rbld(self):
        '''
        Photo/Spectro:
            Build mu_dep matrix which correspond to I(th,mu)==I(Teff,mu)
            with th and mu respectively on [-1,1] and [0,1] Gauss grid
        ESTER/PHX:
            Prepare repertories with I(th,mu) from PHX
        '''
        if 'PHX_{}M_Om{}_Xc{}'.format(self.M,self.Omega,self.Xc) not in sbp.getoutput('ls ./PHX_models/{}_step'.format(self.Etape)).split():
            sbp.Popen('mkdir -p ./PHX_models/{}_step/PHX_{}M_Om{}_Xc{}/Raies'.format(self.Etape,self.M,self.Omega,self.Xc),shell=True)
            sbp.Popen('mkdir -p ./PHX_models/{}_step/PHX_{}M_Om{}_Xc{}/Continuum'.format(self.Etape,self.M,self.Omega,self.Xc),shell=True)
            self.PHX_mod_path='./PHX_models/{}_step/PHX_{}M_Om{}_Xc{}'.format(self.Etape,self.M,self.Omega,self.Xc)
            print('==> It seems the PHOENIX is not rebirth from its ashes.....not yet!!!!')
        else:
            self.PHX_mod_path='./PHX_models/{}_step/PHX_{}M_Om{}_Xc{}'.format(self.Etape,self.M,self.Omega,self.Xc)
            self.PHX_mod_lines=sbp.getoutput('ls '+self.PHX_mod_path+'/Raies').split()
            if self.norm_conti!=False:
            self.PHX_mod_conti=sbp.getoutput('ls '+self.PHX_mod_path+'/Continuum').split()
            if "ls:" in self.PHX_mod_conti:
                self.norm_conti=False
            else:
                self.norm_conti=True
            print('==> PHX model found!')
        if self.Etape!='ESTER/PHX':
            print("==> Computing legendre polynomials")
            #print("cond begor crea mu dep :")
            #print("path = "+self.PHX_mod_path)
            #print("Test = {}".format('mu_dependancies.h5' not in sbp.getoutput('ls '+self.PHX_mod_path).split()))
            if 'mu_dependancies.h5' not in sbp.getoutput('ls '+self.PHX_mod_path).split():
                print("==> Reading PHX.77 outputs")
                self.mu_dep  =self.Crea_mu_dep()
            else:
                print("==> PHX.77 outputs already read")
                self.wavelght=h5py.File(self.PHX_mod_path+'/mu_dependancies.h5')['wavelght']
                self.wavelght_conti=h5py.File(self.PHX_mod_path+'/mu_dependancies.h5')['wavelght_conti']
            #######function envoyant les instruction à phx manquante######
            self.Gauss_weight_grid_PHX=2*np.array([0.17392742256872379,0.32607257743127321, 0.32607257743127321,0.17392742256872379]) # poid(phx)*2 car [0,1] to [-1,1]
            self.leg_poly_PHX       =np.array( [ legendre(i)(2*np.array( [6.9432e-2  ,  3.3001e-01  ,  6.6999e-01  , 9.3057e-01] )-1) for i in range(4)] )   #2*mu(phx)-1 
            self.grid_path          =self.grid_rbld()
            self.poly_path          =self.Poly_leg()
            self.Flux_vis           =self.Total_flux()
            self.V_moy              =self.mean_vproj()
            #self.Average_path       =self.Crea_average_star()
        
        

    def init_mu(self):
        """
        Compute mu, cosine of the angle between the normal to the surface and the line of sight
        (depends on inclination angle, and necessary to compute the visible grid).
        """
        self.mu       = [[(cos(self.phi[i][j])*sin(self.inclr)*(self.r[i]*sin(self.theta[i])
                          - self.rt[i]*cos(self.theta[i])) + cos(self.inclr)
                          * (self.r[i]*cos(self.theta[i]) + self.rt[i]*sin(self.theta[i])))
                          / (self.r[i]*np.sqrt(1 + (self.rt[i]**2 / self.r[i]**2)))
                          for j in range(int(self.nphi[i]))]
                         for i in range(self.nth)]
        self.mu_flat  = np.array(list(itertools.chain(*self.mu)))
        self.vis_mask = np.where(self.mu_flat >= 0.)
        self.mu_vis = self.mu_flat[self.vis_mask]

    def flat(self):
        """Flatten all arrays"""
        rds  = [list(itertools.repeat(self.r[i], self.nphi[i])) for i in range(self.nth)]               
        ds   = [list(itertools.repeat(self.ds[i], self.nphi[i])) for i in range(self.nth)]
        w    = [list(itertools.repeat(self.w[i], self.nphi[i])) for i in range(self.nth)]
        Teff = [list(itertools.repeat(self.Teff[i], self.nphi[i])) for i in range(self.nth)]
        logg = [list(itertools.repeat(self.logg[i], self.nphi[i])) for i in range(self.nth)]
        self.rds_flat  = np.array(list(itertools.chain(*rds)))
        self.ds_flat   = np.array(list(itertools.chain(*ds)))
        self.Teff_flat = np.array(list(itertools.chain(*Teff)))
        self.logg_flat = np.array(list(itertools.chain(*logg)))
        self.w_flat    = np.array(list(itertools.chain(*w)))
        return

    def visgrid(self):
        """Only keep values for the visible surface of the star"""
        self.rds_vis            = self.rds_flat[self.vis_mask]
        self.ds_vis             = self.ds_flat[self.vis_mask]
        #self.surf_weight        = self.nphi/self.nphi[len(self.nphi)//2]
        self.Teff_vis           = self.Teff_flat[self.vis_mask]
        self.logg_vis           = self.logg_flat[self.vis_mask]
        self.w_vis              = self.w_flat[self.vis_mask]
       
    def Total_flux(self):
        return(sum(self.Teff_vis**4*self.mu_vis*self.ds_vis))
        
    def Curv_R(self):
        #finest extended theta
        th=list(self.mod.th[0])[1:len(self.mod.th[0])-1]
        r=list(self.mod.r[-1])[1:len(self.mod.th[0])-1]
        rt=list(self.mod.rt[-1])[1:len(self.mod.th[0])-1]
        rtt_tmp=np.zeros((len(rt),3,3))
        dif=[-1e-5,0,1e-5]
        for k in range(len(rt)):
            for d in range(len(dif)):
                rtt_tmp[k][0][d]=th[k]*(1+dif[d])
            eval_func=[self.mod.leg_eval_matrix(j) for j in rtt_tmp[k][0]]
            rtt_tmp[k][1]=np.dot(self.mod.rt[-1, 1:-1], eval_func).ravel()
            pol=np.polyfit(rtt_tmp[k][0],rtt_tmp[k][1],2)
            rtt_tmp[k][2]=pol
        rtt=[]
        for k in range(len(rt)):
            rtt.append(np.dot(rtt_tmp[k][2],[th[k]**2,th[k],1] ))

        rc=(np.array(r)**2+np.array(rt)**2)**(3/2) /abs(np.array(r)**2+2*np.array(rt)**2-np.array(r)*np.array(rtt))
        plt.plot(th,rc,label='rc')
        plt.plot(th,r,label='r')
        plt.plot(th,rt,label='rt')
        plt.plot(th,rtt,label='rtt')
        #Curvature radius of an ellipse
        plt.plot(np.linspace(0,2*np.pi,100),a**2/b*(1-(a**2-b**2)/a**2*cos(np.linspace(0,2*np.pi,100))**2)**(3/2))
        plt.legend()
        return np.array(rc),np.array(th)
    
    def R_mean_flux(self):
        return sum((self.rds_vis*self.Teff_vis**4*self.mu_vis*self.ds_vis)/self.Flux_vis*self.Rs)
    def R_mean_geo(self):
        geo=self.mu_vis*self.ds_vis
        geo_sum=sum(geo)
        return sum(self.rds_vis*geo/geo_sum*self.Rs)

    def Teff_mean_flux(self):
        return sum((self.Teff_vis**5*self.mu_vis*self.ds_vis)/self.Flux_vis)
    def Teff_mean_geo(self):
        geo=self.mu_vis*self.ds_vis
        geo_sum=sum(geo)
        return sum(self.Teff_vis*geo/geo_sum)
    def Teff_mean_flux4(self):
        geo=self.mu_vis*self.ds_vis
        geo_sum=sum(geo)
        print (sum((self.Teff_vis**4*geo/geo_sum))**(1/4))
        return sum((self.Teff_vis**4*geo/geo_sum))**(1/4)
    

    def logg_mean(self):
        return sum([(self.logg_vis[k]*self.Teff_vis[k]**4*self.mu_vis[k]*self.ds_vis[k])/self.Flux_vis for k in range(len(self.Teff_vis))])

    def B(self, l, T):
        return 2*Star.h*Star.c**2 / (l**5*(np.exp(Star.h*Star.c/(l*Star.k*T))-1))
    def Lp_eq(self):
        l=np.linspace(19,90000,(90000-19))
        Tp=self.Teff[0]
        Teq=self.Teff[-1]
        CNp=self.B(l*1e-8,Tp)
        CNeq=self.B(l*1e-8,Teq)
        return np.pi*(self.r[0]*self.Rs)**2*sum( [ (CNp[j+1]+CNp[j])/2*(l[j+1]-l[j]) for j in range(len(l)-1)]), np.pi*(self.r[-1]*self.Rs)**2*sum( [ (CNeq[j+1]+CNeq[j])/2*(l[j+1]-l[j]) for j in range(len(l)-1)])

    def SED(self, wavelength=np.linspace(1e-5, 2e-2, 10000)):
        flux = []
        for T in self.Teff_vis:
            flux.append(self.B(wavelength, T))
        return wavelength, np.array([sum(f*self.mu_vis*self.ds_vis*self.Rs**2/Star.dist**2)
                                     for f in np.array(flux).T])
    def mean_vproj(self):
        self.Doppler()
        return sum([self.vsini_vis[k]*self.Teff_vis[k]**4*self.mu_vis[k]*self.ds_vis[k]/self.Flux_vis for k in range(len(self.vsini_vis))])
            
            

    def plot_SED(self):
        wave, sed = self.SED()
        fig, ax = plt.subplots(figsize=(19.20, 10.80))

        ax.loglog(wave*1e4, sed*1e-4)

        ax.set_title('SED')
        ax.tick_params(labelsize=16, direction='inout', which='both', length=5, width=1)
        ax.set_xlabel('$\lambda$ ($\mu$m)', fontsize=20)
        ax.set_ylabel('F$_\lambda$ (erg$\cdot$s$^{-1}\cdot$cm$^{-2}\cdot\mu$m$^{-1}$)', fontsize=20)
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10, numticks=15))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        fig.show()
        return

    def Doppler(self):
        grid=[]
        v_vis=[]
        vsini_vis=[]
        
        for i in range(len(self.theta)):
 	        for j in range(len(self.phi[i])):
                 if self.mu[i][j]>=0:
                     grid.append([self.theta[i],self.phi[i][j]])

        if self.incl!=0:
            for k in range(len(grid)):
                v_vis.append    (self.rds_vis[k]* self.w_vis[k]*(self.mod.pc/self.mod.rhoc)**(1/2)*np.sin(self.inclr)*np.sin(grid[k][0])*np.sin(grid[k][1])/self.c)
                vsini_vis.append(self.rds_vis[k]* self.w_vis[k]*(self.mod.pc/self.mod.rhoc)**(1/2)*np.sin(self.inclr))
        self.v_vis_proj=v_vis
        self.vsini_vis=vsini_vis
        return(np.array(v_vis))

    def grid_rbld(self):
                ###Control existing grid rebld       
        if 'Poly_{}M_Om{}_Xc{}'.format(self.M,self.Omega,self.Xc) not in sbp.getoutput('ls ./Poly_models').split():
            os.popen('mkdir ./Poly_models/Poly_{}M_Om{}_Xc{}'.format(self.M,self.Omega,self.Xc))
            while 'Poly_{}M_Om{}_Xc{}'.format(self.M,self.Omega,self.Xc) not in sbp.getoutput('ls ./Poly_models').split():
                time.sleep(0.5)
            print('A new star (Poly folders) is born')

        if self.Etape=='Photo':
            mon_fichier=h5py.File('./Poly_models/Poly_{}M_Om{}_Xc{}/Photo_Pol_leg_{}MOm{}i{}_Xc{}_{}{}.h5'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0),'a')
        elif self.Etape=='Spectro': 
            mon_fichier=h5py.File('./Poly_models/Poly_{}M_Om{}_Xc{}/Spectro_Pol_leg_{}MOm{}i{}_Xc{}_{}{}.h5'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0),'a')
        elif self.Etape=='Line':
            mon_fichier=h5py.File('./Poly_models/Poly_{}M_Om{}_Xc{}/Line_Pol_leg_{}MOm{}i{}_Xc{}_{}{}_lbd{}.h5'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0,self.raie),'a')
        grid=[]
        v_vis=[]
        self.len_phi_vis=[]
        self.vis_grid_mu=[]
        for i in range(len(self.theta)):
            tmp=0
            tmp_mu=[]
            for j in range(len(self.phi[i])):
                if self.mu[i][j]>=0:
                    grid.append([self.theta[i],self.phi[i][j]])
                    tmp=tmp+1
                    tmp_mu.append(self.mu[i][j])
            self.vis_grid_mu.append(tmp_mu)
            self.len_phi_vis.append(tmp)

        for k in range(len(grid)):
            if self.incl!=0 and self.Omega!=0:
                v_vis.append(self.rds_vis[k]* self.w_vis[k]*(self.mod.pc/self.mod.rhoc)**(1/2)*np.sin(self.inclr)*np.sin(grid[k][0])*np.sin(grid[k][1])/self.c)
            else:
                v_vis.append(0)
        old_th=grid[0][0]
        old_vis=v_vis[0]
    
        delta_dop=[]
        for k in range(len(grid)):
            if grid[k][0]!=old_th:
                delta_dop.append(old_vis-v_vis[k])
                old_vis=v_vis[k]
                old_th=grid[k][0]
        
        print('Extract grid info')
        sys.stdout.flush()
        Lp,Leq=self.Lp_eq()
        gdeur=['L_CNp','L_CNeq','Rp','Dist','Rmean_flux','Rmean_geo','Tp','Teq','Tmean_Flux','Tmean_geo','Tmean_Flux4',
               'gp','geq','gmean','mu_vis','ds_vis','R_vis','v_c_vis','grid_vis','T_vis','delta_dop']
#        change=[['L_CNp',Lp],['L_CNeq',Leq],['Rmean_geo',self.R_mean_geo()],['Rmean_flux',self.R_mean_flux()]]
#        for k in range(len(change)):
#            del mon_fichier[change[k][0]]
#            mon_fichier.create_dataset(change[k][0],data=[change[k][1]])
#            print('h5py change={}'.format(mon_fichier[change[k][0]]))
        for gd in gdeur:
            if gd in list(mon_fichier.keys()):
                del mon_fichier[gd]
            
            indexgd=gdeur.index(gd)
            if indexgd==0:
                datagd=Lp
            elif indexgd==1:
                datagd=Leq
            elif indexgd==2:
                datagd=[self.mod.Rp]
            elif indexgd==3:
                datagd=[Star.dist]
            elif indexgd==4:
                datagd=[self.R_mean_flux()]
            elif indexgd==5:
                datagd=[self.R_mean_geo()]
            elif indexgd==6:
                datagd=[max(self.Teff_vis)]
            elif indexgd==7:
                datagd=[min(self.Teff_vis)]
            elif indexgd==8:
                datagd=[self.Teff_mean_flux()]
            elif indexgd==9:
                datagd=[self.Teff_mean_geo()]
            elif indexgd==10:
                datagd=[self.Teff_mean_flux4()]
            elif indexgd==11:
                datagd=[max(self.logg_vis)]
            elif indexgd==12:
                datagd=[min(self.logg_vis)]
            elif indexgd==13:
                datagd=[self.logg_mean()]
            elif indexgd==14:
                datagd=self.mu_vis
            elif indexgd==15:
                datagd=self.ds_vis
            elif indexgd==16:
                datagd=self.rds_vis*self.Rs
            elif indexgd==17:
                datagd=v_vis
            elif indexgd==18:
                datagd=grid
            elif indexgd==19:
                datagd=self.Teff_vis
            elif indexgd==20:
                datagd=1/(1+np.array(v_vis))
            mon_fichier.create_dataset(gd,data=datagd)
            if gd not in list(mon_fichier.keys()):
                 print('===> ERROR : info grid missed {}'.format(gd))
                 print(list(mon_fichier.keys()))
        mon_fichier.close()                
        return './Grids_models/Grids_{}M_Om{}_Xc{}/Grid_incl={}'.format(self.M,self.Omega,self.Xc, self.incl)
        


    def PHX_read(self, phx_line):
        print('############################## Read PHX file function ############################')
        lines=phx_line.readlines()

        if self.Etape=='Photo':
            self.arrond=8
        elif self.Etape=='Spectro' or self.Etape=='Line':
            self.arrond=10
        
        add_jonction=0
        nbr_intruder=0
        add_jonction=0
        nbr_intruder=0
        original_lght=len(lines)
        while '        8\n' in lines:
            del lines[lines.index('        8\n'):lines.index('        8\n')+2]
            add_jonction=add_jonction+1
        wavelght=lines[0::2]
        raw_conv=lines[1::2]
        print("Read wavelength length = {}".format(len(wavelght)))
        intruders=[]
        for j in range(len(wavelght)):
            if j!=0:
                r8_lbd=np.round(float(wavelght[j].replace('D','E'))*1e-8,self.arrond)
                r10_lbd=np.round(float(wavelght[j].replace('D','E'))*1e-8,self.arrond+3)
                if abs(r8_lbd-r10_lbd)==0:        
                    wavelght[j]=float(wavelght[j].replace('D','E'))*1e-8
                    raw_conv[j]=list(np.float_((raw_conv[j].split())[4:]))
                else:
                    wavelght[j]=float(wavelght[j].replace('D','E'))*1e-8
                    raw_conv[j]=list(np.float_((raw_conv[j].split())[4:]))
                    intruders.append(wavelght[j])
                    nbr_intruder=nbr_intruder+1
            else:
                wavelght[j]=float(wavelght[j].replace('D','E'))*1e-8
                raw_conv[j]=list(np.float_((raw_conv[j].split())[4:]))
        for j in intruders:
            del raw_conv[wavelght.index(j)]
            del wavelght[wavelght.index(j)]
        
        print('Number of junctions = {}'.format(add_jonction))
        print('Number of intruders = {}'.format(nbr_intruder))
        print('Waited lists dimensions:')
        print('    lbd  = {}'.format((original_lght-add_jonction*2)/2-nbr_intruder))
        print('    conv = {} * {}'.format((original_lght-add_jonction*2)/2-nbr_intruder,4))
        print("List dimenssion read:")
        print('Dim lbd  = {}'.format(len(wavelght)))
        print('Dim conv = {} * {}'.format(len(raw_conv),len(raw_conv[0])))
        sys.stdout.flush()
        return raw_conv, wavelght

       


    def Crea_mu_dep(self):
        mu_dep=[]
        mu_dep_continuum=[]
        #mu_dep=np.zeros(len(self.wavelght),self.mod.nth*4)
        #mu_dep_continuum=np.zeros(len(self.wavelght),self.mod.nth*4)
        if 'mu_dependancies.h5' in sbp.getoutput('ls '+self.PHX_mod_path).split()==True:
            print('mu_dependancies already built')
            mu_dep=h5py.File(self.PHX_mod_path+'/mu_dependancies.h5')  
            return True , True
        else:
            print('Reading fort.77 files')
            Teff_th,ind_th=[],[]
            print(self.PHX_mod_lines)
            for m in range(len(self.PHX_mod_lines)):
                file_line=open(self.PHX_mod_path+'/Raies/{}'.format(self.PHX_mod_lines[m]))
                conv,wlght=self.PHX_read(file_line)
                
                if self.norm_conti!=False:
                    file_continium=open(self.PHX_mod_path+'/Continuum/{}'.format(self.PHX_mod_conti[m])) 
                    conv_continuum,wlght_conti=self.PHX_read(file_continium)
                #A l'iteration k conv=| I_lbd0(Tk,mu0)   I_lbd0(Tk,mu1) ... I_lbd0(Tk,mup)|
                #                     | I_lbd1(Tk,mu0) ...              ... I_lbd1(Tk,mup)|
                #                     |.                                                 .|
                #                     | I_lbdm(Tk,mu0) ...              ... I_lbdm(Tk,mup)|
                #                     Avec m la taille de l'echantillon en lambda
                if self.raie==0:
                    print(conv[0])
                    mu_dep.append(conv)
                    if self.norm_conti!=False:
                        mu_dep_continuum.append(conv_continuum)

                Teff_th.append(float(self.PHX_mod_lines[m][list(self.PHX_mod_lines[m]).index('T')+1:list(self.PHX_mod_lines[m]).index('r')]))
                ind_th.append((list(np.round(self.mod.Teff[:, 1:-1].ravel(),6))).index((Teff_th[-1])))

            if len(sbp.getoutput('ls '+self.PHX_mod_path+'/Raies').split())!=self.mod.nth and self.Omega==0:
                while len(mu_dep)!=self.mod.nth:
                    mu_dep.append(conv)
            if self.norm_conti!=False:
                if len(sbp.getoutput('ls '+self.PHX_mod_path+'/Continuum').split())!=self.mod.nth and self.Omega==0:
                    while len(mu_dep_continuum)!=self.mod.nth:
                        mu_dep_continuum.append(conv_continuum)
            elif len(sbp.getoutput('ls '+self.PHX_mod_path+'/Raies').split())!=self.mod.nth and self.Omega!=0:
                print("Nth ESTER ne match pas avec le nombre de modèls PHOENIX!!!")
            print("############# Control PHX models theta order ##############" )
            print('Teff read order   = {}'.format(Teff_th))
            print('Theta read order  = {}'.format([(self.mod.th[0][::-1])[k] for k in ind_th]))
            print('Correspondance    = {}'.format(self.mod.Teff[:, 1:-1].ravel()==Teff_th))    
            print('Teffs ESTER order = {}'.format(self.mod.Teff[:, 1:-1].ravel()))
            print('Theta ESTER order = {}'.format(self.mod.th[0][::-1]))
            print("##########################################################" )
            sys.stdout.flush()


            I_k_i_std=np.zeros((len(wlght),self.mod.nth*4))
            mu_dep=np.array(mu_dep)
            if self.norm_conti!=False:
                mu_dep_continuum=np.array(mu_dep_continuum)
                I_k_i_conti_std=np.zeros((len(wlght),self.mod.nth*4))
            for th in range(self.mod.nth):
                for mu in range(4):
                    I_k_i_std[:,th*4+mu]=mu_dep[th,:,mu].ravel()
                    if self.norm_conti!=False:
                        I_k_i_conti_std[:,th*4+mu]=mu_dep_continuum[th,:,mu].ravel()
            #time.sleep(10000)

            if self.Etape!='Line':
                I_k_i=I_k_i_std
                self.wavelght=wlght
                if self.norm_conti!=False:
                    I_k_i_continuum=I_k_i_conti_std
                

            elif self.Etape=='Line':               
                ''' a tester'''
                lbd_min = self.raies*(1-np.max(self.v_vis_proj)/self.c)-50e-8
                lbd_max = self.raies*(1+np.max(self.v_vis_proj)/self.c)+50e-8
                ind_cut=[int(np.max(np.where(self.wavelght<=lbd_min)[0])),int(np.min(np.where(self.wavelght>=lbd_max)[0]))]
                ind_cut_conti=[int(np.max(np.where(self.wavelght_conti<=lbd_min)[0])),int(np.min(np.where(self.wavelght_conti>=lbd_max)[0]))]

                I_k_i=I_k_i_std[ind_cut[0]:ind_cut[1]]
                I_k_i_continuum=I_k_i_conti_std[ind_cut_conti[0]:ind_cut_conti[1]]
                self.wavelght=wlght[ind_cut[0]:ind_cut[1]]
                if self.norm_conti!=False:
                    self.wavelght_conti=wlght[ind_cut_conti[0]:ind_cut_conti[1]]

            print(self.PHX_mod_path+'/mu_dependancies.h5')
            mon_fichier = h5py.File(self.PHX_mod_path+'/mu_dependancies.h5', 'a')
            time.sleep(0.1)
            mu_dep=[]
            mu_dep_continuum=[]
            print("#################### Vérif PHX outputs ####################")
            print('Dim lbd         = {}'.format(len(wlght)))
            print('Dim lbd_conti   = {}'.format(len(wlght)))
            print('Dim Theta       = {}'.format(self.mod.nth))
            print('Dim I_k_i       = {} * {}'.format(len(I_k_i),len(I_k_i[0])))
            if self.norm_conti!=False:
                print('Dim I_k_i_conti = {} * {}'.format(len(I_k_i_continuum),len(I_k_i_continuum[0])))
            print("############################################################")
            sys.stdout.flush()
            mon_fichier.create_dataset('mu_dep',data=I_k_i)
            mon_fichier.create_dataset('wavelght',data=wlght)
            if self.norm_conti!=False:
                mon_fichier.create_dataset('mu_dep_continuum',data=I_k_i_continuum)
                mon_fichier.create_dataset('wavelght_conti',data=wlght)
            mon_fichier.close()
            return self.PHX_mod_path+'/mu_dependancies.h5'
        
#    def creaY(self,w,h5file, conti):
#        Y=[]
#        for tmp in range(self.mod.nth):
#            if conti==False:
#                mu_dep=h5file['mu_dep'][tmp]
#            else:
#                mu_dep=h5file['mu_dep_continuum'][tmp]
#            for m in range(len(mu_dep[w])):
#                Y.append(mu_dep[w][m])
##        for tmp in range(len(mu_dep)):
##            for m in range(len(mu_dep[tmp][w])):
##                Y.append(mu_dep[tmp][w][m])
#        return Y


    def unpack(self,args):
        return self.Ak(*args)
    
    def Ak(self,mu,th,Pl_theta_w):
        mu_prim=2*mu-1
        ####Modif GD ####
        Pl_mu_prim=[ (2*i+1)*legendre(i)(mu_prim)/2 for i in range(4) ]
        #Pl_mu_prim[k]=Pk[mu']
        Pl=[Pl_mu_prim[k] * Pl_theta_w[k] for k in range(4)]
        #Pl[k]=[ Pk(cos(th1)) * w_1 * Pk(mu') , ...... , Pk(cos(th4)) * w_4 * Pk(mu')]
        Ak_mu_tmp=np.sum(Pl,0)
        #Ak_mu_tmp[k]=[  Pk(cos(th1)) * w_1 * Pk(mu') , ...... , Pk(cos(th4)) * w_4 * Pk(mu') ]
        Ai_T=[]
        Ai_T_tmp=[self.mod.leg_eval_matrix(th).ravel()] 
        #Pol_leg=[ np.array([ [ self.mod.leg_eval_matrix(coord[th][0]).ravel()[i]*(2*k+1)*legendre(k)(2*coord[th][1]-1)/2 * Pl_theta_w[k] for k in range(4)] for i in range(self.mod.nth)]).flatten() for th in coord.T[0]]
        for i in range(len(Ai_T_tmp[0])):
            for k in range(len(Ak_mu_tmp)):
                Ai_T.append(Ai_T_tmp[0][i]*Ak_mu_tmp[k])
            #Ai_T.append(Ai_T_tmp[0][i])
        return np.array(Ai_T)
                        
    def Poly_leg(self):
        ###Computation of polynomials 
        print('Compute polynomial coefficient')
        if self.Etape=='Photo':
            self.arrond=8
        elif self.Etape=='Spectro' or self.Etape=='Line' :
            self.arrond=10
        
        ###############################
        if self.Etape=='Photo':
            grid_rbld=h5py.File('./Poly_models/Poly_{}M_Om{}_Xc{}/Photo_Pol_leg_{}MOm{}i{}_Xc{}_{}{}.h5'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0),'r+')
        elif self.Etape=='Spectro':
            grid_rbld=h5py.File('./Poly_models/Poly_{}M_Om{}_Xc{}/Spectro_Pol_leg_{}MOm{}i{}_Xc{}_{}{}.h5'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0),'r+')
        elif self.Etape=='Line':
            grid_rbld=h5py.File('./Poly_models/Poly_{}M_Om{}_Xc{}/Line_Pol_leg_{}MOm{}i{}_Xc{}_{}{}_lbd{}.h5'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0,self.raie),'r+')

        if 'I_k_i' not in list(grid_rbld.keys()):
            Pl_theta_w=self.leg_poly_PHX*self.Gauss_weight_grid_PHX
            th=np.array(grid_rbld['grid_vis']).T[0]
            I_k_i=[]
            I_k_i_continium=[]
            dep=h5py.File(self.PHX_mod_path+'/mu_dependancies.h5','r') 
            
            ####### Attention modif GD/LD #################
            #I=np.array(dep['mu_dep']).T
            #I_conti=np.array(dep['mu_dep_continuum']).T
            #print('mu_dep[0]=len(I)={}'.format(len(I)))
            #print(self.mod.nth)
            #print(I[3::4])
            #print('mu_dep=len(I[0])={}'.format(len(I[0])))
            I=np.array(dep['mu_dep']).T
            if self.norm_conti!=False:
                I_conti=np.array(dep['mu_dep_continuum']).T
            sys.stdout.flush()
            if self.Etape=='Spectro':
                ind_cut=int(np.where(np.round(self.wavelght,self.arrond+1)==1e-4)[0])
                if self.norm_conti!=False:
                    ind_cut_conti=int(np.where(np.round(self.wavelght_conti,self.arrond+1)==1e-4)[0])
                    
            elif self.Etape=='Line':
                lbd_min = self.raies*(1-np.max(self.v_vis_proj)/self.c)-50e-8
                lbd_max = self.raies*(1+np.max(self.v_vis_proj)/self.c)+50e-8
                ind_cut=[int(np.max(np.where(self.wavelght<=lbd_min)[0])),int(np.min(np.where(self.wavelght>=lbd_max)[0]))]
                ind_cut_conti=[int(np.max(np.where(self.wavelght_conti<=lbd_min)[0])),int(np.min(np.where(self.wavelght_conti>=lbd_max)[0]))]
            for lbd in range(len(I)):
                if self.Etape=='Spectro':
                    I_k_i.append(I[lbd][0:ind_cut])
                    I_k_i_continium.append(I_conti[lbd][0:ind_cut_conti])  
                elif self.Etape=='Line':
                    I_k_i.append(I[lbd][ind_cut[0]:ind_cut[1]])
                    I_k_i_continium.append(I_conti[lbd][ind_cut_conti[0]:ind_cut_conti[1]])  
                else:
                    I_k_i.append(I[lbd])
                    if self.norm_conti!=False:
                        I_k_i_continium.append(I_conti[lbd])
            dep.close()
            Pol_leg=[]
#            muth_done=[]
#            order=np.array([0 for h in range(len(th))])
#            t0=time.clock()
#            for i in range(len(th)):
#                if [th[i],mu[i]] not in muth_done:
#                    muth_done.append([th[i],mu[i]])
#                    order[np.where((np.array(th)==muth_done[-1][0])&(np.array(mu)==muth_done[-1][1]))]=len(muth_done) -1
#            print("old exec : ",time.clock()-t0 )
#            t0=time.clock()

            muth_done=[]
            order=np.array([0 for h in range(len(th))])
            add=0
            parity="paire"
            set_th=list(dict.fromkeys(th))
            for k in range(len(set(th))):
                if self.incl!=0:
                    if sum(self.len_phi_vis[0:k])%2==0:
                        parity="paire"
                        len_phi_reduc=sum(self.len_phi_vis[0:k])//2+add
                    else:
                        if parity!="impaire":
                            add=add+1
                        parity="impaire"
                        len_phi_reduc=sum(self.len_phi_vis[0:k])//2+add
                else:
                    len_phi_reduc=k
                muth_done_red=[]
                for j in range(len(self.vis_grid_mu[k])):
                    if [set_th[k],self.vis_grid_mu[k][j]] not in muth_done_red:
                        muth_done_red.append([set_th[k],self.vis_grid_mu[k][j]])
                    order[sum(self.len_phi_vis[0:k])+j]=len_phi_reduc+self.vis_grid_mu[k].index(self.vis_grid_mu[k][j])
                muth_done.extend(muth_done_red)

            for k in range(len(np.array(muth_done).T[0])):
                Pl_mu=np.sum([(2*j+1)*legendre(j)(2*muth_done[k][1]-1)/2 * Pl_theta_w[j] for j in range(4)],0)
                Pl_theta=self.mod.leg_eval_matrix(muth_done[k][0]).ravel()
                tmp=[]
                for i in range(len(Pl_theta)):
                    for k in range(len(Pl_mu)):
                        tmp.append(Pl_theta[i]*Pl_mu[k])
                Pol_leg.append(tmp)
            Pol_leg=np.array(Pol_leg)
                
            print("Dim grid")
            print('#################################')
            print('nth(ester)={}'.format(self.mod.nth))
            print('Ngrid_all={}'.format(len(order)))
            print('Ngrid={}'.format(len(Pol_leg)))
            print('Nmuth={}'.format(len(Pol_leg[0])))
            print('Dim Pol_leg={} x {}'.format(len(Pol_leg),len(Pol_leg[0])))
            print('dim I_k_i={} x {}'.format(len(I_k_i),len(I_k_i[0])))
            print('#################################')
            sys.stdout.flush()
            #################################
    
            grid_rbld.create_dataset('I_k_i',data=I_k_i)
            if self.Etape=='Spectro':
                grid_rbld.create_dataset('I_k_i_continuum',data=I_k_i_continium)
                ind_cut_conti=int(np.where(np.round(np.array(self.wavelght_conti),self.arrond)==1e-4)[0])
                grid_rbld.create_dataset('wavelength_continuum',data=self.wavelght_conti[0:ind_cut_conti])
                ind_cut=int(np.where(np.round(np.array(self.wavelght),self.arrond)==1e-4)[0])
                grid_rbld.create_dataset('wavelength',data=self.wavelght[0:ind_cut])
            elif self.Etape=='Photo':
                grid_rbld.create_dataset('wavelength',data=self.wavelght)
                if self.norm_conti!=False:
                    grid_rbld.create_dataset('I_k_i_continuum',data=I_k_i_continium)
                    grid_rbld.create_dataset('wavelength_continuum',data=self.wavelght_conti)
            grid_rbld.create_dataset('order',data=order)
            grid_rbld.create_dataset('Pol_leg',data=Pol_leg)
            grid_rbld.create_dataset('Ngrid',data=len(Pol_leg))
            grid_rbld.close()

            if self.Etape=='Photo':
                sbp.getoutput('cp ./Poly_models/Poly_{}M_Om{}_Xc{}/Photo_Pol_leg_{}MOm{}i{}_Xc{}_{}{}.h5 ./Jobs2send2Poly_grid/.'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0))
            elif self.Etape=='Spectro':
                sbp.getoutput('cp ./Poly_models/Poly_{}M_Om{}_Xc{}/Spectro_Pol_leg_{}MOm{}i{}_Xc{}_{}{}.h5 ./Jobs2send2Poly_grid/.'.format(self.M,self.Omega,self.Xc,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0))
            elif self.Etape=='Line':
                sbp.getoutput('cp ./Poly_models/Poly_{}M_Om{}_Xc{}/Line{}_Pol_leg_{}MOm{}i{}_Xc{}_{}{}_lbd{}.h5 ./Jobs2send2Poly_grid/.'.format(self.M,self.Omega,self.Xc,self.raie*1e8,self.M,self.Omega,self.incl,self.Xc,self.nth,self.nphi0,self.raie))
            print('Pol has a new leg (M{} Om{} Xc{} nth{} nphi0{} incl={})'.format(self.M,self.Omega,self.Xc,self.nth,self.nphi0,self.incl))
        return ' ./Poly_models/Poly_{}M_Om{}_Xc{}/Pol_legendre_normalized_i={}.h5'.format(self.M,self.Omega,self.Xc,self.incl)
 
        
        
                                                #######Partie PHX######       
    def inputs_PHX(self):
        T=self.mod.Teff[-1, 1:-1]
        g=np.log10(self.mod.gsup[-1, 1:-1])
        r=self.mod.r[-1, 1:-1]
        if self.sph=="True":
            print("Spherical PHX models")
            r,th=self.Curv_R()
        else:
            print("Parralel plan PHX models")
            r=self.mod.r[-1, 1:-1]
            
        Inputs=[]
        for k in range(len(T)):
            Inputs.append([round(T[k],6) , float(("{:.6e}".format(r[k]*self.Rs))[0:("{:.6e}".format(r[k]*self.Rs)).index('e')]), round(g[k],6)])
        return Inputs
        
    def Running_PHX(self, inputs_PHX,mod='Triplets'):
        DB=sbp.getoutput('ls ./PHX_models/Shortcut.20').split()
        T_done=[0 for k in range(len(DB))]
        rg_done=[0 for k in range(len(DB))]
        for k in range(len(DB)):
            if float(DB[k][1:].split('r')[0]) not in T_done:
                T_done[k]=float(DB[k][1:].split('r')[0])
                rg_done[k]=DB[k][1:].split('r')[1]
             
        if '{}MOm{}_Xc{}'.format(self.M,self.Omega,self.Xc) not in sbp.getoutput('ls ./Jobs2send2PHX/').split():
            os.popen('mkdir ./Jobs2send2PHX/{}MOm{}_Xc{}'.format(self.M,self.Omega,self.Xc))

        ######Modify parametresFond.5########
        fichier=open('./standard_PHX_files/parametresFond.5','r')
        to_change=fichier.readlines()[1]
        fichier.close()
        self.replace('./standard_PHX_files/parametresFond.5', to_change ,'  teff = {}, r0 = {}d11, v0=00000, logg={}, pout=1d-6,\n'.format(inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
        if 'T{}r{}g{}'.format(inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]) not in sbp.getoutput('ls ./Jobs2send2PHX/{}MOm{}_Xc{}'.format(self.M,self.Omega,self.Xc)).split():
            os.popen('mkdir ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
            time.sleep(0.5)
        os.popen('cp ./standard_PHX_files/parametresFond.5 ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/.'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))

        #######Launch_T_command########
        
        os.popen('cp ./standard_PHX_files/launch_cvgT_{}.csh ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/launch_cvgT.csh'.format(self.cluster,self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
        while 'launch_cvgT.csh' not in sbp.getoutput('ls ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/.'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2])).split():
            time.sleep(0.5)
        fichier=open('./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/launch_cvgT.csh'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]),'r')
        script=fichier.readlines()
        Name=script[0]
        fichier.close()
        
        os.popen('cp ./standard_PHX_files/launch_cvgT_{}.csh ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/launch_cvgT.csh'.format(self.cluster,self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
        while 'launch_cvgT.csh' not in sbp.getoutput('ls ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/.'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2])).split():
            time.sleep(0.5)
        fichier=open('./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/launch_cvgT.csh'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]),'r')
        if self.cluster=='Titan':
            self.replace('./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/launch_cvgT.csh'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]), Name ,'cp Jobs2send2PHX/jobXc1/{}MOm{}_Xc{}/T{}r{}g{}/job_raies . && sbatch -$1  job_raies\n'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
        elif self.cluster=='Calmip':
            self.replace('./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/launch_cvgT.csh'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]), Name ,'cp Jobs2send2PHX/jobXc1/{}MOm{}_Xc{}/T{}r{}g{}/job_raies . && sbatch  job_raies\n'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
        fichier.close()

        ######Modify job_raies#####
        if self.cluster=="Calmip" and self.Abund=='Vega':
            fichier=open('./standard_PHX_files/job_raies_calmip_Vega','r')
            script=fichier.readlines()
            #Name=script[43]
            Param=script[21]
            if self.sph=="True":
                pp_sph=script[30]
            fichier.close()
            os.popen('cp ./standard_PHX_files/job_raies_calmip_Vega ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/job_raies'.format(self.M,self.Omega,self.Xc,round(inputs_PHX[0],6),round(inputs_PHX[1],6),round(inputs_PHX[2],6)))

        elif self.cluster=="Titan" and self.Abund=='Vega':
            fichier=open('./standard_PHX_files/job_raies_titan_Vega','r')
            script=fichier.readlines()
            Name=script[41]
            Param=script[123]
            if self.sph=="True":
                pp_sph=script[132]
            fichier.close()
            os.popen('cp ./standard_PHX_files/job_raies_titan_Vega ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/job_raies'.format(self.M,self.Omega,self.Xc,round(inputs_PHX[0],6),round(inputs_PHX[1],6),round(inputs_PHX[2],6)))

        elif self.cluster=="Calmip" and self.Abund=='solar':
            fichier=open('./standard_PHX_files/job_raies_calmip_solar','r')
            script=fichier.readlines()
            #Name=script[43]
            Param=script[21]
            if self.sph=="True":
                pp_sph=script[30]
            fichier.close()
            os.popen('cp ./standard_PHX_files/job_raies_calmip_solar ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/job_raies'.format(self.M,self.Omega,self.Xc,round(inputs_PHX[0],6),round(inputs_PHX[1],6),round(inputs_PHX[2],6)))

        elif self.cluster=="Titan" and self.Abund=='solar':   
            fichier=open('./standard_PHX_files/job_raies_titan_solar','r')
            script=fichier.readlines()
            Name=script[41]
            Param=script[123]
            if self.sph=="True":
                pp_sph=script[132]
            fichier.close()
            os.popen('cp ./standard_PHX_files/job_raies_titan_solar ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/job_raies'.format(self.M,self.Omega,self.Xc,round(inputs_PHX[0],6),round(inputs_PHX[1],6),round(inputs_PHX[2],6)))

        while 'job_raies' not in sbp.getoutput('ls ./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/.'.format(self.M,self.Omega,self.Xc,round(inputs_PHX[0],6),round(inputs_PHX[1],6),round(inputs_PHX[2],6))).split():
            time.sleep(1)
        if self.cluster=="Titan":
            self.replace('./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/job_raies'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]), Name , 'export NAME_FILE=T{}r{}g{}\n'.format(inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
        if self.sph=="True":
            self.replace('./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/job_raies'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]), pp_sph , '  ezl = t, model=4, dentyp = 4, irtmth=1, lwdth=1,\n')
        self.replace('./Jobs2send2PHX/{}MOm{}_Xc{}/T{}r{}g{}/job_raies'.format(self.M,self.Omega,self.Xc,inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]), Param , '  teff = {}, r0 = {}d11, v0=00000, logg={}, pout=1d-6,\n'.format(inputs_PHX[0],inputs_PHX[1],inputs_PHX[2]))
        
               
    def replace(self, file_path, pattern, subst):
        #Create temp file
        fh, abs_path = mkstemp()
        with fdopen(fh,'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    new_file.write(line.replace(pattern, subst))
        #Copy the file permissions from the old file to the new file
        copymode(file_path, abs_path)
        #Remove original file
        remove(file_path)
        #Move new file
        move(abs_path, file_path)

    def concat_launcher(self):
        print(sbp.getoutput('(cd ./Jobs2send2PHX/.; ./cat_command.csh)'))
    
 




