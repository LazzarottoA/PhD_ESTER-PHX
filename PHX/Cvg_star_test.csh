#!/bin/bash
##############################################################################
#SBATCH -J phx-dev
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-node=36
#SBATCH --mem=160000
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
##############################################################################
##############################################################################
OMP_DYNAMIC=0
MPC_GANG=OFF
OMP_NUM_THREADS=1
NCPUS=36
export OMP_NUM_THREADS MPC_GANG OMP_DYNAMIC
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
echo NCPUS=$NCPUS
##############################################################################
#module load openmpi/gnu/3.0.2
PATH="${PATH}:/usr/bsd/:/usr/sbin/:${HOME}/bin"


declare -a pid
k="$1"
# Noms des fichiers
###################
export NAME_STAR=model_${k}
echo $'\n'
echo "++++++++++++++++++++++++++"
echo "++++++++++++++++++++++++++"
echo "Star name = ${NAME_STAR}"
echo "++++++++++++++++++++++++++"
echo "++++++++++++++++++++++++++"
echo "Abundance         = $2"
echo "Parituclar suffix = $3"
echo $'\n'
abund=$2
suff=$3


# Definition des repertoires
############################
START_DIR=/tmpdir/lazzarot
INITIALDIR=${START_DIR}
JOB_DIR=${START_DIR}/runs_phoenix/Jobs2send2PHX/jobXc1
CVG_DIR=${START_DIR}/runs_phoenix/raies/${abund}_abund


# Definition des repertoires
############################

PHOENIX_DIR=$HOME/PHOENIX
SCRATCHDIR=$INITIALDIR/runs_phoenix/$SLURM_JOBID
mkdir -p $SCRATCHDIR

# Definition des fichiers Phoenix
#################################
export LINELIST=$INITIALDIR/data/lists/atomic_lines.20070531.bin
export ALGAMDAT2="$INITIALDIR/data/in_sn.150400/*"
export MODELATOMS="$INITIALDIR/data/PhoenixAtoms.150202/*.atom"
export FUZZLINES=$INITIALDIR/data/PhoenixAtoms.150202/fuzz.150202.bin

export PHOENIX=$PHOENIX_DIR/phoenix
export MKINPUT=$PHOENIX_DIR/mkinput

JOBID=$SLURM_JOB_ID

cmpt=0
cd ${JOB_DIR}/${k}
Teff=( $(ls -d T*) )
numjobs=$( ls -ld T* | grep -c "T*" )
init=0

for T in $(ls -d T*)
do
    cd $CVG_DIR
    #if [ ! -f "${CVG_DIR}/${T}.20" ]
    #then
     #   while [  ${#pid[@]} -eq 1  ]
     #   do
     #      for id in ${pid[@]}
     #       do
     #           output=$(ps -p "$id")
     #           if [ $? -ne 0 ]
     #           then
     #               echo "Running jobs = ${pid[*]}"
     #               echo "Job done"
     #               pid=( ${pid[@]/$id} )
     #               echo "Still running jobs = ${pid[*]}"
     #           fi
     #       done
     #   done


        # Noms des fichiers
        ###################

        export NAME_FILE=${T}

        Tmod=${T%r*}
        Tmod=${Tmod%.*} 
        Tmod=${Tmod##*T}  
        mini=10000
        ever_done=0
        for candidate in $( ls -d T*)
        do
            T2test=${candidate%r*}                                    
            T2test=${T2test%.*}
            T2test=${T2test##*T}
            dif=$((Tmod - T2test))
            abs_dif=${dif#-}
 
             if [ $mini -ne $abs_dif ]
             then
                 if [ $mini -gt $abs_dif ]
                 then
                     mini=$abs_dif
                     choosen_mod=$candidate
                     verif_init=$T2test
                 fi
             fi
        done

        echo "Choice of *.20 file"
        echo "Wanted structure    = $Tmod K"
        echo "Starting structure  = $verif_init K"
        echo "Teff difference     = $mini K"
        export REPRIS=${CVG_DIR}/${choosen_mod}
        echo "----------------------"
        echo $'\n'

        # On commence
        #############
   #    if [ $(( $cmpt %2 )) -eq 0 ]
   #    then
   #	T_job=( "$T" )
   #    else
   #	T_job+=( "$T" )
   #    fi

        if [ ! -f "${INITIALDIR}/data/in_sn.150400/algam.txt" ]
        then
            cd ${INITIALDIR}/data/in_sn.150400/
            ls > algam.txt
        fi
        if [ ! -f "${INITIALDIR}/data/PhoenixAtoms.150202/atomlist.txt" ]
        then
            cd ${INITIALDIR}/data/PhoenixAtoms.150202/
            ls *.atom > atomlist.txt
        fi

        echo "Definition du directory de travail"
        cd ${SCRATCHDIR}
        mkdir ${T}
        cd ${T}


        cp ${INITIALDIR}/data/PhoenixAtoms.150202/atomlist.txt .
        cp ${REPRIS} fort.18
        touch fort.10 fort.11
        cp $MKINPUT ./mkinput
        cp ${JOB_DIR}/${k}/${T}/job_raies .
        chmod +x job_raies
        ./job_raies
        cp -f $LINELIST fort.2
        touch fort.1
        ln -sf $FUZZLINES fuzz.bin
        cp -f $ALGAMDAT2 .
        cp -f $MODELATOMS .
        cp -f $PHOENIX ./phoenix

        export MPI_GROUP_MAX=256
        export MPI_TYPE_MAX=15000
        ulimit -s unlimited
        ulimit -a
        F90_BOUNDS_CHECK_ABORT=YES
        echo "Launching $T"
        srun -n36 ./phoenix 2>&1 | tee phoenix.out
        init=$(( $init +1 ))
        pid+=( "$!" )
        sleep 5s
	cmpt=$(( $cmpt +1 ))
	cat algam.txt | xargs -inn rm nn
        cat atomlist.txt | xargs -inn rm nn
        echo "$SCRATCHDIR/../raies/${abund}_abund/$T$suff.20"
        cp fort.20 $SCRATCHDIR/../raies/${abund}_abund/$T$suff.20
        rm fuzz.bin fort.1 fort.10 fort.11 fort.2 fort.71 fort.72 AGridcom.txt 2Drift.data
        rm mkinput fuzz.bin fort.1 fort.10 fort.11 fort.2 fort.71 fort.72 AGridcom.txt 2Drift.data
        gzip -9 fort.*
    #else
#	echo "$T has been already computed"
#    fi
done
wait

echo "xxxxxxxxxxxxxxxxxxxx"
echo "Star fully converged"
echo "xxxxxxxxxxxxxxxxxxxx"
echo $'\n'

