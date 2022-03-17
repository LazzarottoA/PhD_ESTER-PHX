#!/bin/bash
#SBATCH -J phx-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --mem=185000
#SBATCH -t 24:00:00
##############################################################################
OMP_DYNAMIC=0
MPC_GANG=OFF
OMP_NUM_THREADS=1
NCPUS=36
export OMP_NUM_THREADS MPC_GANG OMP_DYNAMIC
echo node=$SLURM_JOB_NODELIST
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
echo NCPUS=$NCPUS
##############################################################################
PATH="${PATH}:/usr/bsd/:/usr/sbin/:${HOME}/bin"

k="$1"
abund=$2
type=$3
step=$4
suff_rep=$3
suff_out=$4
max_job=$(( $( nproc --all ) / 2 ))

# Noms des fichiers
###################
export NAME_STAR=model_${k}
export TYPE_SPEC=$type

echo "#################################################"
echo "Model          = ${NAME_STAR}"
echo "Type           = ${type}" # "Raies" ou "Continuum"
echo "Abund          = ${abund}" # "Solar" ou "Vega"
echo "suff (REPRIS)  = ${suff_rep}" 
echo "base (output)  = base_${type}${suff_out}.5"
echo "Step           = /tmpdir/lazzarot/plt_spec/lbd/${step}_steps" # "Spectro" ou "Photo"
echo "Max_job        = $max_job"
echo "#################################################"
declare -a pid


function wait_job
{
	echo "pid func      = ${pid[*]}"
	echo "len(pid)      = ${#pid[@]}"
	echo "max_job func  = $max_job"
	while [  ${#pid[@]} -eq $max_job  ]
        do
            	for id in ${pid[@]}
                do
                       	output=$(ps -p "$id")
                        if [ $? -ne 0 ]
                        then
				echo "#################################################\n Running jobs = ${pid[*]}\n Job done\n $( ${pid[@]/$id} )\n#################################################"
				pid=( ${pid[@]/$id} )
                        fi
                done
      	done
}
# Definition des repertoires
############################

START_DIR=/tmpdir/lazzarot
INITIALDIR=${START_DIR}/plt_spec
POUBELLEDIR=${START_DIR}/poubelle
JOB_DIR=${START_DIR}/runs_phoenix/Jobs2send2PHX/jobXc1
CVG_DIR=${START_DIR}/runs_phoenix/raies/${abund}_abund

# Definition des fichiers Phoenix
#################################

#export LINELIST=/data1/ahui/v13/data/lists/cd1_f90.bin
export LINELIST=${START_DIR}/data/lists/atomic_lines.20070531.bin
export ALGAMDAT2="${START_DIR}/data/in_sn.150400/*"
export MODELATOMS="${START_DIR}/data/PhoenixAtoms.150401/*.atom"
export FUZZLINES=${START_DIR}/data/PhoenixAtoms.150401/fuzz.150202.bin

#export PHOENIX=$START_DIR/phoenixSer
export PHOENIX=/olympe2/p20025/lazzarot/PHOENIX/phoenix

JOBID=$SLURM_JOB_ID
cd ${JOB_DIR}/${k}
cmpt=0
init=1
for T in $( ls -d T* )
do 
	export NAME=model_${k}
	export REPRIS=${CVG_DIR}/${T}${suff_rep}.20
	
	# On commence
	#############
	JOBID=`echo ${START_DIR} | sed 's/^.*\///' `

 	if [ ! -f "${START_DIR}/data/in_sn.150202/algam.txt" ]
	then
		cd ${START_DIR}/data/in_sn.150202/
		ls > algam.txt
	fi
	
        if [ ! -f "${START_DIR}/data/PhoenixAtoms.150202/atomlist.txt" ]
	then
		cd ${START_DIR}/data/PhoenixAtoms.150202/
		ls *.atom > atomlist.txt
	fi

	echo "Definition du directory poubelle/travail"
	SCRATCHDIR=$POUBELLEDIR
	cd $SCRATCHDIR
	if [ ! -d "${NAME_STAR}" ]
       	then
		mkdir ${NAME_STAR}
	fi
	cd $NAME_STAR
	if [ ! -d "${TYPE_SPEC}" ]
	then
		mkdir ${TYPE_SPEC}
	fi
	cd $TYPE_SPEC
	if [ ! -d "${TYPE_SPEC}" ]
	then
		mkdir ${TYPE_SPEC}
	fi
	if [ ! -d "${T}" ]
        then
            	mkdir ${T}
        fi
	SCRATCHDIR=$POUBELLEDIR/${NAME_STAR}/${TYPE_SPEC}
	WORKDIR=$SCRATCHDIR/${T}
	cd $WORKDIR
	cp -f $PHOENIX ./phoenix
	numjobs=$( ls ${START_DIR}/plt_spec/lbd/${step}_steps/lambda*.5 | wc -l )
        echo "Building repertories"
	for (( i=1 ; i<=$numjobs ; i++ ))
	do
		mkdir $i
		cd $i
		cp $REPRIS fort.18
		cp ${START_DIR}/data/PhoenixAtoms.150202/atomlist.txt .
		touch fort.10 fort.11
		cat ${START_DIR}/plt_spec/base_${type}${suff_out}.5 ${START_DIR}/plt_spec/MPI.5 ${JOB_DIR}/${k}/${T}/parametresFond.5 ${START_DIR}/plt_spec/abond_${abund}.5 ${START_DIR}/plt_spec/lbd/${step}_steps/lambda$i.5 ${START_DIR}/plt_spec/fin.5>fort.5 
		ln -sf $LINELIST fort.2
		touch fort.1
		ln -sf $FUZZLINES fuzz.bin
		ln -sf $ALGAMDAT2 .
		ln -sf $MODELATOMS .
		ln -sf ../phoenix .
		cd ..
	done
        echo "Repertories built"
	export MPI_GROUP_MAX=256
	export MPI_TYPE_MAX=15000
	ulimit -s unlimited 
	ulimit -a
	F90_BOUNDS_CHECK_ABORT=YES

        echo "Building fort77"

	for (( i=1 ; i<=$numjobs ; i++ ))
        do
		echo "Step ${i} launched"
                srun -pps2e -n 1 -N 1 --mem-per-cpu=4000 --chdir ./$i ./phoenix > fort.6 &
                pid+=( "$!" )
		while [  ${#pid[@]} -eq $max_job  ]
                do
                       	for id in ${pid[@]}
                       	do
                       		output=$(ps -p "$id")
                              	val_test=$?
                               	if [ $val_test -ne 0 ]
                               	then
                              		echo "Running jobs = ${pid[*]}"
                                      	echo "Job done"
                                       	pid=( ${pid[@]/$id} )
                                       	echo "Still running jobs = ${pid[*]}"
                               	fi
                       	done
                done
	done
done
wait
cd ${POUBELLEDIR}/${NAME_STAR}/${TYPE_SPEC}
for k in $( ls -d T*)
do
        if [ !  -f "${POUBELLEDIR}/${NAME_STAR}/${TYPE_SPEC}/${TYPE_SPEC}/${k}.77" ]
        then
  	    echo "pwd $PWD"
            cd ${POUBELLEDIR}/${NAME_STAR}/${TYPE_SPEC}/${k}
	    cat 1/fort.7.*>fort.7
       	    cat 1/fort.77>fort.77
       	    for (( i=2 ; i<=$numjobs ; i++ ))
            do
       		cat $i/fort.7.*>>fort.7
              	cat $i/fort.77>>fort.77
       	    done
       	    cp fort.77 ../${TYPE_SPEC}/${k}.77
            cp fort.7 ../${TYPE_SPEC}/${k}.7
       	    rm phoenix
       	    for (( i=1 ; i<=$numjobs ; i++ ))
       	    do
		cd $i
      		rm -frv !("fort.7"|"fort.77"|"fort.6.00000")
		cd .. 
            done
        else
	    echo "${T} already done"
        fi
done
echo "Star ${NAME_STAR} done"
