#!/bin/bash

#!/bin/bash
#SBATCH -q 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --constraint=intel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxx@wayne.edu
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH -t 3:59:59


#for ((jj=6; jj<7; jj++))
#do
mkdir log
mkdir jobs
#mkdir Playgrondi0

for (( ii=0; ii<20; ii++ ))
do
for ((ll=1; ll<5; ll++))
do
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 11 $kk $jj $l1 $l2
done
done


for (( ii=0; ii<20; ii++ ))
do
for ((ll=0; ll<2; ll++))

do
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 12 $kk $jj $l1 $l2
done
done

for (( ii=0; ii<20; ii++ ))
do
for ((ll=0; ll<1; ll++))

do  
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 13 $kk $jj $l1 $l2
done
done


for (( ii=0; ii<10; ii++ ))
do
for ((ll=1; ll<5; ll++))

do  
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 21 $kk $jj $l1 $l2
done
done


for (( ii=0; ii<10; ii++ ))
do
for ((ll=0; ll<2; ll++))
do
  
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 22 $kk $jj $l1 $l2
done
done

for (( ii=0; ii<10; ii++ ))
do
for ((ll=0; ll<1; ll++))

do
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 23 $kk $jj $l1 $l2
done
done


for (( ii=0; ii<5; ii++ ))
do
for ((ll=1; ll<5; ll++))

do
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 31 $kk $jj $l1 $l2
done
done


for (( ii=0; ii<5; ii++ ))
do
for ((ll=0; ll<2; ll++))

do
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 32 $kk $jj $l1 $l2
done
done

for (( ii=0; ii<5; ii++ ))
do
for ((ll=0; ll<1; ll++))

do
    ((jj=$ii*1+1))
    ((kk=$ii*1))
     ((l1=$ll*5))
    ((l2=$ll*5+5))
    ./Sub_Grid_get_xg_kT.py 33 $kk $jj $l1 $l2
done
done


