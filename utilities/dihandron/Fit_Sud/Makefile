CPP = g++
FORTRANFLAGS = -Wall -W -g -c
OPT = -c -O3 -Wall -Werror -m64 -std=c++11

GSLLIB = -lcuba -lgsl -lgslcblas -lm -lgfortran -fopenmp 

LHAPDF6_USE=true
LHAPDF6_CONFIG=lhapdf-config
LHAPDF6_BIN=/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/bin/
LHAPDF6_INCLUDE=-I/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/include
LHAPDF6_LIB=-L/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/lib -Wl,-rpath,/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/lib -lLHAPDF

#SIDIS_with_Sud_only_FBT_Gaussian
SIDIS_with_Sud_only_FBT_Gaussian.exe: SIDIS_with_Sud_only_FBT_Gaussian.o $(LHAPDF_install)
	$(CPP) -o SIDIS_with_Sud_only_FBT_Gaussian.exe SIDIS_with_Sud_only_FBT_Gaussian.o $(LHAPDF_install) $(LHAPDF6_LIB) $(GSLLIB)


	
SIDIS_with_Sud_only_FBT_Gaussian.o: SIDIS_with_Sud_only_FBT_Gaussian.cc
	$(CPP) $(OPT) $(LHAPDF_install) $(LHAPDF6_LIB) -o $@ SIDIS_with_Sud_only_FBT_Gaussian.cc
	@echo "..................done SIDIS_with_Sud_only_FBT_Gaussian."
	
	
	
all:
	make SIDIS_with_Sud_only_FBT_Gaussian.exe

clean:
	rm *.o
	rm *.exe
	
	
