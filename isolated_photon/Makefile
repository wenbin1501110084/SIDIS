CPP = g++
FORTRANFLAGS = -Wall -W -g -c
OPT = -c -O3 -Wall -Werror -m64 -std=c++11
OBJECTS_FTB = FTB.o

GSLLIB = -lcuba -lgsl -lgslcblas -lm -lgfortran -fopenmp 

LHAPDF6_USE=true
LHAPDF6_CONFIG=lhapdf-config
LHAPDF6_BIN=/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/bin/
LHAPDF6_INCLUDE=-I/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/include
LHAPDF6_LIB=-L/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/lib -Wl,-rpath,/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/v2_in_jet/parton_shower_in_pythia8/LHAPDF_install/lib -lLHAPDF

#SIDIS_with_Sub
SIDIS_with_Sub.exe: fDSS.o FINT.o SIDIS_with_Sub.o $(LHAPDF_install) $(OBJECTS_FTB)
	$(CPP) -o SIDIS_with_Sub.exe fDSS.o FINT.o SIDIS_with_Sub.o $(LHAPDF_install) $(LHAPDF6_LIB)  $(OBJECTS_FTB) $(GSLLIB)

exec: fDSS.o FINT.o test.o
	g++ -o exec fDSS.o FINT.o test.o $(LDLIBS)
	
FTB.o:	FBT.cpp
	$(CPP) $(OPT) $(CFLAGS) -o $@ FBT.cpp
	@echo "..................done FBT."

fDSS.o: fDSS.f
	gfortran $(FORTRANFLAGS) fDSS.f
FINT.o: FINT.f
	gfortran $(FORTRANFLAGS) FINT.f
	
SIDIS_with_Sub.o: SIDIS_with_Sub.cc
	$(CPP) $(OPT) $(LHAPDF_install) $(LHAPDF6_LIB) -o $@ SIDIS_with_Sub.cc
	@echo "..................done SIDIS_with_Sub."
	
	
	
all:
	make SIDIS_with_Sub.exe

clean:
	rm *.o
	rm *.exe
	
	
