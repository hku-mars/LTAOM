CXXFLAGS = -g3 -gdwarf2
CCFLAGS = -g3 -gdwarf2

ikd_Forest_Test : ikd_Forest_Test.o ikd_Forest.o 
	g++ -std=c++11 -Wall -O3 -g3 -DRelease ikd_Forest_Test.o ikd_Forest.o -o ikd_Forest_Test -pthread -I/usr/include/hdf5/serial -I/usr/include/eigen3/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib -L/usr/lib/x86_64-linux-gnu -lhdf5 -llz4  

ikd_Forest_Test.o : ikd_Forest_Test.cpp
	g++ -c  -O3 -g3 -DRelease ikd_Forest_Test.cpp 

ikd_Forest.o : ikd_Forest.cpp ikd_Forest.h
	g++ -c ikd_Forest.cpp 

clean:
	rm *.o ikd_Forest_Test