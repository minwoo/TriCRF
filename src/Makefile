.SUFFIXES: .c .cpp

libdir.x86_64 = /usr/lib64
libdir.i686 = /usr/lib

MACHINE := $(shell uname -m)
UNAME_S := $(shell uname -s)

CC=g++
CFLAGS=-I . -I /usr/include/ -g -O2
ifeq ($(UNAME_S),Darwin)
	LIBS = -L/usr/lib
else
	LIBS = -L$(libdir.$(MACHINE))
endif

%.o:	%.cpp
	$(CC) -c -o $@ $(CFLAGS) $<

%.o:	%.c
	gcc -c -o $@ $(CFLAGS) $<

%.o:	%.f
	g77 -c -o $@ $<

target = tricrf
all: $(target)

tricrf: Main.o TriCRF1.o TriCRF2.o TriCRF3.o CRF.o MaxEnt.o Evaluator.o Param.o Data.o LBFGS.o Utility.o
	$(CC) -o $@ Main.o TriCRF1.o TriCRF2.o TriCRF3.o CRF.o MaxEnt.o Evaluator.o Param.o Data.o LBFGS.o Utility.o $(CFLAGS) $(LIBS)
	
clean:
	rm $(target) *.o 

