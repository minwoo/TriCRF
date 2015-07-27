.SUFFIXES: .c .cpp

CC=g++
CFLAGS=-I . -I /usr/include/ -O2
LIBS = -L/usr/lib  

%.o:	%.cpp
	$(CC) -c -o $@ $(CFLAGS) $<

%.o:	%.c
	gcc -c -o $@ $(CFLAGS) $<

%.o:	%.f
	g77 -c -o $@ $<

target = TriCRF
all: $(target)

TriCRF: Main.o TriCRF1.o TriCRF2.o TriCRF3.o CRF.o MaxEnt.o Evaluator.o Param.o Data.o LBFGS.o Utility.o
	$(CC) -o $@ Main.o TriCRF1.o TriCRF2.o TriCRF3.o CRF.o MaxEnt.o Evaluator.o Param.o Data.o LBFGS.o Utility.o $(CFLAGS) $(LIBS)
	
clean:
	rm $(target) *.o 

