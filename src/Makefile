.SUFFIXES: .c .cpp

CC=g++-4.0
CFLAGS=-I . -I /usr/include/ -g -O2
LIBS = -L/usr/lib  

%.o:	%.cpp
	$(CC) -c -o $@ $(CFLAGS) $<

%.o:	%.c
	gcc -c -o $@ $(CFLAGS) $<

%.o:	%.f
	g77 -c -o $@ $<

target = tricrf
all: $(target)

tricrf: Main.o tricrf1.o tricrf2.o tricrf3.o crf.o maxent.o evaluator.o param.o data.o lbfgs.o utility.o
	$(CC) -o $@ Main.o tricrf1.o tricrf2.o tricrf3.o crf.o maxent.o evaluator.o param.o data.o lbfgs.o utility.o $(CFLAGS) $(LIBS)
	
clean:
	rm $(target) *.o 

