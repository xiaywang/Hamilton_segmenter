
#TARGETS = main.c analbeat.c bdac.c classify.c match.c noisechk.c postclas.c qrsdet.c qrsfilt.c rythmchk.c


CC=gcc
CFLAGS=-O3 -I ./inc/
LDFLAGS=-L ./src/ 

all:

	$(CC) $(CFLAGS) ./src/* -o main

run:

	./main

clean:

	rm -rf main ./inc/*.gch

#main:

#	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@

#clean:

#	rm -rf *o main
