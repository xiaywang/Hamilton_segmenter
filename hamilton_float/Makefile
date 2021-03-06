# file: Makefile	G. Moody	10 April 2003
#
# make description file for compiling ourtest & bxbep using gcc
#
# Install the WFDB package (http://www.physionet.org/physiotools/wfdb.shtml)
# before attempting to compile these programs.
#
# To compile these programs, just type 'make' at a command prompt while in
# this directory.  This has been tested under GNU/Linux and MS-Windows/Cygwin,
# and it should also work under MacOS/X (Darwin) and all versions of Unix.
#
# See '00README' in this directory for further information.

# CC is the name of your ANSI/ISO C compiler (might be 'cc' on some platforms).
CFLAGS = -O3 # -std=c11
CC = gcc $(CFLAGS)

ETOBJS = ourtest.o bdac.o classify.o rythmchk.o noisechk.o match.o \
 postclas.o analbeat.o qrsfilt.o tsc_x86.o notch.o

all:		ourtest

ourtest:	$(ETOBJS) qrsdet.o
	$(CC) -o ourtest $(ETOBJS) qrsdet.o

tsc_x86.o:	tsc_x86.c tsc_x86.h
	$(CC) -c tsc_x86.c

notch.o:	notch.c notch.h
	$(CC) -c notch.c

bdac.o:		bdac.c bdac.h qrsdet.h
	$(CC) -c bdac.c

classify.o:	classify.c qrsdet.h bdac.h match.h rythmchk.h analbeat.h \
 postclas.h
	$(CC) -c classify.c

rythmchk.o:	rythmchk.c qrsdet.h
	$(CC) -c rythmchk.c

noisechk.o:	noisechk.c qrsdet.h
	$(CC) -c noisechk.c

match.o:	match.c bdac.h
	$(CC) -c match.c

postclas.o:	postclas.c bdac.h
	$(CC) -c postclas.c

analbeat.o:	analbeat.c bdac.h
	$(CC) -c analbeat.c

qrsfilt.o:	qrsfilt.c qrsdet.h
	$(CC) -c qrsfilt.c

qrsdet.o:	qrsdet.c qrsdet.h
	$(CC) -c qrsdet.c


clean:
	rm -f *.o *~ ourtest
	rm -f ./to_plot/*.csv

run:
	./ourtest
	python compare_result.py
