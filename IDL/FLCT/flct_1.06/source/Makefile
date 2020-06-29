# Generic instructions:  make, then make flct, then make warp,
# then as root, make libflct-install, make flctinstall, and make warpinstall.
# To uninstall, as root, make flctlib-uninstall, then make flctuninstall, then
# make warpuninstall .
# To remove executable, objects, and libflct.a from build directory, make clean.
# You can edit this file to change the choice of compiler CC, 
# install directories
# FLCT_BINDIR, FLCT_MANDIR, FLCT_INCLUDEDIR, FLCT_LIBDIR, compiler 
# options COPTS, or location of
# fftw3 library LIBFFTW3, or location of FFTW include files INCLUDEFFTW3 . 
# If no root permission, will need to change these locations. 
#CC = gcc4
#CC = gcc
CC = gcc
#CC = tcc
# linux:
#FLCT_MANDIR = /Users/rattie/Dev/FLCT_raphael_1.06/man/man1
# macosx:
FLCT_MANDIR = /Users/rattie/Dev/FLCT_raphael_1.06/man/man1
FLCT_BINDIR = /Users/rattie/Dev/FLCT_raphael_1.06/bin/
LIBFFTW3 = /Users/rattie/Dev/fftw/lib
FLCT_LIBDIR = /Users/rattie/Dev/FLCT_raphael_1.06/lib/
FLCT_INCLUDEDIR = /Users/rattie/Dev/FLCT_raphael_1.06/include/
LOPTS = -lflct -lfftw3 -lm
INCLUDEFFTW3 = /Users/rattie/Dev/fftw/include/
# next version of options for a reasonably optimized executable
#COPTS = -O3 -Wall -fomit-frame-pointer -fPIC
COPTS = -O3 -Wall -fomit-frame-pointer -fPIC
# next version of options for debugging with gdb or valgrind
#COPTS = -g
libflct.a:	flctsubs.c flctsubs.h
	$(CC) $(COPTS) -c flctsubs.c -I$(INCLUDEFFTW3) -I.
	ar -rs libflct.a flctsubs.o
flct:	flctmain.c libflct.a 
	$(CC) $(COPTS) flctmain.c -I$(INCLUDEFFTW3) -I. -L. -L$(LIBFFTW3) $(LOPTS) -o flct
warp:	warpmain.c libflct.a
	$(CC) $(COPTS) warpmain.c -I$(INCLUDEFFTW3) -I. -L$(LIBFFTW3) -L. $(LOPTS) -o warp
all:	libflct.a flct warp
flctinstall: flct
	cp ./flct $(FLCT_BINDIR)
	cp ../doc/flct.1 $(FLCT_MANDIR)
libflct-install: libflct.a flctsubs.h
	cp ./libflct.a $(FLCT_LIBDIR)
	cp ./flctsubs.h $(FLCT_INCLUDEDIR)
warpinstall: warp
	cp ./warp $(FLCT_BINDIR)
	cp ../doc/warp.1 $(FLCT_MANDIR)
all-install:	libflct.a flct warp ../doc/flct.1 ../doc/warp.1
	cp ./libflct.a $(FLCT_LIBDIR)
	cp ./flctsubs.h $(FLCT_INCLUDEDIR)
	cp ./flct $(FLCT_BINDIR)
	cp ./warp $(FLCT_BINDIR)
	cp ../doc/flct.1 $(FLCT_MANDIR)
	cp ../doc/warp.1 $(FLCT_MANDIR)

flctuninstall: 
	rm -f $(FLCT_BINDIR)/flct
	rm -f $(FLCT_MANDIR)/flct.1
warpuninstall: 
	rm -f $(FLCT_BINDIR)/warp
	rm -f $(FLCT_MANDIR)/warp.1
libflct-uninstall:
	rm -f $(FLCT_LIBDIR)/libflct.a
	rm -f $(FLCT_INCLUDEDIR)/flctsubs.h
all-uninstall:
	rm -f $(FLCT_LIBDIR)/libflct.a
	rm -f $(FLCT_INCLUDEDIR)/flctsubs.h
	rm -f $(FLCT_BINDIR)/flct
	rm -f $(FLCT_MANDIR)/flct.1
	rm -f $(FLCT_BINDIR)/warp
	rm -f $(FLCT_MANDIR)/warp.1
clean:
	rm -f *.o flct flct.exe warp warp.exe libflct.a
