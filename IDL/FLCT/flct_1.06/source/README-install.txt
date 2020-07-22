COMPILING AND INSTALLING libflct.a, flct, and warp:

The default settings and locations in "Makefile" correspond to the Mac OS X
operating system.  It is assumed you have installed Xcode and/or have an
alternative C compiler available.

Before compiling FLCT, you'll first need to install the FFTW3 library.  Once
installed, make a note of the location of the FFTW3 library, as well as its 
include file, fftw3.h.

The default assumption for the use of fftw3 in flct is that the C99 complex
variable definitions, which can be invoked by including complex.h, is *not*
done.  This decision was made because at least in the past, not all C 
compilers supported complex.h.  However, in the case that you do want to include
complex.h, you can uncomment the line /* #DEFINE COMPLEXH 1 */in the file
flctsubs.h .  The source code contains C pre-processor directives which should
then compile all the source code with complex.h included.

By default, flct will not write out any information about the cross-correlation
function.  But you can choose to output some of this information by uncommenting
the line /* #define CCDATA 1 */.  If this is done, the flct function will
write out two files, deriv2.dat and deriv1.dat, which will contain the 3
second derivatives, and the peak value and first derivatives of the cross
correlation function all evaluated near its peak.

We have also modified the source code to include an experimental "bias
correction" option, which can be invoked by using the option -bc
when running the flct executable.  If this option is
invoked, then bias correction is turned on.  The bias correction works by
estimating the effect of multiplying an unshifted gaussian of width sigma
by the shifted image f2 when computing the sub-images S_1 and S_2 (see Fisher
and Welsch, 2008), and then attempting to correct for it.

To compile the flct and warp executables, first examine the file "Makefile"
and edit it as necessary, paying attention to the definitions of FLCT_BINDIR,
FLCT_MANDIR, FLCT_INCLUDEDIR, FLCT_LIBDIR, which is where the executables,
the man pages, the include file, and the FLCT library file, respectively,
will be installed.  Pay attention also to the C-compiler definition CC,
and the compiler options COPTS, and edit as necessary. You may also need to 
edit the location of the fftw3 library, LIBFFTW3, and the location of its 
include file, INCLUDEFFTW3.

Once these definitions are clarified and corrected (as needed), you can
compile the FLCT library, and the FLCT and WARP executables.  The first
step is to compile the FLCT static library, libflct.a.  To compile this, 
from the build (source) directory, just type "make".

Then, to compile the flct executable, type "make flct".  To compile warp,
type "make warp".  At this point, the executables are in the build 
(source) directory.

To make libflct.a, the executables flct and warp all at once, type "make all".

To install the FLCT library and include file into a "standard" location, type 
"make libflct-install" as root (or type "sudo make libflct-install").
To install the FLCT executable and man pages into their assigned locations,
type "make flctinstall" as root, or type "sudo make flctinstall".  To install
the WARP executable and man pages into their assigned locations, type
"make warpinstall" as root, or type "sudo make warpinstall".

To install the FLCT library, the flct and warp executables and the man pages 
all at once, you can simply type "make all-install" as root, 
or type "sudo make all-install"

To uninstall the flct library, type make libflct-uninstall as root, or type
"sudo make libflct-uninstall".

To uninstall the flct executable and man pages, and then the warp executable
and man page, type "sudo make flctuninstall" and "sudo make warpuninstall",
respectively.

To uninstall the flct and warp executables and man pages, as well as the
library and include files, type "make all-uninstall" as root, or type
"sudo make all-uninstall". 

To clean out the build directory of the library and object files, as well as 
the flct and warp executables, type "make clean".
