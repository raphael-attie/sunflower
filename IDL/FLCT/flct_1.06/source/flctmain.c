/*

 FLCT: http://solarmuri.ssl.berkeley.edu/overview/publicdownloads/software.html
 Copyright (C) 2007-2018 Regents of the University of California

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/

/* 
 * January 25, 2018 - source code has been modified in two ways:  A new
 * "bias correction" algorithm has been added, which can be invoked from the
 * command-line by specifying "-bc" in addition to the other options.  For
 * the flct library functions, the flct and flct_pc functions have an additional
 * argument, biascor, an integer flag which if non-zero, will attempt to do
 * the bias correction.  Also in the source code, a C-preprocessor flag, 
 * CCDATA, can be defined.  If defined, the flct function writes out two
 * additional files, deriv2.dat and deriv1.dat which contain information about
 * the cross-correlation function for each pixel location where the velocity
 * is computed.  The default is to not define CCDATA.
 *
 * December 8, 2017 - source code has been modified to fix a couple of bugs
 * in flct_pc.  An error occurred if sigma=0 in flct_pc, that bug has now
 * hopefully been fixed.  Similarly, a bug occurs if the skip option is used
 * in flct_pc without interpolation.  Now flct_pc will exit if the skip
 * option is used without the interpolation sub-option.

 * November 28, 2017 GHF - source code has been modified to allow for the
 * definition of complex variables in C.  To turn on support for complex 
 * variables, uncomment the line */

 /* DEFINE COMPLEXH 1 */
 /*
 * in the file flctsubs.h. 
 *
 * November 19, 2017 GHF - Source code of flctmain.c modified to allow for
 * specification of latmin, latmax parameters in Plate Carree mode to be input
 * in degrees instead of radians.  To use degrees, append a "d" to the latmin,
 * latmax strings in the command-line.
 *
 * November 8, 2017 GHF - This version of flct (1.04) has the added capability
 * of taking input images in uniform, Plate Carree coordinates (data equally
 * spaced in longitude and latitude), interpolating the images to a Mercator
 * projection, performing the flct operation, then interpolating the velocities
 * back to a Plate Carree projection, while accounting for the cos(latitude)
 * modulation of the velocity amplitudes returned from the Mercator flct 
 * results.  To invoke this option from the command-line input use the
 * option -pc latmin latmax, where latmin, latmax are the minimum and maximum
 * latitudes of the image.  It is assumed latmin, latmax are in radians.
 *
 * October 13, 2017 GHF - This version of flct (1.03) has been re-written so
 * that while the overall functionality of the flct executable is essentially 
 * the same as
 * earlier versions, the structure has been made more modular.  The
 * mathematical and computational tasks of the flct and warp programs are
 * placed into an flct static library, libflct.a .  The flct and warp 
 * main programs now just perform the necessary I/O, then call higher level
 * functions in the library.  The motivation for this change in structure
 * was to enable the use of flct, shift_frac2d, and warp_frac2d in other C
 * or fortran programs, which can then link to the flct library.
 *
 * We are now using a fossil repository for the development of flct.  The
 * latest version of flct can be viewed from this URL:
 * http://cgem.ssl.berkeley.edu/cgi-bin/cgem/FLCT/index .  Click on the Files
 * link at the top of the page to see the contents of the repository.  A copy
 * of the distribution can be obtained from this webpage as well by clicking
 * on the blue hexadecimal link visible near the top of the page visible
 * when clicking the "Files" link.
 *
 * The doc folder now includes a file, libflct_usage.txt, that describes how
 * to use the high level functions from other C or Fortran programs.
 *
 * This version corrects two incorrect loop statements,
 * for initializing outa and outb and also initializing ccorconj.
 * The bug was found be Xudong Sun.  We're grateful for him for finding it.
 * We also have now commented all the places in the
 * source code for where statements have to be changed if complex.h
 * is invoked.  In the code here, complex.h is not invoked.  If you want to
 * invoke complex.h, uncomment the complex.h header in flctsubs.h and then
 * go invoke the necessary corrections to the source code in flctsubs.c .
 *
 * We have also decided to disable the -h option, since its use
 * was made obsolete by the default curve-fitting technique using Taylor
 * expansion as described in Fisher and Welsch (2008).  Tests we have done also
 * show the results are less accurate than the default method.
 * If you attempt to use 
 * the -h option, the code will print a warning message and then exit.
 * There is a new README-install.txt file in the source directory regarding
 * how to compile and install the software.  The README file in the main folder
 * has also been updated.
 * 
 * April 10, 2009
 * Version 1.01
 * Eliminated the g1tmp and g2tmp arrays and computed the g1 and g2 arrays
 * directly from f1, f2, f1bar, f2bar, and gaussian mask (gaussdata) arrays.  
 * Got a big speedup (factor of 9 on a mac using case in tests subdirectory).
 *
 * August 12, 2008
 * This version of flct will now be called 1.0 and supercedes all 
 * test_* versions.
 * July 16, 2008 - Have added the ability to interpolate points via cubic
 * convolution at points that are skipped, potentially
 * saving a great deal of computing effort.
 * Makefile now has 'make install', which will install executable and man page.
 * Found and fixed a bug in interpcc2d in which incorrect edge values were
 * assigned.
 *
 * July 10, 2008 - Included the subtraction of a local mean before each
 * subimage is multiplied by a gaussian.  This, in conjunction with filtering
 * (e.g. -k 0.25) results in much better behavior for noisy images.
 *
 * May 31, 2008 - Added ability to skip every N pixels, with p,q as x,y
 * pixel offsets, in response to suggestion by Karin Muglach.  vm mask is
 * set to zero when calculation is skipped, and vx, vy are set to missing value.
 *
 * Fixed bug in which fabs(x) was previously computed as sqrt(x^2).
 * 
 * April 20, 2007 - Version 1.0 - Supercedes version 12 aka test_12
 * Changes from Version test_12 to Version 1.0:
 * Eliminate any C++ style comments to be compatible with older C compilers
 * Change name of executable from vel_ccor to flct
 * Added a Makefile to compile the executable.
 * To compile now, just type "make".  Before doing that,
 * Make sure that LIBFFTW3 is set to the
 * directory containing libfftw3.a, and make sure INCLUDEFFTW3 is set to the
 * directory containing fftw3.h .
 *
 * Add capability of returning single vx, vy values for whole array by setting
 * sigma = 0. (for Deborah Haber's application)
 *
 *
 * May 12, 2006 - Version 12 
 * Changes from version 10 to version 12:
 * We have changed the algorithm for locating the peak of the cc fn to
 * sub-pixel resolution.  In version 10 and before, the maximum of the 
 * function, interpolated to .1 or .02 pixel (hires mode) was returned.
 * This resulted in a "quantization" effect for velocities computed from
 * very small shifts, corresponding to small time steps.  Now, we have
 * adopted the concept of Taylor-expanding the function in the neighborhood
 * of the peak to find the sub-pixel shift values.  The idea for doing this
 * came from examining the "curve fitting" part of Chae's LCT code, but we
 * have implemented this in a somewhat different way that avoids the need
 * to do any explicit linear algebra solves.  
 *
 * The default mode of operation of the code simply uses the un-interpolated
 * cross-correlation function, in the neighborhood of the peak, to estimate
 * the sub-pixel shift.  The "hires" (-h) flag, if present, turns on cubic
 * convolution interpolation at the 0.1 pixel resolution level, and then
 * uses the Taylor-expansion on the interpolated data to find the peak.  This
 * has the result of a smoother derived velocity field, at the possible expense
 * of some numerical accuracy.
 *
 * May 10, 2005
 * Changes from version 8 to version 10:  
 * The binary I/O is now all done in large endian format.  If the code
 * is run on a small endian machine (e.g. linux or windows), 
 * the data is byteswapped before it
 * is read in or written out.  This eliminates the problem of incompatible
 * binary data format between e.g. SUNs or Macs and linux/windows machines.  
 * This should all be transparent to the user, as long as the new companion 
 * IDL i/o routines
 * are used to prepare the input image arrays and to read the output from
 * vel_ccor.  This version also reads/writes a "signature" integer at the front
 * of the file, so that if an input file that is not meant for vel_ccor is
 * mistakenly entered on the input line, it exits gracefully. *NOTE THIS MEANS
 * THE I/O FILES ARE NOT INTERCHANGEABLE BETWEEN VERSIONS test_8 AND test_10*.
 * 
 * The new IDL companion procedures for reading/writing data for vel_ccor
 * are now called vcimage1in.pro, vcimage2in.pro, vcimage3in.pro (input
 * procedures for 1, 2, or 3 2-d arrays), and vcimage1out.pro, vcimage2out.pro, 
 * and vcimage3out.pro (output procedures for 1, 2, or 3 2-d arrays).
 * The syntax for these procedures is the same as for the old versions
 * described below, except now there is no /bs keyword (no need for it).
 * 
 * The threshold value can now be forced into "absolute" mode by appending
 * an 'a' to the numerical threshold value in the calling argument.
 *
 * The calling arguments for vel_ccor ARE changed from version 8.  The 1st
 * five required arguments are the same.  The optional arguments may now
 * appear in any order, and the syntax is -q (for quiet mode), -h (for
 * hires (high resolution) mode, and -t thresh to specify the threshold value
 * for computing the velocity.  The full syntax is now:
 *
 * vel_ccor infile outfile deltat deltas sigma [-q -h -t thresh]
 * 
 * Running vel_ccor with no arguments now results in some terse documentation,
 * in addition to the expected syntax.  
 *
 * Installation instructions identical to those described below.
 *
 * April 21, 2005 - notes on version test_8:
 *
 * This version of vel_ccor (in C) that seems to have
 * most of the functionality and speed we need.  Even though much improved
 * from earlier versions, I am sure much more could be done to speed this up.
 * This version also introduces the use of a "threshold" below which the LCT
 * velocity is not computed.
 *
 * Almost everywhere, the 2-d arrays are stored and accessed as 1-d arrays
 * using pointer arithmetic to convert i,j indices to a single index.  In
 * general the single index has the form ptr+i*ny+j, where i,j are the x,y
 * indices and ptr is the pointer to the beginning of the array.  Here ny
 * is the "y" limit (the 2nd limit) to the array.  Note that in C, arrays
 * are stored backwards (transposed) from the order in IDL and fortran, so that
 * while the images are being processed, the roles of i,j are reversed from 
 * what we'd think
 * of in IDL.  But when the output velocity arrays are read back into IDL,
 * everything comes back in the expected order.
 *
 * To install from source:  First install fftw version 3 (you can get it from
 * www.fftw.org) by downloading the tarball, unpacking it, and compiling the
 * source code.  If you don't need special compilers or other special options,
 * it should build with just 
 *
 * ./configure
 * make 
 * make install (as root).
 *
 * Then compile this (the vel_ccor) source code.  For me, this is just the
 * single line
 * gcc -O3 <name of this file> -lfftw3 -lm -o vel_ccor
 *
 * Depending on how fftw3 was installed, you may need to add an option 
 * -I <directory containing fftw3.h> and/or
 * -L <directory containing libfftw3.a>
 * depending on whether gcc finds these things on its own or not.  You may or
 * may not need to include the -lm option (but I do).
 * 
 * To build a statically compiled version, the compilation command is
 * gcc -static -O3 <name of this file> -lfftw3 -lm -o vel_ccor
 *
 * FFTW3 and this version of vel_ccor build in linux, windows xp, 
 * and solaris.  In windows I used
 * the mingw/msys environment.  The msys version needs to be at least as recent
 * as 1.10.  The procedure to compile vel_ccor in windows and solaris is
 * identical to the procedure for linux described above.
 *
 * vel_ccor currently runs only from the command line.
 *
 * The command line syntax for running vel_ccor is:
 *
 * vel_ccor infile outfile deltat deltas sigma thresh hires quiet
 *
 * infile and outfile are the input and output files; infile will contain
 * 2 images from which LCT flows are determined; outfile will contain the
 * arrays vx, vy, and a velocity mask vm derived from the 2 images.  
 * deltat is the time between
 * the 2 images, deltas is the unit of pixel spacing (ie length), and 
 * sigma is the width * (in pixels) of the gaussian mask used to 
 * modulate the 2 images.  
 *
 * The parameters thresh, hires and quiet are optional, and default values are 
 * to turn these options off (=0).  If thresh is non-zero, then if the
 * average pixel value between the 2 images is less than thresh, no velocity
 * is computed, and the velocity mask value for that pixel
 * is set to 0.  If the value of
 * thresh is between 0 and 1, thresh is assumed to be given in fractional units
 * of the maximum absolute value of the 2 images.  If thresh > 1, then thresh
 * is assumed to be given in absolute units.  If thresh is set to the
 * noise level of the images, significant increases in run time can be
 * achieved, if one is willing to settle for having no computed velocities at
 * those points.
 *
 * If hires is on (!=0) then .02 pixel precision is used
 * in determining flow velocities, otherwise 0.1 pixel precision is assumed.
 * If quiet keyword (!=0) all printing to the terminal or console is suppressed.
 * Turning hires on can substantially increase run time.
 *
 * To write the input file for vel_ccor containing the pair of images, 
 * use the 'images2out' procedure in IDL.
 * To read the output file from vel_ccor containing the vx, vy arrays into IDL, 
 * use the 'images2in' procedure in IDL.  If you also want to read the velocity
 * mask array vm, use the images3in procedure in IDL.
 *
 * Authors:
 * George H. Fisher, Space Sciences Lab # 7450, University of California,
 * 7 Gauss Way, Berkeley CA 94720-7450 email: fisher at ssl dot berkeley dot edu
 *
 * Brian T. Welsch, Space Sciences Lab # 7450, University of California,
 * 7 Gauss Way, Berkeley CA 94720-7450 email: fisher at ssl dot berkeley dot edu
 *
*/

# include <flctsubs.h>

/* function prototypes: */

int main (int argc, char *argv[]);

int main (int argc, char *argv[])
{

/* BEGIN MAIN PROGRAM */

  char *version ="1.06    ";
  char infile[100], outfile[100], deltats[100], deltass[100], sigmas[100],
    threshs[100], ks[100],skips[100],latmins[100],latmaxs[100];
  char *aloc = NULL;
  char *ploc = NULL;
  char *qloc = NULL;
  char *dloc = NULL;
  char *intloc = NULL;
  i4 iarg, quiet, verbose, hires, expand, filter, absflag;
  i4 nx, nxorig, ny, nyorig ;
  i4 ier, ierflct, ibe;
  i4 poffset=0, qoffset=0, skip=0, skipon=0, degree=0;
  i4 interpolate=0;
  i4 platecarree=0;
  i4 biascor=0;
  double *f1 = NULL, *f2 = NULL, *vx = NULL, *vy = NULL, *vm = NULL;
  double deltat, deltas, sigma, thresh, latmin, latmax, kr=0. ;
  /* char krtest; */
  double pi=3.1415926535897932;
/*      double tol=1e-4; */
  i4 transp = 1; /* This flag is nonzero to transpose input/output arrays */

  /* CODE TO READ IN ARGUMENTS AND OPTIONS: */

  /* check to see if number args is in right range - 
   * if not, print syntax & quit */

  if ((argc < 6) || (argc > 18))
    {
      printf("flct: Version %s Copyright: 2007-2018 University of California\n",
          version);
      printf("Authors: G.H. Fisher, B.T. Welsch, UCB Space Sciences Lab\n\n");
      printf
        ("Syntax: %s ifile ofile deltat deltas sigma -t thr -k kr -s N[pP][qQ][i] -pc latmin latmax -bc -h -q\n\n"
            ,argv[0]);
      printf("ifile - contains 2 images for local correlation tracking\n");
      printf("   (to create ifile use the IDL procedure vcimage2out.pro)\n");
      printf("ofile - contains output vx, vy, vm (x,y velocity and mask\n");
      printf("   arrays).  Mask array 0 where velocity not computed.\n");
      printf("   To read ofile use the IDL procedure vcimage3in.pro)\n");
      printf("deltat - amount of time between images\n");
      printf("deltas - units of length of the side of a single pixel\n");
      printf("   (note velocity is computed in units of deltas/deltat)\n");
      printf("sigma - images modulated by gaussian of width sigma pixels\n");
      printf("(set sigma=0. to just get the overall shift between 2 images)\n");
      printf("Optional parameters:\n");
      printf("-t thr - if avg abs value image pixel < thr, skip calc\n");
      printf("   (if thr between 0 & 1, thr is assumed to be in\n");
      printf("   relative units of the max. abs. pixel value.  To force\n");
      printf("   thr to be in absolute units, append an 'a' to it.)\n");
      printf("-s NpPqQ - skip calc except every N pixels in x and y\n");
      printf("   P is # of offset pixels in x, Q is pixel offset in y\n");
      printf("   e.g. -s 10p5q5 means every 10 pixels, with 5 pixel offsets\n");
      printf("   appending 'i' to string will interpolate at skipped points\n");
      printf("   e.g. -s 5i means skip every 5 points, then interpolate\n");
      printf("-k kr - filter subimages w/gaussian w/ roll-off wavenumber kr\n");
      printf("    - kr is in units of max of kx, ky (typically 0 < kr < 1) \n");
      printf("-pc latmin latmax - assume images are in Plate Carree format\n");
      printf("    with latitude limits latmin, latmax (radians)\n");
      printf("-bc - 'bias correction' is turned on if this is included\n");
      printf("-h  - 'hires' flag - DISABLED\n");
      printf("-q  - flag to suppress printing all non-error messages\n");
      exit (1);
    }

  /* GET THE 5 REQUIRED ARGUMENTS */

  /* get input file name */

  strncpy (infile, argv[1], 99);

  /* get output file name */

  strncpy (outfile, argv[2], 99);

  /* get deltat */

  strncpy (deltats, argv[3], 99);
  deltat = atof (deltats);
/*  deltinv = 1. / deltat; now done inside flct function */

  /* get deltas */

  strncpy (deltass, argv[4], 99);
  deltas = atof (deltass);

  /* get sigma */

  strncpy (sigmas, argv[5], 99);
  sigma = atof (sigmas);

/*  Move the following sigma code fragment to the flct function: */

/*
  if(sigma > 0.) 
  {
     sigminv = 1. / sigma;
     sigmaeq0=0;
  }
  else
  {
     sigmaeq0 = 1;
  }
*/

  /*  GET OPTIONAL ARGUMENTS quiet, hires, expand, thresh, kr, skip : */

  hires = -1;
  quiet = 0;
  expand = 1;
  verbose = 1;
  filter = 0;
  thresh = (double) 0.;
  absflag=0;
  platecarree=0;
  biascor=0;

  if(argc >= 7)
  {
    for (iarg=6; iarg < argc; iarg++)
    {
       if(!strncmp("-h",argv[iarg],2))
       {
          hires=0;
       }
       if(!strncmp("-q",argv[iarg],2))
       {
          quiet=1;
       }
       if(!strncmp("-bc",argv[iarg],3))
       {
          biascor=1;
       }
       if(!strncmp("-t",argv[iarg],2) && ((iarg+1) < argc))
       {
          /* read in threshhold value */

          strncpy (threshs, argv[iarg+1], 99);

          /* Now check to see if there's an 'a' at end of threshold value
           * string.  If so, threshold value is treated as absolute
           * instead of relative even if it's between 0 and 1. */

          aloc = strchr (threshs, 'a');
          /* if threshold string ends in 'a', aloc will not be NULL.
           * Now, remove the 'a' from the string before doing atof(). */
          if (aloc != NULL) 
          {
             *aloc = '\0';
             absflag=1;
          }
          thresh = (double) atof (threshs);
       }
       if(!strncmp("-s",argv[iarg],2) && ((iarg+1) < argc))
       {
          /* read in skip information */

          skip=0;
          poffset=0;
          qoffset=0;
          skipon=0;
          interpolate=0;
          strncpy (skips, argv[iarg+1], 99);

          /* Check to see if there's an 'i' in the skip string.
           * If so, set interpolate flag to 1. */
          intloc = strchr(skips, 'i');
          
          /* if skip string has an 'i', intloc will not be NULL */

          if(intloc != NULL)
          {
             interpolate=1;
             *intloc='\0';
          }

          /* Now check to see if there's a 'q' in the skip 
           * string.  If so, find qoffset (stuff after 'q') */

          qloc = strchr (skips, 'q');

          /* if skip string has a 'q', qloc will not be NULL. */

          if (qloc != NULL) 
          {
            qoffset = (i4) atoi (qloc+1);
            *qloc='\0';
          }

          /* Now check to see if there's a 'p' in the skip 
           * string.  If so, find poffset (stuff after 'p') */

          ploc = strchr (skips, 'p');

          /* if skip string has a 'p', ploc will not be NULL. */
          if (ploc != NULL) 
          {
            poffset = (i4) atoi (ploc+1);
            *ploc='\0';
          }

          /* Finally extract skip integer */

          skip= (i4) atoi (skips);
 
          /* Now check that values of poffset,qoffset, and skip make sense */
 
          if( skip <= 0 )
          {
             printf("flct: skip =%d is zero or negative, fatal\n",
               skip);
             exit(1);
          }

          if((abs(poffset) >= skip) || (abs(qoffset) >= skip))

          {
             printf("flct: p=%d,q=%d, abs(p) or abs(q) >= skip=%d\n",
                   poffset,qoffset,skip);
             exit(1);
          }
       }

       skipon=skip+abs(qoffset)+abs(poffset);

       if(!strncmp("-k",argv[iarg],2) && ((iarg+1) < argc))
       {
          /* read in filter roll-off value, relative to max. kx, ky */

          strncpy (ks, argv[iarg+1], 99);
          /*  following test commented out, replaced by test below
          krtest=(int) isnumber(ks[0]);
          if(!krtest)
          {
            printf("flctmain: Illegal value of kr.  Must be a number\n");
            exit(0);
          }
          */
          kr = (double) atof (ks);
          if((kr <=0.) || (kr > 20.))
          {
            printf("flctmain: Nonsense value of kr, = %g, exiting\n",kr);
            exit(0);
          }
          filter=1;
       }
       
       if(!strncmp("-pc",argv[iarg],3) && ((iarg+2) < argc))
       {
          degree=0;
          strncpy(latmins,argv[iarg+1], 99);
          dloc = strchr (latmins, 'd');
          /* if string ends in 'd', assume in degrees and convert to radians */
          if(dloc != NULL)
          {
            degree=1;
            *dloc='\0';
          }
          latmin = (double) atof (latmins);
          if(degree)
          {
            latmin*= (pi/180.);
            degree=0;
          }
          strncpy(latmaxs,argv[iarg+2], 99);
          dloc = strchr (latmaxs, 'd');
          if(dloc != NULL)
          /* if string ends in 'd', assume in degrees and convert to radians */
          {
             degree=1;
             *dloc='\0';
          }
          latmax = (double) atof (latmaxs);
          if(degree)
          {
             latmax*= (pi/180.);
             degree=0;
          }
          platecarree=1;
       }

     }
  }

  if (quiet) verbose = 0;

 /* DONE FINDING ARGUMENTS AND OPTIONS */

  /* determine if this is a large endian or small endian platform */
  ibe = is_large_endian ();
  if (verbose)
    {
      printf("flct: Version %s Copyright: 2007-2018 University of California\n",
          version);
      if (ibe)
	{
	  printf ("flct: large endian machine; i/o not byteswapped\n");
	}
      else
	{
	  printf ("flct: small endian machine; i/o will be byteswapped\n");
	}
      fflush(stdout);
    }

  /* print out arguments and options */

  if (verbose) printf ("flct: infile = %s\n", infile);
  if (verbose) printf ("flct: outfile = %s\n", outfile);
  if (verbose && transp !=0)
     printf("flct: column major order assumed for data arrays\n");
  if (verbose && transp == 0)
     printf("flct: row major order assumed for data arrays\n");
  if (verbose) printf ("flct: deltat = %g\n", deltat);
  if (verbose) printf ("flct: deltas = %g\n", deltas);
  if (verbose) printf ("flct: sigma = %g\n", sigma);
/* Comment out this warning:
  if (verbose && (sigma < 5.) && (sigma > 0.)) 
       printf ("flct: WARNING: sigma < 5 not recommended\n");
*/
  if (verbose) 
       printf ("flct: threshold image value for LCT is %g\n", thresh);
  if (verbose && aloc) 
       printf ("flct: threshold forced to be in abs. units\n");
  if (verbose && skipon) 
       printf ("flct: skip = %d pixels with p=%d, q=%d\n",skip,poffset,qoffset);
  if (verbose && skipon && ((poffset < 0) || (qoffset < 0))) 
       printf ("flct: p=%d, q=%d: negative p,q will be reset to skip-|p,q|\n",
       poffset,qoffset);
  if (verbose && interpolate) 
       printf ("flct: skipped pixels interpolated with cubic convolution\n");
  if (verbose && filter)
       printf ("flct: filter rolloff value for input images is %g\n", kr);
  if (verbose && biascor)
       printf("flct: bias correction is enabled\n");
  if (verbose && !biascor)
       printf("flct: bias correction not enabled\n");
  if (verbose && platecarree)
       printf ("flct: Plate Carree image format, latmin=%g latmax=%g\n",
       latmin,latmax);

#ifdef COMPLEXH
  if (verbose) printf ("flct: complex.h is included\n");
#else
  if (verbose) printf ("flct: complex.h is not included\n");
#endif

  if (hires == 0) 
     {
       /*  Make this print statement more informative */
       printf ("flct: hires (-h) option on;\n");
       printf ("flct: WE HAVE REMOVED the -h option, exiting.\n");
       exit(1);
     }

  /* For negative poffset or qoffset, set to skip-poffset or skip-qoffset */
 
  if(poffset < 0) poffset=skip-abs(poffset);
  if(qoffset < 0) qoffset=skip-abs(qoffset);
 
  /* Debug 
  printf("flct: poffset = %d, qoffset = %d\n",poffset,qoffset);
  */
  

  /*
   * read nx, ny, and return references to nx and ny to main prog. *
   * NOTE -- roles of nx, ny are reversed from IDL/fortran. In the C version,
   * will work in transposed space. Now, the transpose of nx,ny happens within
   * the flct function, not in the main program, so call read2images with
   * the transp variable set to 0 (no transpose):
   */

  ier = read2images (infile, &nx, &ny, &f1, &f2, (i4) 0);

  /* nx, ny may get set to 1 if sigma=0. so copy original values */
  nxorig=nx;
  nyorig=ny;
  if(sigma == (double) 0)
  {
     /* sigma = 0 means we'll only compute one point */
     nx=1;
     ny=1;
  }

  vx = (double *) malloc (sizeof (double) * nx * ny);
  vy = (double *) malloc (sizeof (double) * nx * ny);
  vm = (double *) malloc (sizeof (double) * nx * ny);

  if (verbose)
    printf ("flct: from input file, nx = %d, ny = %d\n", nx, ny);
  if ((skip >= nx) || (skip >= ny))
  {
     printf("flct: skip = %d is too big compared to nx or ny, fatal\n",skip);
     exit(1);
  }

  /*  If Plate Carree image format, then call flct_pc otherwise call flct. */

  /*  Calling FLCT function now returns vx,vy,vm,nx,ny */
  /* Be sure to set nx,ny calling arguments to nxorig, nyorig */

  if(platecarree)
  {
    ierflct=flct_pc(transp, f1, f2, nxorig, nyorig, deltat, deltas, sigma, 
          vx, vy, vm, thresh, absflag, filter, kr, skip, poffset, qoffset, 
          interpolate, latmin, latmax, biascor, verbose); 
  }
  else
  {
     ierflct=flct(transp, f1, f2, nxorig, nyorig, deltat, deltas, sigma, 
          vx, vy, vm, thresh, absflag, filter, kr, skip, poffset, qoffset, 
          interpolate, biascor, verbose); 
  }

  /* Now do output I/O if no errors.  Note no transpose of nx, ny on output*/
  write3images (outfile, vx, vy, vm, nx, ny, (i4) 0); 

  /* free the the original images, and the
   * velocity output arrays */

  free (f1);
  free (f2);
  free (vx);
  free (vy);
  free (vm);

  return ierflct;
}
