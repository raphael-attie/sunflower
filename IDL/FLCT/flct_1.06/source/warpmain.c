/*

 warp: http://solarmuri.ssl.berkeley.edu/overview/publicdownloads/software.html
 Copyright (C) 2009-2018 Regents of the University of California

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
 * Authors:
 * George H. Fisher, Space Sciences Lab # 7450, University of California,
 * 7 Gauss Way, Berkeley CA 94720-7450 email: fisher at ssl dot berkeley dot edu
 *
 * Brian T. Welsch, Space Sciences Lab # 7450, University of California,
 * 7 Gauss Way, Berkeley CA 94720-7450 email: fisher at ssl dot berkeley dot edu
 *
*/
# include <flctsubs.h>

int main (int argc, char *argv[]);

int main (int argc, char *argv[])
{

/* BEGIN MAIN PROGRAM */

  char *version ="1.06   ";
  char ifile[100], sfile[100], ofile[100];
  i4 quiet, verbose;
  i4 nx, nxdel, ny, nydel;
  i4 ier, ibe;
  double *f1, *f2, *delx, *dely;
  i4 transp = 1; /* This flag is nonzero to transpose input/output arrays */

  /* CODE TO READ IN ARGUMENTS AND OPTIONS: */

  /* check to see if number args is in right range - 
   * if not, print syntax & quit */

  if ((argc < 4) || (argc > 6))
    {
      printf("warp: Version %s Copyright: 2009-2018 University of California\n",
          version);
      printf("Authors: G.H. Fisher, B.T. Welsch, UCB Space Sciences Lab\n\n");
      printf
        ("Syntax: %s ifile sfile ofile [-q]\n\n",argv[0]);
      printf("ifile - contains 1 image for shifting\n");
      printf("sfile - contains 2 values (or arrays) of shifts (or warps)\n");
      printf("ofile - contains output image of shifted or warped array\n");
      printf("-q - optional flag to omit all non-error messages\n");
      exit (1);
    }

  /* GET THE 3 REQUIRED ARGUMENTS */

  /* get input image file name */

  strncpy (ifile, argv[1], 99);

  /* get input  file name */
  strncpy (sfile, argv[2], 99);

  strncpy (ofile, argv[3], 99);

  /*  GET OPTIONAL ARGUMENT quiet : */
  quiet=0;
  verbose=1;
  if(argc == 5)
    {
       if(!strncmp("-q",argv[argc-1],2))
       {
          quiet=1;
          verbose=0;
       }

    }

 /* DONE FINDING ARGUMENTS AND OPTIONS */

  /* determine if this is a large endian or small endian platform */
  ibe = is_large_endian ();
  if (verbose)
    {
      printf("warp: Version %s Copyright: 2009-2018 University of California\n",
          version);
      printf("Authors: G.H. Fisher, B.T. Welsch, UCB Space Sciences Lab\n\n");
      if (ibe)
	{
	  printf ("warp: large endian machine; i/o not byteswapped\n");
	}
      else
	{
	  printf ("warp: small endian machine; i/o will be byteswapped\n");
	}
    }

  /* print out arguments and options */

  if (verbose) printf ("warp: ifile = %s\n", ifile);
  if (verbose) printf ("warp: sfile = %s\n", sfile);
  if (verbose) printf ("warp: ofile = %s\n", ofile);


  /*
   * read nx, ny, and return references to nx and ny to main prog. *
   * NOTE -- roles of nx, ny are reversed from IDL!!!! In the C version,
   * must work in transposed space.  Therefore transp is set to 1.
   */

  ier = readimage (ifile, &nx, &ny, &f1, (i4) 0);

  if (verbose)
    printf ("warp: from input file, nx = %d, ny = %d\n", nx, ny);


  ier = read2images(sfile, &nxdel, &nydel, &delx, &dely, (i4) 0);
  if ((nxdel == nx) && (nydel == ny)) 
  {
     /* in this case we do warping */
     f2=(void *)malloc(sizeof(double) *nx*ny);
     ier=warp_frac2d(f1,delx,dely,f2,nx,ny,transp,verbose);
     free(delx);
     free(dely);
     free(f1);
  }
  else if ((nxdel == 1) && (nydel == 1))
  {
     /* In this case just a uniform shift */
     f2=(void *)malloc(sizeof(double) *nx*ny);
     ier=shift_frac2d(f1,delx[0],dely[0],f2,nx,ny,transp,verbose);
     free(delx);
     free(dely);
     free(f1);
  }
  else
  {
     /* In this case we have an error */
     printf("warp: bad shift arrays: nx=%d, nxdel=%d, ny=%d, nydel=%d\n",
        nx,nxdel,ny,nydel);
     exit(1);
  }

  writeimage (ofile, f2, nx, ny, (i4) 0);

  /* free the gaussian mask array, the original images, and the
   * velocity arrays */

  if (verbose)
    printf ("\nwarp: finished\n");

  /* we're done! */
  return 0;
  /*  END MAIN PROGAM */
}
