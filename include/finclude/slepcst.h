!
!  Include file for Fortran use of the ST object in SLEPc
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     SLEPc - Scalable Library for Eigenvalue Problem Computations
!     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
!
!     This file is part of SLEPc. See the README file for conditions of use
!     and additional information.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
#if !defined(__SLEPCST_H)
#define __SLEPCST_H

#define ST      PetscFortranAddr
#define STType  character*(80)

#define STSHELL     'shell'
#define STSHIFT     'shift'
#define STSINV      'sinvert'
#define STCAYLEY    'cayley'
#define STFOLD      'fold'

      integer STMATMODE_COPY
      integer STMATMODE_INPLACE
      integer STMATMODE_SHELL

      parameter (STMATMODE_COPY          =  0)
      parameter (STMATMODE_INPLACE       =  1)
      parameter (STMATMODE_SHELL         =  2)

#endif
