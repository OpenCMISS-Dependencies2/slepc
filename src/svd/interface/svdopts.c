/*
     SVD routines for setting solver options.
*/
#include "src/svd/svdimpl.h"      /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDSetTransposeMode"
/*@C
   SVDSetTransposeMode - Sets how to handle the transpose of the matrix 
   associated with the singular value problem.

   Collective on SVD and Mat

   Input Parameters:
+  svd  - the singular value solver context
-  mode - how to compute the transpose, one of SVD_TRANSPOSE_EXPLICIT
          or SVD_TRANSPOSE_MATMULT (see notes below)

   Options Database Key:
.  -svd_transpose_mode <mode> - Indicates the mode flag, where <mode> 
    is one of 'explicit' or 'matmult'.

   Notes:
   In the SVD_TRANSPOSE_EXPLICIT mode, the transpose of the matrix is
   explicitly built.

   The option SVD_TRANSPOSE_MATMULT does not build the transpose, but
   handles it implicitly via MatMultTranspose() operations. This is 
   likely to be more inefficient than SVD_TRANSPOSE_EXPLICIT, both in
   sequential and in parallel, but requires less storage.

   The default is SVD_TRANSPOSE_EXPLICIT if the matrix has defined the
   MatTranspose operation, and SVD_TRANSPOSE_MATMULT otherwise.
   
   Level: advanced
   
   .seealso: SVDGetTransposeMode(), SVDSolve(), SVDSetOperator(), SVDGetOperator()
@*/
PetscErrorCode SVDSetTransposeMode(SVD svd,SVDTransposeMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  switch (mode) {
    case SVD_TRANSPOSE_EXPLICIT:
    case SVD_TRANSPOSE_MATMULT:
    case PETSC_DEFAULT:
      svd->transmode = mode;
      svd->setupcalled = 0;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid transpose mode"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetTransposeMode"
/*@
   SVDGetTransposeMode - Gets the mode use to compute the  transpose 
   of the matrix associated with the singular value problem.

   Not collective

   Input Parameter:
+  svd  - the singular value solver context

   Output paramter:
+  mode - how to compute the transpose, one of SVD_TRANSPOSE_EXPLICIT
          or SVD_TRANSPOSE_MATMULT
   
   Level: advanced
   
   .seealso: SVDSetTransposeMode(), SVDSolve(), SVDSetOperator(), SVDGetOperator()
@*/
PetscErrorCode SVDGetTransposeMode(SVD svd,SVDTransposeMode *mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(mode,2);
  *mode = svd->transmode;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetTolerances"
/*@
   SVDSetTolerances - Sets the tolerance and maximum
   iteration count used by the default SVD convergence testers. 

   Collective on SVD

   Input Parameters:
+  svd - the singluar value solver context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -svd_tol <tol> - Sets the convergence tolerance
-  -svd_max_it <maxits> - Sets the maximum number of iterations allowed 
   (use PETSC_DEFAULT to compute a value based on the operator matrix)

   Notes:
   Use PETSC_IGNORE to retain the previous value of any parameter. 

   Level: intermediate

.seealso: SVDGetTolerances()
@*/
PetscErrorCode SVDSetTolerances(SVD svd,PetscReal tol,int maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (tol != PETSC_IGNORE) {
    if (tol < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
    svd->tol = tol;
  }
  if (maxits != PETSC_IGNORE) {
    if (maxits == PETSC_DEFAULT) {
      svd->setupcalled = 0;
    } else {
      if (maxits < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
    }
    svd->max_it = maxits;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetTolerances"
/*@
   SVDGetTolerances - Gets the tolerance and maximum
   iteration count used by the default SVD convergence tests. 

   Not Collective

   Input Parameter:
.  svd - the singular value solver context
  
   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: SVDSetTolerances()
@*/
PetscErrorCode SVDGetTolerances(SVD svd,PetscReal *tol,int *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (tol)    *tol    = svd->tol;
  if (maxits) *maxits = svd->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetDimensions"
/*@
   SVDSetDimensions - Sets the number of singular values to compute
   and the dimension of the subspace.

   Collective on SVD

   Input Parameters:
+  svd - the singular solver context
.  nsv - number of singular values to compute
-  ncv - the maximum dimension of the subspace to be used by the solver

   Options Database Keys:
+  -svd_nsv <nsv> - Sets the number of singular values
-  -svd_ncv <ncv> - Sets the dimension of the subspace

   Notes:
   Use PETSC_IGNORE to retain the previous value of any parameter.

   Use PETSC_DEFAULT for ncv to assign a reasonably good value, which is 
   dependent on the solution method.

   Level: intermediate

.seealso: SVDGetDimensions()
@*/
PetscErrorCode SVDSetDimensions(SVD svd,int nsv,int ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);

  if (nsv != PETSC_IGNORE) {
    if (nsv<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nsv. Must be > 0");
    svd->nsv = nsv;
    svd->setupcalled = 0;
  }
  if (ncv != PETSC_IGNORE) {
    if (ncv<1 && ncv != PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
    svd->ncv = ncv;
    svd->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetDimensions"
/*@
   SVDGetDimensions - Gets the number of singular values to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  svd - the singular value context
  
   Output Parameters:
+  nsv - number of singular values to compute
-  ncv - the maximum dimension of the subspace to be used by the solver

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: SVDSetDimensions()
@*/
PetscErrorCode SVDGetDimensions(SVD svd,int *nsv,int *ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (nsv) *nsv = svd->nsv;
  if (ncv) *ncv = svd->ncv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetWhichSingularTriplets"
/*@
    SVDSetWhichSingularTriplets - Specifies which singular triplets are 
    to be sought.

    Collective on SVD

    Input Parameter:
.   svd - singular value solver context obtained from SVDCreate()

    Output Parameter:
.   which - which singular triplets are to be sought

    Possible values:
    The parameter 'which' can have one of these values:
    
+     SVD_LARGEST  - largest singular values
-     SVD_SMALLEST - smallest singular values

    Options Database Keys:
+   -svd_largest  - Sets largest singular values
-   -svd_smallest - Sets smallest singular values
    
    Level: intermediate

.seealso: SVDGetWhichSingularTriplets()
@*/
PetscErrorCode SVDSetWhichSingularTriplets(SVD svd,SVDWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  switch (which) {
    case SVD_LARGEST:
    case SVD_SMALLEST:
      svd->which = which;
      break;
  default:
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'which' parameter");    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetWhichSingularTriplets"
/*@C
    SVDGetWhichSingularTriplet - Returns which singular triplets are
    to be sought.

    Not Collective

    Input Parameter:
.   svd - singular value solver context obtained from SVDCreate()

    Output Parameter:
.   which - which singular triplets are to be sought

    Notes:
    See SVDSetWhichSingularTriplets() for possible values of which

    Level: intermediate

.seealso: SVDSetWhichSingularTriplets()
@*/
PetscErrorCode SVDGetWhichSingularTriplets(SVD svd,SVDWhich *which) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(which,2);
  *which = svd->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions"
/*@
   SVDSetFromOptions - Sets SVD options from the options database.
   This routine must be called before SVDSetUp() if the user is to be 
   allowed to set the solver type. 

   Collective on SVD

   Input Parameters:
.  svd - the singular value solver context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner

.seealso: 
@*/
PetscErrorCode SVDSetFromOptions(SVD svd)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscTruth     flg,flg2;
  const char     *mode_list[2] = { "explicit", "matmult" };
  PetscInt       i,j;
  PetscReal      r;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  svd->setupcalled = 0;
  ierr = PetscOptionsBegin(svd->comm,svd->prefix,"Singular Value Solver (SVD) Options","SVD");CHKERRQ(ierr);

  ierr = PetscOptionsList("-svd_type","Singular Value Solver method","SVDSetType",SVDList,(char*)(svd->type_name?svd->type_name:SVDEIGENSOLVER),type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SVDSetType(svd,type);CHKERRQ(ierr);
  } else if (!svd->type_name) {
    ierr = SVDSetType(svd,SVDEIGENSOLVER);CHKERRQ(ierr);
  }

  ierr = PetscOptionsName("-svd_view","Print detailed information on solver used","SVDView",0);CHKERRQ(ierr);

  ierr = PetscOptionsEList("-svd_transpose_mode","Transpose SVD mode","SVDSetTransposeMode",mode_list,2,svd->transmode == PETSC_DEFAULT ? "default" : mode_list[svd->transmode],&i,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SVDSetTransposeMode(svd,(SVDTransposeMode)i);CHKERRQ(ierr);
  }   

  r = i = PETSC_IGNORE;
  ierr = PetscOptionsInt("-svd_max_it","Maximum number of iterations","SVDSetTolerances",svd->max_it,&i,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-svd_tol","Tolerance","SVDSetTolerances",svd->tol,&r,&flg2);CHKERRQ(ierr);
  if (flg || flg2) {
    ierr = SVDSetTolerances(svd,r,i);CHKERRQ(ierr);
  }

  i = j = PETSC_IGNORE;
  ierr = PetscOptionsInt("-svd_nsv","Number of singular values to compute","SVDSetDimensions",svd->ncv,&i,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-svd_ncv","Number of basis vectors","SVDSetDimensions",svd->ncv,&j,&flg2);CHKERRQ(ierr);
  if (flg || flg2) {
    ierr = SVDSetDimensions(svd,i,j);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTruthGroupBegin("-svd_largest","compute largest singular values","SVDSetWhichSingularTriplets",&flg);CHKERRQ(ierr);
  if (flg) { ierr = SVDSetWhichSingularTriplets(svd,SVD_LARGEST);CHKERRQ(ierr); }
  ierr = PetscOptionsTruthGroupEnd("-svd_smallest","compute smallest singular values","SVDSetWhichSingularTriplets",&flg);CHKERRQ(ierr);
  if (flg) { ierr = SVDSetWhichSingularTriplets(svd,SVD_SMALLEST);CHKERRQ(ierr); }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (svd->ops->setfromoptions) {
    ierr = (*svd->ops->setfromoptions)(svd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
