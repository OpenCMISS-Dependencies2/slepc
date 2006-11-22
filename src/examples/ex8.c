
static char help[] = "Estimates the 2-norm condition number of a matrix A, that is, the ratio of the largest to the smallest singular values of A. "
  "The matrix is a Grcar matrix.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n\n";

#include "slepcsvd.h"
#include "slepceps.h"

/*
   This example computes the singular values of an nxn Grcar matrix,
   which is a nonsymmetric Toeplitz matrix:

              |  1  1  1  1               |
              | -1  1  1  1  1            |
              |    -1  1  1  1  1         |
              |       .  .  .  .  .       |
          A = |          .  .  .  .  .    |
              |            -1  1  1  1  1 |
              |               -1  1  1  1 |
              |                  -1  1  1 |
              |                     -1  1 |

 */

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  Mat         	 A;		  /* Grcar matrix */
  SVD            svd;             /* singular value solver context */
  EPS         	 eps;		  /* eigenproblem solver context */
  PetscInt    	 N=30, Istart, Iend, i, col[5];
  int         	 nconv1, nconv2;
  PetscScalar 	 value[] = { -1, 1, 1, 1, 1 };
  PetscReal   	 sigma_1, sigma_n;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nEstimate the condition number of a Grcar matrix, n=%d\n\n",N);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Generate the matrix 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for( i=Istart; i<Iend; i++ ) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) {
      ierr = MatSetValues(A,1,&i,4,col+1,value+1,INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      ierr = MatSetValues(A,1,&i,PetscMin(5,N-i+1),col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
             Create the singular value solver and set the solution method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create singular value context
  */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);

  /* 
     Set operator
  */
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);
  /* PENDIENTE: utilizar funciones SVD en lugar de EPS*/
  ierr = SVDEigensolverGetEPS(svd,&eps);CHKERRQ(ierr); 
  ierr = EPSSetDimensions(eps,1,PETSC_DEFAULT);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     First request an eigenvalue from one end of the spectrum
  */
  ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  /* 
     Get number of converged singular values
  */
  ierr = SVDGetConverged(svd,&nconv1);CHKERRQ(ierr);
  /* 
     Get converged singular values: largest singular value is stored in sigma_1.
     In this example, we are not interested in the singular vectors
  */
  if (nconv1 > 0) {
    ierr = SVDGetSingularTriplet(svd,0,&sigma_1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  /*
     Request an eigenvalue from the other end of the spectrum
  */
  ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  /* 
     Get number of converged eigenpairs
  */
  ierr = SVDGetConverged(svd,&nconv2);CHKERRQ(ierr);
  /* 
     Get converged singular values: smallest singular value is stored in sigma_n. 
     As before, we are not interested in the singular vectors
  */
  if (nconv2 > 0) {
    ierr = SVDGetSingularTriplet(svd,0,&sigma_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (nconv1 > 0 && nconv2 > 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Computed singular values: sigma_1=%6f, sigma_n=%6f\n",sigma_1,sigma_n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Estimated condition number: sigma_1/sigma_n=%6f\n\n",sigma_1/sigma_n);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Process did not converge! Try running with a larger value for -eps_ncv\n\n");CHKERRQ(ierr);
  }   
 
  /* 
     Free work space
  */
  ierr = SVDDestroy(svd);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

