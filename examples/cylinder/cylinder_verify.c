#include "TACSAssembler.h"
#include "TACSShellTraction.h"
#include "TACSToFH5.h"
#include "FSDTStiffness.h"
#include "MaterialProperties.h"
#include "KSFailure.h"
#include "InducedFailure.h"
#include "MITCShell.h"
#include "tacslapack.h"

/*
  An FSDTStiffness class of it's own! 

  This class is specifically for testing the cylinder - it's required
  so that the failure criterion is evaluated only at the upper surface
  - and does not return the max of the upper/lower values.
*/
class specialFSDTStiffness : public FSDTStiffness {
 public:
  specialFSDTStiffness( OrthoPly *_ply, int _orthotropic_flag, 
                        TacsScalar _t, TacsScalar _kcorr ){
    ply = _ply; 
    ply->incref();
    
    orthotropic_flag = _orthotropic_flag;
    t = _t;
    kcorr = _kcorr;
  }
  ~specialFSDTStiffness(){
    ply->decref();
  }

  void getPointwiseMass( const double pt[], 
                         TacsScalar mass[] ){
    mass[0] = t*ply->getRho();
    mass[1] = (t*t*t*ply->getRho())/12.0;
  }

  // Compute the stiffness
  TacsScalar getStiffness( const double gpt[],
                           TacsScalar A[], TacsScalar B[],
                           TacsScalar D[], TacsScalar As[] ){
    A[0] = A[1] = A[2] = A[3] = A[4] = A[5] = 0.0;
    B[0] = B[1] = B[2] = B[3] = B[4] = B[5] = 0.0;
    D[0] = D[1] = D[2] = D[3] = D[4] = D[5] = 0.0;
    As[0] = As[1] = As[2] = 0.0;

    TacsScalar Qbar[6], Abar[3];
    ply->calculateQbar(Qbar, 0.0);
    ply->calculateAbar(Abar, 0.0);

    TacsScalar t0 = -0.5*t;
    TacsScalar t1 = 0.5*t;

    TacsScalar a = (t1 - t0);
    TacsScalar b = 0.5*(t1*t1 - t0*t0);
    TacsScalar d = 1.0/3.0*(t1*t1*t1 - t0*t0*t0);
    
    for ( int i = 0; i < 6; i++ ){
      A[i] += a*Qbar[i];
      B[i] += b*Qbar[i];
      D[i] += d*Qbar[i];
    }
    
    for ( int i = 0; i < 3; i++ ){
      As[i] += kcorr*a*Abar[i];
    }

    return 0.5*DRILLING_REGULARIZATION*(As[0] + As[2]);
  }

  // Compute the failure function at the given point
  void failure( const double pt[], const TacsScalar strain[], 
                TacsScalar * fail ){
    TacsScalar z = 0.5*t;
    
    TacsScalar e[3];
    e[0] = strain[0] + z*strain[3];
    e[1] = strain[1] + z*strain[4];
    e[2] = strain[2] + z*strain[5];

    if (orthotropic_flag){
      *fail = ply->failure(0.0, e);
    }
    else {
      *fail = sqrt(ply->failure(0.0, e));
    }
  }

 private:
  OrthoPly * ply;
  TacsScalar t, kcorr;
  int orthotropic_flag;
};

/*
  Compute the coefficients of a single term in a Fourier series for a
  specially orthotropic cylinder subjectedt to a sinusoidally varying
  pressure distribution specified as follows:

  p = sin(alpha*y)*cos(beta*x)

  Note that y = r*theta

  The coefficients U, V, W, theta and phi are for the expressions:
  
  u(x,y) = U*sin(alpha*y)*cos(beta*x)
  v(x,y) = V*cos(alpha*y)*sin(beta*x)
  w(x,y) = W*sin(alpha*y)*sin(beta*x)
  psi_x(x,y) = theta*sin(alpha*y)*cos(beta*x)
  psi_y(x,y) = phi*cos(alpha*y)*sin(beta*x)

  Note that u, v and w correspond to the axial, tangential and normal
  displacements in a reference frame attached to the shell surface -
  they are not the displacements expressed in the global reference
  frame. Likewise, psi_x and psi_y are the rotations of the normal
  along the x-direction and the tangential y-direction.
*/
void compute_coefficients( TacsScalar *U, TacsScalar *V, 
                           TacsScalar *W, TacsScalar *theta, 
                           TacsScalar *phi,
                           double alpha, double beta, 
                           TacsScalar ainv, 
                           TacsScalar A11, TacsScalar A12, 
                           TacsScalar A22, TacsScalar A33,
                           TacsScalar D11, TacsScalar D12, 
                           TacsScalar D22, TacsScalar D33,
                           TacsScalar bA11, TacsScalar bA22, TacsScalar load ){

  TacsScalar A[5*5], A2[5*5]; // 5 x 5 system of linear equations
  int ipiv[5];
  TacsScalar rhs[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  rhs[2] = - load;
 
  TacsScalar B[8*5];
  memset(B, 0, sizeof(B));
  TacsScalar *b = B;

  // Assign the columns for u
  b[0] = -beta;
  b[2] = alpha;
  b[5] = -alpha*ainv;
  b += 8;

  // Assign columns for v
  b[1] = -alpha;
  b[2] = beta;
  b[4] = alpha*ainv;
  b[6] = -ainv;
  b += 8;

  // Assign the columns for w
  b[1] = ainv;
  b[4] = -ainv*ainv;
  b[6] = alpha;
  b[7] = beta;
  b += 8;

  // Assign the columns for psi_x
  b[3] = -beta;
  b[5] = alpha;
  b[7] = 1.0;
  b += 8;

  // Assign the columns for psi_y
  b[4] = -alpha; 
  b[5] = beta;
  b[6] = 1.0;

  for ( int j = 0; j < 5; j++ ){
    TacsScalar *bj = &B[8*j];
    for ( int i = 0; i < 5; i++ ){
      TacsScalar *bi = &B[8*i];

      A2[i + 5*j] = 
        - ((bi[0]*(A11*bj[0] + A12*bj[1]) +  
            bi[1]*(A12*bj[0] + A22*bj[1]) + bi[2]*A33*bj[2]) +
           (bi[3]*(D11*bj[3] + D12*bj[4]) +
            bi[4]*(D12*bj[3] + D22*bj[4]) + bi[5]*D33*bj[5]) +
           bi[6]*bA11*bj[6] + bi[7]*bA22*bj[7]);
    }
  }

  // The first equation for u
  A[0]  = -(A11*beta*beta + A33*alpha*alpha) 
    - D33*ainv*ainv*alpha*alpha;
  A[5]  = -(A33 + A12)*alpha*beta;
  A[10] = A12*beta*ainv;
  A[15] = D33*ainv*alpha*alpha;
  A[20] = D33*ainv*alpha*beta;

  // The second equation for v
  A[1]  = -(A12 + A33)*alpha*beta;
  A[6]  = -(A33*beta*beta + A22*alpha*alpha) 
    - ainv*ainv*bA11 - D22*ainv*ainv*alpha*alpha;
  A[11] = (A22 + bA11)*ainv*alpha + D22*alpha*ainv*ainv*ainv;
  A[16] = D12*ainv*alpha*beta;
  A[21] = bA11*ainv + D22*ainv*alpha*alpha;

  // The third equation for w
  A[2]  = A12*beta*ainv;
  A[7]  = (bA11 + A22)*alpha*ainv + D22*alpha*ainv*ainv*ainv;
  A[12] = -(bA11*alpha*alpha + bA22*beta*beta) 
    - A22*ainv*ainv - D22*ainv*ainv*ainv*ainv;
  A[17] = -bA22*beta - D12*beta*ainv*ainv;
  A[22] = -bA11*alpha - D22*alpha*ainv*ainv;

  // Fourth equation for theta
  A[3]  = D33*ainv*alpha*alpha;
  A[8]  = D12*ainv*alpha*beta;
  A[13] = -bA22*beta - D12*beta*ainv*ainv;
  A[18] = -(D11*beta*beta + D33*alpha*alpha) - bA22;
  A[23] = -(D12 + D33)*alpha*beta;

  // Fifth equation for phi
  A[4]  =  D33*ainv*alpha*beta;
  A[9]  =  bA11*ainv + D22*ainv*alpha*alpha;
  A[14] = -bA11*alpha - D22*alpha*ainv*ainv;
  A[19] = -(D33 + D12)*alpha*beta;
  A[24] = -(D33*beta*beta + D22*alpha*alpha) - bA11;

  int print_equations = 1;
  if (print_equations){
    const char * variables[] = {"U", "V", "W", "theta", "phi"};
    for ( int k = 0; k < 5; k++ ){
      printf("Equation %d\n", k);
      for ( int j = 0; j < 5; j++ ){
        printf(" %15.5f %s", RealPart(A[k + 5*j] - A2[k + 5*j]), 
               variables[j]);
      }
      printf("\n");
    }
  }

  int info = 0;
  int n = 5;
  int nrhs = 1;
  LAPACKgesv(&n, &nrhs, A2, &n, ipiv, rhs, &n, &info);

  // Solve for the coefficients 
  *U = rhs[0];
  *V = rhs[1];
  *W = rhs[2];
  *theta = rhs[3];
  *phi = rhs[4];
}

/*
  Load the cylinder, set the pressure load and solve the problem.
  Note that this will only work for single processor cases.
*/
TACSAssembler *create_tacs_cylinder_model( FSDTStiffness *stiffness,
                                           double alpha, double beta,
                                           int order, int nx, int ny,
                                           double L, double R,
                                           TacsScalar load ){
  if (order < 2){ order = 2; }
  if (order > 4){ order = 4; }

  int nelems = nx*ny;
  int nnodes = ((order-1)*nx + 1)*(order-1)*ny;
  int vars_per_node = 6;
  
  // For now create a complete cylinder
  TACSAssembler *tacs = new TACSAssembler(MPI_COMM_SELF, vars_per_node,
                                          nnodes, nelems);
  
  TACSElement * shell = NULL;
  if (order == 2){ shell = new MITCShell<2>(stiffness); }
  else if (order == 3){ shell = new MITCShell<3>(stiffness); }
  else if (order == 4){ shell = new MITCShell<4>(stiffness); }

  int nnx = (order-1)*nx + 1;
  int nny = (order-1)*ny;

  // Set the connectivity for the elements
  int *conn = new int[ order*order*nelems ];
  int *cptr = conn;

  // Set the node numbers
  for ( int j = 0; j < ny; j++ ){
    for ( int i = 0; i < nx; i++ ){
      // Set the nodes for element i, j
      for ( int jj = 0; jj < order; jj++ ){
        for ( int ii = 0; ii < order; ii++, cptr++ ){
          if (j == ny-1 && jj == order-1){
            cptr[0] = (order-1)*i + ii;
          }
          else {
            cptr[0] = ((order-1)*i + ii) + ((order-1)*j + jj)*nnx;
          }
        }
      }
    }
  }

  int *ptr = new int[ nelems+1 ];
  for ( int i = 0; i < nelems+1; i++ ){
    ptr[i] = order*order*i;
  }

  // Set the element connectivity
  tacs->setElementConnectivity(conn, ptr);
  delete [] conn;
  delete [] ptr;

  // Set the elements
  TACSElement **elements = new TACSElement*[ nelems ];
  for ( int i = 0; i < nelems; i++ ){
    elements[i] = shell;
  }
  tacs->setElements(elements);
  delete [] elements;

  // Add the boundary conditions - the axial displacement is
  // unconstrained
  int root_var_nums[] = {0, 1, 2, 3};
  int var_nums[] = {1, 2, 3};
  for ( int j = 0; j < nny; j++ ){
    int node = j*nnx;
    if (j == 0){
      tacs->addBCs(1, &node, 4, root_var_nums);
    }
    else {
      tacs->addBCs(1, &node, 3, var_nums);
    }

    node += (nnx-1);
    tacs->addBCs(1, &node, 3, var_nums);
  }

  // Initialize TACS
  tacs->initialize();

  // Set the nodal locations
  TACSBVec *X = tacs->createNodeVec();
  X->incref();
  TacsScalar *Xpts;
  X->getArray(&Xpts);
  
  // Set the node locations
  double defect = 0.1;
  for ( int j = 0; j < nny; j++ ){
    double v = -M_PI + (2.0*M_PI*j)/nny;
    for ( int i = 0; i < nnx; i++ ){
      double u = 1.0*i/(nnx - 1);
      double theta = v + 0.25*M_PI*u + defect*sin(v)*cos(2*M_PI*u);
      double x = L*(u + defect*cos(v)*sin(2*M_PI*u));

      int node = i + j*nnx;
      Xpts[3*node]   =  x;
      Xpts[3*node+1] =  R*cos(theta);
      Xpts[3*node+2] = -R*sin(theta);
    }
  }

  // Set the node locations
  tacs->setNodes(X);
  X->decref();

  // Create the surface traction object
  int num_elems = tacs->getNumElements();
  TACSAuxElements *aux = new TACSAuxElements(num_elems);

  // Add the elements
  for ( int j = 0; j < ny; j++ ){
    for ( int i = 0; i < nx; i++ ){
      TacsScalar pval[16];

      // Set the nodes for element i, j
      for ( int jj = 0; jj < order; jj++ ){
        for ( int ii = 0; ii < order; ii++ ){
          int node = ((order-1)*i + ii) + ((order-1)*j + jj)*nnx;
          if (j == ny-1 && jj == order-1){
            node = ((order-1)*i + ii);
          }
          
          // Compute the pressure at this point in the shell
          double x = RealPart(Xpts[3*node]);
          double y =-R*atan2(RealPart(Xpts[3*node+2]), 
                             RealPart(Xpts[3*node+1]));
          pval[ii + order*jj] = load*sin(beta*x)*sin(alpha*y);
        }
      }

      TACSElement *traction = NULL;
      if (order == 2){ traction = new TACSShellPressure<2>(pval); }
      else if (order == 3){ traction = new TACSShellPressure<3>(pval); }
      else if (order == 4){ traction = new TACSShellPressure<4>(pval); }
      aux->addElement(i + nx*j, traction);
    }
  }

  tacs->setAuxElements(aux);

  return tacs;
}

/*
  Compute the error in the solution between the exact solution and the
  finite-element solution.

  This function computes an estimate of the L2 norm of the error:

  error = sqrt( int_{A} (w_{TACS} - w_{exact})^2 dA )
*/
TacsScalar compute_tacs_error( int order, TACSAssembler *tacs, 
                               TacsScalar R, double alpha, double beta,
                               TacsScalar U, TacsScalar V, TacsScalar W, 
                               TacsScalar theta, TacsScalar phi ){
  TacsScalar error = 0.0;
  const double *gauss_pts, *gauss_wts;
  int num_gauss = 
    FElibrary::getGaussPtsWts(order+1, &gauss_pts, &gauss_wts);
  
  int nelems = tacs->getNumElements();
  for ( int k = 0; k < nelems; k++ ){
    TacsScalar vars[6*16], Xpts[3*16];
    TACSElement *shell = tacs->getElement(k, Xpts, vars, NULL, NULL);

    int num_nodes = shell->numNodes();
    for ( int m = 0; m < num_gauss; m++ ){
      for ( int n = 0; n < num_gauss; n++ ){
        double pt[3];
        pt[0] = gauss_pts[n];
        pt[1] = gauss_pts[m];
        double weight = gauss_wts[n]*gauss_wts[m];

        // Get the shape functions
        double N[16];
        shell->getShapeFunctions(N, pt);
   
        TacsScalar h = shell->getDetJacobian(pt, Xpts);
        TacsScalar Xp[3] = {0.0, 0.0, 0.0};
        for ( int i = 0; i < num_nodes; i++ ){
          Xp[0] += Xpts[3*i]*N[i];
          Xp[1] += Xpts[3*i+1]*N[i];
          Xp[2] += Xpts[3*i+2]*N[i];
        }

        // Determine the coordinates on the shell 
        TacsScalar x = Xp[0];
        TacsScalar y =-R*atan2(RealPart(Xp[2]), 
                               RealPart(Xp[1]));
        
        // Evaluate the radial displacement at this point from
        // the exact solution
        
        TacsScalar u[6] = {0.0, 0.0, 0.0, 
                           0.0, 0.0, 0.0};
        for ( int i = 0; i < num_nodes; i++ ){
          u[0] += N[i]*vars[6*i];
          u[1] += N[i]*vars[6*i+1];
          u[2] += N[i]*vars[6*i+2];
          u[3] += N[i]*vars[6*i+3];
          u[4] += N[i]*vars[6*i+4];
          u[5] += N[i]*vars[6*i+5];
        }
        
        // Compute the normal displacement
        TacsScalar ny = cos(y/R);
        TacsScalar nz =-sin(y/R);
        TacsScalar w_tacs = ny*u[1] + nz*u[2];

        // Evaluate the exact solution      
        TacsScalar w_exact = W*sin(beta*x)*sin(alpha*y);

        // Compute the error
        error += weight*h*(w_tacs - w_exact)*(w_tacs - w_exact);
      }
    }
  }

  return sqrt(error);
}

/*
  Compute the p-norm of the von Mises stress at a given reference
  distance from the mid-plane of the shell.

  a = -(beta*Q11(U + z*theta) + Q12*(alpha*(V + z*phi) + W*(ainv + z*ainv*ainv)))
  b = -(beta*Q12(U + z*theta) + Q22*(alpha*(V + z*phi) + W*(ainv + z*ainv*ainv)))
  c = Q33*(alpha*(U + z*theta) + beta*(V + z*phi))

  sigma_vm^2 = 
  (a^2 + b^2 + a*b)*sin^2(alpha*y)*sin^2(beta*x) + 
  3*c^2*cos^2(alpha*y)*cos^2(beta*x)
  
  ||sigma_vm||_p = sqrt(|| sigma_vm^2 ||_(P/2))

  || sigma_vm^2 ||_{K}^{K} = sum_k=0^K (K k)  
*/
TacsScalar evaluate_vm_pnorm( int P, TacsScalar z,
                              TacsScalar Q11, TacsScalar Q12, 
                              TacsScalar Q22, TacsScalar Q33, 
                              double alpha, double beta, 
                              TacsScalar ainv, TacsScalar R, TacsScalar L,
                              TacsScalar U, TacsScalar V, TacsScalar W, 
                              TacsScalar theta, TacsScalar phi,
                              TacsScalar *_vm_max ){
  int K = P/2;
  if (P % 2 != 0){
    printf("Warning! P-norm can only be computed for even P!\n");
  }

  TacsScalar rz = (1.0 - z*ainv);
  TacsScalar a = -(beta*Q11*(U + z*theta) + Q12*(alpha*(V*rz + z*phi) - W*rz*ainv));
  TacsScalar b = -(beta*Q12*(U + z*theta) + Q22*(alpha*(V*rz + z*phi) - W*rz*ainv));
  TacsScalar c = Q33*(alpha*(U*rz + z*theta) + beta*(V + z*phi));

  TacsScalar x = (a*a + b*b - a*b);
  TacsScalar y = 3.0*c*c;
  TacsScalar vm_max1 = sqrt(x*y/(x + y));
  TacsScalar vm_max2 = sqrt(x);
  TacsScalar vm_max3 = sqrt(y);
  
  printf("Max 1: %25.10f\n", RealPart(vm_max1));
  printf("Max 2: %25.10f\n", RealPart(vm_max2));
  printf("Max 3: %25.10f\n", RealPart(vm_max3));

  TacsScalar vm_max = vm_max1; 
  if (RealPart(vm_max2) >= RealPart(vm_max)){ 
    vm_max = vm_max2; 
  }
  if (RealPart(vm_max3) >= RealPart(vm_max)){ 
    vm_max = vm_max3; 
  }
  *_vm_max = vm_max;

  // Compute the area of the cylinder
  TacsScalar A = 2*M_PI*R*L;
  
  TacsScalar pnorm = 0.0;
  for ( int i = 0; i <= K; i++ ){
    // The exponents in the binomial expansion
    int p = i;
    int q = K - i;

    // Compute the binomial coefficient (K / i)
    TacsScalar bterm = 1.0;
    for ( int j = 1; j <= i; j++ ){
      bterm *= (1.0*(K - (i - j)))/j;
    }

    // Evaluate the polynomail term
    for ( int j = 1; j <= p; j++ ){
      bterm *= x;
    }
    for ( int j = 1; j <= q; j++ ){
      bterm *= y;
    }
    
    // Evaluate the integral
    // sin^{2*p}(x)*sin^{2*p}(y)*cos^{2*q}(x)*cos^(2q)(y)
    TacsScalar sc = A;
    for ( int k = 1; k <= p; k++ ){
      TacsScalar t = (2.0*k - 1.0)/(2.0*q + 2.0*k);
      sc *= t*t;
    }
    for ( int k = 1; k <= q; k++ ){
      TacsScalar t = (2.0*k - 1.0)/(2.0*k);
      sc *= t*t;
    }

    pnorm += bterm*sc;
  }

  return pow(pnorm, 1.0/P);
} 

/*
  Evaluate the p-norm of the Tsai--Wu failure criterion

  F(s_1, s_2, s_12) = 
  F1*s_1 + F2*s_2 + 
  F11*s_1**2 + F22*s_2**2 + 2.0*F12*s_1*s_2 + F66*s_12**2
*/
TacsScalar evaluate_tsai_wu_pnorm( int P, int M, int N, TacsScalar z,
                                   TacsScalar Q11, TacsScalar Q12, 
                                   TacsScalar Q22, TacsScalar Q33, 
                                   TacsScalar F1, TacsScalar F2,
                                   TacsScalar F11, TacsScalar F12, 
                                   TacsScalar F22, TacsScalar F66,
                                   double alpha, double beta, TacsScalar ainv,
                                   TacsScalar R, TacsScalar L,
                                   TacsScalar U, TacsScalar V, TacsScalar W, 
                                   TacsScalar theta, TacsScalar phi, 
                                   TacsScalar *tw_max,
                                   TacsScalar *unit_load ){
  if (P % 2 != 0){
    printf("Warning! P-norm can only be computed for even P!\n");
  }

  TacsScalar rz = (1.0 - z*ainv);
  TacsScalar a = -(beta*Q11*(U + z*theta) + Q12*(alpha*(V*rz + z*phi) - W*rz*ainv));
  TacsScalar b = -(beta*Q12*(U + z*theta) + Q22*(alpha*(V*rz + z*phi) - W*rz*ainv));
  TacsScalar c = Q33*(alpha*(U*rz + z*theta) + beta*(V + z*phi));

  TacsScalar pnorm = 0.0;

  // Set the coefficients
  TacsScalar coef[3];
  coef[0] = F1*a + F2*b;
  coef[1] = F11*a*a + F22*b*b + 2.0*F12*a*b;
  coef[2] = F66*c*c;

  int cos_pow[3] = {0, 0, 2};
  int sin_pow[3] = {1, 2, 0};
  
  // Compute all the permutations such that sum t_{k} = P
  int power[3] = {0, 0, 0};

  TacsScalar tw_max1 = coef[0] + coef[1];
  TacsScalar tw_max2 = coef[1] - coef[0];
  TacsScalar tw_max3 = coef[2];
  TacsScalar tw_max4 = (4.0*coef[2]*(coef[1] + coef[2]) - coef[0]*coef[0])/(4.0*(coef[1] + coef[2]));

  printf("Max 1: %25.8f\n", RealPart(tw_max1));
  printf("Max 2: %25.8f\n", RealPart(tw_max2));
  printf("Max 3: %25.8f\n", RealPart(tw_max3));
  printf("Max 4: %25.8f\n", RealPart(tw_max4));
  
  *tw_max = tw_max1;

  // Solve the quadratic equation: alpha*coef[0] + alpha**2*coef[1] - 1.0 = 0
  TacsScalar r1, r2;
  TacsScalar one = 1.0;
  FElibrary::solveQERoots(&r1, &r2, coef[1], coef[0], -one);
  if (RealPart(r1) > RealPart(r2)){ 
    *unit_load = r1; 
  }
  else { *unit_load = r2; }

  if (RealPart(tw_max2) > RealPart(*tw_max)){ 
    *tw_max = tw_max2; 

    // Solve the quadratic equation: alpha*coef[0] - alpha**2*coef[1] - 1.0 = 0
    FElibrary::solveQERoots(&r1, &r2, coef[1], -coef[0], one);
    if (RealPart(r1) > RealPart(r2)){ 
      *unit_load = r1; 
    }
    else { *unit_load = r2; }
  }
  if (RealPart(tw_max3) > RealPart(*tw_max)){ 
    *tw_max = tw_max3; 
    *unit_load = 1.0/sqrt(tw_max3);
  }
  if (RealPart(tw_max4) > RealPart(*tw_max)){ 
    *tw_max = tw_max4; 
  }

  // There is probably a better way to do this, but this was the fastest to code
  for ( int i1 = 0; i1 <= P; i1++ ){
    for ( int i2 = 0; i2 <= P; i2++ ){
      for ( int i3 = 0; i3 <= P; i3++ ){
        if (i1 + i2 + i3 == P){
          power[0] = i1;
          power[1] = i2;
          power[2] = i3;

          // Compute the multi-nomial coefficient
          TacsScalar bterm = 1.0;
          int SK = 0; // The running sum of the powers
          for ( int k = 0; k < 3; k++ ){
            SK += power[k]; 
            
            // Multiply by (SK choose power[k])
            for ( int j = 1; j <= power[k]; j++ ){
              bterm *= (1.0*(SK - (power[k] - j)))/j;
            }
          }
          
          // Compute the polynomial term 
          for ( int j = 0; j < 3; j++ ){
            for ( int k = 1; k <= power[j]; k++ ){
              bterm *= coef[j];
            }
          }
                
          // Compute the sum of the sin and cosine powers
          int sp = 0, cp = 0;
                
          for ( int j = 0; j < 3; j++ ){
            sp += sin_pow[j]*power[j];
            cp += cos_pow[j]*power[j];
          }
                
          // Compute the integral of int_{Omega} sin^{sp}(x) cos^{cp}(x) 
          TacsScalar sc = 0.0;
          if (sp % 2 == 1){
            sc = 1.0/(alpha*beta);
            
            for ( int k = 0; k < (sp-1)/2; k++ ){
              TacsScalar t = (sp - 1.0 - 2.0*k)/(cp + sp - 2.0*k);
              sc *= t*t;
            }
            
            // Multiply by the product with
            // int sin(x) cos^{cp}(x) = -cos^{cp+1}(x)/(cp+1)
            if (cp % 2 == 0){
              if (N % 2 == 1){
                sc *= 2.0/(cp + 1.0);
              }
              else {
                sc = 0.0;
              }
              
              if (M % 2 == 1){
                sc *= 2.0/(cp + 1.0);
              }
              else {
                sc = 0.0;
              }
            }
          }
          else if (sp % 2 == 0 && cp % 2 == 0){
            sc = 2*M_PI*R*L;
            
            for ( int k = 1; 2*k <= sp; k++ ){
              TacsScalar t = (2.0*k - 1.0)/(cp + 2.0*k);
              sc *= t*t;
            }
            for ( int k = 1; 2*k <= cp; k++ ){
              TacsScalar t = (2.0*k - 1.0)/(2.0*k);
              sc *= t*t;
            }
          }
          
          pnorm += bterm*sc;
        }
      }
    }
  }

  return pow(pnorm, 1.0/P);
}
            
/*
  Take the difference between what's in the vector and what the exact
  solution is at each point in the mesh.
*/
void set_tacs_exact( TACSAssembler *tacs, TACSBVec *ans, 
                     TacsScalar R, double alpha, double beta, 
                     TacsScalar U, TacsScalar V, TacsScalar W, 
                     TacsScalar theta, TacsScalar phi ){
  // Get the x,y,z locations of the nodes from TACS
  int nnodes = tacs->getNumNodes();
  
  TACSBVec *X = tacs->createNodeVec();
  X->incref();
  tacs->getNodes(X);

  // Get the nodal array
  TacsScalar *Xpts;
  X->getArray(&Xpts);

  // Get the finite-element solution from TACS
  TacsScalar *u_tacs;
  ans->getArray(&u_tacs);

  for ( int node = 0; node < nnodes; node++ ){
    // Compute the u, v, w, theta, phi solution at this point within
    // the shell
    TacsScalar x = Xpts[3*node];
    TacsScalar y =-R*atan2(RealPart(Xpts[3*node+2]), 
                           RealPart(Xpts[3*node+1]));

    TacsScalar u = U*sin(alpha*y)*cos(beta*x);
    TacsScalar v = V*cos(alpha*y)*sin(beta*x);
    TacsScalar w = W*sin(alpha*y)*sin(beta*x);
    TacsScalar psi_x = theta*sin(alpha*y)*cos(beta*x);
    TacsScalar psi_y = phi*cos(alpha*y)*sin(beta*x);
    
    TacsScalar n[2], t[2];
    n[0] =  cos(y/R);
    n[1] = -sin(y/R);

    t[0] = -sin(y/R);
    t[1] = -cos(y/R);

    // Transform the displacements/rotations into the global ref. frame
    u_tacs[0] = u;
    u_tacs[1] = w*n[0] + v*t[0];
    u_tacs[2] = w*n[1] + v*t[1];
    u_tacs[3] = -psi_y;
    u_tacs[4] = psi_x*t[0];
    u_tacs[5] = psi_x*t[1];

    u_tacs += 6;
  }

  X->decref();
}

/*
  Set the solution to be a linearly varying solution. 

  This can be used to check the strain expressions for the cylinder.
*/
void set_tacs_linear_solution( TACSAssembler *tacs, TACSBVec *ans, 
                               TacsScalar R, double alpha, double beta,
                               TacsScalar coef[] ){
  TACSBVec *X = tacs->createNodeVec();
  X->incref();
  tacs->getNodes(X);

  // Get the x,y,z locations of the nodes from TACS
  int nnodes = tacs->getNumNodes();
  TacsScalar *Xpts;
  X->getArray(&Xpts);

  // Get the finite-element solution from TACS
  TacsScalar * u_tacs;
  ans->getArray(&u_tacs);
  
  // Set the coefficients from the array
  TacsScalar u0, ux, uy, v0, vx, vy, w0, wx, wy;
  TacsScalar psi_x0, psi_xx, psi_xy;
  TacsScalar psi_y0, psi_yx, psi_yy;

  u0 = coef[0], ux = coef[1], uy = coef[2];
  v0 = coef[3], vx = coef[4], vy = coef[5];
  w0 = coef[6], wx = coef[7], wy = coef[8];
  psi_x0 = coef[9],  psi_xx = coef[10], psi_xy = coef[11];
  psi_y0 = coef[12], psi_yx = coef[13], psi_yy = coef[14];

  for ( int node = 0; node < nnodes; node++ ){
    // Compute the u, v, w, theta, phi solution at this point within
    // the shell
    TacsScalar x = Xpts[3*node];
    TacsScalar y =-R*atan2(RealPart(Xpts[3*node+2]), 
                           RealPart(Xpts[3*node+1]));
    
    TacsScalar n[2], t[2];
    n[0] =  cos(y/R);
    n[1] = -sin(y/R);
    t[0] = -sin(y/R);
    t[1] = -cos(y/R);

    // Transform the displacements/rotations into the global ref. frame
    u_tacs[0] = u0 + x*ux + y*uy;
    u_tacs[1] = (w0 + x*wx + y*wy)*n[0] + (v0 + x*vx + y*vy)*t[0];
    u_tacs[2] = (w0 + x*wx + y*wy)*n[1] + (v0 + x*vx + y*vy)*t[1];
    u_tacs[3] = -(psi_y0 + psi_yx*x + psi_yy*y);
    u_tacs[4] = (psi_x0 + psi_xx*x + psi_xy*y)*t[0];
    u_tacs[5] = (psi_x0 + psi_xx*x + psi_xy*y)*t[1];

    u_tacs += 6;
  }

  X->decref();
}

void write_all_linear_solutions( TACSAssembler *tacs,
                                 TACSToFH5 *f5, TacsScalar R, 
                                 double alpha, double beta ){
  // Create a temporary vector
  TACSBVec *ans = tacs->createVec();
  ans->incref();

  TacsScalar coef[15];
  const char * file_names[15] = {"tacs_u0.f5", "tacs_ux.f5", "tacs_uy.f5",
                                 "tacs_v0.f5", "tacs_vx.f5", "tacs_vy.f5",
                                 "tacs_w0.f5", "tacs_wx.f5", "tacs_wy.f5",
                                 "tacs_r0.f5", "tacs_rx.f5", "tacs_ry.f5",
                                 "tacs_t0.f5", "tacs_tx.f5", "tacs_ty.f5"};

  for ( int k = 0; k < 15; k++ ){
    memset(coef, 0, 15*sizeof(TacsScalar));
    coef[k] = 1.0;

    set_tacs_linear_solution(tacs, ans, R, alpha, beta, coef);
    tacs->setVariables(ans);
    f5->writeToFile(file_names[k]);
  }

  ans->decref();
}
                                 
/*
  Print the exact solution to an ASCII file
*/
void write_exact_solution( const char * file_name, 
                           int nnx, int nny, 
                           double R, double L, 
                           double alpha, double beta,
                           double U, double V, double W, 
                           double theta, double phi ){
  FILE * fp = fopen(file_name, "w");

  if (fp){
    fprintf(fp, "Variables = X, Y, Z, u0, v0, w0, rotx, roty, rotz,");
    fprintf(fp, "ex0, ey0, exy0, ex1, ey1, exy1, eyz0, exz0\n");
    fprintf(fp, "Zone T = Cylinder, DATAPACKING = POINT, I =%d J = %d\n", nnx, nny);

    for ( int j = 0; j < nny; j++ ){
      double y = R*(-M_PI + (2.0*M_PI*j)/(nny-1));
      for ( int i = 0; i < nnx; i++ ){
        double x = L*i/(nnx - 1);
        double u = U*sin(alpha*y)*cos(beta*x);
        double v = V*cos(alpha*y)*sin(beta*x);
        double w = W*sin(alpha*y)*sin(beta*x);
        double psi_x = theta*sin(alpha*y)*cos(beta*x);
        double psi_y = phi*cos(alpha*y)*sin(beta*x);
        
        double X = x;
        double Y = R*cos(y/R);
        double Z =-R*sin(y/R);
        
        double n[2], t[2];
        n[0] = Y/R;    
        n[1] = Z/R;

        t[0] = -sin(y/R);
        t[1] = -cos(y/R);
        
        // Print the global displacements
        fprintf(fp, "%e %e %e %e %e %e %e %e %e ",
                X, Y, Z, u, w*n[0] + v*t[0], w*n[1] + v*t[1], -psi_y, 
                psi_x*t[0], psi_x*t[1]);
        
        // Print the local strains
        double ex0 = -U*beta*sin(alpha*y)*sin(beta*x);
        double ey0 = -V*alpha*sin(alpha*y)*sin(beta*x) + w/R;
        double exy0 = (alpha*U + beta*V)*cos(alpha*y)*cos(beta*x);
        
        double kx = -theta*beta*sin(alpha*y)*sin(beta*x);
        double ky = (-phi + V/R)*alpha*sin(alpha*y)*sin(beta*x) + w/(R*R);
        double kxy = (alpha*(theta + U/R) + beta*phi)*cos(alpha*y)*cos(beta*x);
        
        double gyz = (alpha*W + phi)*cos(alpha*y)*sin(beta*x) - v/R;
        double gxz = (beta*W + theta)*sin(alpha*y)*cos(beta*x);
        
        fprintf(fp, "%e %e %e %e %e %e %e %e\n",
                ex0, ey0, exy0, kx, ky, kxy, gyz, gxz);
      }
    }
    fclose(fp);
  }
}                          

/*
  Perform a grid study using a series of nx values. Write the results
  to a file.
*/
void grid_study( const char *file_name, 
		 double P, FSDTStiffness *stiffness, 
		 TacsScalar load, double R, double L, 
		 double alpha, double beta, int order, 
		 int max_quad_elev ){

  FILE * fp = fopen(file_name, "w");
  fprintf(fp, "Variables = nx, KS, DKS, IE, DE, ");
  fprintf(fp, "IE2, DE2, IP, DIP, IP2, DP2, SOC\n");

  for ( int nx = 10; nx <= 60; nx += 2 ){
    // Set the size of the finite-element mesh to use
    int ny = (int)(((2*M_PI*R)*nx)/L);

    TACSAssembler * tacs = 
      create_tacs_cylinder_model(stiffness, alpha, beta,
                                 order, nx, ny, L, R, load);
    tacs->incref();

    // Create the structural matrix/preconditioner
    FEMat *mat = tacs->createFEMat();
    TACSBVec *ans = tacs->createVec();
    TACSBVec *rhs = tacs->createVec();
    mat->incref();
    ans->incref();
    rhs->incref();
    
    // Create the preconditioner
    int lev = 4500;
    double fill = 10.0;
    int reorder_schur = 1;
    PcScMat * pc = new PcScMat(mat, lev, fill, reorder_schur);
    pc->setMonitorFactorFlag(0);
    pc->setAlltoallAssemblyFlag(1);
    
    // Create GMRES object
    int gmres_iters = 15, nrestart = 5, isflexible = 0;
    GMRES * gmres = new GMRES(mat, pc, gmres_iters, nrestart, isflexible);
    gmres->incref();
    
    // Set the GMRES tolerances
    double rtol = 1e-12, atol = 1e-30;
    gmres->setTolerances(rtol, atol);
    
    tacs->zeroVariables();
    tacs->assembleJacobian(1.0, 0.0, 0.0, rhs, mat);
    pc->factor();
    gmres->solve(rhs, ans);
    ans->scale(-1.0);
    tacs->setVariables(ans);
    
    // Set up the KS functional
    TACSKSFailure *ks_func = new TACSKSFailure(tacs, P);
    ks_func->incref();
  
    // Set up the induced function
    TACSInducedFailure *ind_func = new TACSInducedFailure(tacs, P);
    ind_func->incref();

    TACSFunction *ks = ks_func;
    TACSFunction *ind = ind_func;

    // Evaluate the KS functionals
    TacsScalar dks_tacs, ks_tacs;
    ks_func->setKSFailureType(TACSKSFailure::DISCRETE); 
    tacs->evalFunctions(&ks, 1, &dks_tacs);
    
    ks_func->setKSFailureType(TACSKSFailure::CONTINUOUS);
    tacs->evalFunctions(&ks, 1, &ks_tacs);
      
    // Evaluate the induced norms
    TacsScalar ind_exp_tacs, ind_dexp_tacs;
    ind_func->setInducedType(TACSInducedFailure::EXPONENTIAL);
    tacs->evalFunctions(&ind, 1, &ind_exp_tacs);
      
    ind_func->setInducedType(TACSInducedFailure::DISCRETE_EXPONENTIAL);
    tacs->evalFunctions(&ind, 1, &ind_dexp_tacs);
    
    TacsScalar ind_exp2_tacs, ind_dexp2_tacs;
    ind_func->setInducedType(TACSInducedFailure::EXPONENTIAL_SQUARED);
    tacs->evalFunctions(&ind, 1, &ind_exp2_tacs);
    
    ind_func->setInducedType(TACSInducedFailure::DISCRETE_EXPONENTIAL_SQUARED);
    tacs->evalFunctions(&ind, 1, &ind_dexp2_tacs);
    
    // Compute the induced power norms
    TacsScalar ind_pow_tacs, ind_dpow_tacs;
    ind_func->setInducedType(TACSInducedFailure::POWER);
    tacs->evalFunctions(&ind, 1, &ind_pow_tacs);
    
    ind_func->setInducedType(TACSInducedFailure::DISCRETE_POWER);
    tacs->evalFunctions(&ind, 1, &ind_dpow_tacs);
    
    TacsScalar ind_pow2_tacs, ind_dpow2_tacs; 
    ind_func->setInducedType(TACSInducedFailure::POWER_SQUARED);
    tacs->evalFunctions(&ind, 1, &ind_pow2_tacs);
    
    ind_func->setInducedType(TACSInducedFailure::DISCRETE_POWER_SQUARED);
    tacs->evalFunctions(&ind, 1, &ind_dpow2_tacs);

    // Print all the variables to the file
    // P, elev, KS, DKS, IE, DE,
    fprintf(fp, "%d %e %e %e %e ", nx,
            RealPart(ks_tacs), RealPart(dks_tacs), 
            RealPart(ind_exp_tacs), RealPart(ind_dexp_tacs));
      
    double SOC = 
      RealPart(ind_exp_tacs + 0.5*P*(ind_exp2_tacs - 
                                     ind_exp_tacs*ind_exp_tacs));
    
    // IE2, DE2, IP, DP, IP2, DP2, SOC\n");
    fprintf(fp, "%e %e %e %e %e %e %e \n", 
            RealPart(sqrt(ind_exp2_tacs)), 
            RealPart(sqrt(ind_dexp2_tacs)),
            RealPart(ind_pow_tacs), RealPart(ind_dpow_tacs),
            RealPart(sqrt(ind_pow2_tacs)),
            RealPart(sqrt(ind_dpow2_tacs)), SOC);

    tacs->decref();
    ans->decref();
    rhs->decref();
    gmres->decref();
    mat->decref();

    // Decrement the functions
    ks_func->decref();
    ind_func->decref();
  }

  fclose(fp);
}

int main( int argc, char * argv[] ){
  MPI_Init(&argc, &argv);

  // Set the shell geometry parameters
  double t = 1.0;
  double L = 100.0;
  double R = 100.0/M_PI;

  // Set the alpha and beta parameters
  int M = 4;
  int N = 3; // 3*pi/L

  double alpha = 4.0/R;
  double beta = 3*M_PI/L;

  // Compute the p-norm of the failure function
  int P = 20;
  for ( int k = 0; k < argc; k++ ){
    if (sscanf(argv[k], "P=%d", &P) == 1){
      if (P % 2 != 0){
        P += 1;
      }
    }
  }

  int orthotropic_flag = 1, grid_study_flag = 0;
  int nx = 20;
  int order = 3;
  for ( int k = 0; k < argc; k++ ){
    if (sscanf(argv[k], "order=%d", &order) == 1){
      if (order < 2){ order = 2; }
      if (order > 4){ order = 4; }
    }
    if (sscanf(argv[k], "nx=%d", &nx) == 1){
      if (nx < 1){ nx = 1; }
      if (nx > 200){ nx = 200; }
    }
    if (strcmp(argv[k], "isotropic") == 0){
      orthotropic_flag = 0;
    }
    if (strcmp(argv[k], "grid") == 0){
      grid_study_flag = 1;
    }
  }

  // Set the size of the finite-element mesh to use
  int ny = (int)(((2*M_PI*R)*nx)/L);

  if (order == 2){
    printf("Using a 2nd order mesh with %d x %d elements\n", nx, ny);
  }
  else if (order == 3){
    printf("Using a 3rd order mesh with %d x %d elements\n", nx, ny);
  }
  if (order == 4){
    printf("Using a 4th order mesh with %d x %d elements\n", nx, ny);
  }

  OrthoPly * ply = NULL;

  if (orthotropic_flag){
    // Set the material properties to use
    TacsScalar rho = 1.0;
    TacsScalar E1 = 100.0e3;
    TacsScalar E2 =   5.0e3;
    TacsScalar nu12 = 0.25;
    TacsScalar G12 = 10.0e3;
    TacsScalar G13 = 10.0e3;
    TacsScalar G23 = 4.0e3;

    TacsScalar Xt = 100.0;
    TacsScalar Xc = 50.0;
    TacsScalar Yt = 2.5;
    TacsScalar Yc = 10.0;
    TacsScalar S12 = 8.0;

    ply = new OrthoPly(t, rho, E1, E2, nu12, 
                       G12, G23, G13, 
                       Xt, Xc, Yt, Yc, S12);
    printf("Using orthotropic material properties: \n");
  }
  else {
    // Set the material properties to use
    TacsScalar rho = 1.0;
    TacsScalar E = 70e3;
    TacsScalar nu = 0.3;
    TacsScalar ys = 1.0;
  
    ply = new OrthoPly(t, rho, E, nu, ys);
    printf("Using isotropic material properties: \n");
  }

  ply->printProperties();

  // Create the stiffness relationship
  TacsScalar kcorr = 5.0/6.0;
  FSDTStiffness * stiffness = 
    new specialFSDTStiffness(ply, orthotropic_flag, t, kcorr);
  stiffness->incref();

  TacsScalar axis[] = {1.0, 0.0, 0.0};
  stiffness->setRefAxis(axis);
  stiffness->printStiffness();

  // 1. Compute the coefficients with a unit load
  // 2. Compute the normalized load such that ||F||_{infty} = 1.0
  // 3. Recompute the coefficients with the new load value
  TacsScalar load = 1.0;
  TacsScalar U, V, W, theta, phi;
  TacsScalar ainv = 1.0/R;
  TacsScalar A[6], B[6], D[6], As[3];
  double pt[] = {0.0, 0.0};
  stiffness->getStiffness(pt, A, B, D, As); 

  compute_coefficients(&U, &V, &W, &theta, &phi,
                       alpha, beta, ainv, 
                       A[0], A[1], A[3], A[5], 
                       D[0], D[1], D[3], D[5],
                       As[0], As[2], 1.0);

  TacsScalar Q11, Q12, Q22, Q44, Q55, Q66;
  ply->getLaminateStiffness(&Q11, &Q12, &Q22, &Q44, &Q55, &Q66);

  TacsScalar F1, F2, F11, F12, F22, F66;
  ply->getTsaiWu(&F1, &F2, &F11, &F12, &F22, &F66);

  if (!orthotropic_flag){
    TacsScalar vm_max;
    evaluate_vm_pnorm(P, 0.5*t, Q11, Q12, Q22, Q66,
                      alpha, beta, ainv, R, L, 
                      U, V, W, theta, phi, &vm_max);
    load = 1.0/vm_max;
  }
  else {
    TacsScalar tw_max;
    evaluate_tsai_wu_pnorm(P, M, N, 0.5*t, Q11, Q12, Q22, Q66,
                           F1, F2, F11, F12, F22, F66, 
                           alpha, beta, ainv, R, L, U, V, W, theta, phi,
                           &tw_max, &load);
  }

  compute_coefficients(&U, &V, &W, &theta, &phi,
                       alpha, beta, ainv, 
                       A[0], A[1], A[3], A[5], 
                       D[0], D[1], D[3], D[5],
                       As[0], As[2], load);

  if (grid_study_flag){
    char file_name[128];
    if (orthotropic_flag){
      sprintf(file_name, "ortho_grid_study_order=%d_P=%d.dat",
	      order, P);
    }
    else {
      sprintf(file_name, "iso_grid_study_order=%d_P=%d.dat",
	      order, P);
    }
    grid_study(file_name, P, stiffness,
	       load, R, L, alpha, beta, order, 8-order);
  }


  TACSAssembler * tacs = create_tacs_cylinder_model(stiffness, alpha, beta,
                                                    order, nx, ny, L, R, load);
  tacs->incref();

  // Create an TACSToFH5 object for writing output to files
  unsigned int write_flag = (TACSElement::OUTPUT_NODES |
                             TACSElement::OUTPUT_DISPLACEMENTS |
                             TACSElement::OUTPUT_STRAINS |
                             TACSElement::OUTPUT_STRESSES |
                             TACSElement::OUTPUT_EXTRAS |
                             TACSElement::OUTPUT_COORDINATES);
  TACSToFH5 * f5 = new TACSToFH5(tacs, SHELL, write_flag);
  f5->incref();

  // Create the structural matrix/preconditioner/KSM to use in conjunction
  // with the aerostructural solution
  FEMat *mat = tacs->createFEMat();
  TACSBVec *ans = tacs->createVec();
  TACSBVec *rhs = tacs->createVec();
  mat->incref();
  ans->incref();
  rhs->incref();

  // Create the preconditioner
  int lev = 4500;
  double fill = 10.0;
  int reorder_schur = 1;
  PcScMat * pc = new PcScMat(mat, lev, fill, reorder_schur);
  pc->setMonitorFactorFlag(0);
  pc->setAlltoallAssemblyFlag(1);
  
  // Create GMRES object
  int gmres_iters = 15, nrestart = 5, isflexible = 0;
  GMRES * gmres = new GMRES(mat, pc, gmres_iters, nrestart, isflexible);
  gmres->incref();

  // Set the GMRES tolerances
  double rtol = 1e-12, atol = 1e-30;
  gmres->setTolerances(rtol, atol);

  tacs->zeroVariables();
  tacs->assembleJacobian(1.0, 0.0, 0.0, rhs, mat);
  pc->factor();
  gmres->solve(rhs, ans);
  ans->scale(-1.0);
  tacs->setVariables(ans);
  f5->writeToFile("cylindrical_solution.f5");  

  // Compute the error associated with the normal displacement
  TacsScalar err = compute_tacs_error(order, tacs,
                                      R, alpha, beta, 
                                      U, V, W, theta, phi);
  printf("The error in the L2 norm is:   %15.8e\n", RealPart(err));

  char file_name[128];
  if (orthotropic_flag){
    sprintf(file_name, "ortho_cylinder_func_order=%d_nx=%d.dat", order, nx);
  }
  else {
    sprintf(file_name, "iso_cylinder_func_order=%d_nx=%d.dat", order, nx);
  }

  FILE * fp = fopen(file_name, "w");
  fprintf(fp, "Variables = P, KS, DKS, IE, DE, ");
  fprintf(fp, "IE2, DE2, IP, DIP, IP2, DP2, SOC\n");

  // Set up the KS functional
  TACSKSFailure *ks_func = new TACSKSFailure(tacs, P);
  ks_func->incref();
  
  // Set up the induced function
  TACSInducedFailure *ind_func = new TACSInducedFailure(tacs, P);
  ind_func->incref();

  TACSFunction *ks = ks_func;
  TACSFunction *ind = ind_func;

  // Evaluate the KS functionals
  TacsScalar dks_tacs, ks_tacs;
  ks_func->setKSFailureType(TACSKSFailure::DISCRETE); 
  tacs->evalFunctions(&ks, 1, &dks_tacs);
    
  ks_func->setKSFailureType(TACSKSFailure::CONTINUOUS);
  tacs->evalFunctions(&ks, 1, &ks_tacs);
      
  // Evaluate the induced norms
  TacsScalar ind_exp_tacs, ind_dexp_tacs;
  ind_func->setInducedType(TACSInducedFailure::EXPONENTIAL);
  tacs->evalFunctions(&ind, 1, &ind_exp_tacs);
      
  ind_func->setInducedType(TACSInducedFailure::DISCRETE_EXPONENTIAL);
  tacs->evalFunctions(&ind, 1, &ind_dexp_tacs);
    
  TacsScalar ind_exp2_tacs, ind_dexp2_tacs;
  ind_func->setInducedType(TACSInducedFailure::EXPONENTIAL_SQUARED);
  tacs->evalFunctions(&ind, 1, &ind_exp2_tacs);
    
  ind_func->setInducedType(TACSInducedFailure::DISCRETE_EXPONENTIAL_SQUARED);
  tacs->evalFunctions(&ind, 1, &ind_dexp2_tacs);
    
  // Compute the induced power norms
  TacsScalar ind_pow_tacs, ind_dpow_tacs;
  ind_func->setInducedType(TACSInducedFailure::POWER);
  tacs->evalFunctions(&ind, 1, &ind_pow_tacs);
    
  ind_func->setInducedType(TACSInducedFailure::DISCRETE_POWER);
  tacs->evalFunctions(&ind, 1, &ind_dpow_tacs);
    
  TacsScalar ind_pow2_tacs, ind_dpow2_tacs; 
  ind_func->setInducedType(TACSInducedFailure::POWER_SQUARED);
  tacs->evalFunctions(&ind, 1, &ind_pow2_tacs);
    
  ind_func->setInducedType(TACSInducedFailure::DISCRETE_POWER_SQUARED);
  tacs->evalFunctions(&ind, 1, &ind_dpow2_tacs);
    
  printf("TACS discrete KS function with P = %d: %15.8e\n", 
         P, fabs(RealPart(dks_tacs-1.0)));
  printf("TACS KS-functional with P = %d:        %15.8e\n", 
         P, fabs(RealPart(ks_tacs-1.0)));    
  printf("TACS exponential with P = %d:          %15.8e\n", 
         P, fabs(RealPart(ind_exp_tacs-1.0)));
  printf("TACS discrete exp with P = %d:         %15.8e\n", 
         P, fabs(RealPart(ind_dexp_tacs-1.0)));
  printf("TACS exponential2 with P = %d:         %15.8e\n", 
         P, fabs(RealPart(sqrt(ind_exp2_tacs)-1.0)));
  printf("TACS discrete exp2 with P = %d:        %15.8e\n", 
         P, fabs(RealPart(sqrt(ind_dexp2_tacs)-1.0)));
  printf("TACS pow with P = %d:                  %15.8e\n", 
         P, fabs(RealPart(ind_pow_tacs-1.0)));
  printf("TACS discrete pow with P = %d:         %15.8e\n", 
         P, fabs(RealPart(ind_dpow_tacs-1.0))); 
  printf("TACS pow2 with P = %d:                 %15.8e\n", 
         P, fabs(RealPart(sqrt(ind_pow2_tacs)-1.0)));
  printf("TACS discrete pow2 with P = %d:        %15.8e\n", 
         P, fabs(RealPart(sqrt(ind_dpow2_tacs)-1.0)));
    
  printf("Second-order correction P = %d:        %15.8e\n", P, 
	 fabs(RealPart(ind_exp_tacs + 0.5*P*(ind_exp2_tacs - 
                                             ind_exp_tacs*ind_exp_tacs)-1.0)));

  if (!orthotropic_flag){
    TacsScalar vm_max;
    TacsScalar pnorm = evaluate_vm_pnorm(P, 0.5*t, Q11, Q12, Q22, Q66,
					 alpha, beta, ainv, R, L, 
					 U, V, W, theta, phi, &vm_max);
    printf("The maximum von Mises stress:  %15.8e\n", 
           RealPart(vm_max));
    printf("Analytic top p-norm P = %d:    %15.8e\n",
           P, RealPart(pnorm));
  }
  else {
    TacsScalar tw_max, unit_load;
    TacsScalar pnorm = 
      evaluate_tsai_wu_pnorm(P, M, N, 0.5*t, Q11, Q12, Q22, Q66,
			     F1, F2, F11, F12, F22, F66, 
			     alpha, beta, ainv, R, L, U, V, W, theta, phi, 
			     &tw_max, &unit_load);
    printf("The maximum value of Tsai-Wu:  %15.8e\n", 
           RealPart(tw_max));
    printf("Analytic top p-norm P = %d:    %15.8e\n", 
           P, RealPart(pnorm));
  }

  // Decrement the functions
  ks_func->decref();
  ind_func->decref();

  ans->decref();
  rhs->decref();
  gmres->decref();
  mat->decref();
  f5->decref();
  tacs->decref();

  MPI_Finalize();
  return (0);
}
