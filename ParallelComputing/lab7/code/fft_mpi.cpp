# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <mpi.h>

using namespace std;

int main ( int argc, char *argv[] );
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[], double sgn );
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
  double w[], double sgn );
void timestamp ( );
void local_step ( int n, int mj, double x[], double y[], double w[], double sgn, int rank, int chunk_size );
void merge_blocks(double x[], double y[], int mj, int chunk_size, double w[], double sgn);
void distributed_cfft2(int n, double x[], double y[], double w[], double sgn, int rank, int size, MPI_Datatype MPI_MyComplex);

//****************************************************************************80

int main ( int argc, char *argv[] )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for FFT_MPI.
//
//  Discussion:
//
//    The complex data in an N vector is stored as pairs of values in a
//    real vector of length 2*N.
//
{
  double ctime;
  double ctime1;
  double ctime2;
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w;
  double *x;
  double *y;
  double *z;
  double z0;
  double z1;
  
  // MPI variables
  int rank, size;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Create MPI Datatype for complex numbers (pairs of doubles)
  MPI_Datatype MPI_MyComplex;
  MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_MyComplex);
  MPI_Type_commit(&MPI_MyComplex);

  if (rank == 0) {
    timestamp();
    cout << "\n";
    cout << "FFT_MPI\n";
    cout << "  C++ version with MPI\n";
    cout << "\n";
    cout << "  Demonstrate a parallel implementation of the Fast Fourier Transform\n";
    cout << "  of a complex data vector.\n";
    cout << "\n";
    cout << "  Accuracy check:\n";
    cout << "\n";
    cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
    cout << "\n";
    cout << "             N      NITS    Error         Time          Time/Call     MFLOPS    Processes\n";
    cout << "\n";
  }

  seed = 331.0;
  n = 1;
  
  // LN2 is the log base 2 of N. Each increase of LN2 doubles N.
  for (ln2 = 1; ln2 <= 20; ln2++) {
    n = 2 * n;
    
    // Skip problem sizes that are too small for the number of processes
    if (n < size || n % size != 0) {
      if (rank == 0) {
        cout << "  Skipping N = " << n << " as it's too small for " << size << " processes\n";
      }
      continue;
    }
    
    // Allocate storage for the complex arrays W, X, Y, Z
    w = new double[2*n]; //保证w空间足够
    x = new double[2*n];
    y = new double[2*n];
    z = new double[2*n];

    if (rank == 0) {
      for (int i = 0; i < 2*n; i++) {
        x[i] = 0.0;
        y[i] = 0.0;
        z[i] = 0.0;
      }
    }

    int chunk_size = n / size;
    double* local_x = new double[2*chunk_size];
    double* local_y = new double[2*chunk_size];
    double* local_z = new double[2*chunk_size];
    for (int i = 0; i < 2*chunk_size; i++) {
      local_x[i] = 0.0;
      local_y[i] = 0.0;
      local_z[i] = 0.0;
    }

    // Initialize sine and cosine tables
    cffti(n, w);
    first = 1;
    MPI_Bcast(w, n, MPI_MyComplex, 0, MPI_COMM_WORLD);

    for (icase = 0; icase < 2; icase++) {
      // Initialize data
      if (rank == 0) {
        if (first) {
          for (i = 0; i < 2 * n; i = i + 2) {
            z0 = ggl(&seed);
            z1 = ggl(&seed);
            x[i] = z0;
            z[i] = z0;
            x[i+1] = z1;
            z[i+1] = z1;
          }
        } else {
          for (i = 0; i < 2 * n; i = i + 2) {
            z0 = 0.0;
            z1 = 0.0;
            x[i] = z0;
            z[i] = z0;
            x[i+1] = z1;
            z[i+1] = z1;
          }
        }
      }

      MPI_Scatter(x, chunk_size, MPI_MyComplex, local_x, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);
      
      // Transform forward, back
      if (first) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Forward transform
        sgn = +1.0;
        
        // Each process works on its chunk
        for (i = 0; i < 2*chunk_size; i++) {
          local_y[i] = local_x[i];
        }
        
        // Gather results from all processes
        MPI_Gather(local_y, chunk_size, MPI_MyComplex,
                     y, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);
        
        // Perform FFT on the gathered data
        if (rank == 0) {
          cfft2(n, x, y, w, sgn);
        }
        // distributed_cfft2(n, local_x, local_y, w, sgn, rank, size, MPI_MyComplex);
        // MPI_Gather(local_x, chunk_size, MPI_MyComplex, y, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);

        MPI_Scatter(y, chunk_size, MPI_MyComplex, local_y, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);
        // Backward transform
        sgn = -1.0;
        
        // Each process works on its chunk for backward transform
        for (i = 0; i < 2*chunk_size; i++) {
          local_x[i] = local_y[i];
        }
        
        // Gather results for backward transform
        MPI_Gather(local_x, chunk_size, MPI_MyComplex,
                     x, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);
        // Perform FFT on the gathered data
        if (rank == 0) {
          cfft2(n, y, x, w, sgn);

        // Results should be same as initial multiplied by N
          fnm1 = 1.0 / (double)n;
          error = 0.0;
          for (i = 0; i < 2 * n; i = i + 2) {
            error = error 
              + pow(z[i] - fnm1 * x[i], 2)
              + pow(z[i+1] - fnm1 * x[i+1], 2);
          }
          error = sqrt(fnm1 * error);
          cout << "  " << setw(12) << n
               << "  " << setw(8) << nits
               << "  " << setw(12) << error;
        }
        first = 0;
      } else {
        // // Option using MPI_Pack and MPI_Unpack for data transmission
        // double *packed_buffer = new double[4*n]; // Buffer for packed data
        // int position;
        
        // Performance test with timing
        MPI_Barrier(MPI_COMM_WORLD);
        ctime1 = MPI_Wtime();
        
        for (it = 0; it < nits; it++) {
          
          // Forward transform
          sgn = +1.0;
          
          MPI_Scatter(x, chunk_size, MPI_MyComplex, local_x, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);
          if (rank == 0) {
            cfft2(n, x, y, w, sgn);
          }
          MPI_Scatter(y, chunk_size, MPI_MyComplex, local_y, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);
          // Backward transform
          sgn = -1.0;

          MPI_Gather(local_y, chunk_size, MPI_MyComplex,
                     y, chunk_size, MPI_MyComplex, 0, MPI_COMM_WORLD);
          if (rank == 0) {
            cfft2(n, y, x, w, sgn);
          }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        ctime2 = MPI_Wtime();
        ctime = ctime2 - ctime1;
        
        // Calculate and display performance metrics (only on rank 0)
        if (rank == 0) {
          flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);
          mflops = flops / 1.0E+06 / ctime;
          
          cout << "  " << setw(12) << ctime
               << "  " << setw(12) << ctime / (double)(2 * nits)
               << "  " << setw(12) << mflops 
               << "  " << setw(8) << size << "\n";
        }
      
      }
    }
    
    if ((ln2 % 4) == 0) {
      nits = nits / 10;
    }
    if (nits < 1) {
      nits = 1;
    }
    
    delete [] w;
    delete [] x;
    delete [] y;
    delete [] z;
    delete [] local_x;
    delete [] local_y;
    delete [] local_z;
  }

  if (rank == 0) {
    cout << "\n";
    cout << "FFT_MPI:\n";
    cout << "  Normal end of execution.\n";
    cout << "\n";
    timestamp();
  }

  MPI_Type_free(&MPI_MyComplex);
  MPI_Finalize();
  return 0;
}

//****************************************************************************80

void ccopy(int n, double x[], double y[])

//****************************************************************************80
//
//  Purpose:
//
//    CCOPY copies a complex vector.
//
//  Discussion:
//
//    The "complex" vector A[N] is actually stored as a double vector B[2*N].
//
//    The "complex" vector entry A[I] is stored as:
//
//      B[I*2+0], the real part,
//      B[I*2+1], the imaginary part.
//
{
  int i;

  for (i = 0; i < n; i++) {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
  }
  return;
}

//****************************************************************************80

void cfft2(int n, double x[], double y[], double w[], double sgn)

//****************************************************************************80
//
//  Purpose:
//
//    CFFT2 performs a complex Fast Fourier Transform.
//
{
  int j;
  int m;
  int mj;
  int tgle;

  m = (int)(log((double)n) / log(1.99));
  mj = 1;
  
  // Toggling switch for work array.
  tgle = 1;
  step(n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn);

  if (n == 2) {
    return;
  }

  for (j = 0; j < m - 2; j++) {
    mj = mj * 2;
    if (tgle) {
      step(n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn);
      tgle = 0;
    } else {
      step(n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn);
      tgle = 1;
    }
  }
  
  // Last pass thru data: move y to x if needed 
  if (tgle) {
    ccopy(n, y, x);
  }

  mj = n / 2;
  step(n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn);

  return;
}

//****************************************************************************80

void cffti(int n, double w[])

//****************************************************************************80
//
//  Purpose:
//
//    CFFTI sets up sine and cosine tables needed for the FFT calculation.
//
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ((double)n);

  for (i = 0; i < n2; i++) {
    arg = aw * ((double)i);
    w[i*2+0] = cos(arg);
    w[i*2+1] = sin(arg);
  }
  return;
}

//****************************************************************************80

double cpu_time(void)

//****************************************************************************80
//
//  Purpose:
// 
//    CPU_TIME reports the elapsed CPU time.
//
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

//****************************************************************************80

double ggl(double *seed)

//****************************************************************************80
//
//  Purpose:
//
//    GGL generates uniformly distributed pseudorandom numbers.
//
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = *seed;
  t = fmod(16807.0 * t, d2);
  *seed = t;
  value = (t - 1.0) / (d2 - 1.0);

  return value;
}

//****************************************************************************80

void step(int n, int mj, double a[], double b[], double c[],
  double d[], double w[], double sgn)

//****************************************************************************80
//
//  Purpose:
//
//    STEP carries out one step of the workspace version of CFFT2.
//
{
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj = n / mj2;

  for (j = 0; j < lj; j++) {
    jw = j * mj;
    ja = jw;
    jb = ja;
    jc = j * mj2;
    jd = jc;

    wjw[0] = w[jw*2+0]; 
    wjw[1] = w[jw*2+1];

    if (sgn < 0.0) {
      wjw[1] = -wjw[1];
    }

    for (k = 0; k < mj; k++) {
      c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
      c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

      ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
      ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

      d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}

//****************************************************************************80

void timestamp()

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}

void distributed_cfft2(int n, double x[], double y[], double w[], double sgn, int rank, int size, MPI_Datatype MPI_MyComplex) {
    int m = (int)(log((double)n) / log(1.99));
    int mj = 1;
    int chunk_size = n / size;

    for (int level = 0; level < m; level++) {
        if (mj > chunk_size) break;
        local_step(n, mj, x, y, w, sgn, rank, chunk_size);

        int partner = rank ^ (1 << level);
        if (partner >= size) continue;

        // 发送和接收chunk_size个复数（2*chunk_size个double）
        MPI_Sendrecv(
            x + mj * 2, 
            2 * chunk_size,  // 发送元素数量（double数量）
            MPI_DOUBLE, 
            partner, 0,
            y, 
            2 * chunk_size,  // 接收元素数量
            MPI_DOUBLE, 
            partner, 0,
            MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE
        );

        merge_blocks(x, y, mj, chunk_size, w, sgn);
        mj *= 2;
    }
}

void local_step(int n, int mj, double x[], double y[], double w[], double sgn, int rank, int chunk_size) {
    // 本地蝶形计算
    for (int i = 0; i < chunk_size; i += 2*mj) {
        for (int k = 0; k < mj; k++) {
            int idx1 = i + k;
            int idx2 = i + k + mj;
            // 蝶形操作实现
            double real1 = x[idx1*2];
            double imag1 = x[idx1*2+1];
            double real2 = x[idx2*2];
            double imag2 = x[idx2*2+1];
            
            // 应用旋转因子
            double angle = w[k*2] * sgn;
            double wr = cos(angle);
            double wi = sin(angle);
            
            // 计算新的值
            double temp_real = wr * real2 - wi * imag2;
            double temp_imag = wr * imag2 + wi * real2;
            
            x[idx1*2] = real1 + temp_real;
            x[idx1*2+1] = imag1 + temp_imag;
            x[idx2*2] = real1 - temp_real;
            x[idx2*2+1] = imag1 - temp_imag;
        }
    }
}

void merge_blocks(double x[], double y[], int mj, int chunk_size, double w[], double sgn) {
    // 合并接收到的数据块
    for (int i = 0; i < chunk_size; i++) {
        if (mj + i >= chunk_size) break;
        x[(chunk_size + i)*2] = y[i*2];
        x[(chunk_size + i)*2+1] = y[i*2+1];
    }
}