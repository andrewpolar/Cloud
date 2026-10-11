using System;

namespace Cloud
{
    class Complex
    {
        public double re;
        public double im;
        public Complex(double re, double im)
        {
            this.re = re;
            this.im = im;
        }
    }

    internal class Helper
    {
        public static double Pearson(double[] x, double[] y)
        {
            int len = (int)x.Length;
            double xmean = 0.0;
            double ymean = 0.0;
            for (int i = 0; i < len; ++i)
            {
                xmean += x[i];
                ymean += y[i];
            }
            xmean /= len;
            ymean /= len;

            double covariance = 0.0;
            for (int i = 0; i < len; ++i)
            {
                covariance += (x[i] - xmean) * (y[i] - ymean);
            }

            double stdX = 0.0;
            double stdY = 0.0;
            for (int i = 0; i < len; ++i)
            {
                stdX += (x[i] - xmean) * (x[i] - xmean);
                stdY += (y[i] - ymean) * (y[i] - ymean);
            }
            stdX = Math.Sqrt(stdX);
            stdY = Math.Sqrt(stdY);
            return covariance / stdX / stdY;
        }

        static Complex Multiply(Complex a, Complex b)
        {
            return new Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
        }

        public static double[] GenerateComplexFeatures(Random rng, int elements, double min, double max)
        {
            double[] x = new double[elements * 2];
            int cnt = 0;
            for (int j = 0; j < elements; ++j)
            {
                x[cnt++] = rng.NextDouble() * (max - min) + min;
                x[cnt++] = rng.NextDouble() * (max - min) + min;
            }
            return x;
        }

        public static double[] ComputeComplexTargets(double[] V, double errorRate, Random rng)
        {
            int size = V.Length;
            double[] noisyData = new double[size];
            double mean = 0.0;
            for (int j = 0; j < size; ++j)
            {
                noisyData[j] = rng.NextDouble();
                mean += noisyData[j];
            }
            mean /= (size);
            for (int j = 0; j < size; ++j)
            {
                noisyData[j] -= mean;
                noisyData[j] *= errorRate;
            }
            for (int j = 0; j < size; ++j)
            {
                noisyData[j] += V[j];
            }

            int elements = size / 2;
            double[] x = new double[2];

            // Multiplicative identity: 1 + 0i
            Complex product = new Complex(1.0, 0.0);

            for (int j = 0; j < elements; ++j)
            {
                Complex z = new Complex(noisyData[2 * j], noisyData[2 * j + 1]);
                product = Multiply(product, z);
            }
            x[0] = product.re;
            x[1] = product.im;
            return x;
        }

        public static void ShowMatrix(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    Console.Write("{0:0.0000} ", matrix[i, j]);
                }
                Console.WriteLine();
            }
        }

        public static void ShowMatrix(double[][] matrix)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    Console.Write("{0:0.0000} ", matrix[i][j]);
                }
                Console.WriteLine();
            }
        }

        public static void ShowMatrix(int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            int width = 0;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    width = Math.Max(width, matrix[i, j].ToString().Length);
                }
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Console.Write("| {0," + width + "} ", matrix[i, j]);
                }
                Console.WriteLine("|");
            }
        }

        public static int[,] Make2DHistogram(double[][] dataMatrix, int xBuckets, int yBuckets)
        {
            int n = dataMatrix.Length;

            // Find ranges
            double xMin = double.MaxValue;
            double xMax = double.MinValue;
            double yMin = double.MaxValue;
            double yMax = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                double x = dataMatrix[i][0];
                double y = dataMatrix[i][1];

                if (x < xMin) xMin = x;
                if (x > xMax) xMax = x;
                if (y < yMin) yMin = y;
                if (y > yMax) yMax = y;
            }

            if (xMin == xMax || yMin == yMax)
                throw new ArgumentException("Wrong histogram data.");

            double xWidth = (xMax - xMin) / xBuckets;
            double yWidth = (yMax - yMin) / yBuckets;

            int[,] counts = new int[xBuckets, yBuckets];

            // Count points in each rectangle
            for (int i = 0; i < n; i++)
            {
                double x = dataMatrix[i][0];
                double y = dataMatrix[i][1];

                int ix = (int)((x - xMin) / xWidth);
                int iy = (int)((y - yMin) / yWidth);

                // Include the maximum value in the last bucket
                if (ix == xBuckets) ix = xBuckets - 1;
                if (iy == yBuckets) iy = yBuckets - 1;

                counts[ix, iy]++;
            }
            return counts;
        }

        private static double SquaredDistance(double[] a, double[] b)
        {
            double sum = 0;

            for (int i = 0; i < a.Length; i++)
            {
                double d = a[i] - b[i];
                sum += d * d;
            }

            return sum;
        }

        public static int[] KNearest(double[][] matrix, double[] vector, int n)
        {
            if (matrix == null || matrix.Length == 0 || n <= 0)
                return Array.Empty<int>();

            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            int dimensions = vector.Length;

            if (matrix.Any(row => row == null || row.Length != dimensions))
                throw new ArgumentException(
                    "All matrix rows must have the same length as vector.");

            n = Math.Min(n, matrix.Length);

            return matrix
                .Select((row, index) => new
                {
                    Index = index,
                    Distance = SquaredDistance(row, vector)
                })
                .OrderBy(x => x.Distance)
                .Take(n)
                .Select(x => x.Index)
                .ToArray();
        }

        public static double EnergyDistance(double[][] montecarlo, double[][] sample)
        {
            ValidateMatrix(montecarlo, nameof(montecarlo));
            ValidateMatrix(sample, nameof(sample));

            if (montecarlo[0].Length != sample[0].Length)
            {
                throw new ArgumentException(
                    "Both matrices must have the same number of columns.");
            }

            double cross = AveragePairwiseDistance(
                montecarlo, sample);

            double withinPopulation = AveragePairwiseDistance(
                montecarlo, montecarlo);

            double withinSample = AveragePairwiseDistance(
                sample, sample);

            double result =
                2.0 * cross
                - withinPopulation
                - withinSample;

            // Remove tiny negative values caused by floating-point error.
            if (result < 0.0 && result > -1e-12)
                result = 0.0;

            return result;
        }

        private static double AveragePairwiseDistance(
            double[][] A,
            double[][] B)
        {
            double sum = 0.0;
            long count = (long)A.Length * B.Length;

            for (int i = 0; i < A.Length; i++)
            {
                for (int j = 0; j < B.Length; j++)
                {
                    sum += EuclideanDistance(A[i], B[j]);
                }
            }

            return sum / count;
        }

        private static double EuclideanDistance(
            double[] a,
            double[] b)
        {
            double sumSquares = 0.0;

            for (int k = 0; k < a.Length; k++)
            {
                double diff = a[k] - b[k];
                sumSquares += diff * diff;
            }

            return Math.Sqrt(sumSquares);
        }

        private static void ValidateMatrix(
            double[][] matrix,
            string parameterName)
        {
            if (matrix == null || matrix.Length == 0)
            {
                throw new ArgumentException(
                    "Matrix must not be null or empty.",
                    parameterName);
            }

            if (matrix[0] == null || matrix[0].Length == 0)
            {
                throw new ArgumentException(
                    "Matrix rows must not be empty.",
                    parameterName);
            }

            int columns = matrix[0].Length;

            for (int i = 0; i < matrix.Length; i++)
            {
                if (matrix[i] == null ||
                    matrix[i].Length != columns)
                {
                    throw new ArgumentException(
                        "Matrix must be rectangular.",
                        parameterName);
                }

                for (int j = 0; j < columns; j++)
                {
                    if (double.IsNaN(matrix[i][j]) ||
                        double.IsInfinity(matrix[i][j]))
                    {
                        throw new ArgumentException(
                            "Matrix values must be finite.",
                            parameterName);
                    }
                }
            }
        }
    }
}
