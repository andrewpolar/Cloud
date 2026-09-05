using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cloud
{
    internal class MDistance2D
    {
        class Node
        {
            public double[][] Points;
            public int Depth;
            public int SplitDimension;
            public int Path; // 0 = left, 1 = right
        }

        static public double MedianTreeDistance(double[][] sampleA, double[][] sampleB, int minPoints, bool print)
        {
            int dimensions = sampleA[0].Length;

            List<double> allDistances = new List<double>();
            Dictionary<int, List<double>> distancesByDepth = new Dictionary<int, List<double>>();

            List<Node> current = new List<Node>();

            current.Add(new Node
            {
                Points = sampleA,
                Depth = 0,
                SplitDimension = 0,
                Path = 0
            });

            List<Node> currentB = new List<Node>();

            currentB.Add(new Node
            {
                Points = sampleB,
                Depth = 0,
                SplitDimension = 0,
                Path = 0
            });

            int depth = 0;

            while (current.Count > 0)
            {
                List<Node> nextA = new List<Node>();
                List<Node> nextB = new List<Node>();

                int splitDimension = depth % dimensions;

                for (int k = 0; k < current.Count; k++)
                {
                    double[][] a = current[k].Points;
                    double[][] b = currentB[k].Points;

                    if (a.Length < 2 || b.Length < 2)
                        continue;

                    if (a.Length < minPoints ||
                        b.Length < minPoints)
                        continue;

                    // Find median separator in each sample
                    double medianA = Median(a, splitDimension);
                    double medianB = Median(b, splitDimension);

                    double distance = Math.Abs(medianA - medianB);

                    allDistances.Add(distance);

                    if (!distancesByDepth.ContainsKey(depth))
                        distancesByDepth[depth] = new List<double>();

                    distancesByDepth[depth].Add(distance);

                    if (print)
                    {
                        Console.WriteLine(
                            $"Depth {depth}, " +
                            $"dim {splitDimension}: " +
                            $"A median = {medianA:F6}, " +
                            $"B median = {medianB:F6}, " +
                            $"distance = {distance:F6}");
                    }

                    // Split both samples at their own median
                    Split(
                        a,
                        splitDimension,
                        medianA,
                        out double[][] aLeft,
                        out double[][] aRight);

                    Split(
                        b,
                        splitDimension,
                        medianB,
                        out double[][] bLeft,
                        out double[][] bRight);

                    // Corresponding left regions
                    if (aLeft.Length >= minPoints &&
                        bLeft.Length >= minPoints)
                    {
                        nextA.Add(new Node
                        {
                            Points = aLeft,
                            Depth = depth + 1,
                            SplitDimension = (depth + 1) % dimensions,
                            Path = 0
                        });

                        nextB.Add(new Node
                        {
                            Points = bLeft,
                            Depth = depth + 1,
                            SplitDimension = (depth + 1) % dimensions,
                            Path = 0
                        });
                    }

                    // Corresponding right regions
                    if (aRight.Length >= minPoints &&
                        bRight.Length >= minPoints)
                    {
                        nextA.Add(new Node
                        {
                            Points = aRight,
                            Depth = depth + 1,
                            SplitDimension = (depth + 1) % dimensions,
                            Path = 1
                        });

                        nextB.Add(new Node
                        {
                            Points = bRight,
                            Depth = depth + 1,
                            SplitDimension = (depth + 1) % dimensions,
                            Path = 1
                        });
                    }
                }

                current = nextA;
                currentB = nextB;

                depth++;
            }

            if (print)
            {
                Console.WriteLine();
                Console.WriteLine("Average distance by depth:");

                foreach (var item in distancesByDepth)
                {
                    double avg = item.Value.Average();

                    Console.WriteLine(
                        $"Depth {item.Key}: " +
                        $"count = {item.Value.Count}, " +
                        $"average = {avg:F6}");
                }
            }

            return allDistances.Count == 0 ? 0.0 : allDistances.Average();
        }

        static private double Median(double[][] points, int dimension)
        {
            double[] values = points
                .Select(p => p[dimension])
                .OrderBy(v => v)
                .ToArray();

            int n = values.Length;

            if (n % 2 == 1)
                return values[n / 2];

            return (values[n / 2 - 1] + values[n / 2]) / 2.0;
        }

        static private void Split(double[][] points, int dimension, double median, out double[][] left, out double[][] right)
        {
            List<double[]> l = new List<double[]>();
            List<double[]> r = new List<double[]>();

            foreach (double[] p in points)
            {
                if (p[dimension] <= median)
                    l.Add(p);
                else
                    r.Add(p);
            }

            // Protection against pathological duplicate values.
            // If everything went to one side, split by sorted position instead.
            if (l.Count == 0 || r.Count == 0)
            {
                double[][] sorted = points
                    .OrderBy(p => p[dimension])
                    .ToArray();

                int half = sorted.Length / 2;

                l = sorted.Take(half).ToList();
                r = sorted.Skip(half).ToList();
            }

            left = l.ToArray();
            right = r.ToArray();
        }
    }
}
