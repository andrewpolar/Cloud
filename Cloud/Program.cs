// ============================================================================
// KANKAN (Kolmogorov-Arnold Networks KAczmarz-Newton)
//
// Concept: Andrew Polar, Mike Poluektov
// Theory and algorithms are described in peer-reviewed publications.
// https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
// https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
// https://link.springer.com/article/10.1007/s10994-025-06800-6
// https://www.mdpi.com/2673-3951/6/3/88
// https://www.sciencedirect.com/science/article/pii/S0925231226021703
// Additional information, examples and documentation are available at
// OpenKAN.org.
//
// ---------------------------------------------------------------------------
// License
//
// If an end user somehow earns billions of US dollars using this software
// and later encounters the developer asking for spare change outside a
// McDonald's, the end user is under no obligation to buy the developer
// a sandwich.
//
// Symmetry Clause
//
// Likewise, if the developer becomes rich and famous by publishing this
// software and later meets an unfortunate end user who went bankrupt while
// using it, the developer is also under no obligation to buy the end user
// a sandwich.
//----------------------------------------------------------------------------

// The model is Kolmogorov-Arnold networks (KAN). 
// The feautes are Re and Im of several complex numbers, the target is their product. 
// The noise is added to features before computing the targets. 
// The code compares kNN and DDR with vector target and joint distribution of targets
// in aleatoric uncertainty. The goodness of fit test for 2D joint distribution is unique,
// it is introduced by Andrew Polar and Mike Poluektov, published in
// https://www.mdpi.com/2673-3951/6/3/88

using Cloud;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;


namespace Cloud
{
    internal class Program
    {
        static bool ValidateDeterministicModel(KAN kan, double[][] features, double[][] targets, double threshold)
        {
            int nRecords = features.Length;
            int nTargets = targets[0].Length;
            double[][] predictions = new double[nTargets][];
            double[][] observations = new double[nTargets][];

            for (int target = 0; target < nTargets; ++target)
            {
                predictions[target] = new double[nRecords];
                observations[target] = new double[nRecords];
            }

            for (int record = 0; record < nRecords; ++record)
            {
                kan.Predict(features[record]);
                double[] prediction = kan.GetPrediction();
                for (int target = 0; target < nTargets; ++target)
                {
                    predictions[target][record] = prediction[target];
                    observations[target][record] = targets[record][target];
                }
            }

            bool passed = true;
            Console.Write("Pearsons for validation:");
            for (int target = 0; target < nTargets; ++target)
            {
                double pearson = Helper.Pearson(predictions[target], observations[target]);
                Console.Write($" {pearson:0.0000}");
                if (pearson < threshold)
                {
                    passed = false;
                }
            }
            Console.WriteLine();
            return passed;
        }

        public static (double min, double max) FindMatrixMinMax(double[][] m)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            foreach (double[] row in m)
            {
                foreach (double x in row)
                {
                    if (x < min) min = x;
                    if (x > max) max = x;
                }
            }
            return (min, max);
        }

        static int GetNumberOfBuckets(int N, int min)
        {
            if (N <= 0 || min <= 0) return 0;
            int depth = 0;
            int r = N;
            while ((r = r / 2) >= min) ++depth;
            int buckets = 1;
            for (int i = 0; i < depth; ++i) buckets *= 2;
            return buckets;
        }

        static void DeterministicModel(int nFeatures, int nHidden, int nTargets, int nDisjoints,
            double featureMin, double featureMax, double targetMin, double targetMax, double alpha, Random rng,
            int nLoops, int nDisjointSize, int nTrainingRecords,
            double[][] features, double[][] targets, double[][] features_calibration, double[][] targets_calibration,
            double threshold)
        {
            /////////////////////////////////
            //training of deterministic model
            /////////////////////////////////
            Console.WriteLine("Training deterministic model");
            Console.WriteLine("----------------------------");
            Stopwatch sw = Stopwatch.StartNew();
            List<int> network = new List<int>() { nFeatures, nHidden, nTargets };
            List<int> points = new List<int> { 4, 16 };

            KAN[] kans = new KAN[nDisjoints];

            kans[0] = new KAN(network, points, featureMin, featureMax, targetMin,
                targetMax, targetMin, targetMax, alpha, rng);

            for (int m = 1; m < nDisjoints; ++m)
            {
                kans[m] = new KAN(kans[0]);
            }

            for (int loop = 0; loop < nLoops; ++loop)
            {
                Parallel.For(0, nDisjoints, disjoint =>
                {
                    int start = ((loop * nDisjoints + disjoint) * nDisjointSize) % nTrainingRecords;
                    for (int i = 0; i < nDisjointSize; ++i)
                    {
                        int idx = (start + i) % nTrainingRecords;
                        kans[disjoint].Forward(features[idx]);
                        kans[disjoint].Update(targets[idx]);
                    }
                });

                // Merge: accumulate all disjoints into kans[0]
                for (int m = 1; m < nDisjoints; ++m)
                {
                    kans[0].Accumulate(kans[m]);
                }

                // Average
                kans[0].Scale(1.0 / nDisjoints);

                // Copy averaged KAN back to all disjoints
                for (int m = 1; m < nDisjoints; ++m)
                {
                    kans[m] = new KAN(kans[0]);
                }

                Console.WriteLine($"Loop {loop + 1} of {nLoops}");
            }

            bool passed = ValidateDeterministicModel(kans[0], features_calibration, targets_calibration, threshold);
            sw.Stop();
            Console.WriteLine($"Time: {sw.Elapsed.TotalMilliseconds:F3} ms\n");
            if (passed)
            {
                Console.WriteLine("The model uncertainty is acceptable, no need to use stochastic model");
            }
        }

        static void kNNGoodnessOfFitTest(double[][] features, double[][] features_calibration, double[][] targets, 
            double uncertainty, Random rng, int sampleSize, int minMedianBlock, int nCalibrationRecords, 
            int monteCarloSize, int monteCarloSamples)
        {
            ///////////////////////////////////////
            //kNN goodness of fit by 2D median tree
            ///////////////////////////////////////
            int b = GetNumberOfBuckets(sampleSize, minMedianBlock);
            Console.WriteLine("Goodness of fit test by 2D median tree, sample size {0}, min in buckets {1}, {2} buckets",
                sampleSize, minMedianBlock, b);
            int passedTests = 0;
            double meanED = 0.0;
            for (int record = 0; record < nCalibrationRecords; ++record)
            {
                double[][] montecarlo = new double[monteCarloSize][];
                for (int i = 0; i < monteCarloSize; ++i)
                {
                    montecarlo[i] = (double[])Helper.ComputeComplexTargets(features_calibration[record], uncertainty, rng).Clone();
                }

                List<double> MCgoodness = new List<double>();
                for (int k = 0; k < monteCarloSamples; ++k)
                {
                    double[][] MCsample = new double[sampleSize][];
                    for (int i = 0; i < sampleSize; ++i)
                    {
                        MCsample[i] = (double[])Helper.ComputeComplexTargets(features_calibration[record], uncertainty, rng).Clone();
                    }
                    MCgoodness.Add(MDistance2D.MedianTreeDistance(montecarlo, MCsample, minMedianBlock, false));
                }
                MCgoodness.Sort();

                int[] knn_records = Helper.KNearest(features, features_calibration[record], sampleSize);
                double[][] knn = new double[knn_records.Length][];
                for (int i = 0; i < knn_records.Length; ++i)
                {
                    knn[i] = (double[])targets[knn_records[i]].Clone();
                }
                double kNNGoodness = MDistance2D.MedianTreeDistance(knn, montecarlo, minMedianBlock, false);
                double edGoodness = Helper.EnergyDistance(montecarlo, knn);
                meanED += edGoodness;
                if (MCgoodness[MCgoodness.Count - 1] > kNNGoodness) ++passedTests;
                Console.WriteLine("test = {0} of {1}, MC goodness = {2:0.0000}, knn goodness {3:0.0000}, passed = {4}, energy = {5:0.0000}",
                    record + 1, nCalibrationRecords, MCgoodness[MCgoodness.Count - 1], kNNGoodness, passedTests, edGoodness);
            }
            meanED /= nCalibrationRecords;
            Console.WriteLine("Mean energy distance for kNN = {0:0.0000}", meanED);
        }

        static void DDRGoognessOfFitTest(double[][] features, double[][] features_calibration, double[][] targets,
            int nHidden, double uncertainty, int sampleSize, int minMedianBlock, int nCalibrationRecords,
            int monteCarloSize, int monteCarloSamples, double featureMin, double featureMax,
            double targetMin, double targetMax, double alpha, Random rng)
        {
            ///////////////////////////////////////
            //DDR goodness of fit by 2D median tree
            ///////////////////////////////////////
            int nFeatures = features[0].Length;
            int nTargets = targets[0].Length;
            
            Stopwatch sw = Stopwatch.StartNew();

            List<int> network = new List<int>() { nFeatures, nHidden, nTargets };
            List<int> points = new List<int> { 4, 8 };

            KAN kan = new KAN(network, points, featureMin, featureMax, targetMin,
                targetMax, targetMin, targetMax, alpha, rng);

            var modeldata = new ModelData(kan, features, targets);
            var holder = new ModelDataHolder(modeldata);
            holder.SplitAll(50);
            Console.WriteLine("\n{0} blocks are created", holder.GetSize());

            sw.Stop();
            Console.WriteLine($"Time for building ensemble: {sw.Elapsed.TotalMilliseconds:F3} ms\n");

            int passedTests = 0;
            double meanED = 0.0;
            for (int record = 0; record < nCalibrationRecords; ++record)
            {
                double[][] montecarlo = new double[monteCarloSize][];
                for (int i = 0; i < monteCarloSize; ++i)
                {
                    montecarlo[i] = (double[])Helper.ComputeComplexTargets(features_calibration[record], uncertainty, rng).Clone();
                }

                List<double> MCgoodness = new List<double>();
                for (int k = 0; k < monteCarloSamples; ++k)
                {
                    double[][] MCsample = new double[sampleSize][];
                    for (int i = 0; i < sampleSize; ++i)
                    {
                        MCsample[i] = (double[])Helper.ComputeComplexTargets(features_calibration[record], uncertainty, rng).Clone();
                    }
                    MCgoodness.Add(MDistance2D.MedianTreeDistance(montecarlo, MCsample, minMedianBlock, false));
                }
                MCgoodness.Sort();

                double[][] sample = holder.GetPredictedSamples(features_calibration[record]);
                double DDRGoodness = MDistance2D.MedianTreeDistance(sample, montecarlo, minMedianBlock, false);
                double edGoodness = Helper.EnergyDistance(montecarlo, sample);
                meanED += edGoodness;

                if (MCgoodness[MCgoodness.Count - 1] > DDRGoodness) ++passedTests;
                Console.WriteLine("test = {0} of {1}, MC goodness = {2:0.0000}, DDR goodness {3:0.0000}, passed = {4}, energy = {5:0.0000}",
                record + 1, nCalibrationRecords, MCgoodness[MCgoodness.Count - 1], DDRGoodness, passedTests, edGoodness);
            }
            meanED /= nCalibrationRecords;
            Console.WriteLine("Mean energy distance for DDR = {0:0.0000}", meanED);
        }

        static void Main(string[] args)
        {
            //parameters for datasets
            const int nTrainingRecords = 20000;
            const int nCalibrationRecords = 100;
            const int nNumbers = 2;
            const int nFeatures = nNumbers * 2;
            const int nTargets = 2;
            const double featureMin = -1.0;
            const double featureMax = 1.0;
            const double relativeNoiseLevel = 0.5; //when 0.0 the data is exact
            const double uncertainty = (featureMax - featureMin) * relativeNoiseLevel;
            const double threshold = 0.9;

            //parameters for training deterministic model
            const double alpha = 0.1;
            const int nDisjoints = 4;
            const int nDisjointSize = 3000;
            const int nLoops = 8;

            const int nHidden = 8;
            const int sampleSize = 128;
            const int monteCarloSize = 32000;
            const int monteCarloSamples = 100;
            const int minMedianBlock = 3;

            /////////////////
            //data generation
            /////////////////
            Random rng = new Random();
            double[][] features = new double[nTrainingRecords][];
            double[][] targets = new double[nTrainingRecords][];
            for (int record = 0; record < nTrainingRecords; ++record)
            {
                features[record] = Helper.GenerateComplexFeatures(rng, nNumbers, featureMin, featureMax);
                targets[record] = Helper.ComputeComplexTargets(features[record], uncertainty, rng);
            }

            double[][] features_calibration = new double[nCalibrationRecords][];
            double[][] targets_calibration = new double[nCalibrationRecords][];
            for (int record = 0; record < nCalibrationRecords; ++record)
            {
                features_calibration[record] = Helper.GenerateComplexFeatures(rng, nNumbers, featureMin, featureMax);
                targets_calibration[record] = Helper.ComputeComplexTargets(features_calibration[record], uncertainty, rng);
            }

            (double targetMin, double targetMax) = FindMatrixMinMax(targets);

            double[][] montecarlo = new double[monteCarloSize][];
            for (int i = 0; i < monteCarloSize; ++i)
            {
                montecarlo[i] = (double[])Helper.ComputeComplexTargets(features_calibration[0], uncertainty, rng).Clone();
            }

            //int[,] histo = Helper.Make2DHistogram(montecarlo, 10, 10);
            //Helper.ShowMatrix(histo);

            DeterministicModel(nFeatures, nHidden, nTargets, nDisjoints, featureMin, featureMax,
                targetMin, targetMax, alpha, rng, nLoops, nDisjointSize, nTrainingRecords,
                features, targets, features_calibration, targets_calibration, threshold);

            kNNGoodnessOfFitTest(features, features_calibration, targets, uncertainty, rng, sampleSize, minMedianBlock,
                nCalibrationRecords, monteCarloSize, monteCarloSamples);

            DDRGoognessOfFitTest(features, features_calibration, targets, nHidden, uncertainty, sampleSize, minMedianBlock, nCalibrationRecords,
                monteCarloSize, monteCarloSamples, featureMin, featureMax, targetMin, targetMax, alpha, rng);
        }
    }
}
