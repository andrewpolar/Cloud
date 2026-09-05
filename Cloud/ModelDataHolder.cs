using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cloud
{
    class ModelData
    {
        private KAN kan;
        private double[][] features;
        private double[][] targets;
        private double[][] residuals;
        const int nEpochs = 64;
        public ModelData(KAN kan, double[][] features, double[][] targets)
        {
            this.kan = kan;
            this.features = features;
            this.targets = targets;
        }

        public int Count
        {
            get { return targets.Length; }
        }

        public void Train()
        {
            int nRecords = targets.Length;
            for (int epoch = 0; epoch < nEpochs; ++epoch)
            {
                for (int record = 0; record < nRecords; ++record)
                {
                    kan.Forward(features[record]);
                    kan.Update(targets[record]);
                }
            }
        }

        public void GetResiduals()
        {
            int nRecords = targets.Length;
            residuals = new double[nRecords][];
            //double[] t1 = new double[nRecords];
            //double[] t2 = new double[nRecords];
            //double[] m1 = new double[nRecords];
            //double[] m2 = new double[nRecords];
            for (int record = 0; record < nRecords; ++record)
            {
                kan.Predict(features[record]);
                double[] prediction = kan.GetPrediction();
                residuals[record] = new double[prediction.Length];
                for (int j = 0; j < prediction.Length; ++j)
                {
                    residuals[record][j] = targets[record][j] - prediction[j];
                }
                //m1[record] = prediction[0];
                //m2[record] = prediction[1];
                //t1[record] = targets[record][0];
                //t2[record] = targets[record][1];
            }
            //double p1 = Helper.Pearson(m1, t1);
            //double p2 = Helper.Pearson(m2, t2);
            //Console.WriteLine("{0:0.0000} {1:0.0000}", p1, p2);
        }

        public (ModelData left, ModelData right) Split(int component)
        {
            Train();
            GetResiduals();

            // Create indexes so that features/targets
            // can be reordered according to residual.
            int nRecords = targets.Length;

            int[] indices = new int[nRecords];
            for (int i = 0; i < nRecords; ++i) indices[i] = i;

            // Sort records by selected residual component.
            Array.Sort(
                indices,
                delegate (int a, int b)
                {
                    return residuals[a][component].CompareTo(residuals[b][component]);
                });

            // Split exactly in half.
            int middle = nRecords / 2;

            double[][] leftFeatures = new double[middle][];
            double[][] leftTargets = new double[middle][];

            double[][] rightFeatures =
                new double[nRecords - middle][];

            double[][] rightTargets =
                new double[nRecords - middle][];

            for (int i = 0; i < middle; ++i)
            {
                int source = indices[i];

                leftFeatures[i] = features[source];
                leftTargets[i] = targets[source];
            }

            for (int i = middle; i < nRecords; ++i)
            {
                int source = indices[i];
                int destination = i - middle;

                rightFeatures[destination] = features[source];
                rightTargets[destination] = targets[source];
            }

            // Clone the trained model.
            KAN leftKan = new KAN(kan);
            KAN rightKan = new KAN(kan);

            ModelData left = new ModelData(leftKan, leftFeatures, leftTargets);
            ModelData right = new ModelData(rightKan, rightFeatures, rightTargets);

            return (left, right);
        }

        public double[] Predict(double[] features)
        {
            kan.Predict(features);
            double[] prediction = kan.GetPrediction();

            double[] result = new double[prediction.Length];
            Array.Copy(prediction, result, prediction.Length);

            return result;
        }
    }

    internal class ModelDataHolder
    {
        private readonly List<ModelData> leaves = new List<ModelData>();

        public IReadOnlyList<ModelData> Leaves
        {
            get { return leaves; }
        }

        public ModelDataHolder(ModelData initial)
        {
            leaves.Add(initial);
        }

        public int GetSize()
        {
            return leaves.Count;
        }

        public void SplitAll(int minRecords)
        {
            int component = 0;

            while (true)
            {
                bool didSplit = false;
                int originalCount = leaves.Count;

                for (int i = 0; i < originalCount; ++i)
                {
                    ModelData leaf = leaves[i];
                    if (leaf.Count < 2 * minRecords) continue;
                    var result = leaf.Split(component);
                    leaves[i] = result.left;
                    leaves.Insert(i + 1, result.right);
                    didSplit = true;
                    ++i;
                }

                if (!didSplit) break;

                // Alternate component after the whole generation.
                component = 1 - component;
            }
        }

        public double[][] GetPredictedSamples(double[] features)
        {
            double[][] samples = new double[leaves.Count][];
            for (int i = 0; i < leaves.Count; ++i)
            {
                samples[i] = leaves[i].Predict(features);
            }
            return samples;
        }
    }
}
