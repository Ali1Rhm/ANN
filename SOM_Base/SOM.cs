namespace SOM_Base
{
    public class SOM
    {
        private List<(float[] inputs, float label)> train_patterns = new();
        private int features_count = 4;
        private int clusters_count = 2;
        private float alpha = 0.6f;
        private float neighboring_radius = 1.0f;
        private int max_epoch = 200;
        private string topology = "linear"; // Can be "linear", "square", or "hexagon"

        private float[,] w;
        private int iteration = 0;
        private bool stop = false;

        public SOM
            (
            List<(float[] inputs, float label)> train_patterns,
            int features_count = 4,
            int clusters_count = 2,
            float alpha = 0.6f,
            int neighboring_radius = 1,
            int max_epoch = 200,
            string topology = "linear"
            )
        {
            this.train_patterns = train_patterns;
            this.features_count = features_count;
            this.clusters_count = clusters_count;
            this.alpha = alpha;
            this.neighboring_radius = neighboring_radius;
            this.max_epoch = max_epoch;
            this.topology = topology;

            w = new float[features_count, clusters_count];

            for (int i = 0; i < w.GetLength(0); i++)
            {
                float[] random_values = GenerateRandomValues(0.1f, w.GetLength(1));
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    w[i, j] = random_values[j];
                }
            }
        }

        public float[,] Start()
        {

            while (!stop && iteration < max_epoch)
            {
                Train();
                alpha *= MathF.Exp((float) -iteration / max_epoch);
                neighboring_radius *= MathF.Exp(-iteration / (max_epoch / MathF.Log2(clusters_count/2)));

                iteration++;
            }

            //Console.WriteLine($"Iteration Count: {iteration}");

            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    //Console.Write($"{w[i, j]:N2} ");
                }
                //Console.WriteLine();
            }

            return w;
        }

        private void Train()
        {
            foreach (var pattern in train_patterns)
            {
                int min_index = GetBestMatchingUnit(pattern.inputs);
                UpdateWeights(min_index, pattern.inputs);
            }
        }

        public int GetBestMatchingUnit(float[] inputs)
        {
            int min_index = 0;
            float min_value = float.PositiveInfinity;

            for (int i = 0; i < clusters_count; i++)
            {
                float distance = 0;
                for (int j = 0; j < features_count; j++)
                {
                    distance += MathF.Pow(w[j, i] - inputs[j], 2.0f);
                }

                if (distance < min_value)
                {
                    min_value = distance;
                    min_index = i;
                }
            }

            return min_index;
        }

        private void UpdateWeights(int min_index, float[] inputs)
        {
            for (int i = 0; i < clusters_count; i++)
            {
                if (IsWithinNeighborhood(min_index, i, topology))
                {
                    for (int j = 0; j < features_count; j++)
                    {
                        float dw = alpha * (inputs[j] - w[j, i]);
                        w[j, i] += dw;
                    }
                }
            }
        }

        private bool IsWithinNeighborhood(int min_index, int current_index, string topology)
        {
            int distance = Math.Abs(min_index - current_index);

            if (topology == "linear")
            {
                return distance <= neighboring_radius;
            }
            else if (topology == "square")
            {
                int min_row = min_index / (int)Math.Sqrt(clusters_count);
                int min_col = min_index % (int)Math.Sqrt(clusters_count);
                int curr_row = current_index / (int)Math.Sqrt(clusters_count);
                int curr_col = current_index % (int)Math.Sqrt(clusters_count);

                return Math.Abs(min_row - curr_row) <= neighboring_radius && Math.Abs(min_col - curr_col) <= neighboring_radius;
            }
            else if (topology == "hexagon")
            {
                int min_row = min_index / clusters_count;
                int min_col = min_index % clusters_count;
                int curr_row = current_index / clusters_count;
                int curr_col = current_index % clusters_count;

                return Math.Abs(min_row - curr_row) + Math.Abs(min_col - curr_col) <= neighboring_radius;
            }

            return false;
        }

        private float[] GenerateRandomValues(float scale, int count)
        {
            var rand = Random.Shared;
            float[] values = new float[count];

            for (int i = 0; i < values.Length; i++)
                values[i] = (2 * rand.NextSingle() - 1) * scale;

            return values;
        }
    }
}
