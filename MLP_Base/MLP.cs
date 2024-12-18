namespace MLP_Base
{
    public class MLP
    {
        private int input_size = 100;
        private int[] layers_size = { 100, 1 };
        private float alpha = 0.01f;
        private int max_epoch = 100;
        private List<(float[] input, float[] label)> train_patterns = new();
        private List<List<Neuron>> layers = new();

        private int iteration = 0;

        public MLP(int input_size, int[] layers_size, float alpha, int max_epoch, List<(float[] input, float[] label)> train_patterns)
        {
            this.input_size = input_size;
            this.layers_size = layers_size;
            this.alpha = alpha;
            this.max_epoch = max_epoch;
            this.train_patterns = train_patterns;
        }

        public void TrainMLP()
        {
            layers = new();
            for (int i = 0; i < layers_size.Length; i++)
            {
                var layer = new List<Neuron>();
                int weights_count = i == 0 ? input_size : layers_size[i - 1];

                for (int j = 0; j < layers_size[i]; j++)
                {
                    layer.Add(new Neuron(weights_count));
                }

                layers.Add(layer);
            }

            while (iteration < max_epoch)
            {
                iteration += 1;
                Train();
            }

            Console.WriteLine(value: $"Number of iterations: {iteration}\n");
        }

        private void Train()
        {        
            foreach (var pattern in train_patterns)
            {
                for (int i = 0; i < layers.Count; i++)
                {
                    foreach (var neuron in layers[i])
                    {
                        float yni = 0;
                        float fy = 0;
                        float delta = 0;
        
                        for (int j = 0; j < neuron.weights!.Length; j++)
                        {
                            yni += neuron.weights![j] * (i == 0 ? pattern.input[j] : layers[i - 1][j].value);
                        }
                        yni += neuron.bias;
        
                        fy = SigmoidFunction(yni);
                        delta = 0;
        
                        if (i == layers.Count - 1)
                        {
                            delta = (pattern.label[layers[i].IndexOf(neuron)] - fy) * DifferentiatedSigmoidFunction(yni);
                            for (int j = 0; j < neuron.d_weights!.Length; j++)
                            {
                                neuron.d_weights[j] = alpha * delta * layers[i - 1][j].value;
                            }
                            neuron.d_bias = alpha * delta;
                        }
        
                        neuron.net_input = yni;
                        neuron.value = fy;
                        neuron.delta = delta;
                    }
                }
        
                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    foreach (var neuron in layers[i])
                    {
                        if (i == layers.Count - 1)
                        {
                            for (int j = 0; j < neuron.weights!.Length; j++)
                            {
                                neuron.weights[j] += neuron.d_weights[j];
                            }
                            neuron.bias += neuron.d_bias;
                        }
                        else
                        {
                            float D = 0;
                            float error = 0;
                            int neuron_index = layers[i].IndexOf(neuron);
        
                            for (int j = 0; j < layers[i + 1].Count; j++)
                            {
                                D += layers[i + 1][j].weights![neuron_index] * (i + 1 == layers.Count - 1 ? layers[i + 1][j].delta : layers[i + 1][j].error);
                            }
        
                            error = D * DifferentiatedSigmoidFunction(neuron.net_input);
        
                            for (int j = 0; j < neuron.weights!.Length; j++)
                            {
                                neuron.d_weights[j] = alpha * error * (i == 0 ? pattern.input[j] : layers[i - 1][j].value);
                                neuron.weights![j] += neuron.d_weights[j];
                            }
                            neuron.d_bias = alpha * error;
                            neuron.bias += neuron.d_bias;
                        }
                    }
                }
            }
        }

        public List<float> TestMLP(float[] input)
        {
            List<float> predictions = new();

            foreach (var layer in layers)
            {
                int layer_index = layers.IndexOf(layer);

                foreach (var neuron in layer)
                {
                    float yni = 0;
                    float fy = 0;

                    for (int i = 0; i < neuron.weights!.Length; i++)
                    {
                        yni += neuron.weights![i] * (layer_index == 0 ? input[i] : layers[layer_index - 1][i].value);
                    }
                    yni += neuron.bias;

                    fy = SigmoidFunction(yni);
                    neuron.value = fy;

                    if (layer_index == layers.Count - 1)
                        predictions.Add(fy);
                }
            }

            return predictions;
        }
        
        private float SigmoidFunction(float value)
        {
            return 2.0f * (1.0f / (1.0f + (float) Math.Exp(-value))) - 1.0f;
        }
        
        private float DifferentiatedSigmoidFunction(float value)
        {
            float k = MathF.Exp(-value);
            return (2.0f * k) / MathF.Pow(k + 1, 2);
        }


        public static int StepFunction(float yni, float theta)
        {
            return yni <= theta && yni >= -theta ? 0 : (yni > theta ? 1 : -1);
        }
    }

    public class Neuron
    {
        public float[]? weights;
        public float[] d_weights;
        public float bias;
        public float d_bias;
        public float value;
        public float net_input;
        public float delta;
        public float error;

        public Neuron(int weights_count)
        {
            float[] random_values = GenerateRandomValues(1f, weights_count + 1);
            weights = random_values[..^1];
            d_weights = new float[weights_count];
            bias = random_values[^1];
        }

        float[] GenerateRandomValues(float scale, int count)
        {
            var rand = Random.Shared;
            float[] values = new float[count];

            for (int i = 0; i < values.Length; i++)
                values[i] = (2 * rand.NextSingle() - 1) * scale;

            return values;
        }
    }
}