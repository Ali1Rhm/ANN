#region XO Patterns

List<(float[] input, int label)> train_patterns = new();
List<(float[] input, int label)> test_patterns = new();

foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData10Train.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int label = int.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    train_patterns.Add((inputs, label));
}

foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData10Test.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int label = int.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    test_patterns.Add((inputs, label));
}

#endregion

int input_size = 100;
int[] layers_size = {100, 1};
float alpha = 0.01f;
int epoch = 100;

List<List<Neuron>> layers = new();
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

int iteration = 0;
bool stop = false;
while (!stop && iteration < epoch)
{
    iteration += 1;
    MLP(iteration);
}

WriteLine(value: $"Number of iterations: {iteration}\n");

void MLP(int iteration = 1)
{
    stop = true;

    foreach (var pattern in train_patterns)
    { 
        for (int i =  0; i < layers.Count; i++)
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
                delta = (pattern.label - fy) * DifferentiatedSigmoidFunction(yni);

                if (i == layers.Count - 1)
                {
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
                        D += layers[i + 1][j].weights![neuron_index] * layers[i + 1][j].delta;
                    }

                    error = D * DifferentiatedSigmoidFunction(neuron.net_input);

                    if (iteration == 1 || MathF.Abs(error) < MathF.Abs(neuron.error))
                    {
                        stop = false;
                        neuron.error = error;
                    }

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

float SigmoidFunction(float yni)
{
    return 2f * (1f / (1f + MathF.Pow(MathF.E, -yni))) - 1f;
}

float DifferentiatedSigmoidFunction(float yni)
{
    return 0.5f * (1 + SigmoidFunction(yni)) * (1 - SigmoidFunction(yni));
}

static int StepFunction(float yni, float theta)
{
    return yni <= theta && yni >= -theta ? 0 : (yni > theta ? 1 : -1);
}

#region Test

int truePositive = 0;
int trueNegative = 0;
int falsePositive = 0;
int falseNegative = 0;

foreach ((var input, var label) in test_patterns)
{
    float prediction = 0;

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
                prediction = fy;
        }
    }

    prediction = StepFunction(prediction, 0f);

    if (prediction == label && label == 1)
        truePositive++;
    else if (prediction == label && label == -1)
        trueNegative++;
    else if (prediction != label && label == 1)
        falseNegative++;
    else if (prediction != label && label == -1)
        falsePositive++;

    DrawLetter(input);
    WriteLine((prediction == 0 ? "Not Defined" : (prediction > 0) ? "X" : "O") + "\n");
}

double precision = truePositive / (double)(truePositive + falsePositive);
double recall = truePositive / (double)(truePositive + falseNegative);
double accuracy = (truePositive + trueNegative) / (double)(truePositive + trueNegative + falsePositive + falseNegative);
double f1Score = 2 * (precision * recall) / (precision + recall);

// Print the confusion matrix and metrics
WriteLine("Confusion Matrix:");
WriteLine($"                 Predicted X   Predicted O");
WriteLine($"Actual X         {truePositive}            {falseNegative}");
WriteLine($"Actual O         {falsePositive}            {trueNegative}\n");

WriteLine("Metrics:");
WriteLine($"Accuracy : {accuracy:P2}");
WriteLine($"Precision: {precision:P2}");
WriteLine($"Recall   : {recall:P2}");
WriteLine($"F1 Score : {f1Score:P2}");

#endregion

#region Visualize Letters

static void DrawLetter(float[] letter)
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            Write(letter[10 * i + j] == 1 ? "# " : ". ");
        }
        WriteLine();
    }
}

#endregion

class Neuron
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
        float[] random_values = GenerateRandomValues(0.1f, weights_count + 1);
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