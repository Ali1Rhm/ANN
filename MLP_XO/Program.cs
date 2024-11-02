using System.Text;

#region XO Patterns

List<(float[] inputs, int label)> patterns = new();
foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int label = int.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    patterns.Add((inputs, label));
}

#endregion

const int col = 5;
const int row = 5;
int data_size = col * row;

const int z_neurons_count = 1;
const int y_neurons_count = z_neurons_count;

float alpha = 0.01f;

float[,] z_weights = new float[z_neurons_count, data_size];
float[,] y_weights =  new float[y_neurons_count, data_size];

float[,] z_dweights = new float[z_neurons_count, data_size];
float[,] y_dweights = new float[y_neurons_count, data_size];

float[] z_bias = new float[z_neurons_count];
float[] y_bias = new float[y_neurons_count];

float[] z_dbias = new float[z_neurons_count];
float[] y_dbias = new float[y_neurons_count];

float[] z_yni = new float[z_neurons_count];
float[] y_yni = new float[y_neurons_count];

float[] z_fy = new float[z_neurons_count];
float[] y_fy = new float[y_neurons_count];

float[] z_error = new float[z_neurons_count];
float[] y_error = new float[y_neurons_count];

float[] z_prev_error = new float[z_neurons_count];
float[] y_prev_error = new float[y_neurons_count];

AssignRandomValues(ref z_bias, ref z_weights, 0.1f);
AssignRandomValues(ref y_bias, ref y_weights, 0.1f);

int iteration = 0;
bool stop = false; 

while (!stop)
{
    iteration += 1;
    MLP();
}

WriteLine($"Number of iterations = {iteration}");

void MLP()
{
    stop = true;

    foreach ((var inputs, var label) in patterns)
    {
        int i = 0;
        while (i < z_neurons_count)
        {
            z_yni[i] = 0;
            for (int j = 0; j < inputs.Length; j++)
                z_yni[i] += z_weights[i, j] * inputs[j];
            z_yni[i] += z_bias[i];

            z_fy[i] = SigmoidFunction(z_yni[i]);

            i++;
        }

        i = 0;
        while (i < y_neurons_count)
        {
            y_yni[i] = 0;
            for (int j = 0; j < z_fy.Length; j++)
                y_yni[j] += y_weights[i, j] * z_fy[j];
            y_yni[i] += y_bias[i];

            y_fy[i] = SigmoidFunction(y_yni[i]);

            y_error[i] = (label - y_fy[i]) * DifferentiatedSigmoidFunction(y_yni[i]);

            if (iteration == 1 || MathF.Abs(y_error[i]) < MathF.Abs(y_prev_error[i]))
            {
                stop = false;
                y_prev_error[i] = y_error[i];
            }

            for (int j = 0; j < z_fy.Length; j++)
                y_dweights[i, j] = alpha * y_error[i] * z_fy[j];
            y_dbias[i] = alpha * y_error[i];

            i++;
        }

        i = 0;
        while (i < z_neurons_count)
        {
            float d = 0;
            for (int j = 0; j < y_error.Length; j++)
                d += y_error[j] * y_weights[i, j];

            z_error[i] = d * DifferentiatedSigmoidFunction(z_yni[i]);

            if (iteration == 1 || MathF.Abs(z_error[i]) < MathF.Abs(z_prev_error[i]))
            {
                stop = false;
                z_prev_error[i] = z_error[i];
            }


            for (int j = 0; j < inputs.Length; j++)
                z_dweights[i, j] = alpha * z_error[i] * inputs[j];
            z_dbias[i] = alpha * z_error[i];

            i++;
        }

        i = 0;
        while (i < y_neurons_count)
        {
            for (int j = 0; j < z_fy.Length; j++)
                y_weights[i, j] += y_dweights[i, j];

            y_bias[i] += y_dbias[i];

            i++;
        }

        i = 0;
        while (i < z_neurons_count)
        {
            for (int j = 0; j < inputs.Length; j++)
                z_weights[i, j] += z_dweights[i, j];

            z_bias[i] += z_dbias[i];

            i++;
        }
    }
}

static float SigmoidFunction(float yni)
{
    return 2f * (1f / (1f + MathF.Pow(MathF.E, -yni))) - 1f;
}

static float DifferentiatedSigmoidFunction(float yni)
{
    return 0.5f * (1 + SigmoidFunction(yni)) * (1 - SigmoidFunction(yni));
}

static void AssignRandomValues(ref float[] data1D, ref float[,] data2D, float scale)
{
    var rand = Random.Shared;

    if (data1D != null)
    {
        for (int i = 0; i < data1D.Length; i++)
            data1D[i] = (2 * rand.NextSingle() - 1) * scale;
    }

    if (data2D != null)
    {
        for (int i = 0; i < data2D.GetLength(0); i++)
            for (int j = 0; j < data2D.GetLength(1); j++)
                data2D[i, j] = (2 * rand.NextSingle() - 1) * scale;
    }
}

#region Print Final Equataion

StringBuilder equataion_string = new();

for (int i = 0; i < y_weights.GetLength(1); i++)
{
    equataion_string.Append($"{y_weights[0, i]}x{i} + ");
}

equataion_string.Append($"{y_bias[0]}");

WriteLine(equataion_string + "\n");

#endregion

#region Test

foreach ((var inputs, var label) in patterns)
{
    float yni = 0;
    float prediction;

    for (int i = 0; i < inputs.Length; i++)
    {
        yni += y_weights[0, i] * inputs[i];
    }
    yni += y_bias[0];

    prediction = SigmoidFunction(yni);

    DrawLetter(inputs);

    Console.WriteLine((prediction == 0 ? "Not Defined" : (prediction > 0) ? "X" : "O") + "\n");
}

#endregion

#region Visualize Letters

static void DrawLetter(float[] letter)
{
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            Console.Write(letter[5 * i + j] == 1 ? "# " : ". ");
        }
        Console.WriteLine();
    }
}

#endregion