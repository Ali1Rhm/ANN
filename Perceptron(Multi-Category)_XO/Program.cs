using System.Text;

#region XO Patterns

List<(float[] inputs, int x_label, int o_label)> patterns = new();

foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int label = int.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    patterns.Add((inputs, label, -label));
}

#endregion

#region Apply Perceptron Algorithm

const int col = 5;
const int row = 5;
const int data_size = col * row;

float[] x_w = new float[data_size];
float x_bias = 0;

float[] o_w = new float[data_size];
float o_bias = 0;

float alpha = 0.1f;
float theta = 0.2f;

int iteration = 0;
bool stop = false;

while (!stop)
{
    iteration += 1;
    Perceptron(iteration);
}

Console.WriteLine(value: $"Number of iterations: {iteration}\n");

void Perceptron(int iteration = 1)
{
    stop = true;

    foreach ((var inputs, var x_label, var o_label) in patterns)
    {
        float x_yni = 0, o_yni = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            x_yni += x_w[i] * inputs[i];
            o_yni += o_w[i] * inputs[i];
        }
        x_yni += x_bias;
        o_yni += o_bias;

        int x_prediction = StepFunction(x_yni, theta);
        int o_prediction = StepFunction(o_yni, theta);

        if (x_prediction != x_label)
        {
            stop = false;

            for (int i = 0; i < inputs.Length; i++)
                x_w[i] += alpha * inputs[i] * x_label;

            x_bias += alpha * 1 * x_label;
        }

        if (o_prediction != o_label)
        {
            stop = false;

            for (int i = 0; i < inputs.Length; i++)
                o_w[i] += alpha * inputs[i] * o_label;

            o_bias += alpha * 1 * o_label;
        }
    }
}

static int StepFunction(float yni, float theta)
{
    return yni <= theta && yni >= -theta ? 0 : (yni > theta ? 1 : -1);
}
#endregion

#region Print Final Equataion

StringBuilder x_equataion_string = new();
StringBuilder o_equataion_string = new();

for (int i = 0; i < data_size; i++)
{
    x_equataion_string.Append($"{x_w[i]}x{i} + ");
    o_equataion_string.Append($"{o_w[i]}x{i} + ");
}

x_equataion_string.Append($"{x_bias}");
o_equataion_string.Append($"{o_bias}");

Console.WriteLine(x_equataion_string+"\n");
Console.WriteLine(o_equataion_string+"\n");

#endregion

#region Test

foreach ((var inputs, var x_label, var o_label) in patterns)
{
    float x_yni = 0, o_yni = 0;
        
    for (int i = 0; i < inputs.Length; i++)
    {
        x_yni += x_w[i] * inputs[i];
        o_yni += o_w[i] * inputs[i];
    }
    x_yni += x_bias;
    o_yni += o_bias;

    int x_prediction = StepFunction(x_yni, theta);
    int o_prediction = StepFunction(o_yni, theta);

    DrawLetter(inputs);

    Console.WriteLine(x_prediction == 0 ? "Not Defined (in X class)" : (x_prediction == 1) ? "X (in X class)" : "O (in X class)");
    Console.WriteLine((x_prediction == 0 ? "Not Defined (in O class)" : (o_prediction == 1) ? "O (in O class)" : "X (in O class)") + "\n");
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