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

#region Apply Hebb Algorithm

const int col = 5;
const int row = 5;
const int data_size = col * row;

float[] w = new float[data_size];
float bias = 0;

foreach ((var inputs, var label) in patterns)
{
    for (int i = 0; i < inputs.Length; i++)
        w[i] += inputs[i] * label;

    bias += 1 * label;
}

static int StepFunction(float yni, float theta)
{
    return yni <= theta && yni >= -theta ? 0 : (yni > theta ? 1 : -1);
}

#endregion

#region Print Final Equataion

StringBuilder equataion_string = new();

for (int i = 0; i < data_size; i++)
{
    equataion_string.Append($"{w[i]}x{i} + ");
}

equataion_string.Append($"{bias}");

Console.WriteLine(equataion_string + "\n");

#endregion

#region Test

foreach ((var inputs, var label) in patterns)
{
    float yni = 0;
    int prediction;

    for (int i = 0; i < inputs.Length; i++)
    {
        yni += w[i] * inputs[i];
    }
    yni += bias;

    prediction = StepFunction(yni, 0.2f);

    DrawLetter(inputs);

    Console.WriteLine((prediction == 0 ? "Not Defined" : (prediction == 1) ? "X" : "O") + "\n");
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