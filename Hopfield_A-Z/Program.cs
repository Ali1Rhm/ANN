List <(float[] inputs, char label)> train_patterns = new();

int max_line_to_read = 5;
int line_index = 0;
foreach (var line in File.ReadLines(@"C:\Dev\ANN\A-ZData_Train.txt"))
{
    line_index++;

    string[] values = line.Split(new char[] { ' ' });

    char label = char.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    train_patterns.Add((inputs, label));

    if (line_index == max_line_to_read)
        break;
}

List<(float[] inputs, char label)> test_patterns = new();
foreach (var line in File.ReadLines(@"C:\Dev\ANN\A-ZData_Noise.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    char label = char.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    test_patterns.Add((inputs, label));
}

int neurons_count = 100;
float[] y = new float[neurons_count];
float[,] w = new float[neurons_count, neurons_count];

for (int i = 0; i < w.GetLength(0); i++)
{
    for (int j = 0; j < w.GetLength(1); j++)
    {
        if (i == j)
        {
            w[i, j] = 0;
            continue;
        }

        foreach (var pattern in train_patterns)
            w[i, j] += pattern.inputs[i] * pattern.inputs[j];
        w[i, j] /= neurons_count;
    }
}

int[] indexes = new int[neurons_count];
for (int i = 0; i < indexes.Length; i++)
    indexes[i] = i;

foreach (var pattern in train_patterns)
{
    for (int i = 0; i < pattern.inputs.Length; i++)
        y[i] = pattern.inputs[i];

    indexes = indexes.OrderBy(x => Random.Shared.Next()).ToArray();
    for (int i = 0; i < neurons_count; i++)
    {
        float y_ni = y[indexes[i]];
        for(int j = 0; j < neurons_count; j++)
        {
            y_ni += y[j] * w[i, j];
        }
        y[indexes[i]] = MathF.Sign(y_ni);
    }
}

Console.WriteLine("Test:");

foreach (var pattern in test_patterns)
{
    for (int i = 0; i < pattern.inputs.Length; i++)
    {
        y[i] = pattern.inputs[i];
    }

    Console.Write(pattern.label + " ==> ");

    for (int i = 0; i < neurons_count; i++)
    {
        float y_ni = y[i];
        for (int j = 0; j < neurons_count; j++)
        {
            y_ni += y[j] * w[i, j];
        }
        y[i] = MathF.Sign(y_ni);
    }

    foreach (var saved_pattern in train_patterns)
    {
        bool equal = true;
        for (int i = 0; i < saved_pattern.inputs.Length; i++)
        {
            if (saved_pattern.inputs[i] != y[i])
                equal = false;
        }

        if (equal)
        {
            Console.Write(saved_pattern.label);
            break;
        }
        else if (train_patterns.IndexOf(saved_pattern) == train_patterns.Count - 1)
        {
            Console.Write("Not Found");
        }
    }

    Console.WriteLine();
}