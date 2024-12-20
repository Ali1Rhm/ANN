using System;

List<int[]> train_patterns = new();
train_patterns.Add(new[] { 1, -1, 1});
train_patterns.Add(new[] { -1, 1, -1 });

List<int[]> test_patterns = new();
test_patterns.Add(new[] { -1, -1, 1 });
test_patterns.Add(new[] {-1, -1, -1});
test_patterns.Add(new[] { 1, -1, -1 });

int neurons_count = 3;
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
            w[i, j] += pattern[i] * pattern[j];
        w[i, j] /= neurons_count;
    }
}

int[] indexes = new int[neurons_count];
for (int i = 0; i < indexes.Length; i++)
    indexes[i] = i;

foreach (var pattern in train_patterns)
{
    for (int i = 0; i < pattern.Length; i++)
        y[i] = pattern[i];

    indexes = indexes.OrderBy(x => Random.Shared.Next()).ToArray();
    for (int i = 0; i < neurons_count; i++)
    {
        float y_ni = y[indexes[i]];
        for(int j = 0; j < neurons_count; j++)
        {
            y_ni += y[j] * w[i, j];
        }
        y[indexes[i]] = Math.Sign(y_ni);
    }
}

Console.WriteLine("Test:");

foreach (var pattern in test_patterns)
{
    for (int i = 0; i < pattern.Length; i++)
    {
        y[i] = pattern[i];
        Console.Write(y[i] + " ");
    }

    Console.Write("==> ");
    for (int i = 0; i < neurons_count; i++)
    {
        float y_ni = y[i];
        for (int j = 0; j < neurons_count; j++)
        {
            y_ni += y[j] * w[i, j];
        }
        y[i] = Math.Sign(y_ni);
        Console.Write(y[i] + " ");
    }
    Console.WriteLine();
}