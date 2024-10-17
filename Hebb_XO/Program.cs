using System.Text;

#region Generate Letter Patterns

List<int[][]> patterns = new();
int[] outputs = [1, -1]; // 1 = X, -1 = O

patterns.Add([[1, -1, -1, -1, 1], [-1, 1, -1, 1, -1], [-1, -1, 1, -1, -1], [-1, 1, -1, 1, -1], [1, -1, -1, -1, 1]]);
patterns.Add([[-1, 1, 1, 1, -1], [+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1], [-1, 1, 1, 1, -1]]);

#endregion

#region Visualize Letter Patterns

foreach (var p in patterns)
{
    for(int i = 0; i < p.Length; i++)
    {
        for(int j = 0; j < p[i].Length; j++)
        {
            Console.Write(p[i][j] == 1 ? "# " : ". ");
        }
        Console.WriteLine();
    }
    Console.WriteLine();
}

#endregion

#region Apply Hebb Algorithm

const int col = 5;
const int row = 5;
const int data_size = col * row;

int[] weights = new int[data_size], dweights = new int[data_size];
int bias = 0, dbias;

foreach (var p in patterns)
{
    int n = patterns.IndexOf(p);

    for (int i = 0; i < p.Length; i++)
    {
        for (int j = 0; j < p[i].Length; j++)
        {
            int w_index = (col - 1) * i + j;
            dweights[w_index] = p[i][j] * outputs[n];
            weights[w_index] += dweights[w_index];
        }
    }

    dbias = 1 * outputs[n];
    bias += dbias;
}

#endregion

#region Print Final Equataion

StringBuilder equataion_string = new();

for (int i = 0; i < data_size; i++)
{
    equataion_string.Append($"{weights[i]}x{i} + ");
}

equataion_string.Append($"{bias}");

Console.WriteLine(equataion_string);

#endregion