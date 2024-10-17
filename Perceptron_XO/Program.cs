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
    for (int i = 0; i < p.Length; i++)
    {
        for (int j = 0; j < p[i].Length; j++)
        {
            Console.Write(p[i][j] == 1 ? "# " : ". ");
        }
        Console.WriteLine();
    }
    Console.WriteLine();
}

#endregion

#region Apply Perceptron Algorithm

const int col = 5;
const int row = 5;
const int data_size = col * row;

float[] w = new float[data_size], dw = new float[data_size];
float bias = 0, dbias = 0;
float alpha = 1f; // 0 < learning rate <= 1
float theta = 0f;
float yni = 0, y = 0;

int iteration = 0;
bool resume = true;

while(resume)
{
    iteration += 1;
    resume = Perceptron(iteration);
}

bool Perceptron(int iteration = 1)
{
    bool changed = false;
    Console.WriteLine($"Iteration {iteration}");

    foreach (var p in patterns)
    {
        int n = patterns.IndexOf(p);

        for (int i = 0; i < p.Length; i++)
        {
            for (int j = 0; j < p[i].Length; j++)
            {
                int w_index = (col - 1) * i + j;
                yni += w[w_index] * p[i][j];
            }
        }

        yni += bias;
        y = yni <= theta && yni >= -theta ? 0 : (yni > theta ? 1 : -1);

        if (y != outputs[n])
        {
            if (!changed) changed = true;

            for (int i = 0; i < p.Length; i++)
            {
                for (int j = 0; j < p[i].Length; j++)
                {
                    int w_index = (col - 1) * i + j;
                    dw[w_index] = alpha * p[i][j] * outputs[n];
                    w[w_index] += dw[w_index];
                }
            }

            dbias = alpha * 1 * outputs[n];
            bias += dbias;
        }
        else
        {
            dw[0] = dw[1] = dbias = 0;
        }
    }

    return changed;
}

#endregion

#region Print Final Equataion

StringBuilder equataion_string = new();

for (int i = 0; i < data_size; i++)
{
    equataion_string.Append($"{w[i]}x{i} + ");
}

equataion_string.Append($"{bias}");

Console.WriteLine(equataion_string);

#endregion