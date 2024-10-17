using System.Text;

#region Generate Letter Patterns

List<int[][]> patterns = new();
int[] x_outputs = [1, -1]; // 1 = X, -1 = O
int[] o_outputs = [-1, 1]; // 1 = O, -1 = X

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

float[] x_w = new float[data_size], x_dw = new float[data_size];
float x_bias = 0, x_dbias = 0;

float[] o_w = new float[data_size], o_dw = new float[data_size];
float o_bias = 0, o_dbias = 0;

float alpha = 1f; // 0 < learning rate <= 1
float theta = 0f;

float x_yni = 0, o_yni = 0, x_y = 0, o_y = 0;

int iteration = 0;
bool resume = true;

while (resume)
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
                x_yni += x_w[w_index] * p[i][j];
                o_yni += o_w[w_index] * p[i][j];
            }
        }

        x_yni += x_bias;
        x_y = x_yni <= theta && x_yni >= -theta ? 0 : (x_yni > theta ? 1 : -1);

        o_yni += o_bias;
        o_y = o_yni <= theta && o_yni >= -theta ? 0 : (o_yni > theta ? 1 : -1);

        if (x_y != x_outputs[n])
        {
            if (!changed) changed = true;

            for (int i = 0; i < p.Length; i++)
            {
                for (int j = 0; j < p[i].Length; j++)
                {
                    int w_index = (col - 1) * i + j;
                    x_dw[w_index] = alpha * p[i][j] * x_outputs[n];
                    x_w[w_index] += x_dw[w_index];
                }
            }

            x_dbias = alpha * 1 * x_outputs[n];
            x_bias += x_dbias;
        }
        else
        {
            x_dw[0] = x_dw[1] = x_dbias = 0;
        }

        if (o_y != o_outputs[n])
        {
            if (!changed) changed = true;

            for (int i = 0; i < p.Length; i++)
            {
                for (int j = 0; j < p[i].Length; j++)
                {
                    int w_index = 4 * i + j;
                    o_dw[w_index] = alpha * p[i][j] * o_outputs[n];
                    o_w[w_index] += o_dw[w_index];
                }
            }

            o_dbias = alpha * 1 * o_outputs[n];
            o_bias += o_dbias;
        }
        else
        {
            o_dw[0] = o_dw[1] = o_dbias = 0;
        }
    }

    return changed;
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

Console.WriteLine(x_equataion_string);
Console.WriteLine(o_equataion_string);

#endregion