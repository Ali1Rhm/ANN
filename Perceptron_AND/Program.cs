int[] x1 = [1, 1, 0, 0];
int[] x2 = [1, 0, 1, 0];
int[] t = [1, -1, -1, -1];

float[] w = new float[2], dw = new float[2];
float bias = 0, dbias = 0;
float alpha = 1f; // 0 < learning rate <= 1
float theta = 0.2f;
float yni, y;

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

    for (int i = 0; i < 4; i++)
    {
        yni = bias + w[0] * x1[i] + w[1] * x2[i];
        y = yni <= theta && yni >= -theta ? 0 : (yni > theta ? 1 : -1);

        if (y != t[i])
        {
            if (!changed) changed = true;

            dw[0] = alpha * x1[i] * t[i];
            dw[1] = alpha * x2[i] * t[i];
            dbias = alpha * 1 * t[i];

            w[0] += dw[0];
            w[1] += dw[1];
            bias += dbias;
        }
        else
        {
            dw[0] = dw[1] = dbias = 0;
        }

        Console.WriteLine($"x1={x1[i]}, x2={x2[i]}, bias=1, YNI={yni}, y={y}, t={t[i]}, dw1={dw[0]}, dw2={dw[1]}, dbias={dbias}, w1={w[0]}, w2={w[1]}, b={bias}");
    }

    return changed;
}