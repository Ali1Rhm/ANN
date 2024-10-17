int[] x1_binary = [1, 1, 0, 0];
int[] x2_binary = [1, 0, 1, 0];
int[] y_binary = [1, 0, 0, 0];

int[] x1_bipolar = [1, 1, -1, -1];
int[] x2_bipolar = [1, -1, 1, -1];
int[] y_bipolar = [1, -1, -1, -1];

int w1 = 0, w2 = 0, b = 0;
int dw1, dw2, db;

Console.WriteLine("Hebb for AND using binary outputs:");
for(int i = 0; i < 4; i++)
{
    dw1 = x1_binary[i] * y_binary[i];
    dw2 = x2_binary[i] * y_binary[i];
    db = 1 * y_binary[i];

    w1 += dw1;
    w2 += dw2;
    b += db;

    Console.WriteLine($"y = {w1}x1 + {w2}x2 + {b}");
}

w1 = w2 = b = 0;

Console.WriteLine("\nHebb for AND using bipolar outputs:");
for (int i = 0; i < 4; i++)
{
    dw1 = x1_bipolar[i] * y_bipolar[i];
    dw2 = x2_bipolar[i] * y_bipolar[i];
    db = 1 * y_bipolar[i];

    w1 += dw1;
    w2 += dw2;
    b += db;

    Console.WriteLine($"y = {w1}(x1) + {w2}(x2) + {b}");
}