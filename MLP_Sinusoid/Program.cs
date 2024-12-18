using MLP_Base;

List<(float[] input, float[] label)> train_patterns = new();
List<(float[] input, float[] label)> test_patterns = new();

train_patterns = GenerateSamples(1000);
test_patterns = GenerateSamples(50);

int input_size = 2;
int[] layers_size = { 5, 5, 1 };
float alpha = 0.1f;
int max_epoch = 100;

MLP mlp = new(input_size, layers_size, alpha, max_epoch, train_patterns);
mlp.TrainMLP();

float mse = 0;
float mae = 0;

foreach (var pattern in test_patterns)
{
    var predictions = mlp.TestMLP(pattern.input);

    Console.WriteLine(pattern.label[0].ToString() + " ---> " + predictions[0].ToString());

    mse += MathF.Pow(pattern.label[0] - predictions[0], 2.0f);
    mae += MathF.Abs(pattern.label[0] - predictions[0]);
}
mse /= test_patterns.Count;
mae /= test_patterns.Count;
Console.WriteLine($"MSE = {mse}");
Console.WriteLine($"MAE = {mae}");

List<(float[] input, float[] label)> GenerateSamples(int numberOfSamples = 1000)
{
    List<(float[] input, float[] label)> samples = new();

    Random rand = new Random();
    for (int i = 0; i < numberOfSamples; i++)
    {
        float x1 = rand.NextSingle();
        float x2 = rand.NextSingle();

        float y = (float)(Math.Sin(2 * Math.PI * x1) * Math.Sin(2 * Math.PI * x2));

        samples.Add((new float[] { x1, x2 }, new float[] { y }));
    }

    return samples;
}