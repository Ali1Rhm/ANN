using MLP_Base;
#region XO Patterns


List<(float[] input, int[] label)> train_patterns = new();
List<(float[] input, int[] label)> test_patterns = new();

foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData10Train.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int[] label = [int.Parse(values[^1])];
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    train_patterns.Add((inputs, label));
}

foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData10Test.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int[] label = [int.Parse(values[^1])];
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    test_patterns.Add((inputs, label));
}

#endregion

int input_size = 100;
int[] layers_size = { 100, 1 };
float alpha = 0.01f;
int max_epoch = 100;

MLP mlp = new(input_size, layers_size, alpha, max_epoch, train_patterns);
mlp.TrainMLP();

#region Test

int truePositive = 0;
int trueNegative = 0;
int falsePositive = 0;
int falseNegative = 0;

foreach ((var input, var label) in test_patterns)
{

    float prediction = mlp.TestMLP(input)[0];

    prediction = MLP.StepFunction(prediction, 0f);

    if (prediction == label[0] && label[0] == 1)
        truePositive++;
    else if (prediction == label[0] && label[0] == -1)
        trueNegative++;
    else if (prediction != label[0] && label[0] == 1)
        falseNegative++;
    else if (prediction != label[0] && label[0] == -1)
        falsePositive++;

    DrawLetter(input);
    WriteLine((prediction == 0 ? "Not Defined" : (prediction > 0) ? "X" : "O") + "\n");
}

double precision = truePositive / (double)(truePositive + falsePositive);
double recall = truePositive / (double)(truePositive + falseNegative);
double accuracy = (truePositive + trueNegative) / (double)(truePositive + trueNegative + falsePositive + falseNegative);
double f1Score = 2 * (precision * recall) / (precision + recall);

WriteLine("Confusion Matrix:");
WriteLine($"                 Predicted X   Predicted O");
WriteLine($"Actual X         {truePositive}            {falseNegative}");
WriteLine($"Actual O         {falsePositive}            {trueNegative}\n");

WriteLine("Metrics:");
WriteLine($"Accuracy : {accuracy:P2}");
WriteLine($"Precision: {precision:P2}");
WriteLine($"Recall   : {recall:P2}");
WriteLine($"F1 Score : {f1Score:P2}");

#endregion

#region Visualize Letters

static void DrawLetter(float[] letter)
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            Write(letter[10 * i + j] == 1 ? "# " : ". ");
        }
        WriteLine();
    }
}

#endregion