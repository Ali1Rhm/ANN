﻿using System.Text;

#region XO Patterns

List<(float[] inputs, int label)> train_patterns = new();
List<(float[] inputs, int label)> test_patterns = new();


foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData10Train.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int label = int.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    train_patterns.Add((inputs, label));
}

foreach (var line in File.ReadLines(@"C:\Dev\ANN\XOData10Test.txt"))
{
    string[] values = line.Split(new char[] { ' ' });

    int label = int.Parse(values[^1]);
    float[] inputs = new float[values.Length - 1];
    for (int i = 0; i < inputs.Length; i++)
        inputs[i] = float.Parse(values[i]);

    test_patterns.Add((inputs, label));
}

#endregion

#region Apply Perceptron Algorithm

const int col = 10;
const int row = 10;
const int data_size = col * row;

float[] w = new float[data_size];
float bias = 0;
float alpha = 0.1f;
float theta = 0f;

int iteration = 0;
bool stop = false;

while(!stop)
{
    iteration += 1;
    Perceptron(iteration);
}

Console.WriteLine(value: $"Number of iterations: {iteration}\n");

void Perceptron(int iteration = 1)
{
    stop = true;

    foreach ((var inputs, var label) in train_patterns)
    {
        float yni = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            yni += w[i] * inputs[i];
        }
        yni += bias;

        int prediction = StepFunction(yni, theta);

        if (prediction == label)
            continue;

        stop = false;
        for (int i = 0; i < inputs.Length; i++)
        {
            float dw = alpha * label * inputs[i];
            w[i] += dw;
        }

        bias += alpha * label * 1;
    }
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

Console.WriteLine(equataion_string+"\n");

#endregion

#region Test

int truePositive = 0;
int trueNegative = 0;
int falsePositive = 0;
int falseNegative = 0;

foreach ((var inputs, var label) in test_patterns)
{
    float yni = 0;
    int prediction;

    for (int i = 0; i < inputs.Length; i++)
    {
        yni += w[i] * inputs[i];
    }
    yni += bias;

    prediction = StepFunction(yni, theta);

    DrawLetter(inputs);

    Console.WriteLine((prediction == 0 ? "Not Defined" : (prediction == 1) ? "X" : "O") + "\n");

    if (prediction == label && label == 1)
        truePositive++;
    else if (prediction == label && label == -1)
        trueNegative++;
    else if (prediction != label && label == 1)
        falseNegative++;
    else if (prediction != label && label == -1)
        falsePositive++;
}

double precision = truePositive / (double)(truePositive + falsePositive);
double recall = truePositive / (double)(truePositive + falseNegative);
double accuracy = (truePositive + trueNegative) / (double)(truePositive + trueNegative + falsePositive + falseNegative);
double f1Score = 2 * (precision * recall) / (precision + recall);

// Print the confusion matrix and metrics
Console.WriteLine("Confusion Matrix:");
Console.WriteLine($"                 Predicted X   Predicted O");
Console.WriteLine($"Actual X         {truePositive}            {falseNegative}");
Console.WriteLine($"Actual O         {falsePositive}            {trueNegative}\n");

Console.WriteLine("Metrics:");
Console.WriteLine($"Accuracy : {accuracy:P2}");
Console.WriteLine($"Precision: {precision:P2}");
Console.WriteLine($"Recall   : {recall:P2}");
Console.WriteLine($"F1 Score : {f1Score:P2}");

#endregion

#region Visualize Letters

static void DrawLetter(float[] letter)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            Console.Write(letter[row * i + j] == 1 ? "# " : ". ");
        }
        Console.WriteLine();
    }
}

#endregion