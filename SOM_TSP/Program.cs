using SOM_Base;

List<(float[] inputs, float label)> train_patterns = new();
List<Neuron> neurons = new();
List<City> cities = new();

var file_Path = @"C:\Dev\ANN\TSP51.txt";

foreach (var line in File.ReadLines(file_Path))
{
    if (string.IsNullOrWhiteSpace(line)) continue;

    var numbers = line.Split(' ', StringSplitOptions.TrimEntries);
    train_patterns.Add(([float.Parse(numbers[1]), float.Parse(numbers[2])], float.Parse(numbers[0])));
    cities.Add(new City(float.Parse(numbers[1]), float.Parse(numbers[2])));
}

SOM som = new(train_patterns, 2, 230, 0.8f, 230, 200, "linear");
var final_weights = som.Start();

for (int i = 0; i < final_weights.GetLength(1); i++)
    neurons.Add(new Neuron(final_weights[0, i], final_weights[1, i]));

var route = neurons.Select(neuron => FindClosestCity(neuron, cities)).Distinct().ToList();
Console.WriteLine("TSP Route:");
foreach (var city in route)
{
    Console.WriteLine($"City {route.IndexOf(city) + 1}: ({city.X}, {city.Y})");
}

City FindClosestCity(Neuron neuron, List<City> cities)
{
    City? closestCity = null;
    double minDistance = double.MaxValue;
    foreach (var city in cities)
    {
        double distance = Math.Sqrt(Math.Pow(neuron.X - city.X, 2) + Math.Pow(neuron.Y - city.Y, 2));
        if (distance < minDistance)
        {
            minDistance = distance;
            closestCity = city;
        }
    }
    return closestCity!;
}

public class City
{
    public float X { get; set; }
    public float Y { get; set; }
    public City(float x, float y)
    {
        X = x;
        Y = y;
    }
}

public class Neuron
{
    public float X { get; set; }
    public float Y { get; set; }
    public Neuron(float x, float y)
    {
        X = x;
        Y = y;
    }
}