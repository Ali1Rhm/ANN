using SOM_Base;

List <(float[] inputs, float label)> train_patterns = new();

SOM som = new(train_patterns, 1024, 10, 0.6f, 0.5f, 4, 200, "linear");
som.Start();