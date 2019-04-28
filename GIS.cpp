#include <iostream>
#include <cstring>
#include <cmath>

double *getResults(int i, int n, int samples[100][101], int labels[100], double weight[100000], int num[100][100],
                   int pos[100][100])
{ // get classification conditional probability p(y|x)
    double Zx = 0.0;
    for (int j = 0; j < n; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < samples[i][0]; ++k)
            if (num[samples[i][k + 1]][labels[j]] > 0)
                sum += weight[pos[samples[i][k + 1]][labels[j]]];
        Zx += exp(sum);
    }
    double *result = new double[10];
    for (int j = 0; j < n; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < samples[i][0]; ++k)
            if (num[samples[i][k + 1]][labels[j]] > 0)
                sum += weight[pos[samples[i][k + 1]][labels[j]]];
        result[j] = 1.0 / Zx * exp(sum);
    }
    return result;
}

double *GIS(int n, int maxRound)
{
    std::cout << "Hello, World!" << std::endl;
    int samples[100][101] = {0}; //sample data, note than sample[i][0] denotes the size of sample[i]
    int labels[100];
    int num[100][100] = {0}; //counting
    int pos[100][100] = {0}; // get index with tuple
    int tuple[100][2] = {0}; // get tuple with index
    int tupleSize = 0;
    int maxSize = 0; // max sample size
    double expect_train[100000] = {0};
    double expect_test[100000] = {0};
    double weight[100000] = {0};
    double last_weight[100000] = {0};
    double delta = 0.001; //threshold
    srand((int) time(0));
    for (int i = 0; i < n; ++i) // n: sample size
    { //init
        int y = random() % 100;
        int size = random() % 10;
        labels[i] = y;
        if (size > maxSize)
            maxSize = size;
        samples[i][0] = size;  //store the size
        for (int j = 0; j < size; ++j)
        {
            int x = random() % 100;
            samples[i][j + 1] = x;
            if (num[x][y] == 0)
            {
                tuple[tupleSize][0] = x;
                tuple[tupleSize][1] = y;
                pos[x][y] = tupleSize;
                ++tupleSize;
            }
            ++num[x][y];
        }
    }
    for (int i = 0; i < tupleSize; ++i)
    { //init parameter
        expect_train[i] = num[tuple[i][0]][tuple[i][1]] * 1.0 / n;
    }
    for (int e = 0; e < maxRound; ++e)
    { // max round : 1000
        std::cout << "round " << e << " weight: ";
        memcpy(last_weight, weight, sizeof(weight));
        memset(expect_test, 0, tupleSize * sizeof(double));
        for (int i = 0; i < n; ++i)
        { //training
            double *result = getResults(i, n, samples, labels, weight, num, pos);
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < samples[i][0]; ++k)
                    if (num[samples[i][k + 1]][labels[j]] > 0)
                        expect_test[pos[samples[i][k + 1]][labels[j]]] += result[j] * 1.0 / n;
        }
        bool check = false;
        for (int i = 0; i < tupleSize; ++i) //update weights
        {
            weight[i] += 1.0 / maxSize * log(expect_train[i] / expect_test[i]);
            std::cout << weight[i] << " ";
            if (fabs(weight[i] - last_weight[i]) > delta)
                check = true;
        }
        std::cout << std::endl;
        if (!check) //break if not large change
            break;
    }
    return weight;
}

int main()
{
    GIS(10, 1000);
    return 0;
}

