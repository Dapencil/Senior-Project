#include "header/Util.h"
#include <vector>
#include <iostream>

void printVector(const std::vector<int> &vec, const int limit)
{
    for (int i = 0; i < limit; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
};
