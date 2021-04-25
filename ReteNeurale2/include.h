//
// Created by elia on 09/10/2017.
//

#ifndef RETENEURALE2_INCLUDE_H
#define RETENEURALE2_INCLUDE_H

#include <cstdarg>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <vector>
#include <functional>
#include <numeric>
#include <unistd.h>
#include <iomanip>
#include <cassert>
using namespace std;

#define __SIGMOID__(x) 1.00/(1.00 + exp(-x))
#define __D_SIGMOID__(x) x*(1 - x)

void error(string str){
    cout << "error in" << str << endl;
    exit(1);
}

string ntostr(auto num){
    ostringstream strs;
    strs << num;
    string str = strs.str();
    return str;
}

template <typename T>
T vectProd(vector<T> v1, vector<T> v2, T init){
    if(v1.size()!=v2.size())
        error("vectProd: vettori con lunghezza diversa");
    T sum=init;
    for(typename vector<T>::iterator it1=v1.begin(), it2=v2.begin(); it1!=v1.end(); it1++, it2++)
        sum+=(*it1)*(*it2);
    return sum;
}
template <typename T>
void printStrEVect(string str, vector<T> v,string str2, vector<T> v2){
    if(v.size()!=v2.size())
        error("printStrEVect, lunghezze diverse");
    for (typename vector<T>::iterator it = v.begin(), it2=v2.begin(); it!=v.end(); it++, it2++) {
        cout << str << ": " << *it << "  "<< str2 << ": " << *it2 <<endl;
    }
}

#endif //RETENEURALE2_INCLUDE_H
