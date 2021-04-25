#include "include.hpp"
#include "Perceptron.hpp"
#include "Neural.hpp"
#define TYPE double
using namespace std;
int main()
{
    // n(n°in, n° strati)
    Neural<TYPE> rete;
    rete.setKeZeri(1,5);
    rete.setNumeroStratiEIn(2,4,2,3,2,1);
    // creo gli input e gli output
    vector<vector<TYPE>> in={{0,0},{0,1},{1,0},{1,1}};
    vector<vector<TYPE>> out={{1},{0},{1},{1}};
    rete.setinOut(in,out);
    // stampo la rete con gli output iniziali
    rete.run_rete(_in[0]);
    rete.printRete();
    rete.setEta(7);
    rete.setEta2(0.5);
    rete.setMaxEreError(30000,0.00000001);
    //
    rete.training(true,true,true);
    //rete.testTrainingEta2(0.7,0.8,4,5,true);
    // cambio rete
    /*
    rete.setNumeroStratiEIn(2,2);
    vector<int>
    rete.setOgniStrato()*/
    return 0;
}