//
// Created by elia on 09/10/2017.
//

#ifndef RETENEURALE2_PERCEPTRON_H
#define RETENEURALE2_PERCEPTRON_H


#include "include.h"
using namespace std;

template <typename T2>
class Strato;

template <typename T>
class Perceptron{
public:
    Perceptron(){
        _zeri=4;
        _k=1; // pesi da 0 a 2
        _y=0;
        _isIn=false;
        srand(time(NULL));
    };
    //SETTER
    void setPesi(int n, T k,int n_zeri){
        T s;
        if(_w.size()!=0)
            _w.clear(), _dw.clear();
        int zeri=pow(10,n_zeri);
        int p=k;
        float h;
        for(int i=0; i<n; i++){
            s=(rand()%p);
            h=(rand()%zeri);
            h/=zeri;
            s+=(T)h;
            _w.push_back(s);
            _dw.push_back(0);
        }
        // BIAS
        s=(rand()%p);
        h=(rand()%zeri);
        h/=zeri;
        s+=h;
        _w.push_back(-s);
        _dw.push_back(0);
    }
    void setPesi(int n, T k){ setPesi(n,k,_zeri); };
    void setPesi(int n){ setPesi(n,_k,_zeri); };
    void inc_Peso(int i_peso, T value){
        if(i_peso>=_w.size())
            error(" inc_Peso (Perceptron): indice sbagliato");
        _w[i_peso] += value;
    }
    void setDw(int indice_strato_prima, T value){
        if(_dw.size()<indice_strato_prima+1 || indice_strato_prima<0)
            error("setDw: indice fuori");
        _dw[indice_strato_prima]=value;
    }
    // GETTER
    vector<T> getDw(){ return _dw;}
    vector<T> getPesi(){ return _w;}
    // PRINTER
    void printPesi(){
        for(typename vector<T>::iterator it=_w.begin(); it!=_w.end(); it++)
            cout << *it << " " ;
        cout << endl;
    }
    vector<string> printPesi2(){
        vector<string> im;
        for(typename vector<T>::iterator it=_w.begin(); it!=_w.end(); it++)
            im.push_back(ntostr(*it));
        cout << endl;
    }
    ///// COMUNI
    // SETTER
    T Output(vector<T>& v_in){
            if(v_in.size()!=_w.size()-1)
            error(" Output(Perceptron): input non sufficenti");
            vector<T> v_in2=v_in;
            v_in2.insert(v_in2.end(),1);
            _P=vectProd<T>(v_in2, _w,0);
            _isIn=true;
            _y=__SIGMOID__(_P);
            return _y;
    }
    T Output(vector<Perceptron<T>>& stratoPrima){
        vector<T> v_in;
        for(typename vector<Perceptron<T>>::iterator it=stratoPrima.begin(); it!=stratoPrima.end(); it++)
            v_in.push_back((*it).getOut());
        return Output(v_in);
    }
    T Output(Strato<T>& S){
        return Output(S.getStrato());
    }
    // GETTER
    void isIn(){
        if(!_isIn)
            error("isIn: input non dato");
    }
    T getOut(){ isIn();return _y;}

private:
    vector<T> _dw;
    vector<T> _w;
    int _zeri;
    int _k;
    bool _isIn;
    T _y;
    T _P;
};

template <typename T2>
class Strato{
    typedef typename vector<Perceptron<T2>>::iterator v_pi;
    typedef typename vector<T2>::iterator vi;
    typedef typename vector<vector<T2>>::iterator MPi;
    typedef vector<Perceptron<T2>> v_p;
    typedef vector<T2> v;
    typedef vector<vector<T2>> MP;
public:
    Strato(){
        _is_in=false;
        _n_pesi=0;
        _k=1;
        _zeri=4;
    };
    Strato(int n){
        for(int i=0; i<n; i++)
            _SP.push_back(Perceptron<T2>());
        _n_pesi=0;
        _is_in=false;
        _k=1;
        _zeri=4;
    };

    // SETTER
    void setN(int n){
        if(!_SP.empty())
            _SP.clear();
        for(int i=0; i<n; i++)
            _SP.push_back(Perceptron<T2>());
    }
    vector<T2> &Output(vector<T2>& in){
        _out.clear(), _Bout.clear() ;
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++)
            _out.push_back((*it).Output(in)), _Bout.push_back((*it).Output(in));
        _Bout.push_back(1);
        _is_in=true;
        return _out;
    }
    vector<T2> &Output(Strato<T2>& in){
        return Output(in.getOut());
    }
    void setPesi(int n,T2 k=1,int n_zeri=4){
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++)
            (*it).setPesi(n,k,n_zeri);
        _n_pesi=n;
        _k=k;
        _zeri=n_zeri;
    }
    void clearPesiRandom(){
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++)
            (*it).setPesi(_n_pesi,_k,_zeri);
    }
    void inc_peso(int i_p,int i_p_precedente,T2 value){
        if(i_p>=_SP.size())
            error("inc_peso (Strato): indice percettrone sbagliato ");
        _SP[i_p].inc_Peso(i_p_precedente,value);
    }
    void setDw(int i_p,int i_p_precedente,T2 value){
        if(i_p>=_SP.size())
            error("push_dw (Strato): indice fuori");
        _SP[i_p].setDw(i_p_precedente,value);
    }
    // GETTER
    int size(){ return _SP.size();}
    vector<T2> &getOut(){
        if(_out.size()==0)
            error("Output: non sono stati ancora dati i valori di input");
        return _out;
    }
    vector<T2> &getBOut(){
        if(_Bout.size()==0)
            error("Output: non sono stati ancora dati i valori di input");
        return _Bout;
    }
    vector<vector<T2>> getDw(){
        vector<vector<T2>> dw;
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++)
            dw.push_back((*it).getDw());
        return dw;
    }
    v_p getStrato(){ return _SP; }
    bool Isin(){ return _is_in;}
    vector<T2> getPesi(int i_p){
        if(i_p>=_SP.size())
            error("getPesi (Strato): indice fuori");
        return _SP[i_p].getPesi();
    }
    vector<T2> getPesiPrec(int i_p_precedente){
        vector<T2> pesi;
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++)
            pesi.push_back(((*it).getPesi())[i_p_precedente]);
        return pesi;
    }
    // PRINTER
    void printPesi(){
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++){
            cout << "pesi percettrone " << distance(_SP.begin(), it)+1 << " : " ;
            (*it).printPesi();
        }
    }
    void printPesiOutput(vector<T2>& in){
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++){
            cout << "pesi percettrone " << distance(_SP.begin(), it)+1 << " : " ;
            (*it).printPesi();
            cout << "output percettrone "  << distance(_SP.begin(), it)+1 << " : ";
            cout << (*it).Output(in) << endl;
        }
        _is_in=true;
    }
    void printOutput(){
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++){
            cout << "output percettrone"  << distance(_SP.begin(), it)+1 << " : ";
            cout << (*it).getOut() << endl;
        }
    }
    void printDw(){
        MP mdw=getDw();
        for(MPi it=mdw.begin(); it!=mdw.end(); it++){
            cout << "pesi percettrone "  << distance(mdw.begin(), it)+1 << " : ";
            for(vi it2=(*it).begin(); it2!=(*it).end(); it2++)
                cout << (*it2)<< " ";
            cout << endl;
        }
    }
    void printPesiOutput(){
        for(v_pi it=_SP.begin(); it!= _SP.end(); it++){
            cout << "pesi percettrone " << distance(_SP.begin(), it)+1 << " : " ;
            (*it).printPesi();
            cout << "output percettrone "  << distance(_SP.begin(), it)+1 << " : ";
            cout << (*it).getOut() << endl;
        }
    }
private:
    bool _is_in;
    int _n_pesi;
    int _k;
    int _zeri;
    vector<Perceptron<T2>> _SP; // strato di percettroni
    vector<T2> _out;
    vector<T2> _Bout;
};


#endif //RETENEURALE2_PERCEPTRON_H
