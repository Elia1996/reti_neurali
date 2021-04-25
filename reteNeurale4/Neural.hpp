//
// Created by grafito on 25/02/18.
//

#ifndef RETENEURALE4_NEURAL_HPP
#define RETENEURALE4_NEURAL_HPP


#include "include.hpp"
#include "Perceptron.hpp"

using namespace std;

template <typename T>
class Neural{
    typedef typename vector<Strato<T>>::iterator v_Si;
    typedef vector<Strato<T>> v_S;
public:
    Neural(){
        _n_strati=0;
        _n_in=0;
        _n_out=0;
        _eta=0.4;
        _eta2=0.4;
        _max_ere= 100000;
        _max_error= 0.01;
        _debug=false;
    }
    Neural(int n_in, int n_strati,...){
        va_list strati;
        va_start(strati,n_strati);
        setNumeroStratiEIn(n_in,n_strati,strati);
        _eta=0.4;
        _eta2=0.4;
        _max_ere= 100000;
        _max_error= 0.01;
        _debug=false;
    }
    // SETTER
    void setNumeroStratiEIn(int n_in,int n_strati,... ){
        if(!_rete.empty())
            _rete.clear();
        // creo la lista di strati
        va_list strati;
        va_start(strati,n_strati);
        _n_in = n_in;
        _n_strati = n_strati;
        // creo la rete vuota
        vector<int> n_p_strato;
        // va_arg() ritorna ogni volta che è chiamta un nuovo elemento del vettore di strati
        for(int i=0;i<_n_strati; i++)
            n_p_strato.push_back(va_arg(strati,int));
        setNumeroStratiEIn(n_in,n_strati,n_p_strato);
    }
    void setNumeroStratiEIn(int n_in,int n_strati,vector<int>& strati ){
        if(!_rete.empty())
            _rete.clear();
        _n_in = n_in;
        _n_strati = n_strati;
        // creo la rete vuota
        vector<int> n_p_strato;
        n_p_strato.push_back(n_in);
        for(int i=0;i<_n_strati; i++)
            _rete.push_back(Strato<T>()), n_p_strato.push_back(strati[i]);
        _n_out= *--n_p_strato.end();
        // creo gli strati
        for (v_Si it = _rete.begin(); it != _rete.end(); it++) {
            int i = distance(_rete.begin(), it);
            // creo i percettroni
            (*it).setN(n_p_strato[i + 1]);
            // setto il n° di pesi per percettrone
            (*it).setPesi(n_p_strato[i], _k, _n_zeri);
        }
        // creo un vettore di vettori di sigma per più di due strati
        // la lunghezza di ogni vettore è pari al numero di P ci ciesuno strat
        if(!_sigma_strato.empty())
            _sigma_strato.clear();
        for (int k = 0; k < _rete.size(); k++) {
            _sigma_strato.push_back(vector<T>());
            for (int j = 0; j < _rete[k].size(); j++)
                _sigma_strato[k].push_back(0);
        }
    }
    void clearPesiRandom(){
        for(v_Si it=_rete.begin(); it!=_rete.end(); it++)
            (*it).clearPesiRandom();
    }
    // prende l'ingresso è ritorna l'uscita della rete
    vector<T>& run_rete(vector<T>& in){
        _rete[0].Output(in);
        for(v_Si it=++_rete.begin(), itback=_rete.begin(); it!=_rete.end(); it++, itback++){
            (*it).Output(*itback);
        }
    }
    void setKeZeri( T k ,int n_zeri){_k = k; _n_zeri= n_zeri;}
    void setEta(T eta){ _eta=eta;}
    void setEta2(T eta2){ _eta2=eta2;}
    void setMaxEre(int max_ere){ _max_ere= max_ere;}
    void setMaxError(double max_err){ _max_error= max_err;}
    void setMaxEreError(int max_ere,double max_err){_max_ere= max_ere; _max_error= max_err;}
    void setInput(vector<vector<T>> in){ _in = in;}
    void setOutput(vector<vector<T>> out){ _out = out;}
    void setinOut(vector<vector<T>> in,vector<vector<T>> out){_in=in,_out=out;}
    void setDebug(bool debug){ _debug = debug;}
    // GETTER
    vector<T> getOut(){
        return _rete[_n_strati-1].getOut();
    }
    // ritorna
    T back_propagation(){
        if(_in.size()!=_n_in || _out.size()!=_n_out)
            error("back_propagation: vettori sbagliati");
        // eseguo la rete sull'esempio
        run_rete(_in);
        // PARAMETRI CHE USO
        vector<vector<vector<T>>> dw;
        for(int i=0;i<_n_strati; i++)
            dw.push_back(_rete[i].getDw());
        T err=0;
        T dE_dy;
        vector<T> sigma_j(_n_out);
        vector<T> sigma_k(_rete[_n_strati-1-1].size());

        // ciclo sull'ultimo strato
        T delta=0, delta2=0;
        for(int j=0; j<_n_out ; j++){
            T j_out = (_rete[_n_strati-1].getOut())[j];  // output del j-esimo percettrone esterno
            dE_dy = -(_out[j] - j_out);   // -(dj-yj)
            err += dE_dy*dE_dy;  // error= (dj-yj)^2
            _sigma_strato[_n_strati-1][j] = dE_dy*__D_SIGMOID__(j_out);   // sigma_j= -(-(dj-yj))*T'(Pj) = -(-(dj-yj))*T(Pj)*(1-T(Pj))
            // ciclo per scandire tutti i pesi del percettrone e correggere i pesi
            int k;
            T delta=0, delta2=0;
            for(k=0; k<_rete[_n_strati-2].size()+1; k++){
                delta = -_eta*_sigma_strato[_n_strati-1][j]*(_rete[_n_strati-2].getBOut())[k];   // trovo il delta
                delta2 = _eta2*dw[_n_strati-1][j][k];  // trovo il delta che tiene conto del delta precedente
                _rete[_n_strati-1].setDw( j, k, delta + delta2 );  // SALVO Il DELTA W  per il passaggio successivo
                _rete[_n_strati-1].inc_peso( j, k, delta + delta2 );  // INCREMENTO IL PESO JK applicando la correzione deltaw trovata
            }
        }
        //cout <<left <<setprecision(4)<< "1:"<< setw(20)<<abs(delta/_rete[_n_strati-1-1].size())<< " 2:"<<setw(20)<< abs(delta2/_rete[_n_strati-1-1].size());
        //cout <<"e:"<<err<< endl;
        // se la rete ha più di due
        for(int s=_n_strati-2; s>0; s--) {
            // CICLO GENERALE PER UNO STRATO INTERMEDIO
            // k cicla sui percettroni dello strato s
            for(int k=0; k<_rete[s].size(); k++){
                T k_out = (_rete[s].getOut())[k]; // salvo le uscite del primo strato
                _sigma_strato[s][k] = (vectProd<T>(   _sigma_strato[s+1] ,_rete[s+1].getPesiPrec(k), 0))* __D_SIGMOID__(k_out);
                int i;
                // i cicla sui percettroni dello strato precedente (a sinistra) rispetto a s
                for(i=0; i< _rete[s-1].size()+1; i++){
                    _rete[s].setDw(k,i, -_eta*_sigma_strato[s][k]*(_rete[s-1].getBOut())[i]  + _eta2*dw[s][k][i]);
                    _rete[s].inc_peso(   k,i, -_eta*_sigma_strato[s][k]*(_rete[s-1].getBOut())[i]  + _eta2*dw[s][k][i] );
                }
            }
        }
        // ciclo sui percettroni primo strato a sinistra
        // 0-> _n_strati-1-1
        for(int k=0; k<_rete[0].size(); k++){
            T k_out = (_rete[0].getOut())[k];   // salvo le uscite del primo strato
            _sigma_strato[0][k] = (vectProd<T>(   _sigma_strato[_n_strati-1] ,_rete[_n_strati-1].getPesiPrec(k), 0))* __D_SIGMOID__(k_out);
            int i;
            for(i=0; i<_n_in; i++){
                _rete[0].setDw(k,i, -_eta*_sigma_strato[0][k]*_in[i]+ _eta2*dw[0][k][i]);
                _rete[0].inc_peso(   k,i, -_eta*_sigma_strato[0][k]*_in[i]+ _eta2*dw[0][k][i] );
            }
            _rete[0].setDw(k,i, -_eta*_sigma_strato[0][k]+ _eta2*dw[0][k][i]);
            _rete[0].inc_peso(   k,i, -_eta*_sigma_strato[0][k]+ _eta2*dw[0][k][i] );
        }
        return err;

    }
    // ritornano le ere eseguite
    T training( bool print, bool printRete=false, bool allError=false){
        assert(!_in.empty() && !_out.empty());
        T d_err;
        T error=10;
        // ere, ogni era prova tutti gli esempi e corregge i pesi, questo viene ripetuto a ogni era
        int i, iPrint=0;
        float zeri=10;
        if(print)
            cout << "---------- training ------------" << endl;
        for(i=0; i<_max_ere && error>_max_error ; i++, iPrint++ ){
            //cout << "################### ERA " << i << " ############################" << endl;
            error=0;
            // ciclo sugli esempi che ho e correggo i pesi
            for(int j=0; j<_in.size(); j++){
                // eseguo la backpropagation
                error+=this->back_propagation();
                // eseguo la rete
                this->run_rete(_in[j]);
            }
            if(print && (allError || error<(1/zeri)) && iPrint==10000)
                cout << "error: " <<error/4 << endl, zeri*=10, iPrint=0;

            //cout << "###################" << endl;
        }
        if(print) {
            cout << "era : " << i << endl;
            printOut(_in);
            if(printRete)this->printRete();
            cout << "------------- end --------------" << endl;
        }
        this->run_rete(_in[0]);
        return i;
    }
    // testa una variabile più volte per vedere come influenza la velocità della rete
    // le variabili sono: eta, eta2, numero di strati(decrescente)
    void testTrainingEta(T min, T max, int n_eta, int n_prove_per_ogni_eta, bool print=false, bool printRete=false, bool allError=false){
        // trovo la distanza fra due eta consecutive
        T pass = abs(max-min)/(n_eta-1);
        vector<T> ere(n_eta);
        vector<T> eta(n_eta);
        this->clearPesiRandom();
        // ciclo sugli eta
        for (int i = 0; i <n_eta ; ++i) {
            ere[i]=0;
            // setto eta
            eta[i]=min+pass*i;
            this->setEta(eta[i]);
            if(_debug){cout << eta[i] << endl;}
            // ciclo sulle prove per ogni eta
            for (int j = 0; j < n_prove_per_ogni_eta; ++j) {
                //addestro la rete
                ere[i] += training(print,printRete,allError);
                this->clearPesiRandom();
            }
            ere[i]/=n_prove_per_ogni_eta;
        }
        // ora in ogni elemento di ere ho le ere medie fatte per ogni eta
        // le stampo
        printStrEVect("eta",eta,"ere",ere);
    }
    void testTrainingEta2(T min, T max, int n_eta, int n_prove_per_ogni_eta, bool print=false, bool printRete=false, bool allError=false){
        // trovo la distanza fra due eta consecutive
        T pass = abs(max-min)/(n_eta-1);
        vector<T> ere(n_eta);
        vector<T> eta(n_eta);
        this->clearPesiRandom();
        // ciclo sugli eta
        for (int i = 0; i <n_eta ; ++i) {
            ere[i]=0;
            // setto eta
            eta[i]=min+pass*i;
            this->setEta2(eta[i]);
            if(_debug){cout << eta[i] << endl;}
            // ciclo sulle prove per ogni eta
            for (int j = 0; j < n_prove_per_ogni_eta; ++j) {
                //addestro la rete
                ere[i] += training(print,printRete,allError);
                this->clearPesiRandom();
            }
            ere[i]/=n_prove_per_ogni_eta;
        }
        // ora in ogni elemento di ere ho le ere medie fatte per ogni eta
        // le stampo
        printStrEVect("eta",eta,"ere",ere);
    }
    void testTrainintNStrati(int min, int max,int n_strati, bool print=false, bool printRete=false, bool allError=false){
        int n_prove;
        if(min>n_strati)
            n_prove=max-min;
        else
            n_prove=max-n_strati+1;
        vector<vector<int>> reti;

    }
    // PRINTER
    void printOut(vector<vector<T>>& in){
        for(int h=0; h < in.size(); h++) {
            this->run_rete(in[h]);
            cout << "in: ";
            for(int j=0; j < in[h].size() ; j++)
                cout << in[h][j] << " ";

            cout << "| out: ";
            vector<T> out = this->getOut();
            for(int j=0; j < out.size() ; j++)
                cout << out[j] << " " ;
            cout << endl;
        }
    }
    void printRete(){
        cout << "------------- STAMPO RETE NEURALE ----------" << endl;
        for(v_Si it=_rete.begin(); it!=_rete.end(); it++){
            cout << "STRATO " << distance(_rete.begin(),it)+1 << endl;
            if((*it).Isin())
                (*it).printPesiOutput();
            else
                (*it).printPesi();
        }
        cout << "--------------------------------------------" << endl;
    }
private:
    vector<vector<T>> _in;
    vector<vector<T>> _out;
    vector<vector<T>> _sigma_strato;
    vector<Strato<T>> _rete;
    double _max_error;
    int _max_ere;
    int _n_strati;
    int _n_in;
    int _n_out;
    T _k;
    int _n_zeri;
    T _eta;
    T _eta2;
    bool _debug;
};


#endif //RETENEURALE4_NEURAL_HPP
