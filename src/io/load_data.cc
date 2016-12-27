#include "load_data.h"

namespace dml{
void LoadData::load_all_data(){
    fea_matrix.clear();
    while(!fin_.eof()){
        std::getline(fin_, line);
        sample.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
            pline += nchar;
            label.push_back(y);
            while(sscanf(pline, "%d:%ld:%d%n", &fgid, &fid, &val, &nchar) >= 3){
                pline += nchar;
                keyval.fgid = fgid;
                keyval.fid = fid;
                keyval.val = val;
                sample.push_back(keyval);
            }
        }
        fea_matrix.push_back(sample);
    }
    std::cout<<"size : "<<fea_matrix.size()<<std::endl;
}

void LoadData::load_minibatch_data(int num){
    fea_matrix.clear();
    for(int i = 0; i < num; ++i){
        std::getline(fin_, line);
        if(fin_.eof()) break;
        sample.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
            pline += nchar;
            label.push_back(y);
            while(sscanf(pline, "%d:%ld:%d%n", &fgid, &fid, &val, &nchar) >= 3){
                pline += nchar;
                keyval.fgid = fgid;
                keyval.fid = fid;
                keyval.val = val;
                sample.push_back(keyval);
            }
        }
        fea_matrix.push_back(sample);
    }
}

void LoadData::load_mibibatch_hash_data(int num){
    fea_matrix.clear();
    for(int i = 0; i < num; ++i){
        std::getline(fin_, line);
        if(fin_.eof()) break;
        sample.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
   	    pline += nchar;
            label.push_back(y);
	    while(sscanf(pline, "%s", fid_str) >= 1){
      		pline += nchar;
                keyval.fid = h(fid_str);
                sample.push_back(keyval);
            }
        }
        fea_matrix.push_back(sample);
    }
}

void LoadData::load_minibatch_hash_data_fread(int bufsize){
    size_t btop, bmax;
    int buffer_size = bufsize << 20;
    std::vector<char> buf(buffer_size);
    if(bmax < btop){
	memmove(&buf[0], &buf[bmax], (btop - bmax) * sizeof(char));
    }
    btop -= bmax;
    btop += fread(&buf[btop], sizeof(char), buf.size() - 1 - btop, fp_);
    bmax = btop;
    if(btop + 1 == buf.size()){
	while(bmax > 0 && buf[bmax-1] != EOF && buf[bmax-1] != '\n' && buf[bmax-1] != '\r') --bmax;
        if(bmax != 0){
	    buf[bmax - 1] = '\0';
	}
	else{
	    bmax = btop;
	    buf[btop] = '\0';
  	}
    }
    else {
        buf[bmax] = '\0';	
    } 
    
    fea_matrix.clear();
    char *p = &buf[0];
    while(*p != '\0'){
	char *q = p;
	while(*q != '\t') ++q;
        y = std::atoi(p);
        p = q + 1;
        sample.clear();
        while(*q != '\n'){
            while(*q != ' ') ++q;
            *q = '\0';
	    keyval.fid = h(std::string(p));
	    p = q + 1; 
            sample.push_back(keyval);
        }//end while(*q != '\n')
        p = q + 1;
        fea_matrix.push_back(sample);
    }//end while(*p != '\0')
}//load_minibatch_hash_data_fread

}//end namespace
