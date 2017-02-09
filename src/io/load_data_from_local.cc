#include "load_data_from_local.h"

namespace dml{
void LoadData::load_all_data(){
    kv keyval;
    std::vector<kv> sample;
    m_data.fea_matrix.clear();
    while(!fin_.eof()){
        std::getline(fin_, line);
        sample.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
            pline += nchar;
            m_data.label.push_back(y);
            while(sscanf(pline, "%d:%ld:%d%n", &fgid, &fid, &val, &nchar) >= 3){
                pline += nchar;
                keyval.fgid = fgid;
                keyval.fid = fid;
                keyval.val = val;
                sample.push_back(keyval);
            }
        }
        m_data.fea_matrix.push_back(sample);
    }
    std::cout<<"size : "<<m_data.fea_matrix.size()<<std::endl;
}

void LoadData::load_minibatch_data(int num){
    kv keyval;
    std::vector<kv> sample;
    m_data.fea_matrix.clear();
    for(int i = 0; i < num; ++i){
        std::getline(fin_, line);
        if(fin_.eof()) break;
        sample.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
            pline += nchar;
            m_data.label.push_back(y);
            while(sscanf(pline, "%d:%ld:%d%n", &fgid, &fid, &val, &nchar) >= 3){
                pline += nchar;
                keyval.fgid = fgid;
                keyval.fid = fid;
                keyval.val = val;
                sample.push_back(keyval);
            }
        }
        m_data.fea_matrix.push_back(sample);
    }
}

void LoadData::load_all_hash_data(){
    kv keyval;
    std::vector<kv> sample;
    m_data.fea_matrix.clear();
    while(!fin_.eof()){
        std::getline(fin_, line);
        sample.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
            pline += nchar;
            m_data.label.push_back(y);
            while(sscanf(pline, "%s", fid_str) >= 1){
                pline += nchar;
                keyval.fid = h(fid_str);
                sample.push_back(keyval);
            }
        }
        m_data.fea_matrix.push_back(sample);
    }
    std::cout<<"size : "<<m_data.fea_matrix.size()<<std::endl;
}

void LoadData::load_mibibatch_hash_data(int num){
    kv keyval;
    std::vector<kv> sample;
    m_data.fea_matrix.clear();
    for(int i = 0; i < num; ++i){
        std::getline(fin_, line);
        if(fin_.eof()) break;
        sample.clear();
        const char *pline = line.c_str();
        if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
   	    pline += nchar;
            m_data.label.push_back(y);
	    while(sscanf(pline, "%s", fid_str) >= 1){
      		pline += nchar;
                keyval.fid = h(fid_str);
                sample.push_back(keyval);
            }
        }
        m_data.fea_matrix.push_back(sample);
    }
}

void LoadData::load_minibatch_hash_data_fread(){
    kv keyval;
    std::vector<kv> sample;
    m_data.label.clear();
    m_data.fea_matrix.clear();
    if(bmax < btop){
	memmove(&buf[0], &buf[bmax], (btop - bmax) * sizeof(char));
	//char *p = &buf[0];
	//std::cout<<"buf[0] = "<<std::string(&buf[0])<<std::endl;
    }
    btop -= bmax;
    btop += fread(&buf[btop], sizeof(char), buf.size() - 1 - btop, fp_);
    //std::cout<<"buffer_size = "<<buf.size()<<std::endl;
    //std::cout<<"btop = "<<btop<<std::endl;
    bmax = btop;
    if(btop + 1 == buf.size()){
	while(bmax > 0 && buf[bmax-1] != EOF && buf[bmax-1] != '\n') --bmax;
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
    //std::cout<<"bmax = "<<bmax<<std::endl;
 
    char *p = &buf[0];
    //std::cout<<"fisrt buf *p = "<<std::string(p)<<std::endl;
    while(*p != '\0'){
	char *q = p;
	while(*q != '\t') ++q;
        *q = '\0';
        float y_tmp = std::atof(p);
	if(y_tmp > 0.0000001) y = 1;
	else y = 0;
        m_data.label.push_back(y);
        //if(y != 0)std::cout<<"y="<<y<<std::endl;
        ++q;
        p = q;
        sample.clear();
        while(*q != '\n'){
            while(*q != ' ' && *q != '\n' && *q != '\0') ++q;
	    if(*q == '\n'){
		*q = '\0';
                //std::cout<<std::string(p)<<" ";
		keyval.fid = h(std::string(p));
		sample.push_back(keyval);
	        ++q;
		p = q;
		break;
  	    }
	    if(*q == '\0'){
                //std::cout<<std::string(p)<<" ";
                keyval.fid = h(std::string(p));
                sample.push_back(keyval);
                p = q;
                break;
            }
            *q = '\0';
	    //std::cout<<std::string(p)<<" ";
	    keyval.fid = h(std::string(p));
            sample.push_back(keyval);
            ++q;
	    p = q; 
        }//end while(*q != '\n')
        //std::cout<<"================================"<<std::endl;
        m_data.fea_matrix.push_back(sample);
        //break;
    }//end while(*p != '\0')
    //std::cout<<"------------------------------"<<std::endl;
}//load_minibatch_hash_data_fread

}//end namespace
