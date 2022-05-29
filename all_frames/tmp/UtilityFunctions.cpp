
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "UtilityFunctions.h"

bool check_exist(const char* target){
	struct stat sb;
	if(stat(target, &sb) == -1)
		return false; // Not-exist
	else
		return true; // Exist
}

vector<string> split(string org, string separator){
	vector<string> res;
	string::size_type pos = 0, ppos = 0;
	while((pos = org.find(separator, pos)) != string::npos){
		if( pos - ppos > 0 ) // when the first letter is separator
			res.push_back(org.substr(ppos, pos - ppos));
		ppos = pos + 1;
		pos = ppos;
	}
	if(ppos < org.length()) // when the last letter is separator
		res.push_back(org.substr(ppos));
	return res;
}

string int2string(int a){
	char tmp[20];
	sprintf(tmp, "%d", a);
	return string(tmp);
}

long get_timestamp(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return long( (double)(tv.tv_sec)*1000.0 + (double)(tv.tv_usec)*0.001 );
}

double gettimeofday_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

void filecopy(string src, string dst){
	string cp_cmd = string("cp ") + src + string(" ") + dst;
	cout << "Copy file:" << cp_cmd << endl;
	if(system(cp_cmd.c_str()) != 0){
		cout << "ERROR: Cannot run file copy:" << cp_cmd << endl;
		exit(0);
	}
}

void make_directory(string dirname){
	string mkdir_cmd = string("mkdir ") + dirname;
	cout << "Create the directory by " << mkdir_cmd << endl;
	if(system(mkdir_cmd.c_str()) != 0){
		cout << "ERROR: Cannot make the directory:" << mkdir_cmd << endl;
		exit(0);
	}
}

bool cmp_id_ascend(const string &s1, const string &s2){
	
	string::size_type pos1_1 = s1.rfind(".", s1.length()-1);
	string::size_type pos1_2 = s1.rfind("-", pos1_1-1);
	string id1_tmp = s1.substr(pos1_2+1, pos1_1 - pos1_2 - 1);
	for(int i = 0; i < (int)id1_tmp.length(); i++){
		if(isdigit(id1_tmp[i]) == false){
			cout << "!!! ERROR: Fail to get a substring of numbers in " << s1 << endl;
			exit(0);
		}
	}
	int id1 = atoi(id1_tmp.c_str());
	
	string::size_type pos2_1 = s2.rfind(".", s2.length()-1);
	string::size_type pos2_2 = s2.rfind("-", pos2_1-1);
	string id2_tmp = s2.substr(pos2_2+1, pos2_1 - pos2_2 - 1);
	for(int i = 0; i < (int)id2_tmp.length(); i++){
		if(isdigit(id2_tmp[i]) == false){
			cout << "!!! ERROR: Fail to get a substring of numbers in " << s2 << endl;
			exit(0);
		}
	}
	int id2 = atoi(id2_tmp.c_str());
	
	//cout << s1 << "->" << id1 << "(" << id1_tmp << "), "
	//	<< s2 << "->" << id2 << "(" << id2_tmp << ")" << endl;
	
	return id1 < id2;
}

void list_files(string file_restriction, vector<string> &listed_files){
	
	string ls_cmd = string("ls ") + file_restriction;
	cout << ">> List files by " << ls_cmd << endl;
	FILE* list_res;
	if((list_res = popen(ls_cmd.c_str(), "r")) == NULL){
		cout << "!!! ERROR: Cannot list files by " << ls_cmd << endl;
		exit(0);
	}
	
	int file_id = 0;
	char one_res[1000];
	while( fgets(one_res, 999, list_res) != NULL && strlen(one_res) > 0 ){
		listed_files.push_back( strtok(one_res, "\n") );
		//cout << ">> " << file_id << "-th file:" << listed_files[file_id] << endl;
		file_id++;
	}
	
	pclose(list_res);
	
	string ls_cmd_wc = ls_cmd + string(" | wc -w");
	//cout << ">> Check # of listed files by " << ls_cmd_wc << endl;
	FILE* list_res_wc;
	if((list_res_wc = popen(ls_cmd_wc.c_str(), "r")) == NULL){
		cout << "!!! ERROR: Cannot list files (wc version) by " << ls_cmd_wc << endl;
		exit(0);
	}
	
	char file_num_tmp_ch[10];
	int check_tmp = fscanf(list_res_wc, "%s", file_num_tmp_ch);
	if(check_tmp != 1){
		cout << "!!! ERROR: fscanf may be failed (" << check_tmp << endl;
		exit(0);
	}
	int file_num_tmp = atoi(file_num_tmp_ch);
	//cout << "WC:" << file_num_tmp << endl;
	
	pclose(list_res_wc);
		
	if( (int)listed_files.size() != file_num_tmp){
		cout << "!!! ERROR: Inappropriate specification of # of listed files (# of listed files:"
			 << listed_files.size() << ", # of listed files by WC:" << file_num_tmp << ")" << endl;
		exit(0);
	}
	
	sort( listed_files.begin(), listed_files.end(), cmp_id_ascend );
	//exit(0);
	
}

void read_lines(string filename, vector<string> &lines){
	
	ifstream ifs(filename.c_str());
	if(ifs.fail()){
		cout << "!!! ERROR in openning the file " << filename << endl;
		exit(0);
	}
	
	int line_id = 0;
	string line;
	while(getline(ifs, line)){
		lines.push_back(line);
		line_id++;
	}
	
	//cout << ">> # of lines read = " << line_id << " (" << lines.size() << ")" << endl;
	//for(int i = 0; i < (int)lines.size(); i++){
	//	if(i % 10000 == 0 || i == (int)lines.size() - 1)
	//		cout << ">> " << i << "-th line: " << lines[i].substr(0, 50)
	//			<< " ... " << lines[i].substr(lines[i].length()-50) << endl;
	//}
	
}

double compute_eucldian_distance(vector<double> &v1, vector<double> &v2){
	double dist = 0;
	int nb_dims = (int)v1.size();
	for(int i = 0; i < nb_dims; i++)
		dist += ( (v1[i] - v2[i]) * (v1[i] - v2[i]) );
	return dist;
}

// log(exp(m1) + exp(m2)) = log(exp(m_1) + exp(m_1+(m_2 - m_1))
//   (m1 > m2)            = log( exp(m_1) * (1 + exp(sub)) )
//                        = log(exp(m_1)) + log(1+exp(sub))
//                        = m_1 + log(1+exp(sub))
double logSumExp(vector<double> &v){
	double m1, m2, sub;
	double Z = v[0];
	for(int i = 1; i < (int)v.size(); i++){
		if(Z >= v[i]){
			m1 = Z;
			m2 = v[i];
		}
		else{
			m1 = v[i];
			m2 = Z;
		}
		sub = m2 - m1;
		Z = m1 + log(1 + exp(sub));
	}
	return Z;
}

