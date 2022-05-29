
#include <iostream>
#include <vector>
#include "UtilityFunctions.h"

using namespace std;

int main(int argc, char* argv[]){
	
	if(argc != 2){
		cout << "Usage: ./main <Dir name>" << endl;
		return 1;
	}
	
	string targetDir = string(argv[1]);
	cout << ">> Target directory: " << targetDir << endl;

	vector<string> imgFilenames;
	list_files(targetDir + "/*.png", imgFilenames);
	for(int i = 0; i < (int)imgFilenames.size(); i++){
		
		cout << ">> " << i << "th filename: " << imgFilenames[i] << endl;
		
		string::size_type pos1 = imgFilenames[i].rfind(".", imgFilenames[i].length()-1);
        	string::size_type pos2 = imgFilenames[i].rfind("-", pos1-1);
        	string id_tmp = imgFilenames[i].substr(pos2+1, pos1 - pos2 - 1);
        	for(int j = 0; j < (int)id_tmp.length(); j++){
                	if(isdigit(id_tmp[j]) == false){
                        	cout << "!!! ERROR: Fail to get a substring of numbers in " << imgFilenames[i] << endl;
                        	exit(0);
                	}
        	}
        	int id = atoi(id_tmp.c_str());
		cerr << id << endl;
		
	}
	return 0;
	
}

