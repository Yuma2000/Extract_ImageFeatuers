
#ifndef UTIL_FUNC
#define UTIL_FUNC

#include <string>
#include <vector>

using namespace std;

bool check_exist(const char* target);
vector<string> split(string org, string separator);
string int2string(int a);
long get_timestamp();
double gettimeofday_sec();
void filecopy(string src, string dst);
void make_directory(string dirname);
void list_files(string file_restriction, vector<string> &listed_files);
void read_lines(string filename, vector<string> &lines);
double compute_eucldian_distance(vector<double> &v1, vector<double> &v2);
double logSumExp(vector<double> &v);

#endif

