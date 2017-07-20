/*
 * logfilter.cpp
 *
 *  Created on: 20.07.2017
 *      Author: thies
 */

#include <iostream>
#include <fstream>

using namespace std;

int filter(int level, istream& in) {
   for (string line; getline(in, line);) {
      int my_level = 0;
      bool at_colon = false;
      bool contains_double_colon = false;

      for (auto c : line) {
         if (at_colon) {
            my_level++;
            if (c == ':') {
               contains_double_colon = true;
               break;
            } else
               at_colon = false;
         } else if (c == ':')
            at_colon = true;
      }

      if (!contains_double_colon)
         my_level = 0;

      if (my_level <= level) {
         cout << line << endl;
         cout.flush();
      }
   }

   return 0;
}

int print_usage(char* app_name) {
   cout << "Usage: " << endl;

   cout << "  " << app_name << "[level]            : filters stdin up to log level [level]" << endl;
   cout << "  " << app_name << "[level] [filename] : filters [filename] up to log level [level]" << endl;

   return -1;
}

int main(int argc, char** argv) {
   int level = 0;

   if (argc > 1)
      level = stoi(argv[1]);

   if (argc == 3) {
      ifstream ifile(argv[2]);
      if (ifile)
         return filter(level, ifile);
      else
         return print_usage(argv[0]);
   } else if (argc == 2)
      return filter(level, cin);
   else
      return print_usage(argv[0]);
}
