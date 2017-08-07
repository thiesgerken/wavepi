/*
 * Version.h
 *
 *  Created on: 07.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_VERSION_H_
#define INCLUDE_UTIL_VERSION_H_

#include <string>
#include <list>

namespace wavepi {
namespace util {

struct Version {
   public:
      static bool is_git_available();
      static bool is_git_working_dir_dirty();
      static std::string get_git_sha1_long();
      static std::string get_git_sha1();
      static std::string get_git_commit_subject();
      static std::string get_git_branch();
      static std::list<std::string> get_git_status();
      static std::list<std::string> get_untracked_files();
      static std::list<std::string> get_modified_files();

      static std::string get_build_date();
      static std::string get_build_type();

      static std::string get_infos();
      static std::string get_identification();

   private:
      static const std::string GIT_COMMIT_DESCRIPTION;
      static const std::string GIT_COMMIT_SUBJECT;
      static const std::string GIT_COMMIT_DATE;
      static const std::string GIT_BRANCH;
      static const std::string GIT_STATUS;

      static const std::string BUILD_DATE;
      static const std::string BUILD_TYPE;
};

} /* namespace util */
} /* namespace wavepi */

#endif /* INCLUDE_UTIL_VERSION_H_ */
