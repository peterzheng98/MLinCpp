//
// Created by 郑文鑫 on 2019/12/2.
//

#ifndef MLINCPP_EXCEPTION_H
#define MLINCPP_EXCEPTION_H
#include <exception>
#include <iostream>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
namespace peterzheng {
class exception : public std::exception {
private:
  std::string content, file, type;
  int line;
  static std::string getCurrentTimeStr() {
    time_t t = time(NULL);
    char ch[64] = {0};
    strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S",
             localtime(&t)); //年-月-日 时-分-秒
    return ch;
  }

public:
  ~exception() override {}
  const char *what() const noexcept override {
    return (getCurrentTimeStr() + "[" + type + ":" + file + "[" +
            std::to_string(line) + "]: " + content)
        .c_str();
  }
  void handler() {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
//    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);

  }
  exception(const std::string &content, const std::string &file,
            const std::string &type, int line)
      : content(content), file(file), type(type), line(line) {
    std::cerr << std::flush;
    handler();
    std::cerr << std::flush;
  }
};
} // namespace peterzheng

#endif // MLINCPP_EXCEPTION_H
